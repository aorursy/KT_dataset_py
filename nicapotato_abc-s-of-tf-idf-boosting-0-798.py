import time
start = time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import eli5

# models
from sklearn import feature_selection
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing
from scipy.sparse import hstack, csr_matrix
import scipy.stats as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import gc

# Load
train = pd.read_csv("../input/train.csv",index_col="id",low_memory=False,
                    parse_dates=["project_submitted_datetime"])#.sample(1000,random_state=23)
traindex = train.index
test = pd.read_csv("../input/test.csv",index_col="id",low_memory=False,
                    parse_dates=["project_submitted_datetime"])#.sample(1000,random_state=23)
tesdex = test.index
y = train["project_is_approved"].copy()
df = pd.concat([train.drop("project_is_approved",axis=1),test],axis=0)
alldex = df.index

# Resource DF
rc = pd.read_csv("../input/resources.csv",index_col="id").fillna("missingpotato")
# Percentages
print("Approved: {}%\nDenied: {}%".format(*y.value_counts(normalize=True)*100))

# Plot
f, ax = plt.subplots(figsize=[5,4])
sns.countplot(y,ax=ax, palette="rainbow")
ax.set_title("Dependent Variable Imbalance")
ax.set_ylabel("Count")
ax.set_xlabel("Project Approval Status\n0: Denied, 1: Approved")
plt.show()
# Aggregate and Merge
agg_rc = rc.reset_index().groupby('id').agg(
    dict(quantity = ['sum',"mean"],
         price = ["sum","mean","max","min","std"],
         id = 'count',
         description = lambda x: ' nicapotato '.join(x))).rename(columns={"id" : "count"})

# Collapse Multi-index
agg_rc.columns = pd.Index([e[0] +"_"+ e[1] for e in agg_rc.columns.tolist()])
agg_rc.rename({'count_count':"count",'description_<lambda>': "description"}, axis=1,inplace=True)
agg_rc.price_std.fillna(0,inplace=True)

# Merge
df = pd.merge(df,agg_rc, left_index=True, right_index=True, how= "left")
del test, train, rc,agg_rc
gc.collect()
# Feature Engineering
df['text'] = df.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4'])]), axis=1)
%%time

# Text Features for feature extraction
text_cols = ["text","project_resource_summary", "project_title", "description"]

# Sentiment Build
print("Hand Made Text Features..")
SIA = SentimentIntensityAnalyzer()
for cols in text_cols:
    df[cols] = df[cols].astype(str) # Make Sure data is treated as string
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols+'_num_chars'] = df[cols].apply(len) # Count number of Characters
    df[cols+'_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols+'_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    # Count Unique Words
    df[cols+'_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words']*100
    # Unique words to Word count Ratio
    if cols == "text":
        df[cols+"_vader_Compound"]= df[cols].apply(lambda x:SIA.polarity_scores(x)['compound'])
    #     df[cols+'_vader_Neutral']= df[cols].apply(lambda x:SIA.polarity_scores(x)['neu'])
        df[cols+'_vader_Negative']= df[cols].apply(lambda x:SIA.polarity_scores(x)['neg'])
        df[cols+'_vader_Positive']= df[cols].apply(lambda x:SIA.polarity_scores(x)['pos'])
        # Test a Stemmer..
    print("{} Done".format(cols))
# Date Cutoff Variable
df["Date_Cutoff"] = None
df.loc[df["project_submitted_datetime"] > "05/16/2016","Date_Cutoff"] = "After"
df.loc[df["project_submitted_datetime"] <= "05/16/2016","Date_Cutoff"] = "Before"

# Plot
f, ax = plt.subplots(1,4, figsize=[14,4])
for i, plotcol in enumerate(["text_num_words","text_num_unique_words","text_words_vs_unique","text_vader_Compound"]):
    sns.boxplot(data=df, y=plotcol,x="Date_Cutoff",ax=ax[i], palette="rainbow")
    ax[i].set_xlabel("May 17th 2016 Cutoff")
    ax[i].set_ylabel("{}".format(plotcol.replace(r'_',' ').capitalize()))
    ax[i].set_title("{}\nby Date Cutoff".format(plotcol.replace(r'_',' ').capitalize()))
plt.tight_layout(pad=0)
plt.show()
# Time Variables
df["Year"] = df["project_submitted_datetime"].dt.year
df["Date of Year"] = df['project_submitted_datetime'].dt.dayofyear # Day of Year
df["Weekday"] = df['project_submitted_datetime'].dt.weekday
df["Weekd of Year"] = df['project_submitted_datetime'].dt.week
df["Day of Month"] = df['project_submitted_datetime'].dt.day
df["Quarter"] = df['project_submitted_datetime'].dt.quarter

# Split the strings at the comma, and treat them as dummies
df = pd.merge(df, df["project_subject_categories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="left")
df = pd.merge(df, df["project_subject_subcategories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="left")
              
# Teacher ID
teachr_multi_subs = df['teacher_id'].value_counts().reset_index()
df["multi_apps"]= df['teacher_id'].isin(teachr_multi_subs.loc[teachr_multi_subs["teacher_id"]>1,'index'].tolist())
# Percentages
print("Teacher App Distribution:\nTwo Apps: {}%\nOne App: {}%\n".format(*df["multi_apps"].value_counts(normalize=True)*100))

# Teacher Gender
df["Gender"] = None
df.loc[df['teacher_prefix'] == "Mr.","Gender"] = "Male"
df.loc[df['teacher_prefix'] == "Teacher","Gender"] = "Not Specified"
df.loc[(df['teacher_prefix'] == "Mrs.")|(df['teacher_prefix'] == "Ms."),"Gender"] = "Female"

print("Gender Distribution:\nFemale: {}%\nMale: {}%\nNot Specified: {}%".format(*df["Gender"].value_counts(normalize=True)*100))
# Heatmap to see Gender and Teaching Grade of Teachers
f, ax = plt.subplots(figsize=[6,4])
sns.heatmap(pd.crosstab(df.project_grade_category,df.Gender,normalize='columns').mul(100).round(0),
            annot=True, linewidths=.5,fmt='g', cmap="rainbow", vmin=0, vmax=100,ax=ax,
            cbar_kws={'label': '% Percentage'})
ax.set_title("Percentage by Row of Gender in Grade Range")
ax.set_xlabel("")
ax.set_ylabel("")
plt.show;
dumyvars= ["Gender",'school_state','project_grade_category']
timevars = ['Weekday','Weekd of Year','Day of Month','Year','Date of Year',"Quarter"]
encode = ['multi_apps', "Date_Cutoff", 'teacher_prefix',"teacher_id"]

# Decided to go with only encoding, since most of the gradient boosting trees can handle categorical
categorical_features = dumyvars + timevars + encode

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical_features:
     df[col] = lbl.fit_transform(df[col].astype(str))

# Dummies:
# df = pd.get_dummies(df, columns=dumyvars+timevars)
# Text
text_cols = ["project_resource_summary", "project_title","description","text"]

df.drop(['project_subject_categories',"project_subject_subcategories","project_submitted_datetime",
         "project_essay_1","project_essay_2","project_essay_3","project_essay_4"
        ],axis=1,inplace=True)
normalize = ["teacher_number_of_previously_posted_projects","quantity","price"]
gc.collect()
# Lets look at these variables!
print("\nDtypes of DF features:\n",df.dtypes.value_counts())
print("\nDF Shape: {} Rows, {} Columns".format(*df.shape))
tfidf_para = {
    "sublinear_tf":True,
    "strip_accents":'unicode',
    "stop_words":"english",
    "analyzer":'word',
    "token_pattern":r'\w{1,}',
    #"ngram_range":(1,1),
    "dtype":np.float32,
    "norm":'l2',
    "min_df":5,
    "max_df":.9,
    "smooth_idf":False
}
# Thanks To
# https://www.kaggle.com/lopuhin/eli5-for-mercari
# https://www.kaggle.com/jagangupta/understanding-approval-donorschoose-eda-fe-eli5/notebook

def get_col(col_name):
    return lambda x: x[col_name]

df["project_title_count"] = df["project_title"].copy()
textcols = ["text","project_resource_summary","project_title", "project_title_count","description"]
vectorizer = FeatureUnion([
        ('text',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            **tfidf_para,
            preprocessor=get_col('text'))),
        ('project_resource_summary',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=2000,
            preprocessor=get_col('project_resource_summary'))),
        ('project_title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=1500,
            preprocessor=get_col('project_title'))),
        ('project_title_count',CountVectorizer(
            ngram_range=(1, 2),
            max_features=1500,
            preprocessor=get_col('project_title_count'))),
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=2400,
            preprocessor=get_col('description'))),
#         ('Non_text',DictVectorizer())
    ])
start_vect=time.time()
ready_df = vectorizer.fit_transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))
# # Sort and Print
# feature_array = np.array(vectorizer.get_feature_names())
# tfidf_sorting = np.argsort(ready_df.toarray()).flatten()[::-1]
# print("Most Important Words in All Vectorization:\n",feature_array[tfidf_sorting][:20])
df.drop(textcols,axis=1, inplace=True)
# Lets look at these variables!
print("\nDtypes of DF features:\n",df.dtypes.value_counts())
print("\nDF Shape: {} Rows, {} Columns".format(*df.shape))
# Get categorical var position for CatBoost
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
categorical_features_pos = column_index(df,categorical_features)
X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]])
testing = hstack([csr_matrix(df.loc[tesdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();
# Calculating level of imbalance for future models.
imbalance_weight = y.value_counts(normalize=True)[0]/y.value_counts(normalize=True)[1]
print("Imbalance Weight: ",imbalance_weight)
# Training and Validation Set
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=23)

# XGBOOST Sparse Feature Storage
d_train = xgb.DMatrix(X_train, y_train,feature_names=tfvocab)
d_valid = xgb.DMatrix(X_valid, y_valid,feature_names=tfvocab)
d_test = xgb.DMatrix(testing,feature_names=tfvocab)
xgb_params = {'eta': 0.05, 
              'max_depth': 12, 
              'subsample': 0.8, 
              'colsample_bytree': 0.75,
              #'min_child_weight' : 1.5,
              'scale_pos_weight': imbalance_weight,
              'objective': 'binary:logistic', 
              'eval_metric': 'auc', 
              'seed': 23,
              'lambda': 1.5,
              'alpha': .6
             }
modelstart = time.time()
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model = xgb.train(xgb_params, d_train, 2000, watchlist, verbose_eval=50, early_stopping_rounds=50)
xgb_pred = model.predict(d_test)

xgb_sub = pd.DataFrame(xgb_pred,columns=["project_is_approved"],index=tesdex)
xgb_sub.to_csv("xgb_sub.csv",index=True)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

del d_train, d_valid, d_test
gc.collect();
f, ax = plt.subplots(figsize=[7,10])
xgb.plot_importance(model,max_num_features=50,ax=ax)
plt.title("XGBOOST Feature Importance")
plt.show()
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 35,
    #'num_leaves': 500,
    'learning_rate': 0.01,
    'feature_fraction': 0.80,
    'bagging_fraction': 0.80,
    'bagging_freq': 5,
    'max_bin':300,
    #'verbose': 0,
    #'num_threads': 1,
    'lambda_l2': 1.5,
    #'min_gain_to_split': 0,
    'is_unbalance': True
    #'scale_pos_weight':0.15
}  

# LGBM dataset
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab,
                categorical_feature = categorical_features)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab,
                categorical_feature = categorical_features)

modelstart = time.time()
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=2000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=50,
    verbose_eval=150
)
lgbmpred = lgb_clf.predict(testing)
lgbm_sub = pd.DataFrame(lgbmpred,columns=["project_is_approved"],index=tesdex)
lgbm_sub.to_csv("lgbm_sub.csv",index=True)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
del lgvalid, lgtrain
gc.collect();
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.show()
print("Model Correlation: ", pd.Series(lgbmpred).corr(pd.Series(xgb_pred)))
blend = (lgbmpred*.55) + (xgb_pred*.45)
blend_sub = pd.DataFrame(blend,columns=["project_is_approved"],index=tesdex)
blend_sub.to_csv("boost_blend_sub.csv",index=True)
concat_subs = pd.concat([lgbm_sub,xgb_sub,blend_sub], axis=1)
concat_subs.columns = ["LGBM","XGB","BLEND"]
concat_subs.corr()
print("Notebook Runtime: %0.2f Minutes"%((time.time() - start)/60))