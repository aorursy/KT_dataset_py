import pandas as pd
import numpy as np
import pandasql as ps
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
from os import listdir
from os.path import isfile, join
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
# make plots
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
df_raw = pd.read_csv("/Users/karim.arem@ibm.com/Documents/hackathon_2020/2020caohackathon/train_set_usd.csv")
df_test = pd.read_csv("/Users/karim.arem@ibm.com/Documents/hackathon_2020/2020caohackathon/test_set_usd.csv")
df_train = df_raw.drop(columns=['outcome'])
df_train['train_test']='train'
df_test['train_test']='test'
df_all = pd.concat([df_train, df_test], axis=0)
df = df_all
#date related features
from datetime import datetime
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce') #create a website for funding
df['launched_at'] = pd.to_datetime(df['launched_at'], errors='coerce') # lauch the effort to apply fund
df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce') # the end of date of funding effort
from datetime import datetime
# number of time between two days
#col'duration' - it's the # of days the creator takes effort for funding;
#'duration' = (df['deadline'].sub(df['launched_at']))
#The tough the dealine is, the more difficult to win

#The earlier lauched, the more confident to win
df['daysdif_created_to_launched'] = round((df['launched_at'].sub(df['created_at'])) / np.timedelta64(1, 'D'),0)
# add # of times the creator tried
effort= df.groupby(['creator_id','train_test']).agg({"id": pd.Series.nunique}).reset_index().rename(columns={'id':'effort_times'})
df=df.merge(effort, on=['creator_id','train_test'], how='left')
print(df[['id','train_test','creator_id','effort_times']].head())
# determine categorical and numerical features
categorical_cols = ['train_test','main_category', 'sub_category', 'country', 'location_state'
                    , 'location_type','staff_pick', 'disable_communication'
                    , 'show_feature_image','creator_register','launch_year','deadline_year','deadline_month','launch_month']
numeric_cols = ['goal_usd','effort_times','duration','daysdif_created_to_launched','sadness', 'joy'
                , 'fear', 'disgust', 'anger', 'sentiment']
other_cols = ['id','created_at', 'deadline','launched_at', 'country','duration']
index_col = ['id','name', 'blurb', 'slug','currency', 'currency_trailing_code'
             ,'creator_name','creator_id']
text_col = ['blurb']

label=['outcome']
df = df[categorical_cols+numeric_cols+['id']+text_col]
# percentage of missing values
# usually should delete the variables with over 15% of missing data
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
ax = (df.isnull().sum()/df.shape[0]).plot.bar()
pd.set_option('display.max_rows', 80)
df.info()
df.describe()
# check for distributions for all numerics
f = pd.melt(df, value_vars=numeric_cols)
g = sns.FacetGrid(f, col="variable",  col_wrap=8, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
# # fill in missing value by sub_category for training set
['sadness', 'joy', 'fear', 'disgust', 'anger', 'sentiment']
df['sadness'] = df['sadness'].fillna(df.groupby('sub_category')['sadness'].transform('median'))
df['joy'] = df['joy'].fillna(df.groupby('sub_category')['joy'].transform('median'))
df['fear'] = df['fear'].fillna(df.groupby('sub_category')['fear'].transform('median'))
df['disgust'] = df['disgust'].fillna(df.groupby('sub_category')['disgust'].transform('median'))
df['anger'] = df['anger'].fillna(df.groupby('sub_category')['anger'].transform('median'))
df['sentiment'] = df['sentiment'].fillna(df.groupby('sub_category')['sentiment'].transform('median'))
# check for distributions for all numerics again
f = pd.melt(df, value_vars=numeric_cols)
g = sns.FacetGrid(f, col="variable",  col_wrap=8, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
from sklearn.pipeline import FeatureUnion

class TextTransformer(BaseEstimator, TransformerMixin):
    """
    Transform text features
    """
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None, *parg, **kwarg):
        return self
    def transform(self, X):
        return X[self.key]
vec_tdidf = TfidfVectorizer(ngram_range=(1,2), analyzer='word', norm='l2')    
blurb = Pipeline([
                ('transformer', TextTransformer(key = 'blurb')),
                ('vectorizer', vec_tdidf)
                ])
categorical_features = df[categorical_cols]
numeric_features = df[numeric_cols]

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'))])

numeric_transformer = Pipeline([
    ('standard_scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
        #,('text', blurb, text_cols)
    ])

xg = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=20, random_state=42,
                    max_leaf_nodes=2, subsample=0.8,
                    learning_rate=0.005,
                    scale_pos_weight=0.78)

features = FeatureUnion([('prep',preprocessor), 
                         ('blurb', blurb)
                      ])


# define the data preparation and modeling pipeline
pipeline = Pipeline(steps=[('feat', features), ('m', xg)])
df_cleaned = pd.concat([df,df_raw['outcome']], axis=1)
X=df[df['train_test']=='train']
y=df_raw['outcome']
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# fit model
#pipeline.fit(X_train, y_train, **{'m__sample_weight': sample_weight}) #can do a sample weighting if time is enough
pipeline.fit(X_train, y_train)
# pipeline.fit(X, y)
# make predictrions on test set
predictions = pipeline.predict_proba(X_test)
accuracy = pipeline.score(X_test, y_test)
print("Accuracy:", round(accuracy, 3)*100, "%")
from sklearn import metrics
predictions = pipeline.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test,predictions[:,1])
Auc=metrics.auc(fpr, tpr)
print(Auc)
from sklearn.metrics import classification_report
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
##predict the test_set
df_test_submit=df[df['train_test']=='test']
y_predict_df = pipeline.predict_proba(df_test_submit)
# output result
y_predict_df= pd.DataFrame(y_predict_df[:,1], columns=['Predicted'])
y_predict_df.to_csv('y_test_file.csv')
df_test.shape
df_test['Id']=df_test['id']
y_predict_df1 = pd.concat([df_test['Id'],y_predict_df], axis=1)
y_predict_df1.to_csv("Preditions_Submit_Team6_v7.csv", index=False)
# num_col = pipeline.named_steps['prep'].transformers_[0][2]
# cat_col = pipeline.named_steps['prep'].transformers_[1][1].named_steps['one_hot_encoding'].get_feature_names()
# cat_original_col = pipeline.named_steps['prep'].transformers_[1][2]
# num_col
# cat_original_col
# cat_col
# def find_col_name(x):
#     for i in actual_cat_name:
#         if i in x:
#             return '_'.join(str(x).split('_')[:-1])
#     return x
# actual_cat_name = []
# for i in cat_col:
#     x, y = i.split('_', 1)
#     cat_name = cat_original_col[int(x[1:])]
#     actual_cat_name.append(cat_name + '_' + y)
# cols = list(num_col) + list(actual_cat_name)
# print(cols)
# f = pipeline.steps[1][1].feature_importances_
# print(len(f))

# feats = {}  # a dict to hold feature_name: feature_importance
# for feature, importance in zip(cols, f):
#     feats[feature] = importance  # add the name/value pair

# importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
# importances.sort_values(by='Gini-importance', ascending=False)
# #print("Importances",importances)
# importances.to_csv('importance.csv')
# ### summarized feature importance
# importances_df = importances
# importances_df = importances_df.reset_index()
# importances_df['f_name'] = importances_df['index'].apply(lambda x: find_col_name(x))
# importances_df.columns = ['index', 'importance', 'f_name']
# importances_sum = importances_df.groupby(['f_name'])['importance'].sum()
# importances_sum = importances_sum.reset_index()
# importances_sum = importances_sum.sort_values(by='importance', ascending=False).reset_index()
# print(importances_sum)

