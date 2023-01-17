#Import needed libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import lightgbm as lgb

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

#Input data
df = pd.read_csv('../input/voice.csv')
df.head()
#Check ratio of classes in dependent variable
gender_dict = {'male': 0,'female': 1}
gender_list = [gender_dict[item] for item in df.label]

print('Ratio between two classes is:', np.mean(gender_list))
#Lets check the distribution of meanfreq (mean vs women)
plt.figure()
sns.kdeplot(df['meanfreq'][df['label']=='male'], shade=True);
sns.kdeplot(df['meanfreq'][df['label']=='female'], shade=True);
plt.xlabel('meanfreq value')
plt.show()
#Print mean of each category
print('Male mean frequency:', np.mean(df['meanfreq'][df['label']=='male']))
print('Female mean frequency:', np.mean(df['meanfreq'][df['label']=='female']))
#Run the student's t-test
male_df = df[df['label']=='male']
female_df = df[df['label']=='female']
t2, p2 = stats.ttest_ind(male_df['meanfreq'], female_df['meanfreq'])

print('P-value:', p2)
#Check the distribution of meanfun (mean vs women)
plt.figure()
sns.kdeplot(df['meanfun'][df['label']=='male'], shade=True);
sns.kdeplot(df['meanfun'][df['label']=='female'], shade=True);
plt.show()
#Lets construct a heatmap to see which variables are very correlated
no_label_data = df.drop(['label'], axis = 1)
cors = no_label_data.corr()

plt.figure(figsize=(12,7))
sns.heatmap(cors, linewidths=.5, annot=True)
plt.show()
plt.figure(figsize=(8,7))
sns.boxplot(x="label", y="dfrange", data=df)
plt.show()
plt.figure(figsize=(8,7))
sns.violinplot(x="label", y="meanfun", data=df)
plt.show()
sns.lmplot( x="sfm", y="meanfreq", data=df, fit_reg=False, hue='label', legend=False)
plt.show()
#Lets select fewer variables
no_label_data_red = no_label_data[['median', 'skew', 'kurt', 'sp.ent', 
                                   'sfm', 'centroid', 'dfrange', 'modindx']]
sns.pairplot(no_label_data_red)
plt.show()
sns.jointplot(x=df["centroid"], y=df["sfm"], kind='scatter',
              color='m', edgecolor="skyblue", linewidth=1)

plt.show()
#Jointplot alternative: 'hex'. Easily identify the area where two variables are forming a 'cloud' (the peak of their distributions)
sns.jointplot(x=df["centroid"], y=df["sfm"], kind='hex',
              color='m', edgecolor="skyblue", linewidth=1)

plt.show()
#Shuffle data 
df = df.sample(frac=1, random_state = 42)
#Check missing 
df.isnull().sum().sum()
#Replace zeros with null
df.replace(0, np.nan, inplace=True)
#Now lets count nulls again 
df_null = df.isnull().sum()
plt.figure(figsize=(8,7))
g = sns.barplot(df_null.index, df_null.values)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.ylabel('No of missing values')
plt.show()
#Convert text target variable to number
gender_dict = {'male': 0,'female': 1}
gender_list = [gender_dict[item] for item in df.label]

df_final = df.copy() 
df_final['label'] = gender_list
#Split in train and test
train_df, test_df = train_test_split(df_final, test_size=0.2, random_state = 14)
#Get input and output variables
train_x = train_df.drop(['label'], axis = 1)
train_y = train_df['label']

test_x = test_df.drop(['label'], axis = 1)
test_y = test_df['label']
#LGB model
lgb_train = lgb.Dataset(train_x, train_y)

# Specify hyper-parameters as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 16,
    'max_depth': 6,
    'learning_rate': 0.1,
    #'feature_fraction': 0.95,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 5,
    #'reg_alpha': 0.1,
    #'reg_lambda': 0.1,
    #'is_unbalance': True,
    #'num_class': 1,
    #'scale_pos_weight': 3.2,
    'verbose': 1,
}

# Train LightGBM model
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=90,
                #valid_sets= lgb_valid,
                #early_stopping_rounds=40,
                verbose_eval=20
                )
# Plot Importances
print('Plot feature importances...')
importances = gbm.feature_importance(importance_type='gain')  # importance_type='split'
model_columns = pd.DataFrame(train_x.columns, columns=['features'])
feat_imp = model_columns.copy()
feat_imp['importance'] = importances
feat_imp = feat_imp.sort_values(by='importance', ascending=False)
feat_imp.reset_index(inplace=True)

plt.figure()
plt.barh(np.arange(feat_imp.shape[0] - 1, -1, -1), feat_imp.importance)
plt.yticks(np.arange(feat_imp.shape[0] - 1, -1, -1), (feat_imp.features))
plt.title("Feature Importances")
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
pred_lgb = gbm.predict(test_x,num_iteration=50)
pred_01 = np.where(pred_lgb > 0.5, 1, 0)
recall_pred = recall_score(test_y, pred_01)
precision_pred = precision_score(test_y, pred_01)
accuracy_pred = accuracy_score(test_y, pred_01)
print('Recall score: %0.2f' %recall_pred)
print('Precision score: %0.2f' %precision_pred)
print('Overall Accuracy: %0.2f' %accuracy_pred)
confusion_matrix(test_y, pred_01)
