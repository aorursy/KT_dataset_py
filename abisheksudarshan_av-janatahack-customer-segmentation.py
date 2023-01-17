#for data processing

import numpy as np 

import pandas as pd



#for visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
train= pd.read_csv("../input/janatahack-customer-segmentation/Train.csv")

test= pd.read_csv("../input/janatahack-customer-segmentation/Test.csv")
train.head(2)
test.head(2)
train['train_y_n']=1

test['train_y_n']=0

all=pd.concat([train,test])
all.head()
#Visualization to check for missing values

sns.heatmap(all.isna())
all.info()
all.describe()
#ID

all['ID'].value_counts()>1
sum(all['ID'].value_counts()>1)
all[all['train_y_n']==0]['ID'].nunique()
2332/2627
all[all['ID']==462826]
sum(all.groupby(['ID','train_y_n'])['ID'].count()>1)
#Gender

sns.countplot(all['Gender'],hue=all['Segmentation'])
groupby_df = all[all['train_y_n']==1].groupby(['Gender', 'Segmentation']).agg({'Segmentation': 'count'})

groupby_pcts = groupby_df.groupby(level=0).apply(lambda x:round(100 * x / x.sum(),2))

groupby_df,groupby_pcts
#Ever_Married

sns.countplot(all['Ever_Married'],hue=all['Segmentation'])
groupby_df = all[all['train_y_n']==1].groupby(['Ever_Married', 'Segmentation']).agg({'Segmentation': 'count'})

groupby_pcts = groupby_df.groupby(level=0).apply(lambda x:round(100 * x / x.sum(),2))

groupby_df,groupby_pcts
sum(all['Ever_Married'].isnull())
#Age

sns.distplot(all['Age'])
sns.set_style('whitegrid')

sns.distplot(all[all['Segmentation']=='A']['Age'],bins=30,color='blue')

sns.distplot(all[all['Segmentation']=='B']['Age'],bins=30,color='red')

sns.distplot(all[all['Segmentation']=='C']['Age'],bins=30,color='green')

sns.distplot(all[all['Segmentation']=='D']['Age'],bins=30,color='black')

plt.legend(labels=['Seg=A', 'Seg=B', 'Seg=C','Seg=D'])
#Graduated

sns.countplot(all['Graduated'],hue=all['Segmentation'])
groupby_df = all[all['train_y_n']==1].groupby(['Graduated', 'Segmentation']).agg({'Segmentation': 'count'})

groupby_pcts = groupby_df.groupby(level=0).apply(lambda x:round(100 * x / x.sum(),2))

groupby_df,groupby_pcts
#Profession

plt.rcParams['figure.figsize'] = (10, 6)

sns.countplot(all['Profession'],hue=all['Segmentation'])
#Work_Experience

sns.countplot(all['Work_Experience'])
#Spending_Score

sns.countplot(all['Spending_Score'],hue=all['Segmentation'])
groupby_df = all[all['train_y_n']==1].groupby(['Spending_Score', 'Segmentation']).agg({'Segmentation': 'count'})

groupby_pcts = groupby_df.groupby(level=0).apply(lambda x:round(100 * x / x.sum(),2))

groupby_df,groupby_pcts
#Family_Size

sns.countplot(all['Family_Size'],hue=all['Segmentation'])
#Var_1

sns.countplot(all['Var_1'],hue=all['Segmentation'])
all.dtypes
sns.heatmap(all.corr(),annot=True)
feature_cols = all.columns.tolist()

feature_cols.remove('ID')

feature_cols.remove('Segmentation')

feature_cols.remove('train_y_n')

label_col = 'Segmentation'

print(feature_cols)
all.isnull().sum()
#Gender

all=pd.get_dummies(all,prefix='Gender',columns=['Gender'],drop_first=True)
all.head(2)
#Ever_Married

sns.countplot(all['Ever_Married'],hue=all['Family_Size'])
all[all['Ever_Married'].isnull()]['Family_Size'].value_counts()
all['Ever_Married']=all['Ever_Married'].fillna('Yes')
all=pd.get_dummies(all,prefix='Married',columns=['Ever_Married'],drop_first=True)
all.head(2)
#Graduated

sns.countplot(all['Graduated'])
all['Graduated']=all['Graduated'].fillna('Yes')
all=pd.get_dummies(all,prefix='Graduated',columns=['Graduated'],drop_first=True)

all.head(2)
#Profession

all['Profession'].fillna('Unknown',inplace=True)
all['Profession']=all['Profession'].astype('str')
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

all['Profession_en']=le.fit_transform(all['Profession'])
sns.countplot(all['Profession_en'],hue=all['Profession'])
all['Profession_en'].value_counts()
all.drop('Profession',axis=1,inplace=True)
#Work_Experience

all['Work_Experience'].fillna(all['Work_Experience'].mean(),inplace=True)
#Spending_Score

all.loc[all['Spending_Score']=='Low','Spending_Score']=1

all.loc[all['Spending_Score']=='Average','Spending_Score']=2

all.loc[all['Spending_Score']=='High','Spending_Score']=3

all['Spending_Score']=all['Spending_Score'].astype('int')
#Family_Size

all['Family_Size'].fillna(round(all['Family_Size'].mean()),inplace=True)
#Var_1

all['Var_1'].fillna('Cat_6',inplace=True)

all['Var_1']=all['Var_1'].apply(lambda x:x[-1])

all['Var_1']=all['Var_1'].astype('int')
#Train & Test Split

from sklearn.model_selection import train_test_split

df_train, df_eval = train_test_split(all[all['train_y_n']==1], test_size=0.40, random_state=101, shuffle=True, stratify=all[all['train_y_n']==1][label_col])
le = preprocessing.LabelEncoder()

df_train['Segmentation']=le.fit_transform(df_train['Segmentation'])

df_eval['Segmentation']=le.fit_transform(df_eval['Segmentation'])
df_train.info()
df_eval.info()
import lightgbm as lgb

from sklearn import preprocessing

from sklearn.metrics import mean_squared_log_error, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import classification_report

import seaborn as sns

from collections import Counter

sns.set_style('whitegrid')
params = {}

params['learning_rate'] = 0.04

params['max_depth'] = 18

params['n_estimators'] = 3000

params['objective'] = 'multiclass'

params['boosting_type'] = 'gbdt'

params['subsample'] = 0.7

params['random_state'] = 42

params['colsample_bytree']=0.7

params['min_data_in_leaf'] = 55

params['reg_alpha'] = 1.7

params['reg_lambda'] = 1.11

#params['class_weight']: {0: 0.44, 1: 0.4, 2: 0.37}
feature_cols = df_train.columns.tolist()

feature_cols.remove('ID')

feature_cols.remove('Segmentation')

feature_cols.remove('train_y_n')

label_col = 'Segmentation'

print(feature_cols)
cat_cols=['Spending_Score','Family_Size','Var_1','Gender_Male','Married_Yes','Graduated_Yes','Profession_en']
clf = lgb.LGBMClassifier(**params)

    

clf.fit(df_train[feature_cols], df_train[label_col], early_stopping_rounds=100, eval_set=[(df_train[feature_cols], df_train[label_col]), (df_eval[feature_cols], df_eval[label_col])], eval_metric='multi_error', verbose=True, categorical_feature=cat_cols)



eval_score = accuracy_score(df_eval[label_col], clf.predict(df_eval[feature_cols]))



print('Eval ACC: {}'.format(eval_score))
test=all[all['train_y_n']==0]

train=all[all['train_y_n']==1]
#Since there is big overlap between test and train, using train data for all the overlapping IDs

sub=pd.merge(left=test['ID'],right=train[['ID','Segmentation']],how='left',on='ID')
actual_test=(test[test['ID'].isin(train['ID'])==False])
actual_test.shape
pred=clf.predict(actual_test[feature_cols])
pred=le.inverse_transform(pred)

actual_test['Segmentation']=pred
l=actual_test[['ID','Segmentation']]

r=sub[sub['Segmentation'].isnull()==False]

fr=[l,r]

sub=pd.concat(fr)
sub[['ID','Segmentation']].to_csv('submission.csv',index = False)