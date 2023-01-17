import pandas as pd
#.set_option('display.height', 1000)
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import re
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Read data
train = pd.read_csv('../input/train.csv')
train.columns = [c.lower() for c in train.columns]
test = pd.read_csv('../input/test.csv')
test.columns = [c.lower() for c in test.columns]
print('The dimensions of the train dataset are:',train.shape)
print('The dimensions of the test dataset are:',test.shape)
# Concatenate both dataset
y = train.target
train_objs_num = len(train)
data = pd.concat(objs=[train.drop('target',axis=1), test], axis=0)
data.sample(10)
cat_cols = list(data.select_dtypes(include=['object']).columns)
num_cols = list(data.select_dtypes(exclude=['object']).columns)
print('Numeric columns:',num_cols)
print('Categorical columns:',cat_cols)
data.info(verbose=True, null_counts=True)
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.set_context("paper")
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head(10)
print(data.rez_esc.unique())
data.rez_esc.value_counts()
plt.figure(figsize=(10,8))
sns.distplot(data.rez_esc.dropna())
data.rez_esc.fillna(99,inplace=True)
print(data.v18q1.unique())
data.v18q1.value_counts()
data.loc[data.v18q==0, 'v18q1'] = data.loc[data.v18q==0, 'v18q1'].fillna(0)
data.v18q1.fillna(data[:train_objs_num].v18q1.median(),inplace=True)
data.drop(columns='v18q',inplace=True)
plt.figure(figsize=(10,8))
sns.distplot(data.v2a1.dropna())
data['v2a1'].describe()
data.v2a1.fillna(data[:train_objs_num].v2a1.median(),inplace=True)
plt.figure(figsize=(10,8))
sns.distplot(data.meaneduc.dropna())
data.meaneduc.fillna(data[:train_objs_num].meaneduc.mean(),inplace=True)
data.sqbmeaned.fillna(data[:train_objs_num].meaneduc.mean()**2,inplace=True)
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)
print(data[cat_cols].nunique())
data[cat_cols].sample(6)
print(data.edjefe.unique())
print(data.edjefa.unique())
data['edjefe'] = pd.to_numeric(data.edjefe.apply(lambda x: 1 if x=='yes' else 0 if x=='no' else x))
data['edjefa'] = pd.to_numeric(data.edjefa.apply(lambda x: 1 if x=='yes' else 0 if x=='no' else x))
data['idhogar'] = data['idhogar'].astype('category').cat.codes
# Check if there is any data leak, household both in train and test set.
l = data[:train_objs_num]['idhogar'].unique()
p = data[train_objs_num:]['idhogar'].unique()
for i in l:
    if i in p:
        print('data leak')
print(data.dependency.unique())
cols = ['hogar_nin','hogar_adul','hogar_mayor','hogar_total','dependency']
print(data[data.dependency=='no'].hogar_nin.nunique())
print(data[data.dependency=='no'].hogar_mayor.nunique())
data[data.dependency=='no'][cols].head()
# We observe that dependecy = 'no' correspond to Number of children 0 to 19 & Number of individuals 65+ 
# in the household equal to 0.So
data.dependency.replace('no',0,inplace=True)
data.dependency.replace('yes',99,inplace=True)
data['dependency'] = pd.to_numeric(data.dependency)
data.isnull().values.any()
data.set_index('id',inplace=True)
data.head()
train_alt = data[:train_objs_num]
train_alt['target'] = y.values
corrmat = train_alt.corr()
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, vmax=.8, square=True);
mask = abs(train_alt.corr()['target']) >= 0.2
l = [c for c in mask[mask==True].index]
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(train_alt[l].corr(),linewidths=.5,cmap="YlGnBu")
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=42)
rfc.fit(data[:train_objs_num], y)

predictions = rfc.predict(data[train_objs_num:])
submission = pd.DataFrame({ 'Id': test.id,
                            'Target': predictions })
submission.to_csv("submission.csv", index=False)
submission.head()
