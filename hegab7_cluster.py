import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
pd.set_option('display.max_columns',None)
df=pd.read_csv('/kaggle/input/onelasttrain/TRAIN.csv',na_values=['None','?'])
df.head()
df.info()
miss_val=(df.isna().sum()/len(df)*100).sort_values(ascending=False)

miss_val[miss_val>0]
dropped=[]

for col in df.columns:

    if df[col].isna().sum() > 20000:

        dropped.append(col)
df=df.drop(columns=dropped)
df.describe()
df.head()
df['readmitted_NO'].value_counts()
df.gender.value_counts()
df[df['gender']=='Unknown/Invalid']
df=df.drop(30506)

df=df.reset_index(drop=True)
pd.crosstab(df.age,df.readmitted_NO)
pd.crosstab(df.diabetesMed,df.insulin)
pd.crosstab(df.diabetesMed,df.readmitted_NO)
plt.rcParams['figure.figsize'] = (10,6)

sns.countplot(df['number_diagnoses'],palette='twilight_shifted')

plt.show()
sns.countplot(df['time_in_hospital'])

plt.show()
sns.countplot(df.race,palette='Spectral')

plt.show()
sns.countplot(df.age,palette = 'Dark2_r')

plt.show()
df.info()
df.race=df.race.fillna(df.race.mode())
df.diag_1=pd.to_numeric(df.diag_1,errors='coerce')

df.diag_2=pd.to_numeric(df.diag_2,errors='coerce')

df.diag_3=pd.to_numeric(df.diag_3,errors='coerce')
sns.set(style = 'whitegrid')

sns.distplot(df.diag_1)

plt.show()
sns.set(style = 'whitegrid')

sns.distplot(df.diag_2)

plt.show()
sns.set(style = 'whitegrid')

sns.distplot(df.diag_3)

plt.show()
df.diag_1=df.diag_1.fillna(df.diag_1.mean())

df.diag_2=df.diag_2.fillna(df.diag_2.mean())
df.diag_3.mode()
df.diag_3=df.diag_3.fillna(250)
fig,(ax1,ax2)=plt.subplots(1,2)

ax1.scatter(x='num_lab_procedures',y='num_medications',data=df[df['readmitted_NO']==0],color='b')

ax2.scatter(x='num_lab_procedures',y='num_medications',data=df[df['readmitted_NO']==1],color='r')

ax1.set_xlabel('Number of lab procedures')

ax1.set_ylabel('Number of medications')

plt.show()
sns.stripplot(df['age'], df['num_lab_procedures'], palette = 'Purples', size = 10)

plt.show()
sns.boxplot(df.readmitted_NO, df.num_lab_procedures)

plt.show()
cat=['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',

     'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',

     'troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin',

    'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone',

     'metformin-pioglitazone','change','diabetesMed','race','gender','age']
for col in cat:

    df[col]=df[col].astype('category')

    df[col]=df[col].cat.codes
df.head()
df.info()
from sklearn.decomposition import PCA

pca=PCA()
pca.fit(df)
transformed=pca.transform(df)
features=range(pca.n_components_)

plt.bar(features,pca.explained_variance_)

plt.xticks(features)

plt.show()
from sklearn.feature_selection import SelectKBest   #for feature selection

from sklearn.feature_selection import f_classif
test = SelectKBest(score_func=f_classif, k=20)

fit = test.fit(df.drop(columns=['readmitted_NO']), df['readmitted_NO'])

print(sorted(zip(fit.scores_,df.columns),reverse=True))
to_drop=['metformin-pioglitazone','metformin-rosiglitazone','glimepiride-pioglitazone',

       'chlorpropamide','troglitazone','insulin','acetohexamide','glipizide-metformin',

       'tolbutamide','glimepiride','glyburide-metformin','citoglipton','examide','miglitol',

       'diag_1','tolazamide','admission_type_id','rosiglitazone','nateglinide','diag_2',

       'glyburide','acarbose','glipizide']
data=df.drop(columns=to_drop)
from sklearn.preprocessing import StandardScaler
data_scale=StandardScaler().fit_transform(data)
data_scale=pd.DataFrame(data_scale,columns=data.columns)
x_scale=data_scale.drop(columns=['readmitted_NO'])
x=data.drop(columns=['readmitted_NO'])
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
model.fit(x)

labels=model.predict(x)
pd.crosstab(labels,df['readmitted_NO'])
metrics.accuracy_score(labels,df['readmitted_NO'])
model.fit(x_scale)
labels_scale=model.predict(x_scale)
pd.crosstab(labels_scale,df['readmitted_NO'])
from sklearn import metrics

metrics.accuracy_score(labels_scale,df['readmitted_NO'])
from scipy.cluster.vq import whiten
data_whiten=whiten(data)

data_whiten=pd.DataFrame(data_whiten,columns=data.columns)
x_whiten=data_whiten.drop(columns=['readmitted_NO'])
model.fit(x_whiten)

labels_whiten=model.predict(x_whiten)
pd.crosstab(labels_whiten,df['readmitted_NO'])
metrics.accuracy_score(labels_whiten,df['readmitted_NO'])
test=pd.read_csv('/kaggle/input/onelasttrain/TEST.csv',na_values=['None','?'])
test.info()
test=test.drop(columns=dropped)

test.race=test.race.fillna('Caucasian')

test.diag_1=pd.to_numeric(test.diag_1,errors='coerce')

test.diag_2=pd.to_numeric(test.diag_2,errors='coerce')

test.diag_3=pd.to_numeric(test.diag_3,errors='coerce')

test.diag_1=test.diag_1.fillna(test.diag_1.mean())

test.diag_2=test.diag_2.fillna(test.diag_2.mean())

test.diag_3=test.diag_3.fillna(250)
cat=['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',

     'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',

     'troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin',

    'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone',

     'metformin-pioglitazone','change','diabetesMed','race','gender','age']

for col in cat:

    test[col]=test[col].astype('category')

    test[col]=test[col].cat.codes
to_drop=['metformin-pioglitazone','metformin-rosiglitazone','glimepiride-pioglitazone',

       'chlorpropamide','troglitazone','insulin','acetohexamide','glipizide-metformin',

       'tolbutamide','glimepiride','glyburide-metformin','citoglipton','examide','miglitol',

       'diag_1','tolazamide','admission_type_id','rosiglitazone','nateglinide','diag_2',

       'glyburide','acarbose','glipizide']

test=test.drop(columns=to_drop)
test=test.drop(columns=['index'])
test_w=whiten(test)

test_w=pd.DataFrame(test_w,columns=test.columns)

target=model.predict(test_w)
sub=pd.DataFrame(target,columns=['target'])
sub=sub.reset_index()
sub.to_csv('submission_c.csv',index=False)