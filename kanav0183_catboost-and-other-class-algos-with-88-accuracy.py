import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

data = pd.read_csv('../input/adult.csv')

print(data.shape)

data.count()[1]
data.head()
def cc(x):

    return sum(x=='?')

data.apply(cc)
data.loc[data.workclass == '?'].apply(cc)
data.groupby(by='workclass')['hours.per.week'].mean()
df = data[data.occupation !='?']
df.loc[df['native.country']!='United-States','native.country'] = 'non_usa'
for i in df.columns:

    if type(df[i][1])== str:

        print(df[i].value_counts())
df.columns
import seaborn as sns

fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(15,20))

plt.xticks(rotation=45)

sns.countplot(df['workclass'],hue=df['income'],ax=f)

sns.countplot(df['relationship'],hue=df['income'],ax=b)

sns.countplot(df['marital.status'],hue=df['income'],ax=c)

sns.countplot(df['race'],hue=df['income'],ax=d)

sns.countplot(df['sex'],hue=df['income'],ax=e)

sns.countplot(df['native.country'],hue=df['income'],ax=a)
fig, (a,b)= plt.subplots(1,2,figsize=(20,6))

sns.boxplot(y='hours.per.week',x='income',data=df,ax=a)

sns.boxplot(y='age',x='income',data=df,ax=b)
from sklearn.model_selection import train_test_split
df_backup =df
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in df.columns:

    df[i]=le.fit_transform(df[i])
import random

import sklearn

random.seed(100)

train,test = train_test_split(df,test_size=0.2)
l=pd.DataFrame(test['income'])

l['baseline'] =0

k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],l['baseline']))

print(k)

(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

x=train.drop('income',axis=1)

y=train['income']

clf.fit(x,y)
clf.score(x,y)
pred = clf.predict(test.drop('income',axis=1))
import sklearn

k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred))

print(k)
(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])
y_score = clf.fit(x,y).decision_function(test.drop('income',axis=1))



fpr,tpr,the=sklearn.metrics.roc_curve(test['income'],y_score)

sklearn.metrics.roc_auc_score(test['income'],pred)

plt.plot(fpr,tpr,)
sklearn.metrics.roc_auc_score(test['income'],y_score)
col=['age','fnlwgt','capital.gain','capital.loss','hours.per.week','education','education.num','marital.status','relationship','sex']
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(x)





from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=14)

Y_sklearn = sklearn_pca.fit_transform(X_std)



cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()



sklearn_pca.explained_variance_ratio_[:10].sum()



cum_sum = cum_sum*100



fig, ax = plt.subplots(figsize=(8,8))

plt.bar(range(14), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color = 'b',alpha=0.5)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_features=14,min_samples_leaf=100,random_state=10)

clf.fit(x,y)
pred2 = clf.predict(test.drop('income',axis=1))
k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))

print(k)
(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])
from xgboost import XGBClassifier



clf= XGBClassifier()



clf.fit(x,y)



pred2 = clf.predict(test.drop('income',axis=1))



k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))





(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])
from catboost import CatBoostClassifier



clf= CatBoostClassifier(learning_rate=0.04)



clf.fit(x,y)



pred2 = clf.predict(test.drop('income',axis=1))



k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))





(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])
clf.score(x,y)