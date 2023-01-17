%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler,LabelEncoder

from category_encoders import OneHotEncoder

from sklearn.model_selection import train_test_split

import seaborn as sns

import warnings

sns.set(style = 'darkgrid')

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df.head()
df.info()
df.describe(include='all')
workclass = df['workclass'].value_counts()

sns.countplot(df['workclass'],palette='icefire_r')

plt.xticks(rotation = 60)

print(workclass)
print(df['education'].value_counts())

plt.xticks(rotation = 60)

sns.countplot(df['education'])

plt.show()
print(df['marital.status'].value_counts())

sns.countplot(df['marital.status'])

plt.xticks(rotation = 60)

plt.show()
print(df['occupation'].value_counts())

sns.countplot(df['occupation'])

plt.xticks(rotation = 90)

plt.show()
print(df['relationship'].value_counts())

sns.countplot(df['relationship'])

plt.show()
print(df['race'].value_counts())

sns.countplot(df['race'])

plt.show()
print(df['sex'].value_counts())

sns.countplot(df['sex'])

plt.show()
print(df['income'].value_counts())

sns.countplot(df['income'])

plt.show()
print(df['native.country'].value_counts())

sns.countplot(df['native.country'])

plt.xticks(rotation = 90)

plt.show()
attrib, counts = np.unique(df['workclass'], return_counts = True)

most_freq_attrib = attrib[np.argmax(counts, axis = 0)]

df['workclass'][df['workclass'] == '?'] = most_freq_attrib 



attrib, counts = np.unique(df['occupation'], return_counts = True)

most_freq_attrib = attrib[np.argmax(counts, axis = 0)]

df['occupation'][df['occupation'] == '?'] = most_freq_attrib 
sns.distplot(df['hours.per.week'])

plt.show()

sns.violinplot(df['hours.per.week'],)

plt.show()
sns.distplot(df['age'])

plt.show()

sns.boxplot(df['age'])

plt.show()
df['income'] = LabelEncoder().fit_transform(df['income'])
sns.distplot(df['income'],kde = False)

plt.show()
sns.distplot(df['fnlwgt'])

plt.xticks(rotation = 90)

plt.show()

sns.boxplot(df['fnlwgt'])

plt.xticks(rotation = 90)

plt.show()
sns.distplot(df['education.num'])

plt.show()

sns.boxplot(df['education.num'])

plt.show()
sns.distplot(df['capital.gain'],kde = False)

plt.show()

sns.boxplot(df['capital.gain'])

plt.show()
sns.distplot(df['capital.loss'],kde = False)

plt.show()

sns.boxplot(df['capital.loss'])

plt.show()
sns.relplot('capital.gain','capital.loss',data = df,hue = 'income')

plt.show()
sns.catplot('income','capital.gain',data =df,kind = 'violin')

plt.show()
sns.catplot('income','capital.loss',data =df,kind = 'violin')

plt.show()
sns.catplot(y = 'fnlwgt',x = 'income',data = df,kind = 'violin')

plt.show()
sns.catplot(y = 'hours.per.week',x = 'income',data = df,hue = 'income',kind = 'violin')

plt.show()
sns.catplot(y = 'age',x= 'income',data = df,hue = 'income',kind = 'violin')

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot = True,cmap = 'vlag',linewidths=0.25)

plt.show()
df.head(20)
df.drop(['native.country'],axis = 1,inplace = True)

df.head()
sns.countplot('workclass',data = df,hue = 'income' )

plt.xticks(rotation = 60)

plt.show()

(df['workclass'].groupby(by = df['income']).value_counts())
sns.countplot('education',data = df,hue = 'income')

plt.xticks(rotation = 60)

plt.show()

(df['education'].groupby(by = df['income']).value_counts())
sns.countplot('education.num',data = df,hue = 'income')

plt.show()

df['education.num'].groupby(by = df['income']).value_counts()
sns.countplot('marital.status',data = df,hue = 'income')

plt.xticks(rotation = 60)

plt.show()

df['marital.status'].groupby(by = df['income']).value_counts()
sns.countplot('occupation',data = df,hue = 'income')

plt.xticks(rotation = 60)

plt.show()

df['occupation'].groupby(by= df['income']).value_counts()
sns.countplot('relationship',data = df,hue = 'income')

plt.xticks(rotation = 60)

plt.show()

df['relationship'].groupby(by= df['income']).value_counts()
sns.countplot('sex',data = df,hue = 'income')

plt.xticks(rotation = 60)

plt.show()

df['sex'].groupby(by= df['income']).value_counts()
sns.countplot('race',data = df,hue = 'income')

plt.show()

df['race'].groupby(by = df['income']).value_counts()
df.drop('education',axis = 1,inplace = True) ##It's already Ordered encoded in the column 'eduction-num'
df.workclass = df.workclass.map((df.workclass.value_counts()/len(df.workclass)).to_dict())*100
df.rename(columns={'marital.status' : 'marital_status'},inplace = True)
df.marital_status = df.marital_status.map((df['marital_status'].value_counts()/len(df['marital_status'])).to_dict())*100

df.occupation = df.occupation.map((df.occupation.value_counts()/len(df.occupation)).to_dict())*100

df.relationship = df.relationship.map((df.relationship.value_counts()/len(df.relationship)).to_dict())*100

df.race = df.race.map((df.race.value_counts()/len(df.race)).to_dict())*100



df.head()
df = pd.get_dummies(df)
df.head()
df['capital-change'] = df['capital.gain'] - df['capital.loss']

df.drop(['capital.gain','capital.loss'],axis = 1,inplace = True)
y = df.income

df.drop('income',axis = 1,inplace = bool(1))
xtrain,xtest,ytrain,ytest = train_test_split(df,y,random_state = 42)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape
from sklearn.ensemble import GradientBoostingClassifier

gbclf = GradientBoostingClassifier(random_state=42,n_estimators=300,max_depth=5,learning_rate=0.01)
gbclf.fit(xtrain,ytrain)
train_predict = gbclf.predict(xtrain)
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score

print(classification_report(ytrain,train_predict),accuracy_score(ytrain,train_predict),roc_auc_score(ytrain,train_predict))
test_predict = gbclf.predict(xtest)

print(classification_report(ytest,test_predict),accuracy_score(ytest,test_predict),roc_auc_score(ytest,test_predict))
(dict(sorted(zip(df.columns,gbclf.feature_importances_))))