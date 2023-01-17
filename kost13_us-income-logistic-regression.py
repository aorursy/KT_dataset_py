import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.preprocessing import StandardScaler



columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',

           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',

           'Hours/Week','Native country','Income']



train = pd.read_csv('../input/adult-training.csv', names=columns)

test = pd.read_csv('../input/adult-test.csv', names=columns, skiprows=1)



%matplotlib inline
train.head()
train.info()
test.info()
train.replace(' ?', np.nan, inplace=True)

test.replace(' ?', np.nan, inplace=True)
train.isnull().sum()
test.isnull().sum()
train['Income'] = train['Income'].apply(lambda x: 1 if x==' >50K' else 0)

test['Income'] = test['Income'].apply(lambda x: 1 if x==' >50K.' else 0)
plt.hist(train['Age']);
train['Workclass'].fillna(' 0', inplace=True)

test['Workclass'].fillna(' 0', inplace=True)
sns.factorplot(x="Workclass", y="Income", data=train, kind="bar", size = 6, 

palette = "muted")

plt.xticks(rotation=45);
train['Workclass'].value_counts()
train['Workclass'].replace(' Without-pay', ' Never-worked', inplace=True)

test['Workclass'].replace(' Without-pay', ' Never-worked', inplace=True)
train['fnlgwt'].describe()
train['fnlgwt'] = train['fnlgwt'].apply(lambda x: np.log1p(x))

test['fnlgwt'] = test['fnlgwt'].apply(lambda x: np.log1p(x))
train['fnlgwt'].describe()
sns.factorplot(x="Education",y="Income",data=train,kind="bar", size = 7, 

palette = "muted")

plt.xticks(rotation=60);
def primary(x):

    if x in [' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th']:

        return ' Primary'

    else:

        return x
train['Education'] = train['Education'].apply(primary)

test['Education'] = test['Education'].apply(primary)
sns.factorplot(x="Education",y="Income",data=train,kind="bar", size = 6, 

palette = "muted")

plt.xticks(rotation=60);
sns.factorplot(x="Education num",y="Income",data=train,kind="bar", size = 6, 

palette = "muted")

plt.xticks(rotation=60);
sns.factorplot(x="Marital Status",y="Income",data=train,kind="bar", size = 5, 

palette = "muted")

plt.xticks(rotation=60);
train['Marital Status'].value_counts()
train['Marital Status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)

test['Marital Status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)
sns.factorplot(x="Marital Status",y="Income",data=train,kind="bar", size = 6, 

palette = "muted")

plt.xticks(rotation=60);
train['Occupation'].fillna(' 0', inplace=True)

test['Occupation'].fillna(' 0', inplace=True)
sns.factorplot(x="Occupation",y="Income",data=train,kind="bar", size = 8, 

palette = "muted")

plt.xticks(rotation=60);
train['Occupation'].value_counts()
train['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)

test['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)
sns.factorplot(x="Occupation",y="Income",data=train,kind="bar", size = 8, 

palette = "muted")

plt.xticks(rotation=60);
sns.factorplot(x="Relationship",y="Income",data=train,kind="bar", size = 6, 

palette = "muted")

plt.xticks(rotation=60);
train['Relationship'].value_counts()
sns.factorplot(x="Race",y="Income",data=train,kind="bar", size = 6, 

palette = "muted")

plt.xticks(rotation=45);
train['Race'].value_counts()
sns.factorplot(x="Sex",y="Income",data=train,kind="bar", size = 4, 

palette = "muted");
train['Native country'].fillna(' 0', inplace=True)

test['Native country'].fillna(' 0', inplace=True)
sns.factorplot(x="Native country",y="Income",data=train,kind="bar", size = 10, 

palette = "muted")

plt.xticks(rotation=80);
def native(country):

    if country in [' United-States', ' Cuba', ' 0']:

        return 'US'

    elif country in [' England', ' Germany', ' Canada', ' Italy', ' France', ' Greece', ' Philippines']:

        return 'Western'

    elif country in [' Mexico', ' Puerto-Rico', ' Honduras', ' Jamaica', ' Columbia', ' Laos', ' Portugal', ' Haiti',

                     ' Dominican-Republic', ' El-Salvador', ' Guatemala', ' Peru', 

                     ' Trinadad&Tobago', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Vietnam', ' Holand-Netherlands' ]:

        return 'Poor' # no offence

    elif country in [' India', ' Iran', ' Cambodia', ' Taiwan', ' Japan', ' Yugoslavia', ' China', ' Hong']:

        return 'Eastern'

    elif country in [' South', ' Poland', ' Ireland', ' Hungary', ' Scotland', ' Thailand', ' Ecuador']:

        return 'Poland team'

    

    else: 

        return country    
train['Native country'] = train['Native country'].apply(native)

test['Native country'] = test['Native country'].apply(native)
train['Native country'].value_counts()
sns.factorplot(x="Native country",y="Income",data=train,kind="bar", size = 5, 

palette = "muted")

plt.xticks(rotation=60);
#merge datasets

joint = pd.concat([train, test], axis=0)
joint.dtypes
#list of columns with dtype: object

categorical_features = joint.select_dtypes(include=['object']).axes[1]



for col in categorical_features:

    print (col, joint[col].nunique())
#one-hot encode

for col in categorical_features:

    joint = pd.concat([joint, pd.get_dummies(joint[col], prefix=col, prefix_sep=':')], axis=1)

    joint.drop(col, axis=1, inplace=True)
joint.head()
train = joint.head(train.shape[0])

test = joint.tail(test.shape[0])
Xtrain = train.drop('Income', axis=1)

Ttrain = train['Income']



Xtest = test.drop('Income', axis=1)

Ttest = test['Income']



scaler = StandardScaler()

scaler.fit(Xtrain)

Xtrain = scaler.transform(Xtrain)

Xtest = scaler.transform(Xtest)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
model = LogisticRegression()

model.fit(Xtrain, Ttrain)



Ytrain = model.predict(Xtrain)

Ytest = model.predict(Xtest)
print(classification_report(Ttrain, Ytrain))
print(classification_report(Ttest, Ytest))