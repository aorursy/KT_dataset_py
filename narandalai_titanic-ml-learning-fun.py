# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data_train = pd.read_csv("../input/train.csv")

data_train.head()
data_test = pd.read_csv("../input/test.csv")

data_test.head()
# total rows and columns

data_train.shape, data_test.shape
# special attension to the count value, if it does not match with total rows from above, data is missing.

data_train.describe(), data_test.describe()
data_train.info()
# find out missing data

data_train.isnull().sum()
# find out missing data

data_test.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns



data_train.hist(bins=10,figsize=(10,7))
g = sns.FacetGrid(data_train, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age", color = "blue")
g = sns.FacetGrid(data_train, hue="Survived", col="Pclass", margin_titles=True,

                 palette={1:"blue", 0:"green"})

g = g.map(plt.scatter, "Fare", "Age", edgecolor="W").add_legend()
g = sns.FacetGrid(data_train, hue="Survived", col="Sex", margin_titles=True,

                 palette="Set1", hue_kws=dict(marker=["^","v"]))

g.map(plt.scatter, "Fare","Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare')
data_train.Embarked.value_counts().plot(kind='bar', alpha=0.6)

plt.title("Passengers per boarding location")
sns.factorplot(x = 'Embarked',y="Survived", data = data_train,color="r")
corr =  data_train.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr,vmax=1,square=True, annot=True,cmap='cubehelix')

plt.title('Correlation between features')
# correlation of features with target variable.

data_train.corr()["Survived"]
# row that has null Embarked value

data_train[data_train['Embarked'].isnull()]

#data_train[data_train['Ticket']=='113572']
sns.boxplot(x="Embarked",y="Fare",hue="Pclass", data=data_train)
data_train["Embarked"] = data_train["Embarked"].fillna("C")

# Age has many null values

data_train[data_train['Age'].isnull()]

data_train.info()

# sklearn imputing missing values

#from sklearn.preprocessing import Imputer

#imp = Imputer(missing_values="NaN", strategy='mean',axis=0)

#imp.fit_transform(data_train)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEnc = LabelEncoder()

cat_vars=['Embarked','Sex']

for col in cat_vars:

    data_train[col]=labelEnc.fit_transform(data_train[col])

    data_test[col]=labelEnc.fit_transform(data_test[col])

    

data_train.head()
from sklearn.ensemble import RandomForestRegressor



def fill_missing_age(df):

    age_df = df[['Age','Embarked','Pclass','Fare','Parch','SibSp','Sex']]

    

    train = age_df.loc[ (df.Age.notnull()) ]

    test = age_df.loc[ (df.Age.isnull()) ]

    

    y = train.values[:,0]

    X = train.values[:,1::]

    

    rfr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rfr.fit(X,y)

    

    predictedAges = rfr.predict(test.values[:,1::])

    

    df.loc[ (df.Age.isnull()),'Age'] = predictedAges

    

    return df

    
data_train=fill_missing_age(data_train)

#data_test=fill_missing_age(data_test)
data_train.corr()["Survived"]