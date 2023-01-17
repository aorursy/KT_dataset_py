# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_ext autoreload

%autoreload 2

%matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from fastai.imports import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestClassifier

from IPython.display import display

import seaborn as sns



from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load training set

df_raw_train = pd.read_csv('../input/train.csv')

df_raw_train.head()
#Load testing set

df_raw_test = pd.read_csv('../input/test.csv')

df_raw_test.head()
## Join train and test datasets in order to obtain the same number of features during categorical conversion

df_raw =  pd.concat(objs=[df_raw_train, df_raw_test], axis=0, sort=True).reset_index(drop=True)
df_raw.tail()
df_raw.Sex = df_raw.Sex.astype('category')

df_raw.Sex.cat.set_categories(['male', 'female'], ordered=True, inplace=True)
df_raw.Embarked = df_raw.Embarked.astype('category')

df_raw.Embarked.cat.set_categories(['Q', 'S', 'C'], ordered = True, inplace = True)
df_raw.isnull().sum().sort_index()
df_raw["Fare"] = df_raw["Fare"].fillna(df_raw["Fare"].median())
g = sns.distplot(df_raw["Fare"], color="g", label="Skewness : %.2f"%(df_raw["Fare"].skew()))

g = g.legend(loc="best")
# Apply log to Fare to reduce skewness distribution

df_raw["Fare"] = df_raw["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(df_raw["Fare"], color="r", label="Skewness : %.2f"%(df_raw["Fare"].skew()))

g = g.legend(loc="best")
df_raw.Cabin = df_raw.Cabin.str[0:1]
g = sns.catplot("Pclass", col="Cabin", col_wrap=4,

           data=df_raw,

           kind="count", height=2.5, aspect=.8)

g = sns.catplot("Survived", col="Cabin", col_wrap=4,

           data=df_raw,

           kind="count", height=2.5, aspect=.8)
df_raw.Cabin.fillna('U', inplace = True)

df_raw.head()
df_raw.Cabin = df_raw.Cabin.astype('category')

df_raw.Cabin.cat.set_categories(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'U'], ordered=True, inplace=True)
# Get Title from Name

title = [i.split(",")[1].split(".")[0].strip() for i in df_raw.Name]

df_raw["Title"] = title
df_raw.head()
g = sns.countplot(x="Title",data=df_raw)

g = plt.setp(g.get_xticklabels(), rotation=45) 
# Convert to categorical values Title 

df_raw["Title"] = df_raw["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df_raw["Title"] = df_raw["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

df_raw["Title"] = df_raw["Title"].astype(int)
g = sns.factorplot(x="Title",y="Age",data=df_raw,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
#Drop name & age columns

df_raw.drop(['Name', 'Age'], axis=1, inplace=True)

df_raw.head()
df_raw.Embarked.value_counts()
df_raw.Embarked.fillna('S', inplace = True)
df_raw.drop(['Ticket'], axis=1, inplace = True)
df_raw.Sex = df_raw.Sex.cat.codes

df_raw.Cabin = df_raw.Cabin.cat.codes

df_raw.Embarked = df_raw.Embarked.cat.codes

df_raw.head()
## Separate train dataset and test dataset

train_len = len(df_raw_train)

train = df_raw[:train_len]

test = df_raw[train_len:]

test.drop(labels=["Survived"],axis = 1,inplace=True)

print(train.shape)

print(test.shape)
# Separate train features and label 



train["Survived"] = train["Survived"].astype(int)

y = train["Survived"]

df = train.drop(labels = ["Survived", 'PassengerId'],axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.20)
print('Training set: %s, %s' %(X_train.shape, y_train.shape))

print('Validation set: %s, %s' %(X_val.shape, y_val.shape))
m = RandomForestClassifier(n_estimators = 120, min_samples_leaf = 5, n_jobs = -1, max_features=0.5, oob_score=True)

m.fit(X_train, y_train)
m.score(X_train, y_train)
m.score(X_val, y_val)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)
rf_feat_importance(m, X_train)
X_train_keep = X_train.drop(['Parch'], axis = 1)

X_val_keep = X_val.drop(['Parch'], axis = 1)
m.fit(X_train_keep, y_train)
m.score(X_train_keep, y_train)
m.score(X_val_keep, y_val)
rf_feat_importance(m, X_train_keep)
m.fit(df, y)

m.score(df, y)
test.head()
passenger_id = test['PassengerId']

test.drop(['PassengerId'], axis=1, inplace=True)

y_predict = m.predict(test)
titanic_submission = pd.DataFrame({'PassengerId':passenger_id, 'Survived':y_predict})
titanic_submission.head()
titanic_submission.to_csv('rf_Titanic.csv', index=False)