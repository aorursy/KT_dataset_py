import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
titanic_train = pd.read_csv('../input/titanic/train.csv')

titanic_test  = pd.read_csv('../input/titanic/test.csv')
train =  titanic_train.copy()

test  =  titanic_test.copy()
titanic_train.head(10)
def ismarried(df):

    df["Married"] = 0

    df.loc[ df.Name.str.contains("Mrs"),  "Married" ] = 1

    return df 
titanic_train = ismarried(titanic_train) 

titanic_test  = ismarried(titanic_test)
titanic_train = titanic_train.drop(["Name", "Ticket"], axis=1)

titanic_test  = titanic_test.drop(["Name", "Ticket"], axis=1)
titanic_train.info()
titanic_train.Sex.isnull().any()

sex_numbers = titanic_train.Sex.value_counts()/len(titanic_train.Sex)



plt.figure(figsize=(15,8))



plt.subplot(1,2,1)

plt.pie(sex_numbers, labels=["male", "female"], colors=["steelblue", "chocolate"], autopct='%1.1f%%', shadow=True)

plt.title("% of each sex onboard")



plt.subplot(1,2,2)

sns.barplot(x=titanic_train.Sex, y=titanic_train.Survived)

plt.title("Survival rate for each sex group")



plt.show()
titanic_train.Pclass.isnull().any()
Pclass_numbers = titanic_train.Pclass.value_counts()/len(titanic_train.Pclass)

plt.figure(figsize=(15,8))



plt.subplot(1,2,1)

plt.pie(Pclass_numbers, labels=["1", "2", "3"], autopct='%1.1f%%', shadow=True)

plt.title("% of each Pclass")



plt.subplot(1,2,2)

sns.barplot(x=titanic_train.Pclass, y=titanic_train.Survived)

plt.title("Survival rate for each Pclass group")



plt.show()
plt.figure(figsize=(14,8))

sns.distplot(titanic_train.Fare)

plt.title("Distribution of Fare")

plt.show()
titanic_train.Fare.describe()
mu    = titanic_train.Fare.mean()

half_sigma = titanic_train.Fare.std()/2



for k in range(1,6):

    titanic_train.loc[(titanic_train.Fare >= mu + (k-2)*half_sigma) & (titanic_train.Fare < mu + (k-1)*half_sigma) , "FareBin"] = k

titanic_train.loc[titanic_train.Fare < mu  - half_sigma , "FareBin"] = 0

titanic_train.loc[titanic_train.Fare >= mu + 4*half_sigma, "FareBin"] = 6

titanic_train.corrwith(titanic_train.Survived)

labels_FareBin = ['[%d, %d)' % ( mu + half_sigma*(k-2) , mu + half_sigma*(k-1)  ) for k in range(1,6)]

labels_FareBin = ['[0, ' + str(int(mu - half_sigma)) + ')'] + labels_FareBin + ['[' + str(int(mu + half_sigma*4))+ ', 513)']



plt.figure(figsize=(14,8))

sns.barplot(x=titanic_train.FareBin, y=titanic_train.Survived).set_xticklabels(labels_FareBin)

plt.title("Survival rate of different FareBins")



plt.show()
FareBin_size = titanic_train.FareBin.value_counts() / len(titanic_train.FareBin)





plt.figure(figsize=(15,8))



plt.pie(FareBin_size, labels=labels_FareBin, autopct='%1.1f%%', explode =7*[0.05], shadow=True)

plt.title("% of each FareBin")



plt.show()
plt.figure(figsize=(14,8))

sns.barplot(x=titanic_train.Parch, y=titanic_train.Survived)

plt.title("Survival rate of Parch")

plt.show()



print(titanic_train.Parch.value_counts())
titanic_train["nParch"] = 0

titanic_train.loc[titanic_train.Parch > 0, "nParch"] = 1
titanic_train["Mom"] = 0

titanic_train.loc[(titanic_train.Married ==1) & (titanic_train.nParch ==1), "Mom" ] = 1
titanic_train.corrwith(titanic_train.Survived)
plt.figure(figsize=(14,8))

sns.barplot(x=titanic_train.SibSp, y=titanic_train.Survived)

plt.title("Survival rate of SibSp")

plt.show()



print(titanic_train.SibSp.value_counts())
titanic_train["nSibSp"] = 0

titanic_train.loc[titanic_train.SibSp > 0, "nSibSp"] = 1
titanic_train.corrwith(titanic_train.Survived)
titanic_train.Age.isnull().value_counts()
plt.figure(figsize=(14,8))

sns.distplot(titanic_train.Age)

plt.title("Distribution of Age")

plt.show()
titanic_train.loc[titanic_train.Age <= 14, "AgeGroup"] = "child"

titanic_train.loc[titanic_train.Age >= 60, "AgeGroup"] = "senior"

titanic_train.loc[(titanic_train.Age > 14) & (titanic_train.Age < 60), "AgeGroup"] = "adult"

plt.figure(figsize=(14,8))

sns.barplot(x=titanic_train.AgeGroup, y = titanic_train.Survived)

plt.title("Survival rate for different AgeGroup")

plt.show()



print(titanic_train.AgeGroup.value_counts())
def CabinName(df):

    df["Cabin"].fillna("Z", inplace=True)

    cabins = set(df.Cabin.str[0:1])

    

    for cab in cabins:

        df.loc[df.Cabin.str.startswith(cab), "CabinName"] = cab

        

    df["Cabin"].fillna("Z", inplace=True)

    return df
titanic_train = CabinName(titanic_train)
plt.figure(figsize=(14,8))

sns.barplot(x=titanic_train.CabinName, y = titanic_train.Survived)

plt.title("Survival rate for different AgeGroup")

plt.show()



print(titanic_train.CabinName.value_counts())
def Married(df):

    df["Married"] = 0

    df.loc[ df.Name.str.contains("Mrs"),  "Married" ] = 1

    return df 



def CabinName(df):

    df["Cabin"].fillna("Z", inplace=True)

    cabins = set(df.Cabin.str[0:1])

    

    for cab in cabins:

        df.loc[df.Cabin.str.startswith(cab), "CabinName"] = cab

        

    df["Cabin"].fillna("Z", inplace=True)

    return df



def AgeGroup(df):

    df.loc[df.Age <= 14, "AgeGroup"] = "child"

    df.loc[df.Age >= 60, "AgeGroup"] = "senior"

    df.loc[(df.Age > 14) & (df.Age < 60), "AgeGroup"] = "adult"

    

    return df



def nParch(df):

    df["nParch"] = 0

    df.loc[df.Parch > 0, "nParch"] = 1

    return df 



def nSibSp(df):

    df["nSibSp"] = 0

    df.loc[df.SibSp > 0, "nSibSp"] = 1

    return df 



def Mom(df):

    df["Mom"] = 0

    df.loc[(df.Married ==1) & (df.nParch ==1), "Mom" ] = 1

    return df 



mean = train.Fare.mean() 

std  = train.Fare.std()



def FareBin(df, mean , std):

    mu         = mean

    half_sigma = std/2



    for k in range(1,6):

        df.loc[(df.Fare >= mu + (k-2)*half_sigma) & (df.Fare < mu + (k-1)*half_sigma) , "FareBin"] = k



    df.loc[df.Fare < mu  - half_sigma , "FareBin"]  = 0

    df.loc[df.Fare >= mu + 4*half_sigma, "FareBin"] = 6



    return df 

train = Married(train)

train = CabinName(train)

train = FareBin(train, mean , std)

train = nParch(train)

train = nSibSp(train)

train = Mom(train)

train['Age']=train.groupby(['CabinName', 'FareBin'])['Age'].apply(lambda x: x.fillna(x.mean()))

train = AgeGroup(train)



test = Married(test)

test = CabinName(test)

test = FareBin(test, mean , std)

test = nParch(test)

test = nSibSp(test)

test = Mom(test)

test['Age'] = test.groupby(['CabinName', 'FareBin'])['Age'].apply(lambda x: x.fillna(x.mean()))

test = AgeGroup(test)









cols = ['Pclass', 'Sex',  'Embarked', 'Married', 'CabinName', 'AgeGroup', 'nParch', 'nSibSp', 'Mom', 'FareBin']



X = train[cols]

y = train.Survived

X_test = test[cols]

X.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
numerics = ["float64", "int64"]

numerical_cols   = [col for col in X.columns if X[col].dtype in numerics ]

categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

num_transformer = SimpleImputer(strategy="mean")



cat_transformer = Pipeline(steps=[

        ("imputer", SimpleImputer(strategy='most_frequent')),

        ("onehot", OneHotEncoder(handle_unknown = "ignore"))

    ])



preprocessor = ColumnTransformer(

        transformers=[

                ("num", num_transformer, numerical_cols),

                ("cat", cat_transformer, categorical_cols )

    ])
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3)
X_train.head()
def accuracy(model_y_hat):

    acc = model_y_hat == y_valid

    return acc.mean() 
from sklearn.linear_model import LogisticRegression
def logit_get_score(regularization):

    

    logit_pipeline = Pipeline(steps=[

            ("preprocessor", preprocessor), 

            ("model", LogisticRegression(solver="liblinear", C=regularization))

                ])

    logit_pipeline.fit(X_train, y_train)

    

    return logit_pipeline.score(X_valid, y_valid)
logit_score = {}

for k in range(1, 201):

    logit_score[k] = logit_get_score(k/100)



logit_key_max = max(logit_score.keys(), key=(lambda k: logit_score[k]))

opt_C = logit_key_max/100
print("The optimal inverse regularization parameter (between 0 and 2) is", str(opt_C))
# Optimized Logit Model 



logit_pipeline = Pipeline(steps=[

        ("preprocessor", preprocessor),

        ("model", LogisticRegression(solver="liblinear", C=opt_C) )

    ])



# let's fit our model to the data.

logit_pipeline.fit(X_train, np.ravel(y_train))



# Using our fitted model, we can now predict the survival in our validation data

y_pred_logit = logit_pipeline.predict(X_valid)



# The accuracy of our prediction is:

logit_model_score = logit_pipeline.score(X_valid, y_valid)



print("Our Model 1: Logistic Regression yields a score of {0}".format(logit_model_score))
from sklearn.ensemble import RandomForestClassifier
def forest_get_score(estimators_numbers):

    

    forest_pipeline =Pipeline(steps=[

            ("preprocessor", preprocessor), 

            ("model", RandomForestClassifier(n_estimators = estimators_numbers))

                ])

    forest_pipeline.fit(X_train, y_train)



    return forest_pipeline.score(X_valid, y_valid)
forest_score = {}

for k in range(50,500,5):

    forest_score[k] = forest_get_score(k)

    

forest_key_max = max(forest_score.keys(), key=(lambda k: forest_score[k]))
print("The optimal number of estimators (between 50 and 500) is", str(forest_key_max))
forest_pipeline = Pipeline(steps=[

    ("preprocessor", preprocessor), 

    ("model", RandomForestClassifier(n_estimators=forest_key_max) )

    ])



# Fit the model to the data

forest_pipeline.fit(X_train, np.ravel(y_train))



# predict the survival for validation set

forest_y_hat = forest_pipeline.predict(X_valid)

forest_model_score = forest_pipeline.score(X_valid, y_valid)



print("Our Model 2: Random Forest Model yields a score of {0}".format(forest_model_score))
from xgboost import XGBClassifier
# model: 

xgb_pipeline =Pipeline(steps=[

    ("preprocessor", preprocessor), 

    ("model", XGBClassifier(objective = 'reg:squarederror'))

                ])



# Fit the model to the data

xgb_pipeline.fit(X_train, np.ravel(y_train))

xgb_y_hat = xgb_pipeline.predict(X_valid)



xgb_model_score = accuracy(xgb_y_hat)

print("Our Model 3: XGBRegression Model yields a score of {0}".format(xgb_model_score))
from sklearn.neighbors import KNeighborsClassifier
knn_pipeline =Pipeline(steps=[

    ("preprocessor", preprocessor), 

    ("model", KNeighborsClassifier(n_neighbors=5) )

    ])



# Fit the model to the data

knn_pipeline.fit(X_train, np.ravel(y_train))

knn_y_hat = knn_pipeline.predict(X_valid)



# Accuracy of the model

knn_model_score = accuracy(knn_y_hat)



print("Our Model 4: kNN Model yields a score of {0}".format(knn_model_score))
results = pd.DataFrame({

    'Model': ['Logistic Regression', 

              'Random Forest',

              'XGB Classifier',

              'kNN Classifier'],

    'score': [logit_model_score, forest_model_score, xgb_model_score, knn_model_score]})

df = results.sort_values('score', ascending=False).set_index('Model')

df
print("The best model of the four is {0}".format(df.index[0]))
if df.index[0] == 'Logistic Regression':

    logit_pipeline.fit(X, np.ravel(y))

    predictions = logit_pipeline.predict(X_test)

elif df.index[0] == 'Random Forest':

    forest_pipeline.fit(X, np.ravel(y))

    predictions = forest_pipeline.predict(X_test)

elif df.index[0] == 'Random Forest':

    xgb_pipeline.fit(X, np.ravel(y))

    predictions = xgb_pipeline.predict(X_test)    

else:

    knn_pipeline.fit(X, np.ravel(y))

    predictions = knn_pipeline.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': predictions})

output.to_csv('submission.csv', index=False)



print('!! submission.csv has been created !!')