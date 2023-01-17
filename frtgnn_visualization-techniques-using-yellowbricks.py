import numpy as np 

import pandas as pd 

from collections import Counter
df_train = pd.read_csv('../input/titanic/train.csv')

df_test  = pd.read_csv('../input/titanic/test.csv')

df_sample= pd.read_csv('../input/titanic/gender_submission.csv')
df_train.head()
def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers 
Outliers_to_drop = detect_outliers(df_train,2,["Age","SibSp","Parch","Fare"])

df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in df_train["Name"]]



df_train["Title"] = pd.Series(dataset_title)

df_train["Title"] = df_train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df_train["Title"] = df_train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

df_train["Title"] = df_train["Title"].astype(int)



df_train.drop(labels = ["Name"], axis = 1, inplace = True)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age

    

def impute_fare(cols):

    Fare = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Fare):



        if Pclass == 1:

            return 84



        elif Pclass == 2:

            return 20



        else:

            return 13



    else:

        return Fare

    

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)



sex    = pd.get_dummies(df_train['Sex'],drop_first=True)

embark = pd.get_dummies(df_train['Embarked'],drop_first=True)



df_train = pd.concat([df_train,sex,embark],axis=1)



df_train["Family"] = df_train["SibSp"] + df_train["Parch"] + 1

df_train['Single'] = df_train['Family'].map(lambda s: 1 if s == 1 else 0)

df_train['SmallF'] = df_train['Family'].map(lambda s: 1 if  s == 2  else 0)

df_train['MedF']   = df_train['Family'].map(lambda s: 1 if 3 <= s <= 4 else 0)

df_train['LargeF'] = df_train['Family'].map(lambda s: 1 if s >= 5 else 0)

df_train['Senior'] = df_train['Age'].map(lambda s:1 if s>60 else 0)



dataset_title = [i.split(",")[1].split(".")[0].strip() for i in df_test["Name"]]



df_test["Title"] = pd.Series(dataset_title)

df_test["Title"] = df_test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df_test["Title"] = df_test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

df_test["Title"] = df_test["Title"].astype(int)



df_test.drop(labels = ["Name"], axis = 1, inplace = True)

df_test['Age'] = df_test[['Age','Pclass']].apply(impute_age,axis=1)



sex    = pd.get_dummies(df_test['Sex'],drop_first=True)

embark = pd.get_dummies(df_test['Embarked'],drop_first=True)



df_test = pd.concat([df_test,sex,embark],axis=1)



df_test['Fare'].fillna(value=df_test['Fare'].median(),inplace=True)



df_test['Fare'] = df_test[['Fare','Pclass']].apply(impute_fare,axis=1)

df_test["Fare"] = df_test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

df_test["Family"] = df_test["SibSp"] + df_test["Parch"] + 1

df_test['Single'] = df_test['Family'].map(lambda s: 1 if s == 1 else 0)

df_test['SmallF'] = df_test['Family'].map(lambda s: 1 if  s == 2  else 0)

df_test['MedF']   = df_test['Family'].map(lambda s: 1 if 3 <= s <= 4 else 0)

df_test['LargeF'] = df_test['Family'].map(lambda s: 1 if s >= 5 else 0)

df_test['Senior'] = df_test['Age'].map(lambda s:1 if s>60 else 0)



df_train['Person'] = df_train[['Age','Sex']].apply(get_person,axis=1)

df_test['Person']  = df_test[['Age','Sex']].apply(get_person,axis=1)



person_dummies_train  = pd.get_dummies(df_train['Person'])

person_dummies_train.columns = ['Child','Female','Male']

person_dummies_train.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(df_test['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



df_train = df_train.join(person_dummies_train)

df_test  = df_test.join(person_dummies_test)



df_train.drop(['Person'],axis=1,inplace=True)

df_test.drop(['Person'],axis=1,inplace=True)



df_train.drop('male',axis=1,inplace=True)

df_test.drop('male',axis=1,inplace=True)



df_train.drop(['Cabin','Ticket'],axis = 1, inplace= True)

df_test.drop(['Ticket','Cabin'],axis = 1, inplace= True)



df_train.drop(['Sex','Embarked'],axis=1,inplace=True)

df_test.drop(['Sex','Embarked'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_train.drop('Survived',axis=1), 

                                                    df_train['Survived'], test_size=0.20, 

                                                    random_state=101)
X = df_train.drop('Survived',axis=1)

y = df_train['Survived']
import matplotlib.pyplot as plt

from yellowbrick.features import RadViz



fig, ax = plt.subplots(figsize=(12, 12))

rv = RadViz(classes=["died", "survived"],features=X.columns)

rv.fit(X, y)

_ = rv.transform(X)

rv.poof()
from pandas.plotting import radviz

fig, ax = plt.subplots(figsize=(12, 12))

new_df = X.copy()

new_df["target"] = y

radviz(new_df, "target", ax=ax, colormap="PiYG")
from sklearn import ensemble

from yellowbrick.features import RFECV

fig, ax = plt.subplots(figsize=(12, 8))

rfe = RFECV(

    ensemble.RandomForestClassifier(n_estimators=100),cv=5)

rfe.fit(X, y)

rfe.rfe_estimator_.ranking_

rfe.rfe_estimator_.n_features_

rfe.rfe_estimator_.support_

rfe.poof()
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.model_selection import LearningCurve

fig, ax = plt.subplots(figsize=(12, 8))

lc3_viz = LearningCurve(RandomForestClassifier(n_estimators = 10),cv=5)

lc3_viz.fit(X, y)

lc3_viz.poof()
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(

    X_train.values,

    feature_names=X.columns,

    class_names=["died", "survived"]

)

exp = explainer.explain_instance(X_train.iloc[-1].values, rfe.predict_proba)



fig = exp.as_pyplot_figure()

fig.tight_layout()