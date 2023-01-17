import numpy as np

import pandas as pd

from collections import Counter
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')

df_sample= pd.read_csv('../input/gender_submission.csv')
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
import numpy as np

import pandas as pd

from collections import Counter

df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')

df_sample= pd.read_csv('../input/gender_submission.csv')

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
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)
sex = pd.get_dummies(df_train['Sex'],drop_first=True)

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

sex = pd.get_dummies(df_test['Sex'],drop_first=True)

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
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex
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
df_train.head()
df_train.describe()
df_train.info()

print('.............................................')

df_test.info()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_train.drop('Survived',axis=1), 

                                                    df_train['Survived'], test_size=0.20, 

                                                    random_state=101)
from xgboost import XGBClassifier



XGB = XGBClassifier(max_depth=4,learning_rate=0.01,n_estimators=250,n_jobs=-1,min_child_weight=2)

XGB.fit(X_train,y_train)
XGB.score(X_train,y_train)
XGB.score(X_test,y_test)
y_pred = pd.DataFrame(XGB.predict(df_test))

y_pred['Survived'] = y_pred[0]

y_pred.drop(0,axis=1,inplace=True)

y_pred['PassengerId'] = df_test['PassengerId']

y_pred_xgb = y_pred
y_pred.to_csv('titanic_pred_xgb.csv',index=False)
from sklearn.preprocessing import StandardScaler



Scaler1 = StandardScaler()

Scaler2 = StandardScaler()

X_train_scaled = Scaler1.fit_transform(X_train)

df_test_scaled  = Scaler2.fit_transform(df_test)
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression(C=10).fit(X_train,y_train)



y_pred = pd.DataFrame(logmodel.predict(df_test))



y_pred['Survived'] = y_pred[0]

y_pred.drop(0,axis=1,inplace=True)

y_pred['PassengerId'] = df_test['PassengerId']

y_pred_lr = y_pred
logmodel.score(X_train,y_train)
logmodel.score(X_test,y_test)
y_pred.to_csv('titanic_pred_logistic.csv',index=False)
from sklearn.ensemble import RandomForestClassifier



RFC = RandomForestClassifier(n_estimators=750,max_depth=6,min_samples_split=3)

RFC.fit(X_train,y_train)
RFC.score(X_train,y_train)
RFC.score(X_test,y_test)
y_pred = pd.DataFrame(RFC.predict(df_test))



y_pred['Survived'] = y_pred[0]

y_pred.drop(0,axis=1,inplace=True)

y_pred['PassengerId'] = df_test['PassengerId']

y_pred_rf = y_pred

y_pred.to_csv('titanic_pred_rfc.csv',index=False)
from lightgbm import LGBMClassifier



lgb = LGBMClassifier(learning_rate=0.01,max_depth=2,num_leaves=3).fit(X_train,y_train)
lgb.score(X_train,y_train)
lgb.score(X_test,y_test)
y_pred = pd.DataFrame(lgb.predict(df_test))



y_pred['Survived'] = y_pred[0]

y_pred.drop(0,axis=1,inplace=True)

y_pred['PassengerId'] = df_test['PassengerId']

y_pred_lgb = y_pred

y_pred.to_csv('titanic_pred_lgb.csv',index=False)
y_pred_final = y_pred

y_pred_final['Survived'] = round(  0.3 * y_pred_lgb['Survived'] 

                                 + 0.3 * y_pred_rf['Survived'] 

                                 + 0.25 * y_pred_xgb['Survived'] 

                                 + 0.15 * y_pred_lr['Survived'] )



y_pred_final['PassengerId'] = df_test['PassengerId']

y_pred_final['Survived'] = y_pred_final['Survived'].astype(int)

y_pred_final.to_csv('titanic_pred_final.csv',index=False)
y_pred_final['Survived'].value_counts()