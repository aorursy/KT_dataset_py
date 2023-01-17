%reset

%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn import preprocessing

from sklearn.linear_model import Ridge, Lasso

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
titanic_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# Drop useless columns

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df = test_df.drop(['Name','Ticket'], axis=1)



#Auto fill Embarked(only 2 missing values, put most common)

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")



#New variable for age/no age and Cabin/No cabin and Gender

def prep_df(df):

    df["AgeNull"] = 0

    df["NoCabin"] = 0

    df["Gender"] = 0

    df["Class1"] = 0

    df["Class3"] = 0

    df["EmbS"] = 0

    df["EmbC"] = 0

    df.loc[df.Age.isnull(),'AgeNull'] = 1

    df.loc[df.Cabin.isnull(),'NoCabin'] = 1

    df.loc[df.Sex == "female",'Gender'] = 1

    df.loc[df.Pclass == 1,'Class1'] = 1

    df.loc[df.Pclass == 3,'Class3'] = 1

    df.loc[df.Embarked == "S",'EmbS'] = 1

    df.loc[df.Embarked == "C",'EmbC'] = 1

    df = df.drop(['Cabin','Sex','Pclass','Embarked'], axis=1)

    df['Fare'] = np.float32(df['Fare'])

    df['Age'] = np.float32(df['Age'])

    return(df)



titanic_df = prep_df(titanic_df)

test_df = prep_df(test_df)





# Auto fill Fare (the missing is a male in 3rd class)

Fmed = titanic_df[(titanic_df.Gender == 0) & (titanic_df.Class3 == 1)]['Fare'].median()

test_df["Fare"] = test_df["Fare"].fillna(Fmed)



test_df.head()
# Deal with the missing ages

Age = titanic_df[titanic_df.AgeNull == 0].drop(['Survived'], axis=1)

Age_T = test_df[test_df.AgeNull == 0].drop(['PassengerId'], axis=1)

Age = pd.concat([Age, Age_T])

del Age_T



# Separate into training and test (200 rows) df

test_ind = np.random.choice(len(Age), 200, replace=False)

Age_test = Age.iloc[test_ind]

Age_train = Age.drop(test_ind, axis=0)

AgeX_train = np.array(Age_train.drop('Age', axis=1))

AgeY_train = np.array(Age_train['Age'])

AgeX_test = np.array(Age_test.drop('Age', axis=1))

AgeY_test = np.array(Age_test['Age'])





# Use ridge on the train age

def std_age_cv(model):

    std = np.sqrt(-cross_val_score(model, AgeX_train, AgeY_train, scoring="neg_mean_squared_error", cv = 5))

    return(std)



def std_age_val(model):

    cmod = model.fit(AgeX_train,AgeY_train)

    prd = cmod.predict(AgeX_test)

    res = np.sqrt(((prd - AgeY_test)**2).mean())

    return(res)



ntrees = [20,70,110]

cv_age = [std_age_cv(RandomForestRegressor(n_estimators = ntree,

                                             max_features = "sqrt")).mean() for ntree in ntrees]

val_age = [std_age_val(RandomForestRegressor(n_estimators = ntree,

                                             max_features = "sqrt")) for ntree in ntrees]



cv_age = pd.Series(cv_age, index = ntrees)

val_age = pd.Series(val_age, index = ntrees)

cv_age.plot(label = "CV")

val_age.plot(label = "Validation")

plt.xlabel("ntrees")

plt.ylabel("rmse")

plt.legend()
AgeX = np.array(Age.drop('Age', axis=1))

AgeY = np.array(Age['Age'])



model_age = RandomForestRegressor(n_estimators = 100, max_features = "sqrt")



NAge1 = titanic_df[titanic_df.AgeNull == 1].drop(['Survived'], axis=1)

NAge1 = np.array(NAge1.drop('Age', axis=1))

NAge2 = test_df[test_df.AgeNull == 1].drop(['PassengerId'], axis=1)

NAge2 = np.array(NAge2.drop('Age', axis=1))



model_age = model_age.fit(AgeX, AgeY)



YAge1 = model_age.predict(NAge1)

YAge2 = model_age.predict(NAge2)



titanic_df.loc[titanic_df.AgeNull == 1,'Age'] = YAge1

test_df.loc[test_df.AgeNull == 1,'Age'] = YAge2
# Let's  use random forest classifier

test_ind = np.random.choice(len(titanic_df), 200, replace=False)

test = titanic_df.iloc[test_ind]

train = titanic_df.drop(test_ind, axis=0)

X_train = np.array(train.drop('Survived', axis=1))

Y_train = np.array(train['Survived'])

X_test = np.array(test.drop('Survived', axis=1))

Y_test = np.array(test['Survived'])





def std_cv(model):

    std = (cross_val_score(model, X_train, Y_train, scoring="f1", cv = 5))

    return(std)



def std_val(model):

    cmod = model.fit(X_train,Y_train)

    prd = cmod.predict(X_test)

    res = (1-np.sqrt(((prd - Y_test)**2))).mean()

    return(res)
ntrees = [30,90,110,150,200]

cv_rf = [std_cv(RandomForestClassifier(n_estimators = ntree,

                                             max_features = "sqrt")).mean() for ntree in ntrees]

val_rf = [std_val(RandomForestClassifier(n_estimators = ntree,

                                             max_features = "sqrt")) for ntree in ntrees]



cv_rf = pd.Series(cv_rf, index = ntrees)

val_rf = pd.Series(val_rf, index = ntrees)

cv_rf.plot(label = "CV")

val_rf.plot(label = "Validation")

plt.xlabel("ntrees")

plt.ylabel("rmse")

plt.legend()
XX =  np.array(titanic_df.drop('Survived', axis=1))

YY = np.array(titanic_df['Survived'])

final_X = np.array(test_df.drop('PassengerId', axis=1))

rf = RandomForestClassifier(n_estimators = 150, max_features = "sqrt")

rf = rf.fit(XX,YY)

rf_surv = rf.predict(final_X)

rf_sol = pd.DataFrame(test_df['PassengerId'])

rf_sol['Survived'] = rf_surv 
rf_sol.to_csv('rfsolution.csv', index=False)