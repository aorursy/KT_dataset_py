# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler 

from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def impute(col):

    return col.fillna(int(col.dropna().mean()))
def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if substring in big_string:

            return substring

    print(big_string)

    return big_string
def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
def Feature_manipulation(DS):

    

    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']

    

    DS['Title']=DS['Name'].map(lambda x: substrings_in_string(x, title_list))

    DS['Title']=DS.apply(replace_titles, axis=1)

    

    DS['Family_Size']=DS['SibSp']+DS['Parch']

    

    DS['Age*Class']=DS['Age']*DS['Pclass']

    

    #DS['Fare_Per_Person']=DS['Fare']/(DS['Family_Size']+1)

    

    nDS = DS.drop(columns=['Name','SibSp','Parch','Age','Pclass'])

    # import labelencoder

    categorical_feature_mask = nDS.dtypes==object

    # filter categorical columns using mask and turn it into a list

    categorical_cols = nDS.columns[categorical_feature_mask].tolist()

    # instantiate labelencoder object

    le = LabelEncoder()

    # apply le on categorical feature columns

    nDS[categorical_cols] = nDS[categorical_cols].apply(lambda col: le.fit_transform(col))

    

    return nDS

    
from sklearn.preprocessing import MinMaxScaler

def FeatureEngineering(TrainX, TestX, w=False):

    if w == True:

        sc = MinMaxScaler() 



        X_train = sc.fit_transform(TrainX)

        X_test = sc.transform(TestX)

        

        pca = PCA(n_components = 4) 



        X_train = pca.fit_transform(X_train) 

        X_test = pca.transform(X_test) 



        explained_variance = pca.explained_variance_ratio_

        

        # feature selection

        selector = VarianceThreshold()

        train_x = selector.fit_transform(TrainX)

        test_x = selector.transform(TestX)

        return X_train, X_test

    

    return TrainX,TestX
def retUpsampled(TrainX):

    df_majority = TrainX[TrainX.Survived==0]

    df_minority = TrainX[TrainX.Survived==1]

    # Upsample minority class

    df_minority_upsampled = resample(df_minority,replace=True, n_samples=549,random_state=123)

    

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    

    return df_upsampled

    
# After Dealing with NaN values, Text Values, and dropping irrelevent columns on each column

from sklearn.utils import resample



def Preprocess(Train_DS, Test_DS):

    Train_DS.isnull().sum()

    Test_DS.isnull().sum()

    Train_DS.Age = impute(Train_DS.Age)

    Test_DS.Age = impute(Test_DS.Age)

    Test_DS.Fare = impute(Test_DS.Fare)

    Train_DS.Cabin = Train_DS.Cabin.fillna('N')

    Test_DS.Cabin = Test_DS.Cabin.fillna('N')

    Train_DS.Embarked = Train_DS.Embarked.fillna('N')

    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'N']

    #Train_DS['deck']=Train_DS['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))



    #Test_DS['deck']=Test_DS['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

    #TrainY = Train_DS.Survived



    DS = []

    #Train = Train_DS.drop(columns=['Survived'])

    Test_DS['Survived'] = 1

    DS.append(Train_DS)

    DS.append(Test_DS)



    DS = pd.concat(DS, axis = 0)



    nDS = Feature_manipulation(DS)

    #upsample here

    x = Train_DS.shape[0]

    TrainX = nDS.iloc[:x]

    TestX = nDS.iloc[x:]

    

    TrainX = retUpsampled(TrainX)

    

    TrainY = TrainX.Survived

    

    TrainX = TrainX.drop(columns=['Survived'])

    TestX = TestX.drop(columns=['Survived'])

    

    TrainX, TestX = FeatureEngineering(TrainX, TestX, False)

    

    return TrainX,TrainY,TestX

from sklearn.model_selection import GridSearchCV

def gridSearch(X, y, model, params):

    cv = GridSearchCV(estimator=model, param_grid=params, scoring=scoring, iid=False, cv=5, verbose = 5,refit='Accuracy')

    cv.fit(X, y)

    

    return cv.best_score_, cv.best_estimator_

    
# Performing Random forest without the PCA and scaling

def ForecastR(TrainX, TrainY, TestX):

    model = RandomForestClassifier(bootstrap=True,class_weight=None,criterion='gini',max_depth=9,max_features='auto',

                                   max_leaf_nodes=None, min_samples_leaf=3,min_samples_split=10,

                                   min_weight_fraction_leaf=0.0,n_estimators=100,n_jobs=1,oob_score=False,random_state=None,

                                   verbose=0,warm_start=False)

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict_proba(TestX)

    

    return rf_predictions
# Performing Random forest without the PCA and scaling

def ForecastG(TrainX, TrainY, TestX):

    model = XGBClassifier(learning_rate=0.02, n_estimators=100,

                   max_depth= 3, min_child_weight= 1, 

                   colsample_bytree= 0.6, gamma= 0.0, 

                   reg_alpha= 0.001, subsample= 0.8)

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict(TestX)

    

    return rf_predictions
# Performing Random forest without the PCA and scaling

def Forecast_1(TrainX, TrainY, TestX):

    model = RandomForestClassifier()

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict_proba(TestX)

    

    return rf_predictions
from sklearn import svm

def Forecast_2(TrainX, TrainY, TestX):

    model = svm.SVC(C=10, gamma=1, probability=True)

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict_proba(TestX)

    

    return rf_predictions
from sklearn.neighbors import KNeighborsClassifier as knn

def Forecast_3(TrainX, TrainY, TestX):

    model = knn()

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict_proba(TestX)

    

    return rf_predictions
from sklearn.linear_model import LogisticRegression as loreg

def Forecast_4(TrainX, TrainY, TestX):

    model = loreg()

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict_proba(TestX)

    

    return rf_predictions
from sklearn.ensemble import GradientBoostingClassifier as gbc

def Forecast_5(TrainX, TrainY, TestX):

    model = gbc()

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict_proba(TestX)

    

    return rf_predictions

    
def ensembleForecast(TrainX, TrainY, TestX):

    P1 = ForecastG(TrainX, TrainY, TestX)

    #P2 = Forecast_2(TrainX, TrainY, TestX)

    #P3 = Forecast_3(TrainX, TrainY, TestX)

    #P4 = Forecast_4(TrainX, TrainY, TestX)

    P5 = ForecastR(TrainX, TrainY, TestX)

    

    rf_predictions = (P1+P5)/2

    

    predictions = []



    for x in range(0,len(rf_predictions)):

        if rf_predictions[x][0] > rf_predictions[x][1]:

            predictions.append(0)

        else:

            predictions.append(1)

    return predictions
Train_DS = pd.read_csv('/kaggle/input/titanic/train.csv')

Test_DS = pd.read_csv('/kaggle/input/titanic/test.csv')
# Preprocess

TrainX,TrainY,TestX = Preprocess(Train_DS, Test_DS)

pID = Test_DS.PassengerId

TrainX
print(TrainY.value_counts())
params = dict(     

    max_depth = [n for n in [None,9,10,11,12,13,14,15]],     

    min_samples_split = [n for n in range(2, 11)], 

    min_samples_leaf = [n for n in range(1, 5)],     

    n_estimators = [n for n in [100,200,500,750,1000]])



model = RandomForestClassifier()



score, newModel = gridSearch(TrainX, TrainY, model, params)



print(score)

print(newModel)
from xgboost import XGBClassifier

paramsG = {

    'n_estimators': [100,200,500,750,1000],

    'max_depth': [3,5,7,9],

    'min_child_weight': [1,3,5],

    'gamma':[i/10.0 for i in range(0,5)],

    'subsample':[i/10.0 for i in range(6,10)],

    'colsample_bytree':[i/10.0 for i in range(6,10)],

    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 0.1, 1],

    'learning_rate': [0.01, 0.02, 0.05, 0.1]}



modelG = XGBClassifier()



score, newModel = gridSearch(TrainX, TrainY, model, params)



print(score)

print(newModel)
#Training and Predctions

#from xgboost import XGBClassifier

predictions = ForecastR(TrainX, TrainY, TestX)
# Exporting the output

output = pd.DataFrame()

output['PassengerId'] = pID

output['Survived'] = predictions

output.to_csv('submission.csv', index = False)