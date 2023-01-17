import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from dateutil import parser

%matplotlib inline





from sklearn.model_selection import train_test_split



from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.model_selection import GridSearchCV

import pickle

from lightgbm import LGBMClassifier

print('Library Loaded')
df = pd.read_csv('../input/diabetes/diabetes.csv')#,engine='python')

df.shape
df.head()
df.isnull().sum()


# Correlation matrix

corrmat = df.corr()

fig = plt.figure(figsize = (12, 12))



sns.heatmap(corrmat, vmax = 1, square = True,annot=True,vmin=-1)

plt.show()
df.hist(figsize=(12,12))

plt.show()
sns.pairplot(df,hue='Outcome')
#check for unbalance 

df.Outcome.value_counts()
df.columns
print("# rows in dataframe {0}".format(len(df)))

print("-------------------------------------------")

print("# rows missing Glucose: {0}".format(len(df.loc[df.Glucose == 0 ])))

print("# rows missing BloodPressure: {0}".format(len(df.loc[df.BloodPressure == 0 ])))

print("# rows missing SkinThickness: {0}".format(len(df.loc[df.SkinThickness == 0 ])))

print("# rows missing insulin: {0}".format(len(df.loc[df.Insulin == 0 ])))

print("# rows missing bmi: {0}".format(len(df.loc[df.BMI == 0 ])))

print("# rows missing Age: {0}".format(len(df.loc[df.Age == 0 ])))

print("# rows missing Pregnancies: {0}".format(len(df.loc[df.Pregnancies == 0 ])))

print("# rows missing DiabetesPedigreeFunction: {0}".format(len(df.loc[df.DiabetesPedigreeFunction == 0 ])))
X = df.drop('Outcome',axis=1) # predictor feature coloumns

y = df.Outcome





X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)



print('Training Set :',len(X_train))

print('Test Set :',len(X_test))

print('Training labels :',len(y_train))

print('Test Labels :',len(y_test))
# from sklearn.preprocessing import Imputer

from sklearn.impute import SimpleImputer

#impute with mean all 0 readings



fill = SimpleImputer(missing_values = 0 , strategy ="mean")



X_train = fill.fit_transform(X_train)

X_test = fill.fit_transform(X_test)
print('Training Set :',len(X_train))

print('Test Set :',len(X_test))

print('Training labels :',len(y_train))

print('Test Labels :',len(y_test))
def FitModel(X_train,y_train,X_test,y_test,algo_name,algorithm,gridSearchParams,cv):

    np.random.seed(10)

   

    

    grid = GridSearchCV(

        estimator=algorithm,

        param_grid=gridSearchParams,

        cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)

    

    

    grid_result = grid.fit(X_train, y_train)

    best_params = grid_result.best_params_

    pred = grid_result.predict(X_test)

    cm = confusion_matrix(y_test, pred)

   # metrics =grid_result.gr

    print(pred)

    #pickle.dump(grid_result,open(algo_name,'wb'))

   

    print('Best Params :',best_params)

    print('Classification Report :',classification_report(y_test,pred))

    print('Accuracy Score : ' + str(accuracy_score(y_test,pred)))

    print('Confusion Matrix : \n', cm)
# Create regularization penalty space

penalty = ['l1', 'l2']



# Create regularization hyperparameter space

C = np.logspace(0, 4, 10)



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)



FitModel(X_train,y_train,X_test,y_test,'LogisticRegression',LogisticRegression(),hyperparameters,cv=5)
param ={

            'n_estimators': [100, 500, 1000,1500, 2000],

            'max_depth' :[2,3,4,5,6,7],

    'learning_rate':np.arange(0.01,0.1,0.01).tolist()

           

        }



FitModel(X_train,y_train,X_test,y_test,'XGBoost',XGBClassifier(),param,cv=5)
param ={

            'n_estimators': [100, 500, 1000,1500, 2000],

           

        }

FitModel(X_train,y_train,X_test,y_test,'Random Forest',RandomForestClassifier(),param,cv=5)
param ={

            'C': [0.1, 1, 100, 1000],

            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]

        }

FitModel(X_train,y_train,X_test,y_test,'SVC',SVC(),param,cv=5)
y.value_counts()
from imblearn.over_sampling import SMOTE

sm =SMOTE(random_state=42)

X_res_OS , Y_res_OS = sm.fit_resample(X,y)

pd.Series(Y_res_OS).value_counts()
X_train , X_test , y_train , y_test = train_test_split(X_res_OS, Y_res_OS, test_size = 0.20, random_state = 10)



print('Training Set :',len(X_train))

print('Test Set :',len(X_test))

print('Training labels :',len(y_train))

print('Test Labels :',len(y_test))


fill = SimpleImputer(missing_values = 0 , strategy ="mean")



X_train = fill.fit_transform(X_train)

X_test = fill.fit_transform(X_test)
print('Training Set :',len(X_train))

print('Test Set :',len(X_test))

print('Training labels :',len(y_train))

print('Test Labels :',len(y_test))
# Create regularization penalty space

penalty = ['l1', 'l2']



# Create regularization hyperparameter space

C = np.logspace(0, 4, 10)



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)



FitModel(X_train,y_train,X_test,y_test,'LogisticRegression',LogisticRegression(),hyperparameters,cv=5)
param ={

            'n_estimators': [100, 500, 1000,1500, 2000],

            'max_depth' :[2,3,4,5,6,7],

    'learning_rate':np.arange(0.01,0.1,0.01).tolist()

           

        }



FitModel(X_train,y_train,X_test,y_test,'XGBoost',XGBClassifier(),param,cv=5)
param ={

            'n_estimators': [100, 500, 1000,1500, 2000],

           

        }

FitModel(X_train,y_train,X_test,y_test,'Random Forest',RandomForestClassifier(),param,cv=5)
param ={

            'C': [0.1, 1, 100, 1000],

            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]

        }

FitModel(X_train,y_train,X_test,y_test,'SVC',SVC(),param,cv=5)