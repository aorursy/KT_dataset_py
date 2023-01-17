import pandas as pd

from scipy.io import arff

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from sklearn import preprocessing

import numpy as np



from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

import xgboost as xgb

from sklearn import metrics

from sklearn.metrics import mean_squared_error



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
oasis_longitudinal = '/kaggle/input/mri-and-alzheimers/oasis_longitudinal.csv'

oasis_longitudinal = pd.read_csv (oasis_longitudinal)

oasis_longitudinal.head()
oasis_longitudinal.info()
oasis_cross_sectional = '/kaggle/input/mri-and-alzheimers/oasis_cross-sectional.csv'

oasis_cross_sectional = pd.read_csv (oasis_cross_sectional)

oasis_cross_sectional.rename(columns={'Educ': 'EDUC'}, inplace=True)

oasis_cross_sectional.head()
oasis_cross_sectional.info()
oasis_cross_sectional['CDR'].isnull().sum()
oasis_cross_sectional.dropna(subset=['CDR'], inplace=True)
oasis_cross_sectional.info()
oasis_longitudinal.drop(['Subject ID'], axis = 1, inplace = True, errors = 'ignore')

oasis_longitudinal.drop(['MRI ID'], axis = 1, inplace = True, errors = 'ignore')

oasis_longitudinal.drop(['Visit'], axis = 1, inplace = True, errors = 'ignore')

oasis_longitudinal.drop(['Group'], axis = 1, inplace = True, errors = 'ignore')

oasis_longitudinal.drop(['Hand'], axis = 1, inplace = True, errors = 'ignore')

oasis_longitudinal.drop(['MR Delay'], axis = 1, inplace = True, errors = 'ignore')

oasis_longitudinal.head()
oasis_cross_sectional.drop(['ID'], axis = 1, inplace = True, errors = 'ignore')

oasis_cross_sectional.drop(['Delay'], axis = 1, inplace = True, errors = 'ignore')

oasis_cross_sectional.drop(['Hand'], axis = 1, inplace = True, errors = 'ignore')

oasis_cross_sectional.head()
frames = [oasis_longitudinal, oasis_cross_sectional]

dataset_final = pd.concat(frames)

dataset_final.head()
dataset_final.info()
data_na = (dataset_final.isnull().sum() / len(dataset_final)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Lost proportion (%)' :round(data_na,2)})

missing_data.head(20)
from sklearn.impute  import SimpleImputer

# We perform it with the most frequent value 

imputer = SimpleImputer ( missing_values = np.nan,strategy='most_frequent')



imputer.fit(dataset_final[['SES']])

dataset_final[['SES']] = imputer.fit_transform(dataset_final[['SES']])



# We perform it with the median

imputer = SimpleImputer ( missing_values = np.nan,strategy='median')



imputer.fit(dataset_final[['MMSE']])

dataset_final[['MMSE']] = imputer.fit_transform(dataset_final[['MMSE']])
# 1= M, 0 = F

dataset_final['M/F'] = dataset_final['M/F'].replace(['M', 'F'], [1,0])  

dataset_final.head(3)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(dataset_final['CDR'].values)

le.classes_
dataset_final['CDR'] = le.transform(dataset_final['CDR'].values) 
# The classes are heavily skewed we need to solve this issue later.

print('Label 0 :', round(dataset_final['CDR'].value_counts()[0]/len(dataset_final) * 100,2), '% of the dataset')

print('Label 0.5 :', round(dataset_final['CDR'].value_counts()[1]/len(dataset_final) * 100,2), '% of the dataset')

print('Label 1 :', round(dataset_final['CDR'].value_counts()[2]/len(dataset_final) * 100,2), '% of the dataset')

print('Label 2 :', round(dataset_final['CDR'].value_counts()[3]/len(dataset_final) * 100,2), '% of the dataset')
dataset_final = dataset_final.drop(dataset_final[dataset_final['CDR']==3].index)

dataset_final.info()
X = dataset_final.drop(["CDR"],axis=1)

y = dataset_final["CDR"].values # 0,0.5=1,1=2,2=3
# We divide our data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    random_state = 42)
print("{0:0.2f}% Train".format((len(X_train)/len(dataset_final.index)) * 100))

print("{0:0.2f}% Test".format((len(X_test)/len(dataset_final.index)) * 100))
print(len(X_train))

print(len(X_test))
FOLDS =10



parametros_gb = {

    "loss":["deviance"],

    "learning_rate": [0.01, 0.025, 0.005,0.5, 0.075, 0.1, 0.15, 0.2,0.3,0.8,0.9],

    "min_samples_split": [0.01, 0.025, 0.005,0.4,0.5, 0.075, 0.1, 0.15, 0.2,0.3,0.8,0.9],

    "min_samples_leaf": [1,2,3,5,8,10,15,20,40,50,55,60,65,70,80,85,90,100],

    "max_depth":[3,5,8,10,15,20,25,30,40,50],

    "max_features":["log2","sqrt"],

    "criterion": ["friedman_mse",  "mae"],

    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    "n_estimators":range(1,100)

    }



model_gb= GradientBoostingClassifier()





gb_random = RandomizedSearchCV(estimator = model_gb, param_distributions = parametros_gb, n_iter = 100, cv = FOLDS, 

                               verbose=2, random_state=42,n_jobs = -1, scoring='f1_macro')

gb_random.fit(X_train, y_train)



gb_random.best_params_
model_gb = GradientBoostingClassifier(subsample = 1.0,n_estimators= 23,

                 min_samples_split = 0.025,

                 min_samples_leaf = 3,

                 max_features = 'log2',

                 max_depth =10,

                 loss = 'deviance',

                 learning_rate = 0.8,

                 criterion= 'mae')

model_gb.fit(X_train,y_train)



Predicted_gb= model_gb.predict(X_test)

Predicted_gb_tr= model_gb.predict(X)





acc = accuracy_score(Predicted_gb, y_test)

acc_tr = accuracy_score(Predicted_gb_tr, y)



test_score = cross_val_score(model_gb, X_train, y_train, cv=FOLDS, scoring='accuracy').mean()

test_f1 = cross_val_score(model_gb, X_train, y_train, cv=FOLDS, scoring='f1_macro').mean()

print("Accuracy Test",acc)

print("Accuracy Training",acc_tr)

print("Accuracy Cross_validate",test_score)

print("F1 Cross_validate",test_f1)

print("F1 Macro:",f1_score(y_test, Predicted_gb, average='macro'))

print("F1 Micro:",f1_score(y_test, Predicted_gb, average='micro'))  

print("F1 Weighted:",f1_score(y_test, Predicted_gb, average='weighted'))



#print(f1_score(y, Predicted_gb_tr, average='macro'))

#print(f1_score(y, Predicted_gb_tr, average='micro'))  

#print(f1_score(y, Predicted_gb_tr, average='weighted'))



print("\nMatrix of confusion")

Predicted_gb= model_gb.predict(X)

confusion_matrix(y, Predicted_gb)

from joblib import dump, load

dump(model_gb, 'model_gb.joblib') 