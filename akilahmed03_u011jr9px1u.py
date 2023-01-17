# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside4of the current session
from sklearn import preprocessing

import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold



import warnings

warnings.filterwarnings('ignore')



import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/summeranalytics2020/train.csv')

df_test = pd.read_csv("../input/summeranalytics2020/test.csv")

df.head()
print(df['Attrition'].value_counts())

df['Behaviour'].value_counts()
df = df.drop_duplicates(subset = ["Age", "Attrition", "BusinessTravel", "Department", "DistanceFromHome", "Education", "EducationField", "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "CommunicationSkill"])
# THE NEW SHAPE OF THE COLUMN

df.shape
# CHANGING CATEGORICAL DATA TO NUMERICAL DATA

from sklearn.preprocessing import LabelEncoder



labelencoder = LabelEncoder()



df['BusinessTravel'] = labelencoder.fit_transform(df['BusinessTravel'])

df['Department'] = labelencoder.fit_transform(df['Department'])

df['EducationField'] = labelencoder.fit_transform(df['EducationField'])

df['Gender'] = labelencoder.fit_transform(df['Gender'])

df['JobRole'] = labelencoder.fit_transform(df['JobRole'])

df['MaritalStatus'] = labelencoder.fit_transform(df['MaritalStatus'])

df['OverTime'] = labelencoder.fit_transform(df['OverTime'])



df_test['BusinessTravel'] = labelencoder.fit_transform(df_test['BusinessTravel'])

df_test['Department'] = labelencoder.fit_transform(df_test['Department'])

df_test['EducationField'] = labelencoder.fit_transform(df_test['EducationField'])

df_test['Gender'] = labelencoder.fit_transform(df_test['Gender'])

df_test['JobRole'] = labelencoder.fit_transform(df_test['JobRole'])

df_test['MaritalStatus'] = labelencoder.fit_transform(df_test['MaritalStatus'])

df_test['OverTime'] = labelencoder.fit_transform(df_test['OverTime'])
# DROPPING "Behaviour" COLUMN, CAUSE IT IS OF NO USE

df_test = df_test.drop(["Behaviour"], axis = 1)

df = df.drop(["Behaviour"], axis = 1)
# SEPARATING FEAURE AND TARGET COLUMN

X =  df.loc[:, df.columns != 'Attrition']

y = df.loc[:, df.columns == 'Attrition']
# NOW LET'S SCALE THE DATA

from sklearn.preprocessing import scale

X = scale(X)

df_test_final = scale(df_test)
# WITH HEAT MAP WE CAN FIND RELATION SHIPS BETWEEN EACH COLUMNS

plt.figure(figsize=(20,20))

sns.heatmap(df.corr(), annot=True, fmt='.0%')
# LET'S HISTOGRAM FOR NUMERICAL DATA OF DATAFRAME

df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)# ; avoid having the matplotlib verbose informations
# SPLITTING THE DATA

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 1)
# CHECKING THE SHAPE OF DATA

print(x_train.shape)

print(y_train.shape)

print(x_val.shape)
from sklearn.metrics import accuracy_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score
# PREDICTING WITH LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.5, penalty='l2', random_state=1)

logreg_train = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.5, penalty='l2', random_state=1)

logreg_full = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.5, penalty='l2', random_state=1)



logreg_full = logreg_full.fit(X, y)

logreg_train = logreg_train.fit(x_train, y_train)



y_pred_train = logreg_full.predict(x_train)

y_pred_val = logreg_train.predict_proba(x_val) 

y_pred_val = y_pred_val[:,1]



y_pred = logreg_full.predict_proba(df_test_final)

df_logreg_final = y_pred[:,1]



print("------logreg-------")

print("accuracy", accuracy_score(y_train, y_pred_train))

print("roc_auc", roc_auc_score(y_train, y_pred_train))

print("roc_auc", roc_auc_score(y_val, y_pred_val))
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(loss = "deviance", learning_rate = 1, n_estimators = 5500, criterion = "friedman_mse")

gbc_train = GradientBoostingClassifier(loss = "deviance", learning_rate = 1, n_estimators = 5500, criterion = "friedman_mse")

gbc_full = GradientBoostingClassifier(loss = "deviance", learning_rate = 1, n_estimators = 5500, criterion = "friedman_mse")



gbc_train = gbc_train.fit(x_train, y_train) 

gbc_full = gbc_full.fit(X, y)



y_pred_train = gbc_full.predict(x_train)

y_pred_val = gbc_train.predict_proba(x_val) 

y_pred_val = y_pred_val[:, 1]



y_pred = gbc_full.predict_proba(df_test_final)

df_gbc_final = y_pred[:,1]



print("------gbc-------")

print("accuracy", accuracy_score(y_train, y_pred_train))

print("roc_auc", roc_auc_score(y_train, y_pred_train))

print("roc_auc", roc_auc_score(y_val, y_pred_val))
from sklearn.svm import SVC



svc = SVC(C = 1, kernel = 'rbf',random_state = 42, probability = True)

svc_train = SVC(C = 1, kernel = 'rbf',random_state = 42, probability = True)

svc_full = SVC(C = 1, kernel = 'rbf',random_state = 42, probability = True)



svc_full = svc_full.fit(X, y)

svc_train = svc_train.fit(x_train, y_train)



y_pred_train = svc_full.predict(x_train) 



y_pred_val = svc_train.predict_proba(x_val) 

y_pred_val = y_pred_val[:,1]



y_pred = svc_full.predict_proba(df_test_final)

df_svc_final = y_pred[:,1]



print("accuracy of traing: ",accuracy_score(y_train, y_pred_train))

print("roc_auc of training: ", roc_auc_score(y_train, y_pred_train))

print("roc_auc of validation: ", roc_auc_score(y_val, y_pred_val))
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes=(4,2,2), activation='relu', solver='adam', max_iter=1000, alpha = 0.0007, random_state = 56, learning_rate = "invscaling")

mlp_test = MLPClassifier(hidden_layer_sizes=(4,2,2), activation='relu', solver='adam', max_iter=1000, alpha = 0.0007, random_state = 56, learning_rate = "invscaling")

mlp_full = MLPClassifier(hidden_layer_sizes=(4,2,2), activation='relu', solver='adam', max_iter=1000, alpha = 0.0007, random_state = 56, learning_rate = "invscaling")



mlp_full = mlp_full.fit(X, y)

mlp_test = mlp_test.fit(x_train, y_train)



y_pred_train = mlp_full.predict(x_train) 



y_pred_val = mlp_test.predict_proba(x_val) 

y_pred_val = y_pred_val[:,1]



y_pred = mlp_full.predict_proba(df_test_final)

df_svc_final = y_pred[:,1]



print("accuracy of traing: ",accuracy_score(y_train, y_pred_train))

print("roc_auc of training: ", roc_auc_score(y_train, y_pred_train))

print("roc_auc of validation: ", roc_auc_score(y_val, y_pred_val))
from sklearn.ensemble import VotingClassifier 



estimator = [] 

estimator.append(('LR', logreg)) 

estimator.append(('SV', svc))

estimator.append(('MLP', mlp))

estimator.append(('NB', gbc))



vot_test = VotingClassifier(estimators = estimator, voting ='soft') 

vot_test = vot_test.fit(x_train, y_train) 

y_pred_test = vot_test.predict_proba(x_val) 



vot_full = VotingClassifier(estimators = estimator, voting ='soft') 

vot_full = vot_full.fit(X, y) 

y_pred = vot_test.predict(x_train) 



print("accuracy of traing: ",accuracy_score(y_train, y_pred))

print("roc_auc of training: ", roc_auc_score(y_train, y_pred))

print("roc_auc of validation: ", roc_auc_score(y_val, y_pred_test[:,1]))
y_vc_pred = vot_full.predict_proba(df_test_final)

y_vc_pred_final = y_vc_pred[:, 1]

print(len(y_vc_pred))
my_submission = pd.DataFrame({'Id': df_test.Id, 'Attrition': y_vc_pred_final})

my_submission.to_csv('submission.csv', index=False)