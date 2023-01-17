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
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Packages used to view data analysis graphs
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Metrics packages to analyze the efficiency of the algorithm 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
 
# Packages to standardize, normalize data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Package to share traning and test data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 

# Package to generating the predective model report
from sklearn.metrics import classification_report

# Pacotes de Modelos preditivos
# Predective models packages
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Attribute selection and dimensioning reduction packages 
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

# package to balance classes
from imblearn.over_sampling import SMOTE

# Pipeline package
from sklearn.pipeline import Pipeline

# Package to not show the warnings
import warnings
warnings.filterwarnings("ignore")
# importing the training file with the target variable
x_train = pd.read_csv("/kaggle/input/santander-customer-satisfaction/train.csv")
# Loading the TARGET training variable
y_train = x_train.TARGET
# Transforming the TARGET variable file into DataFrame
y_train = y_train.to_frame()
# Importing the test file  
x_test = pd.read_csv("/kaggle/input/santander-customer-satisfaction/test.csv")
# Loanding the TARGET file
y_test = pd.read_csv("/kaggle/input/santander-customer-satisfaction/sample_submission.csv")
y_test.TARGET.value_counts()
# Veriying the the shape of the data
x_train.shape, x_test.shape, y_test.shape, y_train.shape
# Excluding column ID
x_train.drop(['ID','TARGET'], axis=1, inplace=True)
x_test.drop(['ID'], axis=1, inplace=True)
y_test.drop(['ID'], axis=1, inplace=True)
# Checking the shape of the data
x_train.shape, x_test.shape, y_test.shape, y_train.shape
type(y_train)
type(y_test)
# Loading training data and test data in one Dataset
X_dados = pd.concat((x_train.loc[:,'var3':'var38'], 
                      x_test.loc[:,'var3':'var38']))
Y_dados = pd.concat((y_train, y_test))
y_train.head(), type(y_train)
# Checking and deleting duplicate columns:
# checking the shape, decreased
col_duplicates = []
columns = X_dados.columns
for i in range(len(columns)-1):
    s = X_dados[columns[i]].values
    for j in range(i+1, len(columns)):
        if np.array_equal(s, X_dados[columns[j]].values):
            col_duplicates.append(columns[j])
        
x_train.drop(col_duplicates, axis=1, inplace=True)
x_test.drop(col_duplicates, axis=1, inplace=True)
X_dados.drop(col_duplicates, axis=1, inplace=True)

# Checking shape
x_train.shape, x_test.shape, X_dados.shape
# Removes columns with constant values

cols_Remove = []
for col in X_dados.columns:
    if X_dados[col].std() == 0:
        cols_Remove.append(col)
        
x_train.drop(cols_Remove, axis=1, inplace=True)
x_test.drop(cols_Remove, axis=1, inplace=True)
X_dados.drop(cols_Remove, axis=1, inplace=True)

x_train.shape, x_test.shape, X_dados.shape
# top 5 values more communs
# Note that the negative value -999999 is incompatible to enter the predictive model algorithm
x_train.var3.value_counts()[:5], x_test.var3.value_counts()[:5], X_dados.var3.value_counts()[:5]
# 116 values of -999999 were found in column var3 which is suspected to be the client's nationality and
# the value -999999 is can be said to be the nationality unknown to the client or has not been placed


x_train.loc[x_train.var3==-999999].shape, x_test.loc[x_test.var3==-999999].shape, X_dados.loc[X_dados.var3==-999999].shape 
# So we are going to replace the value -999999 by the value 2 fashion, which is the most repeated value  

x_train.var3 = x_train.var3.replace(-999999,2)
x_test.var3 = x_test.var3.replace(-999999,2)
X_dados.var3 = X_dados.var3.replace(-999999,2)

x_train.loc[x_train.var3==999999].shape, x_test.loc[x_test.var3==999999].shape, X_dados.loc[X_dados.var3==999999].shape
# Gathering training data x and y
xy_train = x_train.copy()
xy_train['TARGET'] = y_train['TARGET']
xy_train.shape, type(xy_train)
# Checking shape
x_train.shape, x_test.shape, y_test.shape, y_train.shape, X_dados.shape, Y_dados.shape, xy_train.shape
# Data Normalization:
x_train_normalizados = x_train.apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
x_test_normalizados = x_test.apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
X_dados_normalizados = X_dados.apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
xy_train_normalizados = xy_train.apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
# Calculating the standard deviation of each column and removing columns with a standard deviation less than or equal to 0.07
# then 52 columns remain.
# Based on the standard deviation of the VAR3 variable, 
# any variable that has a standard deviation less than 0.041125 will be excluded.

XNR = X_dados_normalizados.copy()
cols_Remove = []

for col in XNR.columns:
    if XNR[col].std() < 0.07:
        cols_Remove.append(col)
        
XNR.drop(cols_Remove, axis=1, inplace=True)

Col_Excluidas = len(X_dados_normalizados.columns) - len(XNR.columns)

msg = '%s columns were excluded  \nand %s colunms left' % (len(cols_Remove), len(XNR.columns))

print(msg)
# x_train.drop(cols_Remove, axis=1, inplace=True)
# x_test.drop(cols_Remove, axis=1, inplace=True)
# X_dados.drop(cols_Remove, axis=1, inplace=True)

# x_train.shape, x_test.shape, X_dados.shape
# Happy customers have TARGET==0, unhappy custormers have TARGET==1.
# The most customers are classified satisfied customers, almost 4% are just dissatisfied customers.
# The TARGET variable is not balanced.
df = pd.DataFrame(xy_train.TARGET.value_counts())
df['Porcentagem'] = 100*df['TARGET']/xy_train.shape[0]
df
%%time

# The CART algorithm showed the best precision among the others
array = XNR
labels = array.columns

X = array[labels]
y = Y_dados
    
X_resampled, y_resampled = SMOTE(sampling_strategy=0.2).fit_resample(X, y)   

X_train, X_test, y_train, y_test = train_test_split(X_resampled, 
                                                    y_resampled, 
                                                    test_size=0.2, 
                                                    random_state=1)

model = DecisionTreeClassifier()
modelo = model.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
msg = " Accuracy: %.2f%%" % ((accuracy * 100.0))    


print(msg)
# Making predictions and building the report
report = classification_report(y_test, y_pred)

# Printing the report
print(report)
%%time
# Confusion Matrix
# Now let's check the accuracy in a table format with DecisionTreeClassifier (CART)

# loading and share data in predictive variables and the target variable
array = XNR
labels = array.columns

X = array[labels]
y = Y_dados


# Splitting data into training and testing
model = SMOTE()
X_resampled, y_resampled = model.fit_resample(X, y)   

X_train, X_test, y_train, y_test = train_test_split(X_resampled, 
                                                    y_resampled, 
                                                    test_size=0.2, 
                                                    random_state=1)

# Creating a model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions and building the Confusion Matrix
previsoes = model.predict(X_test)
matrix = confusion_matrix(y_test, previsoes)

# Printing Confusion Matrix
print(matrix)
print(previsoes)
print(model.feature_importances_)
%%time

# Loading datas

array = XNR
labels = array.columns

X = array[labels]
y = Y_dados
    
# Separating training data and test data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1)

# Division of training data into training data and validation data
x_train_res, x_val, y_train_res, y_val = train_test_split(X_train, y_train,
                                                  test_size = .1,
                                                  random_state=12)

# Applying SMOTE for class balancing
modeloSMOTE = SMOTE(sampling_strategy='all', k_neighbors=5)
X_resampled, y_resampled = modeloSMOTE.fit_sample(x_train_res, y_train_res)
        
# Creating the model
model = DecisionTreeClassifier()
modelo = model.fit(X_resampled, y_resampled)
y_pred = modelo.predict(X_test)
          
# Evaluating the model and updating the accuracy list
score = model.score(x_val, y_val)
print("Accuracy is = %.2f%%" % ( score * 100))

# Making predictions and building the report
report = classification_report(y_test, y_pred)

# Printing the report
print(report)