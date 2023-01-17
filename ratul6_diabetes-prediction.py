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
import matplotlib as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/diabetescsv/diabetes.csv')
df.describe()
# Eye ball the imported dataset
df.info()

# Take away : No missing values. All dimesions are numerics. So, no conversion required.
# However, there are 0 values for Glucose, BllodPressure, SkinThickness, Insulin, BMI , Which cannot be correct. So, needs to 
# be treated.
df.shape # print dimension
df.Outcome.value_counts() # there are more data points for non-diabetics compared to diabetics,model which will be trained 
# using this data ideally it should be good in predicting the non-diabetics patients first.
sns.countplot(x='Outcome' , data =df);
# Check data types of dataset

df.dtypes # all data types are numeric. So, encoding is needed.
df.describe()

#There are incorrect values i.e.0's in Glucose, BloodPressure, SkinThickness, Insulin, BMI. 
# replacing 0 with median of corresponding column.
dataframe_temp = df.drop(["Pregnancies","Outcome"],axis = 1)
dataframe_temp
medians = dataframe_temp.median()
print("medians",medians)
dataframe_nonzero = dataframe_temp.replace(0,medians)
dataframe_nonzero["Pregnancies"] = df["Pregnancies"]
dataframe_nonzero["Outcome"] = df["Outcome"]
dataframe_nonzero
corr = dataframe_nonzero.corr()
corr

# Takeaway : outcome is positively corelated to Glucose feature.
# Age & no. of pregencies have positive corelation.
# BMI & Skin thickness has positive corelation
# No other strong negetive corelation is observed.
sns.heatmap(corr)
sns.pairplot(dataframe_nonzero, diag_kind='kde', hue="Outcome") # plotting pairplot
from sklearn.model_selection import train_test_split
X = dataframe_nonzero.drop('Outcome', axis=1)
Y = dataframe_nonzero['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
from sklearn import svm
from sklearn.svm import SVC

clf = svm.SVC(C = 100,gamma= "scale")
clf.fit(X_train,Y_train)
score1 = clf.score(X_test,Y_test)
score1
from sklearn import metrics
Y_pred = clf.predict(X_test)  
print( metrics.confusion_matrix(Y_test,Y_pred))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#Zscore
from scipy.stats import zscore 
X_train_z = X_train.apply(zscore) # converting to Z score
X_test_z = X_test.apply(zscore)
# Model score on Minmax scaled values
clf = svm.SVC(C = 10,gamma= "scale")
clf.fit(X_train_scaled,Y_train)
score2 = clf.score(X_test_scaled,Y_test)
score2
# Model score using zscore  values
clf = svm.SVC(C = 10,gamma= "scale")
clf.fit(X_train_z,Y_train)
score3 = clf.score(X_test_z,Y_test)
score3
clf = svm.SVC(C = 1000,gamma= "scale")
clf.fit(X_train,Y_train)
score4 = clf.score(X_test,Y_test)
print("Model score for non-scaled datapoints", score4)

# model accuracy has increased on non-scaled data,however for scaled values with c = 1000, model accuracy is decreasing.
import multiprocessing 
from sklearn.model_selection import GridSearchCV
param_grid = [    {        
     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],        
     'C': [ 0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0 ]    } ]
gs = GridSearchCV(estimator=SVC(), param_grid=param_grid,scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
gs.fit(X_train_scaled, Y_train)
gs.best_estimator_
gs.best_score_
from sklearn.metrics import roc_auc_score,roc_curve
auc = roc_auc_score(Y_test,Y_pred)
print("AUC %0.3f" %auc)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
seed = 7
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RFC', RandomForestClassifier()))
results = []
names = []
scoring = 'accuracy'
import warnings
warnings.filterwarnings("ignore")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,
cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
