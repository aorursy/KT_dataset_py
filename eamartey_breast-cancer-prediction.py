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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.diagnosis.replace("M", 1, inplace = True)
df.diagnosis.replace("B", 0, inplace = True)
df.drop(['Unnamed: 32','id'], axis = 1, inplace = True)
df.head()
df.isna().sum()
df.head()
dfcorr = df.corr(method = 'pearson')
dfcorr
sns.heatmap(dfcorr)
 
plt.subplots(figsize = (12,8))
sns.heatmap(dfcorr)

sns.scatterplot('radius_mean', 'texture_mean', data = df)
sns.scatterplot('radius_mean', 'perimeter_mean', data = df) ## The two features have a one-to-one correlation
sns.scatterplot('radius_mean', 'area_mean', data = df)
sns.scatterplot('perimeter_mean', 'area_mean', data = df) # Perimeter_mean and radius_mean are having the same effect on area_mean
sns.scatterplot('perimeter_worst', 'radius_worst', data = df) # These two also have an almost one-to-one correlation
sns.scatterplot('area_worst', 'radius_worst', data = df)
sns.scatterplot('area_worst', 'perimeter_worst', data = df)
sns.scatterplot('area_se', 'perimeter_se', data = df) # Perimeter_se looks like it has a one to one effect on area_se
sns.scatterplot('area_se', 'radius_se', data = df) # Radius_se has more of an exponential effect on the area_se
sns.scatterplot('radius_se', 'perimeter_se', data = df)
df.drop(['perimeter_worst', 'perimeter_mean'], axis = 1, inplace = True)
dfcorr = df.corr(method = 'pearson')
plt.figure(figsize = (20, 15))
sns.heatmap(dfcorr)
y = df.iloc[:, :1]
x = df.iloc[:, 1:]
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
dt = DecisionTreeClassifier(random_state = 1)
rf = RandomForestClassifier(random_state = 1)
ss = StandardScaler()
lr = LinearRegression()
lor = LogisticRegression()
gr = GradientBoostingRegressor(learning_rate= 0.9)
gc = GradientBoostingClassifier()
knn = KNeighborsClassifier()
svc = SVC(C = 0.5, kernel='linear')
xtrain,xtest,ytrain,ytest = train_test_split(x,y, random_state = 2, test_size = 0.15)
pipe = Pipeline([('ss',ss),('rf', rf)]) # Created a pipeline using Standard Scalar and Random Forest
pipe.fit(xtrain, ytrain)
pipe.score(xtrain, ytrain)
pipe.score(xtest, ytest)
 
rf.fit(xtrain, ytrain) # Random Forest
rf.score(xtrain, ytrain)
rf.score(xtest, ytest)
dt.fit(xtrain,ytrain)  # Decision Tree
dt.score(xtrain, ytrain)
dt.score(xtest, ytest)
lor.fit(xtrain,ytrain)  # Logistic Regression
lor.score(xtrain, ytrain)
lor.score(xtest, ytest)
gc.fit(xtrain,ytrain) # Gradient Boosting Classifier
gc.score(xtrain, ytrain)
gc.score(xtest, ytest)
k_range = range(1,10)   ## KNearest Neighbor
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(xtrain,ytrain)
    knn.score(xtrain, ytrain)
    score_knn = round(knn.score(xtest, ytest) * 100, 2)
    scores_list.append(score_knn)

print("The accuracy score achieved using KNN is: " + str(max(scores_list)) + " %")

estimators = [('knn', knn), ('svc',svc)]
stack = StackingClassifier(estimators, final_estimator= None)
stack.fit(xtrain, ytrain)
stack.score(xtrain, ytrain)
stack.score(xtest, ytest)
svc = SVC(C = 0.9, kernel='linear') # SVC
svc.fit(xtrain,ytrain)
svc.score(xtrain, ytrain)
svc.score(xtest, ytest)
svc_poly = SVC(C = 0.9, kernel='poly') # SVC
svc_poly.fit(xtrain,ytrain)
svc_poly.score(xtrain, ytrain)
svc_poly.score(xtest, ytest)
svc_rbf = SVC(C = 0.9, kernel='rbf') # SVC
svc_rbf.fit(xtrain,ytrain)
svc_rbf.score(xtrain, ytrain)
svc_rbf.score(xtest, ytest)


import tensorflow as tf
from sklearn.metrics import accuracy_score
df.head()
df1 = pd.DataFrame.to_numpy(df)
y = df1[:, 0]
x = df1[:,1:]
xtrain,xtest,ytrain,ytest = train_test_split(x,y, random_state = 2, test_size = 0.15)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1, activation = 'hard_sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(xtrain, ytrain, batch_size = 50, epochs = 150)
ypred = ann.predict(xtest)
ypred = (ypred>0.5)
accuracy_score(ytest, ypred)
