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
import pandas as pd # data processing

%matplotlib inline 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import mpld3 as mpl
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold   #K-fold cross validation
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
filename = "../input/breastdata/data.csv"
print(filename)
df = pd.read_csv(filename,header = 0)
df.head()
df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
# size of the data
len(df)
df.diagnosis.unique()
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()
#PCA to determine the important feature
pca = PCA(n_components=0.95)
pca.fit(df)
print(pca.explained_variance_ratio_)
traindf, testdf = train_test_split(df, test_size = 0.2)
#Fit model
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
    
  #Make predictions on training and testing set:
  predictions = model.predict(data[predictors])
  predictions1 = model.predict(testdf[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  accuracy1 = metrics.accuracy_score(predictions1,testdf[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))
  print("Accuracy on the test set : %s" % "{0:.3%}".format(accuracy1))


  #k-fold cross-validation
  kf = KFold(n_splits=5)
  error = []
  for train, test in kf.split(data):
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))    
  model.fit(data[predictors],data[outcome]) 
#LogisticRegression
predictor_var = ['radius_mean']
outcome_var = 'diagnosis'
model=LogisticRegression(C=0.01,class_weight={1:0.6,0:0.4})
classification_model(model,traindf,predictor_var,outcome_var)
#KNN
predictor_var = ['radius_mean']
outcome_var = 'diagnosis'
k_range = range(1,31)
for k in k_range:
    print("k = ",k)
    model=KNeighborsClassifier(n_neighbors=k)
    classification_model(model,traindf,predictor_var,outcome_var)
#DecisionTree
predictor_var = ['radius_mean']
outcome_var = 'diagnosis'
model=DecisionTreeClassifier()
classification_model(model,traindf,predictor_var,outcome_var)