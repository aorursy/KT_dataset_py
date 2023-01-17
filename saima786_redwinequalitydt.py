import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
quality_mapping={3:0,
                4:1,
                5:2,
                6:3,
                7:4,
                8:5
                }
df.loc[:,"quality"]=df.quality.map(quality_mapping)
df.shape
df_train=df.head(1000)
df_test=df.tail(599)
from sklearn import tree
from sklearn import metrics
df.columns
cols=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']

clf=tree.DecisionTreeClassifier(max_depth=3
                               )
clf.fit(df_train[cols],df_train.quality)
train_pred=clf.predict(df_train[cols])
test_pred=clf.predict(df_test[cols])
train_acc=metrics.accuracy_score(df_train.quality,train_pred)
test_acc=metrics.accuracy_score(df_test.quality,test_pred)
    
df_test.quality
test_pred
