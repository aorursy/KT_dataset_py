# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd



%matplotlib inline 

import matplotlib.pyplot as plt 

# subplots

import matplotlib.gridspec as gridspec 

import mpld3 as mpl



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics



df = pd.read_csv("../input/data.csv",header = 0)

df.head()



df.drop('id',axis=1,inplace=True)

df.drop('Unnamed: 32',axis=1,inplace=True)

len(df)



df.diagnosis.unique()



df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

df.head()



df.describe



plt.hist(df['diagnosis'])

plt.title('Diagnosis (M=1 , B=0)')

plt.show()



features_mean=list(df.columns[1:11])

dfM=df[df['diagnosis'] ==1]

dfB=df[df['diagnosis'] ==0]



plt.rcParams.update({'font.size': 8})

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))

axes = axes.ravel()

for idx,ax in enumerate(axes):

    ax.figure

    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50

    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])

    ax.legend(loc='upper right')

    ax.set_title(features_mean[idx])

plt.tight_layout()

plt.show()



traindf, testdf = train_test_split(df, test_size = 0.3)





predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']

outcome_var='diagnosis'

model=LogisticRegression()

classification_model(model,traindf,predictor_var,outcome_var)



predictor_var = ['radius_mean']

model=LogisticRegression()

classification_model(model,traindf,predictor_var,outcome_var)



predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']

model = DecisionTreeClassifier()

classification_model(model,traindf,predictor_var,outcome_var)



predictor_var = ['radius_mean']

model = DecisionTreeClassifier()

classification_model(model,traindf,predictor_var,outcome_var)



#Random forest



predictor_var = features_mean

model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)

classification_model(model, traindf,predictor_var,outcome_var)



featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)

print(featimp)



predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean',]

model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)

classification_model(model,traindf,predictor_var,outcome_var)



predictor_var =  ['radius_mean']

model = RandomForestClassifier(n_estimators=100)

classification_model(model, traindf,predictor_var,outcome_var)



#Using on the test data set¶

predictor_var = features_mean

model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)

classification_model(model, testdf,predictor_var,outcome_var)


