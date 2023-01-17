# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/Admission_Predict.csv")
print(df.info())
print(df.describe().to_string())
print(df.columns)

def Admissible(value):
    mean = df['Chance of Admit '].mean()
    if value >= mean:
        return(1)
    else:
        return(0)

df["AdmitOrNot"] = [Admissible(x) for x in df['Chance of Admit '].values]    
df.hist()
plt.show()
correlations = df.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
names = df.columns
ticks = np.arange(0,len(names),1)

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
plt.xticks(rotation=90)
ax.set_yticklabels(names)
plt.show()

factors = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']
X = df[factors].values
y = df["AdmitOrNot"]

print(X.shape)
print(y.shape)

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

scaling = {"Normalizer" : Normalizer(norm="l2"),
           "Standard Scaler" : StandardScaler(),
           "MinMaxScaler" : MinMaxScaler(feature_range=(0,1)),
           "PCA" : PCA(n_components=3, random_state=0, whiten=False)
            }

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

classifiers = {"K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=3),
               "Random Forest" : RandomForestClassifier(n_estimators=10, random_state=0),
               #"Support Vector Clf" : LinearSVC(penalty="l2", random_state=0), THIS ONE CAUSES ERRORS IN KAGGLE
               "Logistic Regression" : LogisticRegression(penalty="l2", random_state=0),
               "Perceptron" : Perceptron(penalty="l2", random_state=0),
               "Decision Tree" : DecisionTreeClassifier(random_state=0),
               "Naive Bayes" : GaussianNB()
              }

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Let's get learning
fold_splits = 5
rounding_prec = 4 # digits after the decimal
#results_dict = {"Clf Name" : {"Scaling Method":"Accuracy"}}
results_dict = {}
for clfy_name, clfy in classifiers.items(): 

    acc_dict = {}
    acc_dict["Non-scaled"] = round(cross_val_score(clfy, X, y, cv=fold_splits
                                  ).mean(), rounding_prec) 

    for scl_name, sclr in scaling.items(): 
        pipln = []
        pipln.append((scl_name, sclr)) 
        pipln.append(("clf", clfy))

        pip = Pipeline(pipln)
        acc_dict[scl_name] = round(cross_val_score(pip, X, y, cv=fold_splits
                                                  ).mean(), rounding_prec)
    results_dict[clfy_name] = acc_dict

# MAKE DF
df_ML_acc = pd.DataFrame(results_dict)
df_ML_acc.name = "Machine Learning Models Accuracy Scores"
print(df_ML_acc.to_string())
print('---*---Best Results:')
print(df_ML_acc.max())


