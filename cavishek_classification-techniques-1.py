from IPython.display import Image

Image('decision_tree.png')
from IPython.display import Image

Image('ex2.png')
from IPython.display import Image

Image('dt-1.png')
from IPython.display import Image

Image('wine.png')
import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix

%matplotlib inline

wine_df = pd.read_csv("wineQualityReds.csv")

wine_df.drop(wine_df.columns[0],axis=1,inplace=True)

wine_df.head()
wine_df.info()
wine_df.describe()
plt.figure(figsize=(10,8))

sns.heatmap(wine_df.corr(),

            annot=True,

            linewidths=.5,

            center=0,

            cbar=False,

            cmap="YlGnBu")

plt.show()

def pairplot(cols_keep):

    g = sns.pairplot(wine_df,vars=cols_keep,kind='reg',hue='quality')

    return
cols_keep = list(wine_df.columns[0:3])

pairplot(cols_keep)

cols_keep = list(wine_df.columns[3:7])

pairplot(cols_keep)
cols_keep = list(wine_df.columns[7:11])

pairplot(cols_keep)
X_train, X_test, y_train, y_test =train_test_split(wine_df.drop('quality',axis=1), wine_df['quality'], test_size=.25,

                                                   random_state=22)

X_train.shape,X_test.shape
model1=DecisionTreeClassifier()
model1.fit(X_train, y_train)

preds = model1.predict(X_test)
print(accuracy_score(y_test,preds))

print(recall_score(y_test,preds,average="weighted"))

print(precision_score(y_test,preds,average="weighted"))

print(f1_score(y_test,preds,average="weighted"))

print(metrics.classification_report(y_test,preds))
model2=DecisionTreeClassifier(criterion='entropy')
model2.fit(X_train, y_train)
preds = model1.predict(X_test)
print(accuracy_score(y_test,preds))

print(recall_score(y_test,preds,average="weighted"))
print(precision_score(y_test,preds,average="weighted"))

print(f1_score(y_test,preds,average="weighted"))

print(metrics.classification_report(y_test,preds))