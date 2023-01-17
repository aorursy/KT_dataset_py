import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import sklearn.metrics as metrics

import pylab

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df= pd.read_csv("/kaggle/input/german-credit-risk-with-classes/gcr_classes.csv")

df1 = pd.read_csv("/kaggle/input/german-credit-dataset-without-vacationothers/german_credit_data1.csv")

df = df.iloc[:,1:11]
df.head()
df1.head()
pylab.scatter(df1.Age, df.Age)

pylab.show()
pylab.scatter(df1["Credit amount"], df["Credit amount"])

pylab.show()
pylab.scatter(df1["Duration"], df["Duration"])

pylab.show()
degiskenler = ['Job', 'Credit amount', 'Age', "Duration","Housing",

       'Saving accounts', 'Checking account', 'Sex_female']

kumeleme = df.drop(degiskenler,axis=1)

kumeleme.head()
k_means = KMeans(n_clusters = 4).fit(kumeleme)

cluster = k_means.labels_

df["Purpose"] = cluster

df.head(12)
y = df["Risk_good"]

X = df.drop(["Risk_good"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=984)
xgb_tuned = XGBClassifier(learning_rate= 0.01, 

                                max_depth= 3,

                          min_child_weight = 22,

                                n_estimators= 500, 

                                subsample= 0.8).fit(X_train, y_train)

y_pred = xgb_tuned.predict(X_test)
print(classification_report(y_test, y_pred))
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
feature_imp = pd.Series(xgb_tuned.feature_importances_,

                        index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index,color="Purple")

plt.xlabel('Değişken Önem Skorları')

plt.ylabel('Değişkenler')

plt.title("Değişken Önem Düzeyleri")

plt.show()