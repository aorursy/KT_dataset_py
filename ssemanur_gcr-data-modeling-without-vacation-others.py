import numpy as np

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

from yellowbrick.cluster import KElbowVisualizer

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import classification_report

import sklearn.metrics as metrics





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/german-credit-data-with-risk/german_credit_data.csv")

df.drop(df.columns[[0]],axis=1,inplace=True)
sns.set(style="darkgrid")

sns.boxenplot(x="Purpose", y="Credit amount",

              color="r",

              scale="linear", data=df);
df1=pd.read_csv("../input/german-credit-dataset-without-vacationothers/german_credit_data1.csv")

df1.head()
df1["Saving accounts"].fillna("no_account",inplace=True)

df1["Checking account"].fillna("no_account",inplace=True)
add = pd.DataFrame(

        {'Housing': pd.Categorical(

              values =  df1["Housing"],

              categories=["free","rent","own"]),



         'Saving accounts': pd.Categorical(

             values = df1["Saving accounts"],

             categories=["no_account","little","moderate","rich","quite rich"]),



         'Checking account': pd.Categorical(

             values = df1["Checking account"],

             categories=["no_account","little","moderate","rich"]),

         'Purpose': pd.Categorical(

             values = df1["Purpose"],

             categories=["repairs","domestic appliances","furniture/equipment"

                         ,"radio/TV","education","business","car"])

        }

    )
add = add.apply(lambda x: x.cat.codes)

add.head()
del df1["Saving accounts"]

del df1["Checking account"]

del df1["Housing"]

del df1["Purpose"]

df2 = pd.concat([df1,add],axis=1)

df2.head()
df2=pd.get_dummies(df2, columns = ["Sex"], prefix = ["Sex"])

df2=pd.get_dummies(df2, columns = ["Risk"], prefix = ["Risk"])
del df2["Sex_male"]

del df2["Risk_bad"]

df2.rename(columns={"Risk_good":"Risk",

                  "Sex_female":"Sex"},inplace=True)

from sklearn.cluster import KMeans

degiskenler = ['Checking account', 'Risk', 'Duration', 'Purpose',"Credit amount",

       'Saving accounts', 'Housing', 'Sex']

kumeleme = df2.drop(degiskenler,axis=1)

kumeleme
kmeans = KMeans()

visu = KElbowVisualizer(kmeans, k = (2,20))

visu.fit(kumeleme)

visu.poof()
k_means = KMeans(n_clusters = 3).fit(kumeleme)

cluster = k_means.labels_

df2["Age"] = cluster
degiskenler = ['Job', 'Age', 'Duration', 'Risk',"Housing",

       'Saving accounts', 'Checking account', 'Sex']

kumeleme = df2.drop(degiskenler,axis=1)

kumeleme
kmeans = KMeans()

visu = KElbowVisualizer(kmeans, k = (2,20))

visu.fit(kumeleme)

visu.poof()
k_means = KMeans(n_clusters = 4).fit(kumeleme)

cluster = k_means.labels_

df2["Credit amount"] = cluster
degiskenler = ['Job', 'Age', 'Checking account', 'Purpose',"Housing",

       'Saving accounts', 'Credit amount', 'Sex']

kumeleme = df2.drop(degiskenler,axis=1)

kumeleme
kmeans = KMeans()

visu = KElbowVisualizer(kmeans, k = (2,20))

visu.fit(kumeleme)

visu.poof()
k_means = KMeans(n_clusters = 3).fit(kumeleme)

cluster = k_means.labels_

df2["Duration"] = cluster
degiskenler = ['Job', 'Age', 'Checking account', 'Duration',"Housing",

       'Saving accounts', 'Credit amount', 'Sex']

kumeleme = df2.drop(degiskenler,axis=1)

kumeleme
k_means = KMeans(n_clusters = 4).fit(kumeleme)

cluster = k_means.labels_

df2["Purpose"] = cluster
df2.head()
y = df2["Risk"]

X = df2.drop(["Risk"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30, 

                                                    random_state=536)
xgb_tuned = XGBClassifier(learning_rate= 0.01, 

                                max_depth= 3,

                          min_child_weight = 38,

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



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Importance')

plt.ylabel('Features')

plt.title("Features Skor Levels")

plt.show()