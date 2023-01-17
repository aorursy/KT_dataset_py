import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline
df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

attrition = {"No":0,"Yes":1}
df.Attrition = df.Attrition.map(attrition)

df.head()
dummies = pd.get_dummies(df)

cols = dummies.columns.tolist()

cols.remove("Attrition")



X = dummies[cols]

y = dummies.Attrition.values.reshape(-1,)
pca = PCA()



X_train, X_test, y_train, y_test = train_test_split(X,y)
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.fit_transform(X_test)
X_train_pca = pca.fit_transform(X_train_std)

X_test_pca = pca.transform(X_test_std)
lr = LogisticRegressionCV(cv=10)

lr.fit(X_train_pca,y_train)
y_pred = lr.predict(X_test_pca)



cm_df = pd.DataFrame(confusion_matrix(y_pred=y_pred,y_true=y_test).T)

cm_df.columns.name = "True"

cm_df.index.name = 'Predicted'

print(cm_df)

print(classification_report(y_pred = y_pred, y_true=y_test))