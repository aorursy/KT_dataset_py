import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

!pip install multi_imbalance

from multi_imbalance.resampling.mdo import MDO

pd.set_option("display.max_rows",None)
src = pd.read_csv("../input/fish-market/Fish.csv")
src.head()
src.dtypes
src.isnull().mean()
src.shape
src.Species.value_counts()
src.hist(figsize=(10,10))

plt.show()
sns.pairplot(src,diag_kind="kde")

plt.show()
sns.heatmap(src.corr(),annot=True,mask=np.triu(src.corr()))

plt.show()
X = src.iloc[:,1:].copy()

y = src.iloc[:,0].copy()

le = LabelEncoder()

ylab= le.fit_transform(y)

labels = pd.DataFrame({"y":y,"ylabel":ylab})

labels.drop_duplicates(inplace=True)

labels = labels.sort_values(by="ylabel")

labels
X_train, X_test, y_train, y_test = train_test_split(X,ylab,test_size=0.2,random_state=11)

X_train.shape
pd.Series(y_train).value_counts()
dtree_model = DecisionTreeClassifier(random_state=11)

dtree_grid = GridSearchCV(estimator=dtree_model,

                       param_grid={"max_leaf_nodes":list(np.arange(2,50,1)),

                                  "criterion":["gini","entropy"]}

                                  ,scoring="f1_weighted",cv=5,n_jobs=-1)

m = dtree_grid.fit(X_train,y_train)

print("Best Model: "+str(m.best_estimator_))

print("Decision Tree CV F1 Score: "+str(m.best_score_))

print("Decision Tree Test F1 Score: "+str(m.score(X_test,y_test)))
final_dtree = m.best_estimator_

final_dtree.fit(X_train,y_train)

plt.figure(figsize=(17,20))

plot_tree(final_dtree,filled=True,feature_names=list(X.columns))

plt.show()
mdo = MDO(k=3, k1_frac=0, seed=0)

X_train_mdo, y_train_mdo = mdo.fit_resample(X_train, y_train)

pd.Series(y_train_mdo).value_counts()
dtree_model = DecisionTreeClassifier(random_state=11)

dtree_grid = GridSearchCV(estimator=dtree_model,

                       param_grid={"max_leaf_nodes":list(np.arange(2,50,1)),

                                  "criterion":["gini","entropy"]}

                                  ,scoring="f1_weighted",cv=5,n_jobs=-1)

m = dtree_grid.fit(X_train_mdo,y_train_mdo)

print("Best Model: "+str(m.best_estimator_))

print("Decision Tree CV F1 Score: "+str(m.best_score_))

print("Decision Tree Test F1 Score: "+str(m.score(X_test,y_test)))
final_dtree_mdo = m.best_estimator_

final_dtree_mdo.fit(X_train_mdo,y_train_mdo)

plt.figure(figsize=(17,20))

plot_tree(final_dtree_mdo,filled=True,feature_names=list(X.columns))

plt.show()
pred = final_dtree_mdo.predict(X_test)
fig = plt.figure(figsize=(8,7))

sns.heatmap(confusion_matrix(y_test,pred,labels=list(labels["ylabel"])),annot=True,xticklabels=labels["y"],yticklabels=labels["y"])

plt.show()
print(accuracy_score(y_test,pred))