import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv("../input/glass.csv")
data.head(5)     # it seems like columns are showing possible elements and rows are showing the amount of them. And Type column is or label
data.tail()     # I see that data is sorted by label.   Data needs shuffling before training and splitting.
data.info()          # all of the features are in float format and non-null which is perfect
data.describe()             # I see that some columns has generally big values (Si), while some has very small values (Fe).  Data needs normalization before training
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(24,16))

plt.subplot(3,3,1)
sns.violinplot(data.Type,data.RI)

plt.subplot(3,3,2)
sns.violinplot(data.Type,data.Na)

plt.subplot(3,3,3)
sns.violinplot(data.Type,data.Mg)

plt.subplot(3,3,4)
sns.violinplot(data.Type,data.Al)

plt.subplot(3,3,5)
sns.violinplot(data.Type,data.Si)

plt.subplot(3,3,6)
sns.violinplot(data.Type,data.K)

plt.subplot(3,3,7)
sns.violinplot(data.Type,data.Ca)

plt.subplot(3,3,8)
sns.violinplot(data.Type,data.Ba)

plt.subplot(3,3,9)
sns.violinplot(data.Type,data.Fe)

plt.show()
plt.figure(figsize=(24,20))
sns.heatmap(data.corr(),annot=True,linecolor="white",linewidths=(1,1),cmap="winter")
plt.show()
sns.pairplot(data=data,hue="Type",vars=["RI", "Na","Mg","Al","Si","K","Ca","Ba","Fe"])
plt.show()
data = data.sample(frac=1,random_state=22)
data.head()  # as you can see below, it has changed rows randomly (even indexes are same as what they corresponded before)
y = data.Type.values.reshape(-1,1)
data.drop(["Type"],axis=1,inplace=True)

x = data.values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.3)
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
log_res_model = LogisticRegression(solver="newton-cg",max_iter=400,multi_class="multinomial",random_state=42)

log_res_model.fit(x_train,y_train.ravel())
y_pred = log_res_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Logistic Regression: ",log_res_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()
from sklearn.tree import DecisionTreeClassifier
dec_tree_model = DecisionTreeClassifier(min_samples_split=4,random_state=42)
dec_tree_model.fit(x_train,y_train)
y_pred = dec_tree_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Decision Tree Classifier: ",dec_tree_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()
from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=600,random_state=42,max_leaf_nodes=36)
rfc_model.fit(x_train,y_train.ravel())
y_pred = rfc_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Random Forest Classifier: ",rfc_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(x_train,y_train.ravel())
y_pred = nb_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Naive Bayes: ",nb_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1,p=1)
knn_model.fit(x_train,y_train.ravel())
y_pred = knn_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of K Nearest Neighbors: ",knn_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()
from sklearn.svm import SVC
svc_model = SVC(random_state=42,C=2)
svc_model.fit(x_train,y_train.ravel())
y_pred = svc_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Support Vector Machine: ",svc_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()


