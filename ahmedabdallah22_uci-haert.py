import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/heart-disease-uci/heart.csv")

df
df.describe()




plt.figure(figsize=(14,8))

sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=.1)

plt.show()
X = df.drop(columns = ['fbs', 'chol'])

X=X.iloc[:,:-1]

y=df.iloc[:,-1]

X
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

X_train

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier( random_state = 0, max_depth = 5, min_samples_leaf = 3)

model_tree.fit(X_train,y_train)

Y_tree = model_tree.predict(X_test)
from sklearn.metrics import accuracy_score

print ("Accuracy score is " , accuracy_score(y_test,Y_tree)*100)
plt.scatter(y_test,'o',Y_tree,'_')
from sklearn.metrics import plot_confusion_matrix 



disp = plot_confusion_matrix(model_tree, X_test, y_test,

                              display_labels=['Yes','No'],

                              cmap=plt.cm.Blues)


from sklearn.ensemble import RandomForestClassifier

model_forset = RandomForestClassifier(n_estimators=125)

model_forset.fit(X_train,y_train)

Y_forset=model_forset.predict(X_test)





print ("Accuracy score is " , accuracy_score(y_test,Y_forset)*100)




disp = plot_confusion_matrix(model_forset, X_test, y_test,

                              display_labels=['Yes','No'],

                              cmap=plt.cm.Blues)
from sklearn.neighbors import KNeighborsClassifier

model_KNN = KNeighborsClassifier(n_neighbors=7)

model_KNN.fit(X_train,y_train)

Y_KNN = model_KNN.predict(X_test)


print ("Accuracy score is " , accuracy_score(y_test,Y_KNN)*100)

disp = plot_confusion_matrix(model_KNN, X_test, y_test,

                              display_labels=['Yes','No'],

                              cmap=plt.cm.Blues)
plt.scatter(y_test,'o',Y_KNN,'_')




from sklearn.svm import SVC

model_SVC=SVC(kernel='linear',C=5)

model_SVC.fit(X_train,y_train)

Y_SVC = model_SVC.predict(X_test)
print ("Accuracy score is " , accuracy_score(y_test,Y_SVC)*100)
disp = plot_confusion_matrix(model_SVC, X_test, y_test,

                              display_labels=['Yes','No'],

                              cmap=plt.cm.Blues)
from sklearn.ensemble import GradientBoostingClassifier

model_GBC = GradientBoostingClassifier(learning_rate=.4)

model_GBC.fit(X_train,y_train)

y_Gpred=model_GBC.predict(X_test)

print ("Accuracy score is " , accuracy_score(y_test,y_Gpred)*100)

disp = plot_confusion_matrix(model_GBC, X_test, y_test,

                              display_labels=['Yes','No'],

                              cmap=plt.cm.Blues)
from catboost  import CatBoostClassifier

model_cat=CatBoostClassifier()

model_cat.fit(X_train,y_train)

Y_cat=model_cat.predict(X_test)
print ("Accuracy score is " , accuracy_score(y_test,Y_cat)*100)
disp = plot_confusion_matrix(model_cat, X_test, y_test,

                              display_labels=['Yes','No'],

                              cmap=plt.cm.Blues)
df
from sklearn.naive_bayes import GaussianNB

model_GNB = GaussianNB()

model_GNB.fit(X_train,y_train)

y_GNB = model_GNB.predict(X_test)

print ("Accuracy score is " , accuracy_score(y_test,y_GNB)*100)