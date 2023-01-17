# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
import sklearn.model_selection as ms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.naive_bayes as NB
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read csv file
df = pd.read_csv("../input/diabetes.csv", delimiter = ",")
#check if the file has entries
print(df.head())

#check if there are any NaN
print("Checking if the values are null") 
print(df.isnull().any())

#explore the data 
print("shape of dataset:\n", df.shape) #number of rows and columns

print("Statistics of the data\n") #statistics of each column
for i in range (0,8): #not considering the Outcome column 
    print(df.columns[i])
    print(df.iloc[i].describe())
    print("\n")
#diagnosis of patients
diag = df['Outcome']
diag1 = (diag == 0).sum()
diag2 = (diag == 1).sum()
a=[diag1,diag2]
print(a)
b=['No diabetes','Diabetes']
df2 = pd.DataFrame(a, index = b)
ax = df2.plot(kind='bar', legend = False, width = .5, rot = 0,color = "plum", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.ylim(0, 1000)
plt.xlabel('Diagnosis')
plt.ylabel('All cases')
plt.title('Diagnosis of patients')
plt.show()
#plot historgram for each columns

print("Features")
features_mean= list(df.columns[0:8])
bins = 12
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    
    plt.subplot(rows, 2, i+1)
    
    sns.distplot(df[df['Outcome']==1][feature], bins=bins, color='red', label='0 = Negative');
    sns.distplot(df[df['Outcome']==0][feature], bins=bins, color='blue', label='1 = Positive');
    
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
#boxplot
print("Boxplot of the data")
features_mean= list(df.columns[0:8])
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    plt.subplot(rows,2, i+1)
    sns.boxplot(x='Outcome', y=feature, data=df, palette="Set1")

plt.tight_layout()
plt.show()
sns.pairplot(df,hue="Outcome")
plt.show()
#X and y values
X = df.iloc[:,0:7]
y = df['Outcome']
#random forest
print("Random forest Classifier")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X_train,y_train)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y_test, model.predict(X_test))
print ("Random Forest = ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X_test, y_test, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y_test,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y_test,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y_test,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y_test, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y_test,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y_test,model.predict(X_test)))
#knn
print("KNN")
model = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X_train,y_train)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y_test, model.predict(X_test))
print ("KNN= ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X_test, y_test, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y_test,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y_test,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y_test,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y_test, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y_test,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y_test,model.predict(X_test)))
#Logistic Regression
print("Logistic Regression")
model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X_train,y_train)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y_test, model.predict(X_test))
print ("Logistic Regression = ", rf_roc_auc)
#print(metrics.classification_report(y_test,model.predict(X_test)))
predCV = ms.cross_val_predict(model, X_test, y_test, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y_test,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y_test,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y_test,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y_test, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y_test,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y_test,model.predict(X_test)))
#important feature in the dataset
X = df.iloc[:,0:8]
y = df['Outcome']

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,max_features=None,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking for all factors:")

for f in range(X.shape[1]):
    print("%d. feature %d : %f" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest 
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()