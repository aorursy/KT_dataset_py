# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/glass/glass.csv")
df.isna().sum(axis=0) #No missing values
df.hist()
descr_stats = df.describe()
descr_stats
df["Type"].value_counts()
df.boxplot(column="Ba",by="Type",vert=False)
df.boxplot(column="Fe",by="Type",vert=False)
df.boxplot(column="K",by="Type",vert=False)
ba_fe_k = df.loc[(df["Ba"]==0)&(df["Fe"]==0)&(df["K"]==0)]
len(ba_fe_k.loc[df["Type"]==6]),len(ba_fe_k)
corr_table = df.corr()
corr_table
#Creating separate DFs for dependent and independent variables
target = df["Type"]
x_vars = df.drop(columns="Type")
from sklearn.cluster import KMeans
#Number of clusters is same as number of types of glass
kmeans_init = KMeans(init="random",n_clusters=len(df["Type"].value_counts()),n_init=10,random_state=42) 
kmeans_algo = kmeans_init.fit(df)
kmeans_clust_nums = kmeans_algo.predict(df)
compare = pd.crosstab(target,kmeans_clust_nums) #Compare the cluster numbering and actual glass types
'''
csum = np.cumsum(np.array(compare),axis=1)
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
clrmap = plt.get_cmap('RdBu')
cNorm  = colors.Normalize(vmin=0, vmax=len(compare.index))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=clrmap)
plt.figure()
for c in compare.columns:
    for i in range(0,len(compare.index)):
        if i==0:
            plt.bar(c,compare[c][compare.index[i]],color=str(0.13*i))
        else:
            plt.bar(c,compare[c][compare.index[i]],color=str(0.13*i),bottom=compare[c][compare.index[i-1]]) #Why is one color getting repeated?
        
'''
compare
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_vars,target,train_size=0.8,random_state=64) #80% training data, 20% testing data
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(x_train,y_train)
knn_pred = knn_model.predict(x_test) #Classified into 6 different types
knn_compare = pd.crosstab(y_test,knn_pred)
knn_dsum = 0
for c in knn_compare.columns:
    knn_dsum = knn_dsum + knn_compare[c][c]
knn_accuracy = knn_dsum/knn_compare.sum().sum() #56% accuracy

#Logistic regression
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()
logreg_model.fit(x_train,y_train) #Failed to converge
logreg_pred = logreg_model.predict(x_test)
logreg_compare = pd.crosstab(y_test,logreg_pred)
logreg_dsum = 0
for c in logreg_compare.columns:
    logreg_dsum = logreg_dsum + logreg_compare[c][c]
logreg_accuracy = logreg_dsum/logreg_compare.sum().sum() #44% accuracy

#Decision trees
from sklearn import tree
dtree_gini_model = tree.DecisionTreeClassifier(criterion="gini")
dtree_entropy_model = tree.DecisionTreeClassifier(criterion="entropy")

dtree_gini_model.fit(x_train,y_train)
dtree_gini_pred = dtree_gini_model.predict(x_test) #Classified into 6 different categories
dtree_gini_compare = pd.crosstab(y_test,dtree_gini_pred)
dtree_gini_dsum = 0
for c in dtree_gini_compare.columns:
    dtree_gini_dsum = dtree_gini_dsum + dtree_gini_compare[c][c]
dtree_gini_accuracy = dtree_gini_dsum/dtree_gini_compare.sum().sum() #53% accuracy

dtree_entropy_model.fit(x_train,y_train)
dtree_entropy_pred = dtree_entropy_model.predict(x_test) #Classified into 5 different categories
dtree_entropy_compare = pd.crosstab(y_test,dtree_entropy_pred)
dtree_entropy_dsum = 0
for c in dtree_entropy_compare.columns:
    dtree_entropy_dsum = dtree_entropy_dsum + dtree_entropy_compare[c][c] #Correctly classified counts - diagonal elements only
dtree_entropy_accuracy = dtree_entropy_dsum/dtree_entropy_compare.sum().sum() #74% accuracy
acc_cmp1 = pd.DataFrame([[knn_accuracy],[logreg_accuracy],[dtree_gini_accuracy],[dtree_entropy_accuracy]])
acc_cmp1.index = ["KNN","Logistic reg","D_trees - Gini","D_trees - Entropy"]
acc_cmp1
knn_compare
logreg_compare
dtree_gini_compare
dtree_entropy_compare
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score

kfold = RepeatedStratifiedKFold(n_splits=10,n_repeats=5,random_state=42) #Divide data set into 10 parts, and perform 5 iterations with different train/test splits
knn_kfold = cross_val_score(knn_model,x_vars,target,scoring="accuracy",cv=kfold) #Median accuracy around 68%, large variation 50%-86%
logreg_kfold = cross_val_score(logreg_model,x_vars,target,scoring="accuracy",cv=kfold) #Median accuracy around 60%, large variation 45%-75%
dtree_gini_kfold = cross_val_score(dtree_gini_model,x_vars,target,scoring="accuracy",cv=kfold) #Median accuracy around 69%, large variation 45%-90%
dtree_entropy_kfold = cross_val_score(dtree_entropy_model,x_vars,target,scoring="accuracy",cv=kfold) #Median accuracy around 70%, large variation 52%-90%

plt.figure()
plt.title("With RI included")
pd.concat([pd.DataFrame(knn_kfold),pd.DataFrame(logreg_kfold),pd.DataFrame(dtree_gini_kfold),pd.DataFrame(dtree_entropy_kfold)],axis=1).boxplot()
plt.xticks((1,2,3,4),("KNN","Logistic\nregression","Decision trees\nGini","Decision trees\nEntropy"))
x_noRI = x_vars.drop(columns="RI") #This will be required for k-fold cross validation
#Not splitting into train and test again because some issues observed with random_state not giving consistent results
#Subsetting the original test and train data to obtain new data
x_train_noRI = x_train.drop(columns="RI")
x_test_noRI = x_test.drop(columns="RI")

#KNN
knn_model = KNeighborsClassifier()
knn_model.fit(x_train_noRI,y_train)
knn_pred = knn_model.predict(x_test_noRI)
knn_compare2 = (knn_compare,pd.crosstab(y_test,knn_pred))
knn_dsum = 0
for c in knn_compare2[1].columns:
    knn_dsum = knn_dsum + knn_compare2[1][c][c]
knn_accuracy2 = (knn_accuracy,knn_dsum/knn_compare2[1].sum().sum())

#Logistic regression
logreg_model = LogisticRegression()
logreg_model.fit(x_train_noRI,y_train)
logreg_pred = logreg_model.predict(x_test_noRI)
logreg_compare2 = (logreg_compare,pd.crosstab(y_test,logreg_pred))
logreg_dsum = 0
for c in logreg_compare2[1].columns:
    logreg_dsum = logreg_dsum + logreg_compare2[1][c][c]
logreg_accuracy2 = (logreg_accuracy,logreg_dsum/logreg_compare2[1].sum().sum())

#Decision trees
dtree_gini_model = tree.DecisionTreeClassifier(criterion="gini")
dtree_entropy_model = tree.DecisionTreeClassifier(criterion="entropy")

dtree_gini_model.fit(x_train_noRI,y_train)
dtree_gini_pred = dtree_gini_model.predict(x_test_noRI)
dtree_gini_compare2 = (dtree_gini_compare,pd.crosstab(y_test,dtree_gini_pred))
dtree_gini_dsum = 0
for c in dtree_gini_compare2[1].columns:
    dtree_gini_dsum = dtree_gini_dsum + dtree_gini_compare2[1][c][c]
dtree_gini_accuracy2 = (dtree_gini_accuracy,dtree_gini_dsum/dtree_gini_compare2[1].sum().sum())

dtree_entropy_model.fit(x_train_noRI,y_train)
dtree_entropy_pred = dtree_entropy_model.predict(x_test_noRI)
dtree_entropy_compare2 = (dtree_entropy_compare,pd.crosstab(y_test,dtree_entropy_pred))
dtree_entropy_dsum = 0
for c in dtree_entropy_compare2[1].columns:
    dtree_entropy_dsum = dtree_entropy_dsum + dtree_entropy_compare2[1][c][c]
dtree_entropy_accuracy2 = (dtree_entropy_accuracy,dtree_entropy_dsum/dtree_entropy_compare2[1].sum().sum())
acc_cmp2 = pd.DataFrame([[knn_accuracy2],[logreg_accuracy2],[dtree_gini_accuracy2],[dtree_entropy_accuracy2]])
acc_cmp2.index = ["KNN","Logistic reg","D_trees - Gini","D_trees - Entropy"]
acc_cmp2
x_3removed = x_noRI.drop(columns=["Ba","Fe","K"]) #This will be required for k-fold cross validation
#Not splitting into train and test again because some issues observed with random_state not giving consistent results
#Subsetting the original test and train data to obtain new data
x_train_3removed = x_train_noRI.drop(columns=["Ba","Fe","K"])
x_test_3removed = x_test_noRI.drop(columns=["Ba","Fe","K"])

#KNN
knn_model = KNeighborsClassifier()
knn_model.fit(x_train_3removed,y_train)
knn_pred = knn_model.predict(x_test_3removed)
knn_compare3 = (knn_compare2,pd.crosstab(y_test,knn_pred))
knn_dsum = 0
for c in knn_compare3[1].columns:
    knn_dsum = knn_dsum + knn_compare3[1][c][c]
knn_accuracy3 = (knn_accuracy2[1],knn_dsum/knn_compare3[1].sum().sum())

#Logistic regression
logreg_model = LogisticRegression()
logreg_model.fit(x_train_3removed,y_train)
logreg_pred = logreg_model.predict(x_test_3removed)
logreg_compare3 = (logreg_compare2,pd.crosstab(y_test,logreg_pred))
logreg_dsum = 0
for c in logreg_compare3[1].columns:
    logreg_dsum = logreg_dsum + logreg_compare3[1][c][c]
logreg_accuracy3 = (logreg_accuracy2[1],logreg_dsum/logreg_compare3[1].sum().sum())

#Decision trees
dtree_gini_model = tree.DecisionTreeClassifier(criterion="gini")
dtree_entropy_model = tree.DecisionTreeClassifier(criterion="entropy")

dtree_gini_model.fit(x_train_3removed,y_train)
dtree_gini_pred = dtree_gini_model.predict(x_test_3removed)
dtree_gini_compare3 = (dtree_gini_compare2,pd.crosstab(y_test,dtree_gini_pred))
dtree_gini_dsum = 0
for c in dtree_gini_compare3[1].columns:
    dtree_gini_dsum = dtree_gini_dsum + dtree_gini_compare3[1][c][c]
dtree_gini_accuracy3 = (dtree_gini_accuracy2[1],dtree_gini_dsum/dtree_gini_compare3[1].sum().sum())

dtree_entropy_model.fit(x_train_3removed,y_train)
dtree_entropy_pred = dtree_entropy_model.predict(x_test_3removed)
dtree_entropy_compare3 = (dtree_entropy_compare2,pd.crosstab(y_test,dtree_entropy_pred))
dtree_entropy_dsum = 0
for c in dtree_entropy_compare3[1].columns:
    dtree_entropy_dsum = dtree_entropy_dsum + dtree_entropy_compare3[1][c][c]
dtree_entropy_accuracy3 = (dtree_entropy_accuracy2[1],dtree_entropy_dsum/dtree_entropy_compare3[1].sum().sum())
acc_cmp3 = pd.DataFrame([[knn_accuracy3],[logreg_accuracy3],[dtree_gini_accuracy3],[dtree_entropy_accuracy3]])
acc_cmp3.index = ["KNN","Logistic reg","D_trees - Gini","D_trees - Entropy"]
acc_cmp3
knn_compare3[1]
logreg_compare3[1]
dtree_gini_compare3[1]
dtree_entropy_compare3[1]