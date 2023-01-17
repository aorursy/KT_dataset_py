# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math
%matplotlib inline
plt.rcParams["figure.figsize"] = [10,6]

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# EDA
data = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")
print(data.columns)
print(data.shape)
print(data.describe())
# Plot 1
data["Gender"].value_counts().plot.barh(color="blue",
                                        title="Count by Gender")
plt.xlabel("Count")
plt.ylabel("Gender")
plt.show()
# Plot 2
data["Age"].describe()
 
# Create bins for age
def create_age_bins(x,bins=10):
    max = int(math.floor((x.max()+10)/10))*10
    min = int(math.floor((x.min()-10)/10))
    bin_width = (max-min)/bins
    intervals = [min+x*bin_width for x in range(bins+1)]
    labels = [i for i in range(1,len(intervals))]
    return pd.cut(x,bins=intervals, labels=labels)
 
data["Age Bins"] = create_age_bins(data["Age"],bins=6)
data["Age Bins"].head()
 
data["Age Bins"].value_counts().plot.barh(color="purple",
                                          title="Count by Age Gap")
plt.xlabel("Count")
plt.ylabel("Age Gap")
plt.show()
# Check for missing values
pd.isnull(data).sum() # None
# Dtypes
data.dtypes
data.info()
# See if can convert some dtypes from object to int or float
data["Vehicle_Age"].unique() # Yes
data["Vehicle_Age"]=data["Vehicle_Age"].replace({"> 2 Years":2, "1-2 Year":1,"< 1 Year":0})
cat_type = pd.CategoricalDtype(categories=data["Vehicle_Age"].unique()
                              ,ordered=True)
data["Vehicle_Age"] = data["Vehicle_Age"].astype(cat_type)
data["Vehicle_Damage"].unique() #Yes
data["Vehicle_Damage"]=data["Vehicle_Damage"].replace({"Yes":1, "No":0})
 
data["Gender"] = data["Gender"].replace({"Male":0,"Female":0})
# Modeling
# Predict Response based on other variables
 
# Drop id, Age
data = data.drop(columns=["id","Age"])
 
# Split into train, validation, and test sets
# Use a 70/10/20 split
 
def train_val_test_split(train_per=.70, val_per=.10,test_per=.20,nrows=0):
    train_split = math.floor(nrows*train_per)
    val_split = math.floor(nrows*val_per)
   
    train_index = np.random.choice(np.arange(0,nrows),
                                   train_split,
                                   replace=False)
    train_mask = np.isin(np.arange(0,nrows), train_index )
    val_index=np.random.choice(np.arange(0,nrows)[~train_mask],
                     val_split,
                     replace=False)
    val_mask = np.isin(np.arange(0,nrows),
                       np.concatenate([train_index,val_index]))
    test_index = np.arange(0, nrows)[~val_mask]
   
    return train_index, val_index,test_index
 
train_index, val_index, test_index = train_val_test_split(nrows=data.shape[0])
 
train_index.shape
val_index.shape
test_index.shape
 
 
# Split training set into X,Y variables
X_train = data.iloc[train_index,:].drop(columns=["Response"])
Y_train = data.iloc[train_index,:]["Response"]
X_val = data.iloc[val_index,:].drop(columns=["Response"])
Y_val = data.iloc[val_index,:]["Response"]
X_test = data.iloc[test_index,:].drop(columns=["Response"])
Y_test = data.iloc[test_index,:]["Response"]
print(Y_train.value_counts())
print(Y_val.value_counts())
print(Y_test.value_counts())
# Fit Model
from sklearn.linear_model import LogisticRegression
LOGI_FIT_1 = LogisticRegression(max_iter=2500)
LOGI_FIT_1.fit(X_train,Y_train)
 
# Predict classes using val set
val_probs=LOGI_FIT_1.predict_proba(X_val)
find_class = np.vectorize(lambda x: 0 if x>0.5 else 1)
val_preds = find_class(val_probs[:,0])
 
 
def misclass_rate(preds, real):
    cross = pd.crosstab(preds,real)
    cross.columns.name = "Predictions"
    misclass_rate =  (cross.iloc[0,1]+cross.iloc[1,0])/(cross.iloc[0,0]+cross.iloc[1,1])
    print("The misclassification rate is %0.0f percent."%(misclass_rate*100))
 
# Predict classs using test set
test_preds = LOGI_FIT_1.predict(X_test)
misclass_rate(test_preds, Y_test)
 
test_probs=LOGI_FIT_1.predict_proba(X_test)
test_preds=find_class(test_probs[:,0])
misclass_rate(test_preds, Y_test)
# Decision Tree
from sklearn import tree
TREE_FIT_1 = tree.DecisionTreeClassifier()
TREE_FIT_1.fit(X_train,Y_train)
val_preds=TREE_FIT_1.predict(X_val)
test_preds = TREE_FIT_1.predict(X_test)
 
misclass_rate(val_preds,Y_val)
misclass_rate(test_preds,Y_test)
# Bagging
from sklearn.ensemble import BaggingClassifier
BAGGER = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
                           n_estimators=100,random_state=0).fit(X_train,Y_train)
 
val_preds = BAGGER.predict(X_val)
test_preds = BAGGER.predict(X_test)
misclass_rate(val_preds,Y_val)
misclass_rate(test_preds,Y_test)
# ROC curve
from sklearn import metrics
logi_probs = LOGI_FIT_1.predict_proba(X_test)[:,1]
tree_probs = TREE_FIT_1.predict_proba(X_test)[:,1]
bag_probs  = BAGGER.predict_proba(X_test) [:,1]
 
fpr1, tpr1, threshold1 = metrics.roc_curve(Y_test, logi_probs)
roc_auc1 = metrics.auc(fpr1, tpr1)
 
fpr2, tpr2, threshold2 = metrics.roc_curve(Y_test, tree_probs)
roc_auc2 = metrics.auc(fpr2, tpr2)
 
fpr3, tpr3, threshold1 = metrics.roc_curve(Y_test, bag_probs)
roc_auc3 = metrics.auc(fpr1, tpr1)
 
plt.plot(fpr1,tpr1,"b-",label="LOGI AUC %0.2f"%roc_auc1)
plt.plot(fpr2,tpr2,"k-",label="Tree AUC %0.2f"%roc_auc2)
plt.plot(fpr3,tpr3,"g-",label="Bag AUC %0.2f"%roc_auc3)
 
 
plt.plot([0,1],[0,1] ,"r",linestyle="--")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(loc="lower right")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
# Plot ROC using seaborn
sns.set_style("whitegrid")
plt.plot(fpr1,tpr1,"b-",label="LOGI AUC %0.2f"%roc_auc1)
plt.plot(fpr2,tpr2,"k-",label="Tree AUC %0.2f"%roc_auc2)
plt.plot(fpr3,tpr3,"g-",label="Bag AUC %0.2f"%roc_auc3)
plt.plot([0,1],[0,1] ,"r",linestyle="--")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(loc="lower right")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

