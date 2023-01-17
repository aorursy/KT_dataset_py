# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
data = pd.read_csv("../input/titanic/train.csv")
data.head()
data.shape
data.columns
data.dtypes
data.isnull().sum()
data["Cabin"].value_counts()
data["Age"].value_counts()
plt.figure(figsize = (8,5))
plt.title("Age distribution")
plt.ylabel("Density")
plt.xlabel("Age")
sns.distplot(data["Age"],kde = False,norm_hist = True)
data["Age"].fillna(0,inplace = True)
data["Embarked"].value_counts()
data["Embarked"].fillna(data["Embarked"].mode()[0],inplace = True)
data["Title"] = 0
miss_check = data["Name"].str.contains("Miss")
miss_array = miss_check.loc[miss_check == True].index.values 
mr_check = data["Name"].str.contains("Mr")
mr_array = mr_check.loc[mr_check == True].index.values
mrs_check = data["Name"].str.contains("Mrs")
mrs_array = mrs_check.loc[mrs_check == True].index.values
master_check = data["Name"].str.contains("Master")
master_array = master_check.loc[master_check == True].index.values
for i in miss_array:
    data.iloc[i,-1] = "Miss"
for i in mr_array:
    data.iloc[i,-1] = "Mr"
for i in mrs_array:
    data.iloc[i,-1] = "Mrs"
for i in master_array:
    data.iloc[i,-1] = "Master"
data.loc[data.Title == 0,["Name","Title","Sex"]].sort_values(by = "Sex")
for i in range(0,890):
    if data.iloc[i,-1] == 0:
        if data.iloc[i,4] == "male":
            data.iloc[i,-1] = "Mr"
        else:
            data.iloc[i,-1] = "Mrs" 
data.Title.value_counts()
data[["Title","Age"]].groupby("Title").mean().sort_values(by = "Age",ascending = False)
for i in range(0,890):
    if data.iloc[i,5] == 0:
        if data.iloc[i,-1] == "Mr":
            data.iloc[i,5] = 26
        elif data.iloc[i,-1] == "Mrs":
            data.iloc[i,5] = 31
        elif data.iloc[i,-1] == "Miss":
            data.iloc[i,5] = 17
        elif data.iloc[i,-1] == "Master":
            data.iloc[i,5] = 4
data_train = data[["Sex","Pclass","Age","SibSp","Parch","Fare","Title","Embarked","Survived"]]
data_train
data_train["Sex"].value_counts()
sex_data = data_train[["Sex","Survived"]].groupby(data_train["Sex"]).sum()
sex_data
sns.barplot(x = sex_data.index,y = sex_data["Survived"])
data_train[["Sex","Survived"]].groupby(data_train["Sex"]).mean()
data_train[["Pclass","Survived"]].groupby(data_train["Pclass"]).mean()
pclass_data = data_train[["Pclass","Survived"]].groupby(data_train["Pclass"]).sum()
pclass_data.drop(["Pclass"],axis = 1)
sns.barplot(x = pclass_data.drop(["Pclass"],axis = 1).index,y = pclass_data.drop(["Pclass"],axis = 1)["Survived"])
data_train["Pclass"].value_counts()
sex_pclass1 = data_train.loc[(data_train["Sex"] == "female") & (data_train["Pclass"] == 1),["Sex","Pclass","Survived"]]
sex_pclass1.value_counts()
sex_pclass2 = data_train.loc[(data_train["Sex"] == "female") & (data_train["Pclass"] == 2),["Sex","Pclass","Survived"]]
sex_pclass2.value_counts()
sex_pclass3 = data_train.loc[(data_train["Sex"] == "female") & (data_train["Pclass"] == 3),["Sex","Pclass","Survived"]]
sex_pclass3.value_counts()
sex_by_class = data_train[["Sex","Pclass","Survived"]].pivot_table(index = "Sex",columns = "Pclass",aggfunc = np.mean)
plt.figure(figsize = (11,7))
plt.title("Survived percentage - Sex by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Sex")
sns.heatmap(sex_by_class,annot = True)
sns.scatterplot(x = data_train["Age"],y = data_train["Survived"],s = 100)
age_survived = data_train.loc[data_train["Survived"]==1]
age_not = data_train.loc[data_train["Survived"]==0]
sns.distplot(age_survived["Age"],kde = False,norm_hist = True)
sns.distplot(age_not["Age"],kde = False,norm_hist = True)
data_train["SibSp"].value_counts()
data_train.loc[data_train["Survived"] == 1,["SibSp"]].value_counts()
data_train["Parch"].value_counts()
data_train.loc[data_train["Survived"] == 1,["Parch"]].value_counts()
data_train[["SibSp","Survived"]].groupby(data_train["SibSp"]).mean()
data_train[["Parch","Survived"]].groupby(data_train["Parch"]).mean()
data_train.loc[data_train["Fare"] == 0,["Pclass","Fare"]]
data_train.loc[data_train["Pclass"] == 1,["Pclass","Fare"]].max()
data_train.loc[data_train["Pclass"] == 2,["Pclass","Fare"]].max()
data_train.loc[data_train["Pclass"] == 3,["Pclass","Fare"]].max()
data_train["Embarked"].value_counts()
data_train[["Embarked","Survived"]].groupby(data_train["Embarked"]).mean()
data_train.loc[data_train["Pclass"] == 1,["Embarked","Pclass"]].value_counts()
data_train.loc[(data_train["Pclass"] == 1)&(data_train["Survived"] == 1),["Embarked","Pclass"]].value_counts()
data_train.loc[(data_train["Sex"] == "female"),["Embarked","Sex"]].value_counts()
X = data_train.iloc[:,:-1].values
y = data_train.iloc[:,-1].values
X
y
data_train.dtypes
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X[:,0] = encoder.fit_transform(X[:,0])
X
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
enc = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[6,7])],remainder = "passthrough")
X = enc.fit_transform(X)
X
X[0:2]
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.2,random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X[:,7:] = sc.fit_transform(train_X[:,7:])
test_X[:,7:] = sc.transform(test_X[:,7:])
train_X[0]
from sklearn.linear_model import LogisticRegression
lr_class = LogisticRegression()
lr_class.fit(train_X,train_y)
lr_pred = lr_class.predict(test_X)
lr_prob = lr_class.predict_proba(test_X)
from sklearn.naive_bayes import GaussianNB
nb_class = GaussianNB()
nb_class.fit(train_X,train_y)
nb_pred = nb_class.predict(test_X)
nb_prob = nb_class.predict_proba(test_X)
from sklearn.neighbors import KNeighborsClassifier as KNN
kn_class = KNN()
kn_class.fit(train_X,train_y)
kn_pred = kn_class.predict(test_X)
from sklearn.svm import SVC
svc_class = SVC(kernel = "linear")
svc_class.fit(train_X,train_y)
svc_pred = svc_class.predict(test_X)
from sklearn.svm import SVC
svck_class = SVC(kernel = "rbf")
svck_class.fit(train_X,train_y)
svck_pred = svck_class.predict(test_X)
from sklearn.tree import DecisionTreeClassifier as DTC
dt_class = DTC()
dt_class.fit(train_X,train_y)
dt_pred = dt_class.predict(test_X)
from sklearn.ensemble import RandomForestClassifier as RFC
rf_class = RFC()
rf_class.fit(train_X,train_y)
rf_pred = rf_class.predict(test_X)
from xgboost import XGBClassifier as XGB
xbg_class = XGB()
xbg_class.fit(train_X,train_y)
xbg_pred = xbg_class.predict(test_X)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 5,activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 5,activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 1,activation = "sigmoid"))
ann.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
ann.fit(np.array(train_X,dtype=np.float),train_y,epochs = 100)
ann_prob = ann.predict(np.array(test_X,dtype = np.float))
ann_pred = (ann_prob>0.5)
from sklearn.metrics import confusion_matrix,accuracy_score
lr_matrix = confusion_matrix(test_y,lr_pred)
lr_acc = accuracy_score(test_y,lr_pred)
nb_matrix = confusion_matrix(test_y,nb_pred)
nb_acc = accuracy_score(test_y,nb_pred)
kn_matrix = confusion_matrix(test_y,kn_pred)
kn_acc = accuracy_score(test_y,kn_pred)
svc_matrix = confusion_matrix(test_y,svc_pred)
svc_acc = accuracy_score(test_y,svc_pred)
svck_matrix = confusion_matrix(test_y,svck_pred)
svck_acc = accuracy_score(test_y,svck_pred)
dt_matrix = confusion_matrix(test_y,dt_pred)
dt_acc = accuracy_score(test_y,dt_pred)
rf_matrix = confusion_matrix(test_y,rf_pred)
rf_acc = accuracy_score(test_y,rf_pred)
xbg_matrix = confusion_matrix(test_y,xbg_pred)
xbg_acc = accuracy_score(test_y,xbg_pred)
ann_matrix = confusion_matrix(test_y,ann_pred)
ann_acc = accuracy_score(test_y,ann_pred)
from sklearn.model_selection import cross_val_score
lr_cross = cross_val_score(estimator = lr_class,X = train_X,y = train_y,cv = 10,scoring = "accuracy")
nb_cross = cross_val_score(estimator = nb_class,X = train_X,y = train_y,cv = 10,scoring = "accuracy")
kn_cross = cross_val_score(estimator = kn_class,X = train_X,y = train_y,cv = 10,scoring = "accuracy")
svc_cross = cross_val_score(estimator = svc_class,X = train_X,y = train_y,cv = 10,scoring = "accuracy")
svck_cross = cross_val_score(estimator = svck_class,X = train_X,y = train_y,cv = 10,scoring = "accuracy")
dt_cross = cross_val_score(estimator = dt_class,X = train_X,y = train_y,cv = 10,scoring = "accuracy")
rf_cross = cross_val_score(estimator = rf_class,X = train_X,y = train_y,cv = 10,scoring = "accuracy")
xbg_cross = cross_val_score(estimator = xbg_class,X = train_X,y = train_y,cv = 10,scoring = "accuracy")
models_list = ["Logistic Regression","Naive Bayes","K-Nearest Neighbors","Linear SVC","Kernel SVC",
               "Decision Tree","Random Forest","XGBoost","Artificial Neural Networks"]
col1 = [lr_matrix[0,0],nb_matrix[0,0],kn_matrix[0,0],svc_matrix[0,0],svck_matrix[0,0],dt_matrix[0,0],rf_matrix[0,0],
       xbg_matrix[0,0],ann_matrix[0,0]]
col2 = [lr_matrix[1,0],nb_matrix[1,0],kn_matrix[1,0],svc_matrix[1,0],svck_matrix[1,0],dt_matrix[1,0],rf_matrix[1,0],
       xbg_matrix[1,0],ann_matrix[1,0]]
col3 = [lr_matrix[1,1],nb_matrix[1,1],kn_matrix[1,1],svc_matrix[1,1],svck_matrix[1,1],dt_matrix[1,1],rf_matrix[1,1],
       xbg_matrix[1,1],ann_matrix[1,1]]
col4 = [lr_matrix[0,1],nb_matrix[0,1],kn_matrix[0,1],svc_matrix[0,1],svck_matrix[0,1],dt_matrix[0,1],rf_matrix[0,1],
       xbg_matrix[0,1],ann_matrix[0,1]]
col5 = [lr_acc*100,nb_acc*100,kn_acc*100,svc_acc*100,svck_acc*100,dt_acc*100,rf_acc*100,xbg_acc*100,ann_acc*100]
col6 = [lr_cross.mean()*100,nb_cross.mean()*100,kn_cross.mean()*100,svc_cross.mean()*100,svck_cross.mean()*100,dt_cross.mean()*100,
        rf_cross.mean()*100,xbg_cross.mean()*100,"None"]
performance = pd.DataFrame({"True Negatives":col1,"False Negatives":col2,"True Positives":col3,"False Positives":col4,
                            "Accuracy":col5,"K-fold cross":col6}, index = models_list)
performance
from sklearn.model_selection import GridSearchCV
parameters = [{"kernel":["rbf","sigmoid","poly"],"gamma":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],"C":[0.25,0.5,0.75,1]}]
boost = GridSearchCV(estimator = svck_class,param_grid = parameters,scoring = "accuracy",cv = 10,n_jobs = -1)
boost.fit(train_X,train_y)
best_score = boost.best_score_
best_params = boost.best_params_
print(best_score)
print(best_params)