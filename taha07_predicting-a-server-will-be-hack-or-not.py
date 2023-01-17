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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
%matplotlib inline
train_data = pd.read_csv("/kaggle/input/novartis-data/Train.csv")
test_data = pd.read_csv("/kaggle/input/novartis-data/Test.csv")
train_data.head()
test_data.head()
train_data.info()
test_data.info()
train_data.describe()
train_data.isnull().sum()
test_data.isnull().sum()
train_data["X_12"] = train_data["X_12"].ffill()
test_data["X_12"] = test_data["X_12"].ffill()
train_data["X_12"] = train_data["X_12"].bfill()
test_data["X_12"] = test_data["X_12"].bfill()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.drop(["INCIDENT_ID","DATE"],axis=1,inplace=True)
test_data.drop(["INCIDENT_ID","DATE"],axis=1,inplace=True)
train_data.head()
test_data.head()
sns.set()
train_data.hist(figsize=(20,10),bins=15,color="purple")
plt.title("Distribution of Features")
plt.show()
sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data= train_data,palette = "Set3")
plt.show()
sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data= test_data,palette = "Set3")
plt.show()
lower_limit = train_data["X_8"].mean() - 3*train_data["X_8"].std()
upper_limit = train_data["X_8"].mean() + 3*train_data["X_8"].std()
df_train1 = train_data[(train_data["X_8"] > lower_limit) & (train_data["X_8"] < upper_limit)]
train_data.shape[0] - df_train1.shape[0]
lower_limit = test_data["X_8"].mean() - 3*test_data["X_8"].std()
upper_limit = test_data["X_8"].mean() + 3*test_data["X_8"].std()
df_test1 = test_data[(test_data["X_8"] > lower_limit) & (test_data["X_8"] < upper_limit)]
test_data.shape[0] - df_test1.shape[0]
lower_limit = df_train1["X_10"].mean() - 3*df_train1["X_10"].std()
upper_limit = df_train1["X_10"].mean() + 3*df_train1["X_10"].std()
df_train2 = df_train1[(df_train1["X_10"] > lower_limit) & (df_train1["X_10"] < upper_limit)]
df_train1.shape[0] - df_train2.shape[0]
lower_limit = df_test1["X_10"].mean() - 3*df_test1["X_10"].std()
upper_limit = df_test1["X_10"].mean() + 3*df_test1["X_10"].std()
df_test2 = df_test1[(df_test1["X_8"] > lower_limit) & (df_test1["X_8"] < upper_limit)]
df_test1.shape[0] - df_test2.shape[0]
lower_limit = df_train2["X_11"].mean() - 3*df_train2["X_11"].std()
upper_limit = df_train2["X_11"].mean() + 3*df_train2["X_11"].std()
df_train3 = df_train2[(df_train2["X_11"] > lower_limit) & (df_train2["X_11"] < upper_limit)]
df_train2.shape[0] - df_train3.shape[0]
lower_limit = df_test2["X_11"].mean() - 3*df_test2["X_11"].std()
upper_limit = df_test2["X_11"].mean() + 3*df_test2["X_11"].std()
df_test3 = df_test2[(df_test2["X_11"] > lower_limit) & (df_test2["X_11"] < upper_limit)]
df_test2.shape[0] - df_test3.shape[0]
lower_limit = df_train3["X_12"].mean() - 3*df_train3["X_12"].std()
upper_limit = df_train3["X_12"].mean() + 3*df_train3["X_12"].std()
df_train4 = df_train3[(df_train3["X_12"] > lower_limit) & (df_train3["X_12"] < upper_limit)]
df_train3.shape[0] - df_train4.shape[0]
lower_limit = df_test3["X_12"].mean() - 3*df_test3["X_12"].std()
upper_limit = df_test3["X_12"].mean() + 3*df_test3["X_12"].std()
df_test4 = df_test3[(df_test3["X_12"] > lower_limit) & (df_test3["X_12"] < upper_limit)]
df_test3.shape[0] - df_test4.shape[0]
lower_limit = df_train4["X_13"].mean() - 3*df_train4["X_13"].std()
upper_limit = df_train4["X_13"].mean() + 3*df_train4["X_13"].std()
df_train5 = df_train4[(df_train4["X_13"] > lower_limit) & (df_train4["X_13"] < upper_limit)]
df_train4.shape[0] - df_train5.shape[0]
lower_limit = df_test4["X_13"].mean() - 3*df_test4["X_13"].std()
upper_limit = df_test4["X_13"].mean() + 3*df_test4["X_13"].std()
df_test5 = df_test4[(df_test4["X_13"] > lower_limit) & (df_test4["X_13"] < upper_limit)]
df_test4.shape[0] - df_test5.shape[0]
df_train5.head()
df_test5.head()
df_train5.info()
df_test5.info()
n = msno.bar(df_train5,color="purple")
x = df_train5.drop("MULTIPLE_OFFENSE",axis=1)
y = df_train5["MULTIPLE_OFFENSE"]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
model_params ={
    "svm": {
        "model" : SVC(gamma="auto"),
        "params": {
            "C" : [1,10,20],
            "kernel": ["rbf"],
            "random_state":[0,10,100]
        }
    },
    
    "decision_tree":{
        "model": DecisionTreeClassifier(),
        "params":{
            "criterion": ["entropy","gini"],
            "max_depth": [5,8,9],
            "random_state":[0,10,100]
        }
    },
    "random_forest":{
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators" : [1,5,10],
            "max_depth" : [5,8,9],
            "random_state":[0,10,100]
        }
    },
    "naive_bayes":{
        "model": GaussianNB(),
        "params": {}
    },
    
    "logistic_regression":{
        "model" : LogisticRegression(solver='liblinear',multi_class = 'auto'),
        "params":{
            "C" : [1,5,10],
            "random_state":[0,10,100]
        }
    },
    "knn" : {
        "model" : KNeighborsClassifier(),
        "params": {
            "n_neighbors" : [5,12,13]
        }
    }
    
    
}
scores =[]
for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=12,return_train_score=False)
    clf.fit(x,y)
    scores.append({
        "Model" : model_name,
        "Best_Score": clf.best_score_,
        "Best_Params": clf.best_params_
    })
result_score = pd.DataFrame(scores, columns = ["Model","Best_Score","Best_Params"])
result_score
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
clf_dt = DecisionTreeClassifier(criterion= "gini",max_depth = 9,random_state=0)
clf_dt.fit(x_train,y_train)
y_pred = clf_dt.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
result = pd.DataFrame({"Actual_Value": y_test, "Predicted_Value": y_pred})
result
