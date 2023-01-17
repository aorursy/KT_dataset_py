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
train_df = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test_df = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

train_df.columns
train_df.head(10)
train_df.describe()
train_df.info()
train_df.isnull().sum()
# libraries for Visualization
import seaborn as sns
import matplotlib.pyplot as plt
pd.crosstab(train_df.Age,train_df.Gender).plot(kind="bar",figsize=(30,8))
plt.title('Age Frequency for Genders')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
plt.subplot
g = sns.FacetGrid(train_df, col = "Response", height = 6)
g.map(sns.distplot, "Age", bins = 50)
plt.show()
def bar_plot(variable):
    """
        input: variable ex: "Vehicle_Age"
        output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()   
    
    #visualize
    plt.figure(figsize =(6,6))
    labels = varValue.index
    colors = ['#2C4447','#F3EC86','#679B75','red','green','brown']
    plt.pie(varValue, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.ylabel("Rate")
    plt.title(variable)
    plt.show()
    
    #print("{}: \n {}".format(variable,varValue))
    
category1 = ["Gender","Vehicle_Age","Vehicle_Damage"]
for c in category1:
    bar_plot(c)
train_df.head(10)
test_df.head()
train_df = pd.get_dummies(train_df, columns = ["Driving_License"])
test_df = pd.get_dummies(test_df, columns = ["Driving_License"])
train_df.head()
plt.figure(figsize=(30,8))
sns.countplot(x="Region_Code", data = train_df)
plt.xticks(rotation = 60)
plt.show()
train_df = pd.get_dummies(train_df, columns = ["Region_Code"], prefix = "RC")
test_df = pd.get_dummies(test_df, columns = ["Region_Code"], prefix = "RC")
train_df.head()
plt.figure(figsize=(30,8))
sns.countplot(x="Policy_Sales_Channel", data = train_df)
plt.xticks(rotation = 60)
plt.show()
train_df.Policy_Sales_Channel.value_counts().head(10)
train_df["Policy_Sales_Channel"] = [i if i == 152.0 or i == 26.0 or i == 124.0 or i == 160.0 or i == 156.0 or i==122.0 or i == 157.0 or i == 154.0 else 200 for i in train_df.Policy_Sales_Channel]
test_df["Policy_Sales_Channel"] = [i if i == 152.0 or i == 26.0 or i == 124.0 or i == 160.0 or i == 156.0 or i==122.0 or i == 157.0 or i == 154.0 else 200 for i in test_df.Policy_Sales_Channel]
train_df.Policy_Sales_Channel.value_counts().head(10)
plt.figure(figsize=(30,8))
sns.countplot(x="Policy_Sales_Channel", data = train_df)
plt.xticks(rotation = 60)
plt.show()
train_df = pd.get_dummies(train_df, columns = ["Policy_Sales_Channel"], prefix = "SC")
test_df = pd.get_dummies(test_df, columns = ["Policy_Sales_Channel"], prefix = "SC")
train_df.head()
train_df = pd.get_dummies(train_df, columns = ["Vehicle_Damage"], prefix = "VD")
test_df = pd.get_dummies(test_df, columns = ["Vehicle_Damage"], prefix = "VD")
train_df.head()
train_df = pd.get_dummies(train_df, columns = ["Vehicle_Age"], prefix = "VA")
test_df = pd.get_dummies(test_df, columns = ["Vehicle_Age"], prefix = "VA")
train_df.head()
train_df = pd.get_dummies(train_df, columns = ["Gender"], prefix = "G")
test_df = pd.get_dummies(test_df, columns = ["Gender"], prefix = "G")
train_df.head()
train_df.info()
test_df.info()
train_df.drop(labels = ["id"], axis = 1, inplace = True)
train_df.columns
y = train_df.Response.values
x_data = train_df.drop(["Response"],axis=1)

# normalization 
x = ( x_data - np.min(x_data) ) / ( np.max(x_data) - np.min(x_data) ).values
# %% split data
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.1, random_state = 42) # validation data = 0.1 data
print("X_train",len(x_train))
print("x_val",len(x_val))
print("y_train",len(y_train))
print("y_val",len(y_val))

print("test",len(test_df))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
acc_log_train = round(logreg.score(x_train, y_train)*100,2)
acc_log_val = round(logreg.score(x_val, y_val)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_val))
# import models
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
# %% split data

x_train1, x_ss, y_train1, y_ss = train_test_split(x_train,y_train,test_size = 0.01, random_state = 42) # subset data = 0.01 training data
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier(),
             GaussianNB()]

dt_param_grid = {"min_samples_split" : range(10,100,20),
                "max_depth": range(1,20,4)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [ 0.01, 0.1, 1],
                 "C": [1,10,50,100,500]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}

naive_param_grid = {}

classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid,
                   naive_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(x_ss,y_ss)     
    cv_result.append(clf.best_score_) # save best scores
    best_estimators.append(clf.best_estimator_) # save best estimators
    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier","GaussianNB"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")
plt.show()
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(x_train, y_train) 
print(accuracy_score(votingC.predict(x_val),y_val))
test_df_id = test_df.id
test_df.drop(labels = ["id"], axis = 1, inplace = True)
test_df_response = pd.Series(votingC.predict(test_df), name = "Response").astype(int)

results = pd.concat([test_df_id, test_df_response],axis = 1)

results.to_csv("cross_sell_prediction.csv", index = False)