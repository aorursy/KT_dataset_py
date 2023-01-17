import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix 
import warnings as ws
ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")
df.head()
df.drop(columns = "PassengerId", inplace = True)
def clean_name(name):
    return name.lower().strip().replace(" ", "_")
df.rename(columns = clean_name, inplace =True)
df.head()
X = df.drop(columns = "survived")
y = df.survived
# Performing Some EDA to get more insights 
frame = df.copy()
frame.head()
# Checking for null values 
df.isna().sum()
frame.dtypes
# As the firstname and lastname are the unique values for the every person it doesn't gives us any significancef
frame.drop(columns  = ["firstname", "lastname"], inplace = True)
frame.head()
sns.set()
plt.figure(figsize = (9,9))
plt.title("Coutrywise passengers count", fontdict={'fontsize' : 20})
sns.countplot(y = frame.country, palette="plasma")
plt.show()
len(list(frame.country.unique()))
frame.sex.replace({'M': 1, 'F' : 0 }, inplace = True)
frame.head()
frame["is_passenger"] = pd.get_dummies(df.category, drop_first= True).rename(columns= {'P': "is_passenger"})
frame.country.value_counts()
frame.head()
temp = pd.get_dummies(frame, drop_first = True)
sns.set()
plt.figure(figsize = (20,20))
sns.heatmap(temp.corr(), annot= True)
plt.show()
col_finalize = ["sex", "age", "country_Sweden"]
X = temp[col_finalize]
X.head()
# Splititng  the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,  test_size = 0.2, stratify = y)
# Fitting the Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
classifier = [KNeighborsClassifier(), GaussianNB(), LogisticRegression(),DecisionTreeClassifier(), RandomForestClassifier()]
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
for i in classifier:
    i.fit(X_train, y_train)
    score = cross_val_score(i, X_train, y_train, cv =5)
    y_pred = i.predict(X_test) 
    acc = accuracy_score(y_test, y_pred)
    if(i.__class__.__name__ == "GaussianNB"):
        print(i.__class__.__name__ , "\t\t  Has accuracy of ", round(acc * 100, 3), "\t CV score is ", round(score.mean()*100, 3))
    else:
        print(i.__class__.__name__ , "\t  Has accuracy of ", round(acc * 100, 3), "\t CV score is ", round(score.mean()*100, 3))
# By looking at the  cv and accuracy_score  we will Logistic Regression for our further anlaysis
lr = LogisticRegression()
lr.fit (X_train,y_train)
y_predict = lr.predict(X_test)
print("Accuracy score is ",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Tuning the Logistic Regression
from sklearn.model_selection import GridSearchCV
logit  = LogisticRegression()
parm = {
    'C': np.linspace(0.1, 1.0, 10),
    'class_weight': ['balanced','None'],
    'penalty': ['l1', 'l2', 'none'],
    'solver' :['liblinear']
}
gsv = GridSearchCV(logit, parm, scoring = 'roc_auc', n_jobs= -1, cv = 5)
gsv.fit(X_train, y_train)
gsv.best_score_
rfe = RandomForestClassifier(max_depth  = 4, random_state=42)
rfe.fit(X_train, y_train)
y_pred_rfe = rfe.predict(X_test) 
print(accuracy_score(y_test, y_pred_rfe))
print(confusion_matrix(y_test, y_pred_rfe))
# Tuning the random forest
forest = RandomForestClassifier()
param  = {
    'n_estimators': [100, 300, 500, 800, 1200],
    'max_depth' : [5, 8, 15, 25, 30],
    'min_samples_split' : [2, 5, 10, 15, 100],
    'min_samples_leaf' : [1, 2, 5, 10]
}
gsv = GridSearchCV(forest, param, cv = 3, verbose = 1, 
                      n_jobs = -1)
gsv.fit(X_train, y_train)
gsv.best_score_
final_param = gsv.best_params_
final_param
forest1 = RandomForestClassifier(**final_param, random_state=42)
forest1.fit(X_train, y_train)
y_pred = forest1.predict(X_test)
print(accuracy_score(y_test, y_pred))
cnf = confusion_matrix(y_test, y_pred)
print(cnf)
sns.set()
plt.figure(figsize = (6,6))
sns.heatmap(cnf, annot = True, fmt = "2g")
