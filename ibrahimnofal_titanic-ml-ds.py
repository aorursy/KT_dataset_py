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
data_train=pd.read_csv("/kaggle/input/titanic/train.csv")

data_test=pd.read_csv("/kaggle/input/titanic/test.csv")

data_train.info()
data_train.describe()
data_train.isna().sum()
data=data_train["Sex"].value_counts()

data
data=data_train.Sex[data_train["Survived"]==1].value_counts()

data.plot.bar(figsize=(15,5),color=["salmon","lightblue"]);

plt.title("Number of Survived on Sex")

plt.xlabel("Sex")

plt.ylabel("Number of Survived");

data=data_train.Sex[data_train["Survived"]==0].value_counts()

data.plot.bar(figsize=(15,5),color=["red","green"]);

plt.title("Number of not Survived on Sex")

plt.xlabel("Sex")

plt.ylabel("Number of not  Survived");
data=data_train.groupby("Pclass")["Sex"].value_counts()

data
fig,ax=plt.subplots(figsize=(15,5))

ax.bar(data_train['Survived'].value_counts().index,data_train['Survived'].value_counts().values,color=["lightblue","salmon"])

plt.title('Survival counts')

plt.xlabel('Survived')

plt.ylabel('No of passengers')

plt.xticks(rotation = 0);

plt.show()
data=data_train.Pclass.value_counts()

data.plot(kind="bar",color=["salmon","lightblue"],figsize=(10,6));

plt.title("Type of passenger class")

plt.xlabel("Pclass Type")

plt.ylabel("Total of Passengers")

plt.xticks(rotation = 0);

 
pd.crosstab(data_train.Survived,data_train.Sex).plot(kind="bar",figsize=(10,6), 

                                    color=["salmon", "lightblue"]);

# Add some attributes to it

plt.title("Survival Rate  for Sex")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Counts")

plt.legend(["Female", "Male"])

plt.xticks(rotation=0); # keep the labels on the x-axis vertical

data_train.tail()
data_train["SibSp"].value_counts().index,data_train["SibSp"].value_counts().values
data=data_train.SibSp.value_counts()

data.plot(kind="bar",color=["salmon","lightblue"],figsize=(15,6));

plt.title("Number of siblings/spouses aboard")

plt.xlabel("Type")

plt.ylabel("Total of Passengers")

plt.xticks(rotation = 0);

 
data_train.tail()


data=data_train.Parch.value_counts()

data.plot(kind="bar",color=["salmon","lightblue"],figsize=(15,6));

plt.title("Number of parents/childrens aboard")

plt.xlabel("Count")

plt.ylabel("Total of Passengers")

plt.xticks(rotation = 0);
pd.crosstab(data_train.Pclass,data_train.Survived)
pd.crosstab(data_train.Pclass,data_train.Survived).plot(kind="bar",figsize=(10,6), 

                                    color=["salmon", "lightblue"]);

# Add some attributes to it

plt.title("Survival Rate  for P Class")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Counts")

plt.legend(["Not Survived", "Survived"])

plt.xticks(rotation=0); # keep the labels on the x-axis vertical

pd.crosstab(data_train.Embarked,data_train.Survived)
pd.crosstab(data_train.Embarked,data_train.Survived).plot(kind="bar",figsize=(10,6), 

                                    color=["salmon", "lightblue"]);

# Add some attributes to it

plt.title("Survival Rate  for Embarked")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Counts")

plt.legend(["Not Survived", "Survived"])

plt.xticks(rotation=0); # keep the labels on the x-axis vertical
data_train.Age.hist();
data_train.head()
data_train.corr()
import seaborn as sns



corr_matrix=data_train.corr()

fig,ax=plt.subplots(figsize=(15,10))

ax=sns.heatmap(corr_matrix,

               annot=True,

               linewidth=0.5,

               fmt=".2f",cmap="YlGnBu");
data_train.columns
# Deleting unnecessary columns

X=data_train.drop(['Ticket','Cabin','Name','Fare','Embarked'],axis=1)

X.head()

X_data=X.drop("Survived",axis=1)

X_data.tail()
test_data=data_test.drop(['Ticket','Cabin','Name','Fare','Embarked'],axis=1)

test_data
from sklearn.preprocessing import LabelEncoder

conv = LabelEncoder()
X_data.isna().sum()
X_data.Sex=conv.fit_transform(X_data.Sex)

X_data.head()
X_data.dtypes
X_data.isna().sum()
X_data.fillna(X_data.Age.median(),inplace=True)

X_data.isna().sum()
test_data
test_data.Sex=conv.fit_transform(test_data.Sex)
test_data.tail(2)
test_data.isna().sum()
test_data.fillna(test_data.Age.median(),inplace=True)
test_data.tail(3)
test_data.isna().sum()
X_data.shape,test_data.shape
X=X_data

X
y=data_train.Survived

y
# Models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

# Import LinearSVC from sklearn's svm module

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier



# Import SVC from sklearn's svm module

from sklearn.svm import SVC





# Note: we don't have to import RandomForestClassifier, since we already have

# Model Evaluation

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import precision_score,recall_score,f1_score

from sklearn.metrics import plot_roc_curve

#########
from sklearn.tree import DecisionTreeClassifier

models = {"LinearSVC": LinearSVC(),

          "KNN": KNeighborsClassifier(),

          "SVC": SVC(),

          "LogisticRegression": LogisticRegression(),

          "RandomForestClassifier": RandomForestClassifier(),

          "Ga":DecisionTreeClassifier()}

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20)
def fit_and_score(model,X_train,X_test,y_train,y_test):

    """

    Fits and evaluates given machine learning models.

    models : a dict of different Scikit-Learn machine learning models

    X_train : training data

    X_test : testing data

    y_train : labels assosciated with training data

    y_test : labels assosciated with test data

    """

    np.random.seed(42)

    model_scores={}

    for model_name,model in models.items() :

        model.fit(X_train,y_train)

        model_scores[model_name]=model.score(X_val,y_val)

    

    return(model_scores)

        
model_scores=fit_and_score(models,X_train,X_val,y_train,y_val)

model_scores
modal_compare=pd.DataFrame(model_scores,index=["accuracy"])

modal_compare.T.plot.bar();
## Tuning KNN

train_scores=[]

test_scores=[]



nieghbors=range(1,21)

Knn=KNeighborsClassifier()



for i in nieghbors :

    Knn.set_params(n_neighbors=i)

    

    Knn.fit(X_train,y_train)

    

    train_scores.append(Knn.score(X_train,y_train))

    test_scores.append(Knn.score(X_val,y_val))
train_scores
test_scores
plt.plot(nieghbors, train_scores, label="Train score")

plt.plot(nieghbors, test_scores, label="Test score")

plt.xticks(np.arange(1, 21, 1))

plt.xlabel("Number of neighbors")

plt.ylabel("Model score")

plt.legend()



print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")
log_reg_grid={"C":np.logspace(-4,4,20),

             "solver":["liblinear"]}

rf_grid = {"n_estimators": np.arange(10, 1000, 50),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2)}
np.random.seed(42)

rs_log_reg=RandomizedSearchCV(LogisticRegression(),

                              param_distributions=log_reg_grid,

                              cv=5,

                              n_iter=20,

                              verbose=True)

rs_log_reg.fit(X_train,y_train)
rs_log_reg.best_params_
rs_log_reg.score(X_val,y_val)
np.random.seed(42)

rs_log_cv2=RandomizedSearchCV(RandomForestClassifier(),

                             param_distributions=rf_grid,

                             n_iter=20,

                             cv=5,

                             verbose=True)

rs_log_cv2.fit(X_train,y_train)
rs_log_cv2.best_params_
from sklearn.metrics import plot_roc_curve

plot_roc_curve(rs_log_reg,X_val,y_val);
y_preds=rs_log_reg.predict(X_val)
confusion_matrix(y_val,y_preds)
sns.set(font_scale=1.5)

def plot_conf_mat(y_test,y_preds):

    fig,axe=plt.subplots(figsize=(3,3))

    ax=sns.heatmap(confusion_matrix(y_test,y_preds),

                   annot=True,

                   cbar=False)

    plt.xlabel("True Table")

    plt.ylabel("Predicted Table")

    

plot_conf_mat(y_test,y_preds)
print(classification_report(y_val,y_preds))
clf=LogisticRegression(solver= 'liblinear', C= 0.615848211066026)

crx_val=cross_val_score(clf,X,y,cv=5,scoring="accuracy")

crx_val
accu=np.mean(crx_val)

accu
precison=cross_val_score(clf,

                        X,

                        y,

                        cv=5,

                        scoring="precision")

mpr=precison.mean()

mpr
recal=np.mean(cross_val_score(clf,X,y,scoring="recall"))

recal
cv_f1=np.mean(cross_val_score(clf,X,y,cv=5,scoring="f1"))

cv_f1
cv_metrics=pd.DataFrame({"Accuracy":accu,

                        "Precision":mpr,

                        "Recall":recal,

                        "F1 Score": cv_f1},index=[0])

cv_metrics.T.plot.bar(figsize=(15,6),title="Cross-Validated Metrics", legend=False);
clf.fit(X_train,y_train)
clf.coef_
f_dict=dict(zip(X.columns,list(clf.coef_[0])))

f_dict
df1=pd.DataFrame(f_dict,index=[0])

df1.T.plot.bar(title="Feature Importance",legend=False);
data_test.head()
y_preds=rs_log_reg.predict(test_data)
y_preds
# Format predictions into the same format Kaggle is after

df_preds = pd.DataFrame()

df_preds["PassengerId"] = data_test["PassengerId"]

df_preds["Survived"] = y_preds

df_preds
# Export prediction data

df_preds.to_csv("submession.csv", index=False)