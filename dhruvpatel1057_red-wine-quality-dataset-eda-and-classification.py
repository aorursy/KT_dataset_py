import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

### for multiple outputs in cell
from IPython.core.interactiveshell import InteractiveShell  ## getting
InteractiveShell.ast_node_interactivity = "all"


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from sklearn.pipeline import Pipeline



from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score




data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()
data.columns
data.shape

data.groupby("quality")["quality"].count()   ## here is the distribution of the 

data[data["quality"]==4].shape
data.info()
data.isnull().sum()
data.describe()
plot = plt.figure(figsize=(5,5))
sns.barplot(data=data,x="quality",y="volatile acidity",palette="rocket")
plot = plt.figure(figsize=(5,5))
sns.barplot(data=data,x="quality",y="fixed acidity",palette="rocket")
plot = plt.figure(figsize=(5,5))
sns.barplot(data=data,x="quality",y="fixed acidity",palette="rocket")
plot = plt.figure(figsize=(5,5))
sns.barplot(data=data,x="quality",y="citric acid",palette="rocket")

plot = plt.figure(figsize=(5,5))
sns.barplot(data=data,x="quality",y="residual sugar",palette="rocket")

plot = plt.figure(figsize=(5,5))
sns.barplot(data=data,x="quality",y="chlorides",palette="rocket")

plot = plt.figure(figsize=(5,5))
sns.barplot(data=data,x="quality",y="pH",palette="vlag")

plot = plt.figure(figsize=(5,5))
sns.barplot(data=data,x="quality",y="density",palette="rocket")


# Plot
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(data.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

# Decorations
plt.title('Correlogram of mtcars', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plot = plt.figure(figsize=(5,5))

sns.jointplot(data=data,x="quality",y="alcohol", kind="reg", truncate=False,
                  
                  color="r", height=7)
## now it's   time to preprocess our data

data.groupby("quality")["quality"].count()
## we can now divide them into 3 segments : bad, average, good

bins =(2,4,6,8)
names =["bad","average","good"]

data["quality"] = pd.cut(data["quality"],bins=bins,labels=names)
data.groupby("quality")["quality"].count()
data.head()
### now we can see  there is average, bad ,and good labels in quality columns

sns.countplot(data["quality"])
le = LabelEncoder()

data["quality"] = le.fit_transform(data["quality"])

data.head()
data.groupby("quality")["quality"].count()
### now we can see  the average, bad ,and good labels in quality columns are encoded into numeric 

sns.countplot(data["quality"])
X=data.drop("quality",axis=1)
y = data["quality"]
X.head()
y.value_counts()
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=0)

scaler = StandardScaler()

model_rf=RandomForestClassifier(n_estimators=100,random_state=0)
model_xgb = XGBClassifier(n_estimators=200,random_state=0)



rf_pipeline =Pipeline(steps=[
    ("sc",scaler),
    ("rf",model_rf)
])

xgb_pipeline = Pipeline(steps=[
    ("sc",scaler),
    ("xgb",model_xgb)
])


parameters_xgb = {'xgb__n_estimators':[i for i in range(100,1000,100)]}

parameters_rf = {'rf__n_estimators':[i for i in range(100,1000,100)]}
cv = GridSearchCV(xgb_pipeline,parameters_xgb,cv=5)

cv.fit(X_train,y_train)
preds = cv.predict(X_valid)
print(classification_report(y_valid,preds))
print(confusion_matrix(y_valid,preds))
rf_cv = GridSearchCV(rf_pipeline,parameters_rf,cv=5)

rf_cv.fit(X_train,y_train)
rf_preds = rf_cv.predict(X_valid)

print(confusion_matrix(y_valid,rf_preds))
print(classification_report(y_valid,rf_preds))
rf_cv.score(X_valid,y_valid)
cv.score(X_valid,y_valid)

dt_classifier = DecisionTreeClassifier(criterion = 'gini', max_features=6, max_leaf_nodes=400, random_state = 0)

dt_pipeline = Pipeline(steps=[
    ("sc",scaler),
    ("dt",dt_classifier)
])


params_dt = {"dt__criterion":["gini","entropy"],
          "dt__max_features":[3,4,5,6,7,8,9,10],
          "dt__max_leaf_nodes":[200,300,400,500]
         }
dt_cv = GridSearchCV(dt_pipeline,params_dt,cv=5)

dt_cv.fit(X_train,y_train)
dt_cv.best_params_
dt_cv.score(X_valid,y_valid)
dt_preds=dt_cv.predict(X_valid)
print(classification_report(y_valid,dt_preds))
print(confusion_matrix(y_valid,dt_preds))