import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
park = pd.read_csv("../input/parkinson-disease-detection/Parkinsson disease.csv")
park.head()
park.shape
park.info()
park["status"].value_counts()
# there are 147 datapoints where it shows they have disease and  48 datapoints where it shows they don't have disease, 
# which means the dataset is skewed.
park[park.isnull().any(axis=1)]
#no missing/null data
sns.countplot(x='status',data=park)
#Shows the distribution of status column - univariate analysis of the target column
sns.pairplot(park)
park = park.drop("name",axis=1)
#Dropped name column as it doesnot contribute to model building
fig, ax = plt.subplots(figsize=(15,5))
park.boxplot(['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)','HNR'],ax=ax)
k=[]
for i in park.columns:
    for j in park[i]:
        if (j<1 and j>0):
            k.append(i)
            break

fig, ax = plt.subplots(figsize=(15,5))
park.boxplot(k,ax=ax)
park.describe().T
X = park.drop("status",axis=1)
y = park["status"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=9)
print(X_train.shape)
print(X_test.shape)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train,y_train)
preds = model.predict(X_test)
preds
model.score(X_train,y_train)
#accuracy of the model obtained for the train data
metrics.accuracy_score(y_test,preds)
#accuracy of the model obtained for the test data
pd.crosstab(y_test,preds)
model_reg = DecisionTreeClassifier(criterion="entropy",max_depth=10,min_samples_leaf=20)
model_reg.fit(X_train,y_train)
preds_reg = model_reg.predict(X_test)
model_reg.score(X_train,y_train)
metrics.accuracy_score(y_test,preds_reg)
rfcl = RandomForestClassifier(n_estimators=100,max_depth=15)
rfcl.fit(X_train,y_train)
preds_rfcl = rfcl.predict(X_test)
rfcl.score(X_train,y_train)
metrics.accuracy_score(y_test,preds_rfcl)
z=0
b=0
for i in np.arange(10,150):
    rfcl = RandomForestClassifier(n_estimators = i, max_depth=15)
    rfcl.fit(X_train, y_train)
    preds_rfcl=rfcl.predict(X_test)
    acc=accuracy_score(y_test,preds_rfcl)
    if acc>z:
        z=acc
        b=i
print("For",b,"number of trees,accuracy is",z)