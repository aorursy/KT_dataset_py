import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
test=pd.read_csv("../input/test.csv")
train=pd.read_csv("../input/train.csv")
#gender=pd.read_csv("../input/gender_submission.csv")
#gender file is only an example how a submission should look like format-wise
# Step 2a: Descriptive statistics
# Lets only focus on the available train-data as if we would have had one set that we split into test_train and that the model is then put to production
train.info()
train.describe()
# Lets look into concrete data
train.head()
train.Cabin.dropna()[5:40]
train[(train.PassengerId==27)|(train.PassengerId==88)]
# But for the fun of it:
train.at[26,"Age"]=26
train.at[87,"Age"]=20
sns.countplot(train.Pclass)
sns.distplot(train.Age.dropna())
# Many babies on board!
# Left-skewed
# not many kids/teenagers
# young, affluent people
sns.countplot(train.Sex)
sns.countplot(train.SibSp)
# For family size lets make a short sensecheck
train[train.SibSp==8]
# only 7 people have 8 siblings. One is missing (probably in test). But a family. No one survived.
# All on same Ticket nr -> so these numbers are not unique
sns.countplot(train.Parch)
sns.heatmap(train.corr(),cmap="coolwarm")
sns.boxplot(data=train,y="Fare",x="Pclass", hue="Survived")
sns.boxplot(data=train,y="Age",x="Pclass", hue="Survived")
sns.boxplot(data=train,y="Fare",x="Pclass", hue="Sex")
sns.boxplot(data=train,y="Age",x="Pclass", hue="Sex")
sns.countplot(train["Sex"],hue=train["Survived"])
#Lets check the fare outlier
train.sort_values("Fare",ascending=False)
# Actually the 15 people with fare = 0 worry me more than the 3 that paid 500. I leave it like this for now
# But the passenger ID I dont need
# Wait a moment, before I delete: I need to do operations both on test and train. Therefore, lets combine them. I can separate them later by whether there is a Survived value or not
data=pd.concat([train, test])
data.info()
data.drop("PassengerId",inplace=True, axis=1)
data.info()
data.drop("Name",inplace=True, axis=1)
data.drop("Embarked",inplace=True, axis=1)
data.Sex2=np.nan
data.loc[data["Sex"]=="male","Sex2"]=1
data.loc[data["Sex"]=="female","Sex2"]=0
data.head()
data.drop("Ticket",inplace=True,axis=1)
# So far so good. What were other ideas?
data["Age_cohort"]=np.nan
data["Age_cohort"][(data["Age"]>=0)&(data["Age"]<=2)]=0
data["Age_cohort"][(data["Age"]>2)&(data["Age"]<=6)]=1
data["Age_cohort"][(data["Age"]>6)&(data["Age"]<=10)]=2
data["Age_cohort"][(data["Age"]>10)&(data["Age"]<=16)]=3
data["Age_cohort"][(data["Age"]>16)&(data["Age"]<=30)]=4
data["Age_cohort"][(data["Age"]>30)&(data["Age"]<=50)]=5
data["Age_cohort"][(data["Age"]>50)]=6
data.head(30)
sns.countplot(data.Age_cohort)
# Ok lets see if it makes so much sense to split up the kids so much
# But now lets get the decks out of the room numbers
data["Cabin"].fillna(" ", inplace=True)
data["Deck"]=""
data["Deck"]=data["Cabin"].apply(lambda x:str(x)[0])
data.head()
sns.countplot(data.Deck)
sns.countplot(data["Deck"],hue=data["Survived"])
#Let's engineer a new feature of only adult sex to account for male and female kids being treated rather as "kids" than as "female" or "male" in prioritization of life boats
data["Sex_adults"]=np.nan
data["Sex_adults"][(data["Sex"]==1)&(data["Age"]>16)]=1
data["Sex_adults"][(data["Sex"]==0)&(data["Age"]>3)]=0
data.head(20)
sns.countplot(data["Age_cohort"],hue=data["Survived"])
train=data[(data["Survived"]==0)|(data["Survived"]==1)]
train.info()
# Make one version of a full dataset
del train["Sex"]
del train["Sex_adults"]
del train["Deck"]
del train["Cabin"]
train.head()
train.dropna(axis=0,how="any",inplace=True)
train.info()
# Now lets check the first classification algorithm
# Make 1 version of cleaned data
sns.heatmap(train.isnull(),cbar=None,cmap="viridis")
test=data[data["Survived"].isnull()]
del test["Sex"]
del test["Sex_adults"]
del test["Deck"]
del test["Cabin"]
del test["Survived"]
test.info()
X_train=train.drop("Survived", axis=1)
y_train=train["Survived"]

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
logmodel.coef_
X_train.columns

predictions=logmodel.predict(X_train)
from sklearn.metrics import classification_report
print(classification_report(y_train,predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train,predictions))
sns.heatmap(test,cmap="viridis",cbar=False)
test.info()
test.fillna(test.mean(),inplace=True)
test.info()
predictions=logmodel.predict(test).astype("int")
type(predictions[0])
solution=pd.DataFrame({"PassengerId":list(range(892,1310)), "survived":predictions})
solution.to_csv("solution.csv", index=False)
#Now lets try a KNN approach
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=50)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
features=scaler.transform(X_train)
X_train_scaled = pd.DataFrame(features,columns=X_train.columns)

scaler.fit(test)
features=scaler.transform(test)
test_scaled = pd.DataFrame(features,columns=test.columns)

X_train_scaled.head()
test_scaled.head()
knn.fit(X_train_scaled,y_train)
predictions=knn.predict(X_train)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predictions))
print(classification_report(y_train, predictions))
predictions=knn.predict(test_scaled).astype("int")
solution=pd.DataFrame({"PassengerId":list(range(892,1310)), "survived":predictions})
solution.to_csv("solution.csv", index=False)
# Now lets use a decision tree and random forest approach
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# Lets check how good the model fits
predictions= dtree.predict(X_train)
print(confusion_matrix(y_train,predictions))
print("/n")
print(classification_report(y_train, predictions))

#Now lets predict on test-data
predictions=dtree.predict(test).astype("int")
solution=pd.DataFrame({"PassengerId":list(range(892,1310)), "survived":predictions})
solution.to_csv("solution.csv", index=False)
# Wow - the precision is enormous on train data
# Result on the test-data: 0.66985
# Very clear case of overfitting

# Let's see if Random Forest improves on this
# but before, lets have a look into the Tree that the model thought predicts best:
"""from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(X_train.columns[1:])
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())"""

# No module pydot available in Kaggle
# Lets check later how to look into the tree

# NOW: Random Forest time:
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5)
rfc.fit(X_train, y_train)

# Lets check how good the model fits
predictions= rfc.predict(X_train)
print(confusion_matrix(y_train,predictions))
print("/n")
print(classification_report(y_train, predictions))
# Equally enormous fit to training-data

# Now with test-data
predictions = rfc.predict(test).astype("int")
solution=pd.DataFrame({"PassengerId":list(range(892,1310)), "survived":predictions})
solution.to_csv("solution.csv", index=False)

# Better, not not as good as n_neighbors
# n_estimator=5: 0.71291
# n_estimators=10: 0.73684! (Best)
# n_estimators=100: 0.72248
# n_estimators=1000: 0.71770
# Now lets try Support Vector Machine Classifier:
from sklearn.svm import SVC
model=SVC()
model.fit(X_train, y_train)

# Lets check how good the model fits
predictions= model.predict(X_train)
print(confusion_matrix(y_train,predictions))
print("/n")
print(classification_report(y_train, predictions))
# The fit is not as huge (overfitting) as the Tree-Models

# Now with test-data
predictions = model.predict(test).astype("int")
solution=pd.DataFrame({"PassengerId":list(range(892,1310)), "survived":predictions})
solution.to_csv("solution.csv", index=False)

# Result:
# Very bad: 0.59330
# But makes sense: Why would the population be "geometrically" splittable?
model.get_params()
# Just learned about GridSearchCV, lets try to improve the so far best model knn
knn.get_params()
from sklearn.model_selection import GridSearchCV
# documentation knn: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# I search rather full than partially (thx to my only partial understanding as of now of the parameters)
param_grid = {'n_neighbors': np.arange(10,100,10), 'p': np.arange(1,10,2), 'leaf_size':np.arange(10,50,10), 'weights': ['uniform','distance'],'algorithm':['auto','ball_tree', 'kd_tree', 'brute']} 

grid=GridSearchCV(knn,param_grid,refit=True, verbose=2)
grid.fit(X_train,y_train)
predictions = grid.predict(test).astype("int")
solution=pd.DataFrame({"PassengerId":list(range(892,1310)), "survived":predictions})
solution.to_csv("solution.csv", index=False)
# Result: Much worse than before. Clear case of overfitting. 0,66
