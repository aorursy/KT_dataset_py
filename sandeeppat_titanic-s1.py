import pandas as pd
df = pd.read_csv("../input/train.csv")
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
pf = pd.read_csv("../input/train.csv")
pf.describe()
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import Imputer
import numpy as np
imp_mean = SimpleImputer(missing_values=np.nan, strategy = 'mean')
#imp_mean = Imputer(strategy = 'mean')
pf["Age"] = imp_mean.fit_transform(pf["Age"].values.reshape(-1,1))
pf0 = pd.get_dummies(pf["Sex"])
pf1 = pd.get_dummies(pf["Embarked"])
pf2 = pf[["Pclass","Parch","SibSp","Fare","Age"]]
pf3 = pd.concat([pf0,pf1],axis = 1)
pf4 = pd.concat([pf3,pf2],axis = 1)
pf4.describe()
pf4[["Age","C","Fare","Parch","Pclass","Q","S","SibSp","female","male"]] = scaler.fit_transform(pf4[["Age","C","Fare","Parch","Pclass","Q","S","SibSp","female","male"]])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pf4, df["Survived"], test_size=0.3, random_state=42)
##############################  Logistic Regression W/O GridSearchCV   ###############################
#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train,y_train)
##############################   Ridge Regression    #####################################
#clf = Ridge(alpha=1.0)
####################################   SVR using GridSearchCV  ##########################################
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100],'gamma':(0.001,'auto')}
svr = SVR()
clf = GridSearchCV(svr, parameters, cv = 5)
###################################  Logistic Regression using GridSearchCV  ######################################
#from sklearn.linear_model import LogisticRegression
#parameters = {'solver':('lbfgs','newton-cg','sag','liblinear','saga')}
#logR = LogisticRegression(random_state = 0,multi_class = 'ovr')
#clf = GridSearchCV(logR, parameters)
##################################################### K Neighbours #############################################################
#from sklearn.neighbors import KNeighborsClassifier
#parameters = {'n_neighbors':[1,50]}
#neigh = KNeighborsClassifier()
#clf = GridSearchCV(neigh, parameters)
#################################################### Random Forest ###################################################
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
#################################################### AdaBoost ########################################
#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators = 200, random_state = 0)
########################### Decision Tree ######################################
#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(random_state=0)
############################ SVC ##############################################
#from sklearn.svm import SVC
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100],'gamma':(0.001,'auto')}
#svc = SVC()
#clf = GridSearchCV(svc, parameters)

clf.fit(X_train,y_train)
y_test_pred = clf.predict(X_test)
y_test_pred[y_test_pred >= 0.5] = 1
y_test_pred[y_test_pred < 0.5] = 0
from sklearn.metrics import accuracy_score
accuracy_score(y_test_pred,y_test)
#df.head()
#df1 = pd.get_dummies(df["Sex"])
#df2 = df[["Pclass","Age","SibSp","Parch","Fare"]]
#df3 = pd.get_dummies(df["Embarked"])
#df4 = pd.concat([df1,df2],axis=1)
#df0 = pd.concat([df3,df4],axis=1)
#Name,Ticket,Cabin
#df0[df0["Fare"].isnull()]
#df5 = df0.corr(method='pearson',min_periods = 1)
#df5["Age"]
#dfTrain = df0
#dfTrain.head()
#dfTrain.describe()
#dfNullTest = dfTrain[dfTrain["Age"].isnull()]
#dfNullTrain = dfTrain[dfTrain["Age"].notnull()]
#dfNullTrain[["Pclass","Parch","SibSp"]] = scaler.fit_transform(dfNullTrain[["Pclass","Parch","SibSp"]])
#from sklearn.svm import SVR
#from sklearn.model_selection import GridSearchCV
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100],'gamma':(0.001,'auto')}
#svr = SVR()
#clf = GridSearchCV(svr, parameters)
#clf.fit(dfNullTrain[["Pclass","Parch","SibSp"]],dfNullTrain["Age"])
#dfNullTest[["Pclass","Parch","SibSp"]] = scaler.fit_transform(dfNullTest[["Pclass","Parch","SibSp"]])
#dfNullTest["Age"] = clf.predict(dfNullTest[["Pclass","Parch","SibSp"]])
#dfReplTrain = dfTrain[dfTrain["Age"].isnull()]
#dfReplTrain.describe()
#dfReplTrain = dfReplTrain.drop(["Age"],axis = 1)
#dfReplFinal = pd.concat([dfReplTrain,dfNullTest["Age"]],axis = 1)
#dfTrain = dfTrain[pd.notnull(df["Age"])]
#dfTrainFinal = pd.concat([dfTrain,dfReplFinal],axis = 0)
#dfTrainFinal[["Age","C","Fare","Parch","Pclass","Q","S","SibSp","female","male"]] = scaler.fit_transform(dfTrainFinal[["Age","C","Fare","Parch","Pclass","Q","S","SibSp","female","male"]])
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import Ridge

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(dfTrainFinal, df["Survived"], test_size=0.3, random_state=42)
##############################  Logistic Regression W/O GridSearchCV   ###############################
#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train,y_train)
##############################   Ridge Regression    #####################################
#clf = Ridge(alpha=1.0)
####################################   SVR using GridSearchCV  ##########################################
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100],'gamma':(0.001,'auto')}
#svr = SVR()
#clf = GridSearchCV(svr, parameters)
###################################  Logistic Regression using GridSearchCV  ######################################
#parameters = {'solver':('lbfgs','newton-cg','sag','liblinear','saga')}
#logR = LogisticRegression(random_state = 0,multi_class = 'ovr')
#clf = GridSearchCV(logR, parameters)

#clf.fit(X_train,y_train)
#y_test_pred = clf.predict(X_test)
#y_test_pred[y_test_pred >= 0.5] = 1
#y_test_pred[y_test_pred < 0.5] = 0
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test,y_test_pred)
########################### THE MISSING AGE COLUMN FILLED BY CONSIDERING THE OTHER FEATURES,PREDICTION #########################
#dfTrainFinal.describe()
######################################### WE FINALLY HAVE THE MODEL FOR STAGE 1 #####################################
tf = pd.read_csv("../input/test.csv")
tf1 = pd.get_dummies(tf["Sex"])
tf2 = tf[["Pclass","Age","SibSp","Parch","Fare"]]
tf3 = pd.get_dummies(tf["Embarked"])
tf4 = pd.concat([tf1,tf2],axis=1)
tf0 = pd.concat([tf3,tf4],axis=1)
#from sklearn.preprocessing import Imputer
#imp_mean = Imputer(strategy = 'mean')
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy = 'mean')
tf0["Age"] = imp_mean.fit_transform(tf0["Age"].values.reshape(-1,1))
tf0["Fare"] = imp_mean.fit_transform(tf0["Fare"].values.reshape(-1,1))

Y_Pred = clf.predict(tf0)
Y_Pred[Y_Pred >= 0.5] = 1
Y_Pred[Y_Pred < 0.5] = 0
Y_pred = pd.Series(Y_Pred)
FINAL = pd.concat([tf["PassengerId"],Y_pred],axis=1)
FINAL.to_csv("submission-s1.csv", encoding='utf-8', index=False)