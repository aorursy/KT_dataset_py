# import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score,r2_score

#Reading in my student information file
SpO=pd.read_csv("../input/StudentsPerformance.csv")
#Computing the Total score as a summation of the Math, Reading and Writing scores
SpO["TotalScore"]=SpO[["math score","reading score","writing score"]].sum(axis=1)
#Copy of SpOriginal file
Sp=SpO

#Setting the Level of Student criteria 
Sp["Level"]=np.NaN

for ind,TS in zip(Sp.index,Sp["TotalScore"]):
    if (TS<=175.000000):
        Sp["Level"][ind]=0
        
    elif (TS>175.000000 and TS<=205.000000):
        Sp["Level"][ind]=1
        
    elif (TS>205.000000 and TS<=233.000000):
        Sp["Level"][ind]=2
    else:
        Sp["Level"][ind]=3
        
#Using Label Encoder on the Categorical Columns
#Gender
le=LabelEncoder()
le.fit(Sp["gender"])
Sp["gender"]=le.transform(Sp["gender"])

#Race
le.fit(Sp["race/ethnicity"])
Sp["race/ethnicity"]=le.transform(Sp["race/ethnicity"])

#Parents level of edu
le.fit(Sp["parental level of education"])
Sp["parental level of education"]=le.transform(Sp["parental level of education"])

#Lunch
le.fit(Sp["lunch"])
Sp["lunch"]=le.transform(Sp["lunch"])

#Test prep
le.fit(Sp["test preparation course"])
Sp["test preparation course"]=le.transform(Sp["test preparation course"])

#Print Sp with the label Encoder values now converted 
#print(Sp)

#Plot
plt.scatter(Sp["TotalScore"],Sp["Level"],c=Sp["Level"])
plt.show()

#Implementing Random Forest Classifier on Sp
X=Sp[["gender","race/ethnicity","parental level of education","lunch","test preparation course","math score","reading score","writing score","TotalScore"]]
y=Sp["Level"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
random_forest=RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train,y_train)
Y_pred=random_forest.predict(X_test)  ## predicting Level of students in test data

#ACCESSING PERFORMANCE
#Accuracy Score
accscore=accuracy_score(y_test,Y_pred)
print("Accuracy score of RFC based on all predictors using LabelEncoder:")
print(accscore)



#Confusion matrix
print("Confusion matrix for RFC based on all predictors using LabelEncoder:")
cf=confusion_matrix(y_test,Y_pred)
print(cf)
#Classification Report
report = classification_report(y_test, Y_pred)
print("Classification Report for RFC based on all predictors using LabelEncoder:")
print(report)
sns.heatmap(X.corr(), vmin=0, vmax=1)
plt.title("HeatMap of feature correlation with Label Encoder",fontsize=15)
plt.show()

##Feature selection using RFE and ExtratreesClassifier
#RFE

rfe=RFE(random_forest,1)
fit=rfe.fit(X_train,y_train)
print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)
print("\n\nFeature importance using RFE:")
for col,rank in zip(X.columns,fit.ranking_):
    print(col,rank)

#Extra Trees Classifiers
forest=ExtraTreesClassifier(n_estimators=200)
forest.fit(X_train,y_train)
print("Feature importance using ExtraTreesClassifiers:")
for col1,imp in zip(X.columns,forest.feature_importances_):
    print(col1,imp)
    

#####Feature importance plotting
featimp=pd.DataFrame()
featimp["feat_name"]=pd.Series(X.columns)
featimp["feature_imp"]=pd.Series(forest.feature_importances_)
featimp.sort_values(by=['feature_imp'], inplace=True)


#Plot Cumulative feat imp:

featimp["Impcumsum"]=featimp["feature_imp"].cumsum()
print(featimp)

fig,ax=plt.subplots()
fig.set_size_inches(9,7, forward=True)
ax.plot(featimp["feat_name"],featimp["Impcumsum"])
ax.set_title(" Cumulative Importances for the features after Label Encoding",fontsize=12)
ax.set_xticklabels(featimp["feat_name"],rotation=90)
plt.grid()
plt.hlines(y = 0.90, xmin=0, xmax=len(featimp["feat_name"])-1, color = 'r', linestyles = 'dashed')
plt.savefig("featimp-labelencoder.png")
fig.tight_layout(pad=5.0)
plt.show()

#Conducting RFE on Selected Features
#Without Total Score
X=Sp[["math score","reading score","writing score","gender","race/ethnicity","parental level of education","lunch","test preparation course"]]

y=Sp["Level"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
random_forest=RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train,y_train)
Y_pred=random_forest.predict(X_test)  ## predicting Level of students in test data

#ACCESSING PERFORMANCE
#Accuracy Score
accscore=accuracy_score(y_test,Y_pred)
print("\n\nAccuracy score of RFC without Total Score as predictor,using LabelEncoder:")
print(accscore)

#Confusion matrix
print("\nConfusion matrix for RFC without Total Score as predictor,using LabelEncoder:")
cf=confusion_matrix(y_test,Y_pred)
print(cf)
#Classification Report
report = classification_report(y_test, Y_pred)
print("\nClassification Report for RFC without Total Score as predictor,using LabelEncoder:")
print(report)

#Conducting RFE on Selected Features
#Without Total Score, Math score,Writing Score and Reading score
X=Sp[["gender","race/ethnicity","parental level of education","lunch","test preparation course"]]

y=Sp["Level"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
random_forest=RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train,y_train)
Y_pred=random_forest.predict(X_test)  ## predicting Level of students in test data

#ACCESSING PERFORMANCE
#Accuracy Score
accscore=accuracy_score(y_test,Y_pred)
print("\n\nAccuracy score of RFC without Total Score,Math Score,Writing Score and Reading Score as predictors,using LabelEncoder:")
print(accscore)

#Confusion matrix
print("Confusion matrix for RFC without Total Score,Math Score,Writing Score and Reading Score as predictors,using LabelEncoder:")
cf=confusion_matrix(y_test,Y_pred)
print(cf)
#Classification Report
report = classification_report(y_test, Y_pred)
print("Classification Report for RFC without Total Score,Math Score,Writing score and Reading Score as predictors,using LabelEncoder:")
print(report)

#PLOT without scores
def grosscol(col):
    if col<=175:
        return("indianred")
    elif (col>175 and col<=205):
        return("tomato")
    elif (col>205 and col<=233):
        return ("red")
    else:
        return("gold")


fig,ax=plt.subplots()

for ind,lev in zip(X_test.index,y_test):
    ax.scatter(ind,lev,color=grosscol(lev))
ax.scatter(X_test.index,Y_pred,c=Y_pred,s=Y_pred*2)

plt.show()

import pandas as pd
from sklearn.preprocessing import LabelBinarizer
Sp=pd.read_csv("../input/StudentsPerformance.csv")
#Computing the Total score as a summation of the Math, Reading and Writing scores
Sp["TotalScore"]=SpO[["math score","reading score","writing score"]].sum(axis=1)
Sp

#print(Sp)

#Setting the Level of Student criteria 
Sp["Level"]=np.NaN

for ind,TS in zip(Sp.index,Sp["TotalScore"]):
    if (TS<=175.000000):
        Sp["Level"][ind]=0
        
    elif (TS>175.000000 and TS<=205.000000):
        Sp["Level"][ind]=1
        
    elif (TS>205.000000 and TS<=233.000000):
        Sp["Level"][ind]=2
    else:
        Sp["Level"][ind]=3
        
        
#Gender
a=pd.get_dummies(Sp['gender'], prefix='Gender')
Sp1 = pd.concat([Sp, a], axis=1).drop(["gender"], axis=1)

#Race
a=pd.get_dummies(Sp1['race/ethnicity'], prefix='Race')
Sp1 = pd.concat([Sp1, a], axis=1).drop(["race/ethnicity"], axis=1)

#Parental level of Education
b=pd.get_dummies(Sp1['parental level of education'], prefix='PEd')
Sp1 = pd.concat([Sp1, b], axis=1).drop(["parental level of education"], axis=1)

#Lunch
c=pd.get_dummies(Sp1['lunch'], prefix='lun')
Sp1 = pd.concat([Sp1, c], axis=1).drop(["lunch"], axis=1)

#Test Prep

d=pd.get_dummies(Sp1['test preparation course'], prefix='TP')
Sp1 = pd.concat([Sp1, d], axis=1).drop(["test preparation course"], axis=1)



#Implementing Random Forest Classifier on Sp1 [get_dummies]
X=Sp1[["math score", "reading score", "writing score", "TotalScore", "Level",
       "Gender_female", "Gender_male", "Race_group A", "Race_group B",
       "Race_group C", "Race_group D", "Race_group E",
       "PEd_associate's degree", "PEd_bachelor's degree", "PEd_high school",
       "PEd_master's degree", "PEd_some college", "PEd_some high school",
       "lun_free/reduced", "lun_standard", "TP_completed", "TP_none"]]
y=Sp1["Level"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
random_forest=RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train,y_train)
Y_pred=random_forest.predict(X_test)  ## predicting Level of students in test data

#ACCESSING PERFORMANCE
#Accuracy Score
accscore=accuracy_score(y_test,Y_pred)
print("Accuracy score of RFC based on all predictors using get_dummies:")
print(accscore)

#Confusion matrix
print("Confusion matrix for RFC based on all predictors using get_dummies:")
cf=confusion_matrix(y_test,Y_pred)
print(cf)
#Classification Report
report = classification_report(y_test, Y_pred)
print("Classification Report for RFC based on all predictors using get_dummies:")
print(report)
sns.heatmap(X.corr(), vmin=0, vmax=1)
plt.title("HeatMap of feature correlation with get_dummies",fontsize=15)
plt.show()



#####Feature importance plotting
featimp=pd.DataFrame()
featimp["feat_name"]=pd.Series(X.columns)
featimp["feature_imp"]=pd.Series(forest.feature_importances_)
featimp.sort_values(by=['feature_imp'], inplace=True)


#Plot Cumulative feat imp:

featimp["Impcumsum"]=featimp["feature_imp"].cumsum()
print(featimp)

fig,ax=plt.subplots()
fig.tight_layout(pad=4.0)
fig.set_size_inches(10,8, forward=True)
ax.plot(featimp["feat_name"],featimp["Impcumsum"])
ax.set_title("Cumulative Importances for the features with get_dummies",fontsize=12)
ax.set_xticklabels(featimp["feat_name"],rotation=90)
plt.grid()
plt.hlines(y = 0.90, xmin=0, xmax=len(featimp["feat_name"])-1, color = 'r', linestyles = 'dashed')
plt.savefig("featimp-dummies.png")
plt.show()

##Feature selection using RFE and ExtratreesClassifier
#RFE

rfe=RFE(random_forest,1)
fit=rfe.fit(X_train,y_train)
print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)
print("\n\nFeature importance using RFE for get_dummies table:")
for col,rank in zip(X.columns,fit.ranking_):
    print(col,rank)

#Extra Trees Classifiers
forest=ExtraTreesClassifier(n_estimators=200)
forest.fit(X_train,y_train)
print("Feature importance using ExtraTreesClassifiers for get_dummies table:")
for col1,imp in zip(X.columns,forest.feature_importances_):
    print(col1,imp)

#Conducting RFE on Selected Features
#Without Total Score
X=Sp1[["math score", "reading score", "writing score", "Level","Gender_female", "Gender_male", "Race_group A", "Race_group B","Race_group C", "Race_group D", "Race_group E","PEd_associate's degree", "PEd_bachelor's degree", "PEd_high school","PEd_master's degree", "PEd_some college", "PEd_some high school","lun_free/reduced", "lun_standard", "TP_completed", "TP_none"]]
y=Sp1["Level"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
random_forest=RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train,y_train)
Y_pred=random_forest.predict(X_test)  ## predicting Level of students in test data

#ACCESSING PERFORMANCE
#Accuracy Score
accscore=accuracy_score(y_test,Y_pred)
print("Accuracy score of RFC without Total Score predictor using get_dummies:")
print(accscore)

#Confusion matrix
print("Confusion matrix for RFC without Total Score predictor using get_dummies:")
cf=confusion_matrix(y_test,Y_pred)
print(cf)
#Classification Report
report = classification_report(y_test, Y_pred)
print("Classification Report for RFC without Total Score predictor using get_dummies:")
print(report)
sns.heatmap(X.corr(), vmin=0, vmax=1)
plt.show()


#Conducting RFE on Selected Features
#Without Total Score,Math Score,Writing Score and Reading Score as predictors
X=Sp1[["Level","Gender_female", "Gender_male", "Race_group A", "Race_group B","Race_group C", "Race_group D", "Race_group E","PEd_associate's degree", "PEd_bachelor's degree", "PEd_high school","PEd_master's degree", "PEd_some college", "PEd_some high school","lun_free/reduced", "lun_standard", "TP_completed", "TP_none"]]

y=Sp1["Level"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
random_forest=RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train,y_train)
Y_pred=random_forest.predict(X_test)  ## predicting Level of students in test data

#ACCESSING PERFORMANCE
#Accuracy Score
accscore=accuracy_score(y_test,Y_pred)
print("Accuracy score of RFC without Total Score,Math Score,Writing Score and Reading Score as predictors using get_dummies:")
print(accscore)

#Confusion matrix
print("Confusion matrix for RFC without Total Score,Math Score,Writing Score and Reading Score as predictors using get_dummies:")
cf=confusion_matrix(y_test,Y_pred)
print(cf)
#Classification Report
report = classification_report(y_test, Y_pred)
print("Classification Report for RFC without Total Score,Math Score,Writing Score and Reading Score as predictors using get_dummies:")
print(report)
sns.heatmap(X.corr(), vmin=0, vmax=1)
plt.show()



#Plot
def grosscol(col):
    if col<=175:
        return("indianred")
    elif (col>175 and col<=205):
        return("tomato")
    elif (col>205 and col<=233):
        return ("red")
    else:
        return("gold")
for ind,lev in zip(X_test.index,y_test):
    plt.scatter(ind,lev,color=grosscol(lev))
plt.scatter(X_test.index,Y_pred,c=Y_pred)
plt.show()