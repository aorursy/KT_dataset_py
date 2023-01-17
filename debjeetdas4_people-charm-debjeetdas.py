import pandas as pd

import numpy as np

data= pd.read_csv("/kaggle/input/People Charm case.csv")

data.head()
# Information about the dataset

data.info()
# checking the null values in the dataset

data.isnull().sum()/len(data)
import matplotlib.pyplot as plt

%matplotlib inline

plt.boxplot([data["satisfactoryLevel"],data["timeSpent.company"],data["lastEvaluation"]])

plt.xticks([1,2,3],["satisfactoryLevel","timeSpent.company","lastEvaluation"])

plt.show()
# Capping the outliers

Q1 = data["timeSpent.company"].quantile(0.25)

Q3 = data["timeSpent.company"].quantile(0.75)

IQR = Q3 - Q1

out_ls=[]

for i in data["timeSpent.company"]:

    if(i<Q1-(1.5*IQR)):

        i=Q1-(1.5*IQR)

        out_ls.append(i)

    elif (i>Q3+(1.5*IQR)):

        i=Q3+(1.5*IQR)

        out_ls.append(i)

    else:

        out_ls.append(i)
# Dropping the original column which had outliers and adding a new column with the capped values for the outliers.



data=data.drop(columns=["timeSpent.company"])

data["timeSpent.company"]=out_ls
# Salary count on compared with the target variable

salarycount=data.pivot_table(index= ["dept"],values ="left",aggfunc=np.size).sort_values(by="left",ascending = False)

salarycount
# dept. substituted by the count corresponding to its output



salarycount=data.pivot_table(index= ["dept"],values ="left",aggfunc=np.size).sort_values(by="left",ascending = False)

salaryseries= pd.Series(salarycount["left"],index=salarycount.index)

data["dept"]=data["dept"].map(salaryseries)

data["dept"]
# label encoding on the numerically converted data

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["dept"]=le.fit_transform(data["dept"])

# unique values in dept column after encoding

data["dept"].unique()
# salary count when compared to the target variable

salarycount2=data.pivot_table(index= ["salary"],values ="left",aggfunc=np.size).sort_values(by="left",ascending = False)

salarycount2
# substituting salary with count of target variable on compared with salary

salarycount2=data.pivot_table(index= ["salary"],values ="left",aggfunc=np.size).sort_values(by="left",ascending = False)

salaryseries2= pd.Series(salarycount2["left"],index=salarycount2.index)

data["salary"]=data["salary"].map(salaryseries2)

data["salary"]
# Unique values in salary column after converting its categories into nemrical count of when compared with output variable

data["salary"].unique()
# Label encoding the numerical transformed variable salary

le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
# unique values after label encoding

data["salary"].unique()
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, MultinomialNB , BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve

import seaborn as sns

import matplotlib.pyplot as plt
# creating the target variable column at the end and dropping the duplicate

data["Target_left"]=data["left"]

data=data.drop("left",axis=1)
# checking the corelation

plt.figure(figsize=(15,10))

sns.heatmap(data.corr(),annot=True,cmap="Blues")
sns.distplot(data.skew(),hist=False)

plt.show
import seaborn as sns

sns.countplot(data["Target_left"])

plt.xlabel("Classes")

plt.ylabel("frequency")

plt.title(" Class Frequncy")
# splitting the model into training & testing datasets

real_x = data[["satisfactoryLevel","salary","workAccident","timeSpent.company","avgMonthlyHours","promotionInLast5years","numberOfProjects"]]

real_y = data.iloc[:,-1]

log_x= data[["satisfactoryLevel","salary","workAccident","timeSpent.company","avgMonthlyHours","promotionInLast5years","numberOfProjects"]].apply(lambda x : np.log(x+1))

trainxl,testxl,trainyl,testyl = train_test_split(log_x,real_y,test_size=0.2,random_state = 42)

trainx,testx,trainy,testy = train_test_split(real_x,real_y,test_size=0.2,random_state = 42)
# KNN

# performing model fitting on the skewed  and unscaled data

knn= KNeighborsClassifier(n_neighbors =9,weights='distance')

knn.fit(trainx,trainy)

knn_pred = knn.predict(testx)

knn_conf = confusion_matrix(testy,knn_pred)

knn_accuracy = accuracy_score(testy,knn_pred)

print("confusion matrix :" ,knn_conf,sep="\n")

print("Accuracy : ",knn_accuracy,sep="\n")

print("Classification Report : ",classification_report(testy,knn_pred),sep="\n")



pred_probknn = knn.predict_proba(testx)

auc_knn= roc_auc_score(testy,pred_probknn[:,1])

fprkn,tprkn,thresholdskn = roc_curve(testy,pred_probknn[:,1])

plt.plot(fprkn,tprkn,label="AUC = %.2f" %  auc_knn )

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for KNN model")

plt.plot([0,1],[0,1],"--")

plt.legend()
# KNN

# performing model fitting on the log transformed

knn= KNeighborsClassifier(n_neighbors =9,weights='distance')

knn.fit(trainxl,trainyl)

knn_predl = knn.predict(testxl)

knn_confl = confusion_matrix(testyl,knn_predl)

knn_accuracyl = accuracy_score(testyl,knn_predl)

print("confusion matrix :" ,knn_confl,sep="\n")

print("Accuracy : ",knn_accuracyl,sep="\n")

print("Classification Report : ",classification_report(testyl,knn_predl),sep="\n")
# Gaussian NB

# performing model fitting log transformed

gnb= GaussianNB()

gnb.fit(trainxl,trainyl)

gnb_predl = gnb.predict(testxl)

gnb_confl = confusion_matrix(testyl,gnb_predl)

gnb_accuracyl= accuracy_score(testyl,gnb_predl)

print("confusion matrix :" ,gnb_confl,sep="\n")

print("Accuracy : ",gnb_accuracyl,sep="\n")

print("Classification Report : ",classification_report(testyl,gnb_predl),sep="\n")



pred_probgnb = gnb.predict_proba(testx)

auc_gnb = roc_auc_score(testy,pred_probgnb[:,1])

fprgnb,tprgnb,thresholdsgnb = roc_curve(testy,gnb_predl)

plt.plot(fprgnb,tprgnb,label="AUC = %.2f" %  auc_gnb)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for GaussianNB model")

plt.plot([0,1],[0,1],"--")

plt.legend()
# ADABOOST

# performing model fitting on the skewed  and unscaled data

from sklearn.tree import DecisionTreeClassifier

ab= AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10,learning_rate=1.0,random_state=3)

ab.fit(trainx,trainy)

ab_pred = ab.predict(testx)

ab_conf = confusion_matrix(testy,ab_pred)

ab_accuracy= accuracy_score(testy,ab_pred)

print("confusion matrix : " ,ab_conf,sep="\n")

print("Accuracy : ",ab_accuracy,sep="\n")

print("Classification Report : ",classification_report(testy,ab_pred),sep="\n")



pred_probab = ab.predict_proba(testx)

auc_ab = roc_auc_score(testy,pred_probab[:,1])

fprab,tprab,thresholdsab = roc_curve(testy,pred_probab[:,1])

plt.plot(fprab,tprab,label = "AUC = %.2f" % auc_ab)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Adaboost ")

plt.plot([0,1],[0,1],"--")

plt.legend()
# GradientBOOST

# performing model fitting on the skewed  and unscaled data

from sklearn.tree import DecisionTreeClassifier

gb= GradientBoostingClassifier(n_estimators=10,learning_rate=0.9)

gb.fit(trainx,trainy)

gb_pred = gb.predict(testx)

gb_conf = confusion_matrix(testy,gb_pred)

gb_accuracy = accuracy_score(testy,gb_pred)

print("confusion matrix :" ,gb_conf,sep="\n")

print("Accuracy : ",gb_accuracy,sep="\n")

print("Classification Report : ",classification_report(testy,gb_pred),sep="\n")



pred_probgb = gb.predict_proba(testx)

auc_gb = roc_auc_score(testy,pred_probgb[:,1])

fprgb,tprgb,thresholdsgb = roc_curve(testy,pred_probgb[:,1])

plt.plot(fprgb,tprgb,label = "AUC = %.2f" % auc_gb)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Gradientboost ")

plt.plot([0,1],[0,1],"--")

plt.legend()
# XGBOOST

# performing model fitting on the skewed  and unscaled data

xgb= XGBClassifier(n_estimators=800,learning_rate=0.8)

xgb.fit(trainx,trainy)

xgb_pred = xgb.predict(testx)

xgb_conf = confusion_matrix(testy,xgb_pred)

xgb_accuracy = accuracy_score(testy,xgb_pred)

print("confusion matrix :" ,xgb_conf,sep="\n")

print("Accuracy : ",xgb_accuracy,sep="\n")

print("Classification Report : ",classification_report(testy,xgb_pred),sep="\n")





pred_prob_xgb = xgb.predict_proba(testx)

auc_xgb = roc_auc_score(testy,pred_prob_xgb[:,1])

fprxgb,tprxgb,thresholdsxgb = roc_curve(testy,pred_prob_xgb[:,1])

plt.plot(fprxgb,tprxgb,label = "AUC = %.2f" % auc_xgb)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for XGboost ")

plt.plot([0,1],[0,1],"--")

plt.legend()
# Decision Tree

# performing model fitting on the skewed  and unscaled data

DT= DecisionTreeClassifier(criterion="entropy",max_depth=32)

DT.fit(trainx,trainy)

DT_pred = DT.predict(testx)

DT_conf = confusion_matrix(testy,DT_pred)

DT_accuracy = accuracy_score(testy,DT_pred)

print("confusion matrix :" ,DT_conf,sep="\n")

print("Accuracy : ",DT_accuracy,sep="\n")

print("Classification Report : ",classification_report(testy,DT_pred),sep="\n")





pred_prob_DT = DT.predict_proba(testx)

auc_DT = roc_auc_score(testy,pred_prob_DT[:,1])

fprDT,tprDT,thresholdsDT = roc_curve(testy,pred_prob_DT[:,1])

plt.plot(fprDT,tprDT,label = "AUC = %.2f" % auc_DT)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for XGboost ")

plt.plot([0,1],[0,1],"--")

plt.legend()
# RandomForest

rf= RandomForestClassifier(criterion="entropy",max_depth=32)

rf.fit(trainx,trainy)

rf_pred = rf.predict(testx)

rf_conf = confusion_matrix(testy,rf_pred)

rf_accuracy = accuracy_score(testy,rf_pred)



from sklearn.metrics import roc_curve ,roc_auc_score



pred_prob_rf = rf.predict_proba(testx)

auc_rf = roc_auc_score(testy,pred_prob_rf[:,1])

fprrf,tprrf,thresholdsrf = roc_curve(testy,pred_prob_rf[:,1])





print("confusion matrix :" ,rf_conf,sep="\n")

print("Accuracy : ",rf_accuracy,sep="\n")

print("Classification Report : ",classification_report(testy,rf_pred),sep="\n")



import matplotlib.pyplot as plt

plt.plot(fprrf,tprrf,label = "AUC = %.2f" % auc_rf)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Random Forest")

plt.plot([0,1],[0,1],"--")

plt.legend()
# Performing logistic regression on the training data

log=LogisticRegression(class_weight="balanced")

log.fit(trainx,trainy)

log_pred = log.predict(testx)

log_conf = confusion_matrix(testy,log_pred)

log_accuracy = accuracy_score(testy,log_pred)

print("confusion matrix :" ,log_conf,sep="\n")

print("Accuracy : ",log_accuracy,sep="\n")

print("Classification Report : ",classification_report(testy,log_pred),sep="\n")



from sklearn.metrics import roc_curve ,roc_auc_score



pred_prob_log = log.predict_proba(testx)



fpr,tpr,thresholds = roc_curve(testy,pred_prob_log[:,1])

logit_roc_auc = roc_auc_score(testy,pred_prob_log[:,1])



import matplotlib.pyplot as plt

plt.plot(fpr,tpr,label = "AUC = %.2f" % logit_roc_auc)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Logistic Regression ")

plt.plot([0,1],[0,1],"--")

plt.legend()



print("Area under the curve is {}".format(logit_roc_auc))
# thus we will use this threshold value for the logistic regression



threshold_df = pd.DataFrame({"fpr":fpr.round(2),"tpr":tpr.round(2),"Threshold":thresholds.round(2)})

threshold_df[(threshold_df["fpr"]==0.25) & (threshold_df["tpr"]==0.85)]
# These are the predicted probabilities

probabilities= log.predict_proba(testx)

probabilities
# We use binarize to set the threshold to a desired values to reduce the FPR.

# binarize takes  2 arguments- binarize([predicted_proba],threshold = 0.47)



from sklearn.preprocessing import binarize

pred_prob = probabilities[:,1]

pred_new_y = binarize([pred_prob],threshold =0.47)

pred_new_y=pred_new_y.flatten()



# Now we use this new predicted values which were determined by using 0.47 as threshold values for prediction of model

# Here we can see the False positives has reduced, also there is a reduction in the in the accuracy



log_accuracy_new = accuracy_score(testy,pred_new_y)

log_conf_new = confusion_matrix(testy,pred_new_y)

print("confusion matrix :" ,log_conf_new,sep="\n")

print("Accuracy : ",log_accuracy_new,sep="\n")

print("Classification Report : ",classification_report(testy,pred_new_y),sep="\n")
list_ac=[xgb_pred,gb_pred,ab_pred,gnb_predl,knn_pred,DT_pred,rf_pred,pred_new_y]

list_area=[]

list_acc= []

for i in list_ac:

    list_acc.append(accuracy_score(testy,i))

    list_area.append(roc_auc_score(testy,i))

    

area_auc_DF = pd.DataFrame({"model":["XGBoost","GradientBoost","AdaBoost","GaussianNB","KNNClassifier","DecisionTree","RandomForest","LogisticRegression"],"AUC":list_area,"Accuracy":list_acc}) 

area_auc_DF.sort_values(by=["AUC","Accuracy"],ascending=[False,False]) 
from sklearn.model_selection import cross_val_score
# CROSS VALIDATION SCORE FOR LOGISTIC :



list1=[LogisticRegression(),RandomForestClassifier(),KNeighborsClassifier(),XGBClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),DecisionTreeClassifier(),GaussianNB()]

ls=[]

for i in list1:

    c= cross_val_score(i,real_x,real_y,cv=7)

    ls.append(c)
# Calculating the mean accuracy for each classification model:

mean_acc=[]

for i in ls:

    mean_acc.append(i.mean())
# mean accuracy of models:

mean_acc
# Creating dataframe with models and average mean using K-fold Cross validation

list4=["LogisticRegression","RandomForestClassifier","KNeighborsClassifier","XGBClassifier","AdaBoostClassifier","GradientBoostingClassifier","DecisionTreeClassifier","GaussianNB"]



model_df=pd.DataFrame({"Model":list4,"Avg_Accuracy":mean_acc})

model_df.sort_values(by= "Avg_Accuracy",ascending=False)