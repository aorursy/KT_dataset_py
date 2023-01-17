# Data analysis tools

import pandas as pd

import numpy as np



# Data Visualization Tools

import seaborn as sns

import matplotlib.pyplot as plt



# Data Pre-Processing Libraries

from sklearn.preprocessing import LabelEncoder,StandardScaler



# For Train-Test Split

from sklearn.model_selection import train_test_split



# Libraries for various Algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.tree import DecisionTreeClassifier



# Metrics Tools

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score, f1_score



#For Receiver Operating Characteristic (ROC)

from sklearn.metrics import roc_curve ,roc_auc_score, auc



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/people-charm/People Charm case.csv')
df.head(10)
df.info()
df.isnull().sum()
len(df[df.duplicated()])
# Removing all duplicates

df=df.drop_duplicates(subset=None, keep='first', inplace=False)

len(df[df.duplicated()])
sns.countplot(df["left"])

plt.xlabel("Class")

plt.ylabel("frequency")

plt.title("Checking imbalance")
sns.distplot(df.skew(),hist=False)

plt.show()
# Printing interquartile range (IQR) for each column

Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
# Boxplot visualization for columns with high IQR



plt.boxplot([df["numberOfProjects"]])

plt.xticks([1],["numberOfProjects"])

plt.show()

plt.boxplot([df["timeSpent.company"]])

plt.xticks([1],["timeSpent.company"])

plt.show()

plt.boxplot([df["avgMonthlyHours"]])

plt.xticks([1],["avgMonthlyHours"])

plt.show()
# Identifying the Ideal min and maximum value



print(df['timeSpent.company'].quantile(0.10))

print(df['timeSpent.company'].quantile(0.90))
# Capping and Flooring of Outliers



df["timeSpent.company"] = np.where(df["timeSpent.company"] <2.0, 2.0,df['timeSpent.company'])

df["timeSpent.company"] = np.where(df["timeSpent.company"] >5.0, 5.0,df['timeSpent.company'])



df.head()
cols=['dept', 'salary']

for label in cols:

    df[label]=LabelEncoder().fit_transform(df[label])

df.head()
# Correlation 

df.corr()["left"]
plt.figure(figsize = (10,5))

sns.heatmap(df.corr(), cmap = "RdYlGn", annot = True)
# satisfactoryLevel vs left

fig, ax = plt.subplots(3,3,figsize = (15,15))

satisfactoryLevel = pd.crosstab(df['satisfactoryLevel'],df['left'])

satisfactoryLevel.div(satisfactoryLevel.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax=ax[0,0])



# lastEvaluation vs left

lastEvaluation = pd.crosstab(df['lastEvaluation'],df['left'])

lastEvaluation.div(lastEvaluation.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True, ax=ax[0,1])



# numberOfProjects vs left

numberOfProjects = pd.crosstab(df['numberOfProjects'],df['left'])

numberOfProjects.div(numberOfProjects.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True, ax=ax[0,2])



# avgMonthlyHours vs left

avgMonthlyHours = pd.crosstab(df['avgMonthlyHours'],df['left'])

avgMonthlyHours.div(avgMonthlyHours.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,ax=ax[1,0])



# timeSpent vs left

timeSpent = pd.crosstab(df['timeSpent.company'],df['left'])

timeSpent.div(timeSpent.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True, ax=ax[1,1])



# workAccident vs left

workAccident = pd.crosstab(df['workAccident'],df['left'])

workAccident.div(workAccident.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True, ax=ax[1,2])



# promotionInLast5years vs left

promotionInLast5years= pd.crosstab(df['promotionInLast5years'],df['left'])

promotionInLast5years.div(promotionInLast5years.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,ax=ax[2,0])



# dept vs left

dept= pd.crosstab(df['dept'],df['left'])

dept.div(dept.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,ax=ax[2,1])



# salary vs left

salary= pd.crosstab(df['salary'],df['left'])

salary.div(salary.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,ax=ax[2,2])
scaler=StandardScaler()
X=df.drop(["left"],axis=1)

y=df["left"]



X =scaler.fit_transform(X)
# Train-Test Split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
#Fitting the model



logistic_Regression = LogisticRegression(max_iter=3000,random_state=0,class_weight="balanced",solver = "saga")

logistic_Regression.fit(x_train,y_train)
# Applying the model to the x_test



y_pred = logistic_Regression.predict(x_test)

y_pred
# Finding Accuracy



log = accuracy_score(y_pred,y_test)*100
# Confusion Matrix



cmlr=confusion_matrix(y_pred,y_test)

print(cmlr)
# Classification Report that computes various

# metrics like Precision, Recall and F1 Score



print(classification_report(y_pred,y_test))
# Plotting the ROC Curve



prob_lr=logistic_Regression.predict_proba(x_test)

auc_lr = roc_auc_score(y_test,prob_lr[:,1])

fprlr,tprlr,_ = roc_curve(y_test,prob_lr[:,1])

roc_auc=auc(fprlr,tprlr)

plt.plot(fprlr,tprlr,label = "AUC = %.2f" % auc_lr)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Logistic Regression")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
#Fitting the model



knn = KNeighborsClassifier(n_neighbors=35)

knn.fit(x_train,y_train)
# Applying the model to the x_test



pred_knn = knn.predict(x_test)

pred_knn
# Finding Accuracy



KNN = accuracy_score(pred_knn,y_test)*100
# Confusion Matrix



cm_knn=confusion_matrix(pred_knn,y_test)

print(cm_knn)
# Classification Report that computes various

# metrics like Precision, Recall and F1 Score



print(classification_report(pred_knn,y_test))
# Plotting the ROC Curve



prob_knn= knn.predict_proba(x_test)

auc_knn = roc_auc_score(y_test,prob_knn[:,1])

fprknn,tprknn,_= roc_curve(y_test,prob_knn[:,1])

roc_auc_knn=auc(fprknn,tprknn)

plt.plot(fprknn,tprknn,label = "AUC = %.2f" % auc_knn)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for KNN")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
#Fitting the model



gnb=GaussianNB()

gnb.fit(x_train,y_train)
# Applying the model to the x_test



pred_gnb = gnb.predict(x_test)

pred_gnb
# Finding Accuracy



GNB = accuracy_score(pred_gnb,y_test)*100
# Confusion Matrix



cm_gnb=confusion_matrix(pred_gnb,y_test)

print(cm_gnb)
# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(pred_gnb,y_test))
# Plotting the ROC Curve



prob_gnb= gnb.predict_proba(x_test)

auc_gnb = roc_auc_score(y_test,prob_gnb[:,1])

fprgnb,tprgnb,_= roc_curve(y_test,prob_gnb[:,1])

roc_auc_gnb=auc(fprgnb,tprgnb)

plt.plot(fprgnb,tprgnb,label = "AUC = %.2f" % auc_gnb)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Naive-Bayes")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
#Fitting the model



svc = SVC(probability=True)

svc.fit(x_train,y_train)



# Applying the model to the x_test

pred_svc = svc.predict(x_test)

pred_svc
# Finding Accuracy



SVC = accuracy_score(pred_svc,y_test)*100
# Confusion Matrix



cm_svc=confusion_matrix(pred_svc,y_test)

print(cm_svc)
# Classification Report that computes various 

#metrics like Precision, Recall and F1 Score



print(classification_report(pred_svc,y_test))
# Plotting the ROC Curve



prob_svc= svc.predict_proba(x_test)

auc_svc = roc_auc_score(y_test,prob_svc[:,1])

fprsvc,tprsvc,_= roc_curve(y_test,prob_svc[:,1])

roc_auc_svc=auc(fprsvc,tprsvc)

plt.plot(fprsvc,tprsvc,label = "AUC = %.2f" % auc_svc)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for SVM")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
#Fitting the model



dtree_en = DecisionTreeClassifier()

clf = dtree_en.fit(x_train,y_train)
# Applying the model to the x_test



pred_dt = clf.predict(x_test)

pred_dt
# Finding Accuracy



DTREE = accuracy_score(pred_dt,y_test)*100
# Confusion Matrix



cm_dt=confusion_matrix(y_test,pred_dt)

print(cm_dt)



# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(y_test,pred_dt))
# Plotting the ROC Curve



prob_dt= dtree_en.predict_proba(x_test)

auc_dt = roc_auc_score(y_test,prob_dt[:,1])

fprdt,tprdt,_= roc_curve(y_test,prob_dt[:,1])

roc_auc_dt=auc(fprdt,tprdt)

plt.plot(fprdt,tprdt,label = "AUC = %.2f" % auc_dt)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Decision Tree")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
#Fitting the model



GBC=GradientBoostingClassifier(n_estimators=150)

GBC.fit(x_train,y_train)
# Applying the model to the x_test



Y_predict=GBC.predict(x_test)

Y_predict
# Finding Accuracy



gbc = accuracy_score(y_test,Y_predict)*100
# Confusion Matrix



cm_gbc=confusion_matrix(y_test,Y_predict)

print(cm_gbc)



# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(y_test,Y_predict))
# Plotting the ROC Curve



prob_GBC= GBC.predict_proba(x_test)

auc_GBC = roc_auc_score(y_test,prob_GBC[:,1])

fprGBC,tprGBC,_= roc_curve(y_test,prob_GBC[:,1])

roc_auc_GBC=auc(fprGBC,tprGBC)

plt.plot(fprGBC,tprGBC,label = "AUC = %.2f" % auc_GBC)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Gradient Boosting")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
#Fitting the model



rfc = RandomForestClassifier(n_estimators=30,criterion='gini',random_state=1,max_depth=10)

rfc.fit(x_train, y_train)
# Applying the model to the x_test



pred_rf= rfc.predict(x_test)

pred_rf
# Finding Accuracy



RFC = accuracy_score(y_test,pred_rf)*100
# Confusion Matrix



cm_rf=confusion_matrix(pred_rf,y_test)

print(cm_rf)
# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(pred_rf,y_test))
# Plotting the ROC Curve



prob_rfc= rfc.predict_proba(x_test)

auc_rfc = roc_auc_score(y_test,prob_rfc[:,1])

fprrfc,tprrfc,_= roc_curve(y_test,prob_rfc[:,1])

roc_auc_rfc=auc(fprrfc,tprrfc)

plt.plot(fprrfc,tprrfc,label = "AUC = %.2f" % auc_rfc)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Random Forest")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
#Fitting the model. Base model is chosen to be Decision Tree



model = DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=0)

adaboost = AdaBoostClassifier(n_estimators=80, base_estimator=model,random_state=0)

adaboost.fit(x_train,y_train)
# Applying the model to the x_test



pred = adaboost.predict(x_test)

pred
# Finding Accuracy



ada = accuracy_score(y_test,pred)*100
# Confusion Matrix



cm_ada=confusion_matrix(pred,y_test)

print(cm_ada)



# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(pred,y_test))
# Plotting the ROC Curve



prob_adaboost= adaboost.predict_proba(x_test)

auc_adaboost = roc_auc_score(y_test,prob_adaboost[:,1])

fpradaboost,tpradaboost,_= roc_curve(y_test,prob_adaboost[:,1])

roc_auc_adaboost=auc(fpradaboost,tpradaboost)

plt.plot(fpradaboost,tpradaboost,label = "AUC = %.2f" % auc_adaboost)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for AdaBoost (Entropy-Decision Tree)")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
#Fitting the model



xgb =  XGBClassifier(learning_rate =0.000001,n_estimators=1000,max_depth=5,min_child_weight=1,

                     subsample=0.8,colsample_bytree=0.8,nthread=4,scale_pos_weight=1,seed=27)

xgb.fit(x_train, y_train)
# Applying the model to the x_test





predxg = xgb.predict(x_test)



# Finding Accuracy

xg = accuracy_score(y_test,predxg)*100

# Confusion Matrix



cm_xg=confusion_matrix(predxg,y_test)

print(cm_xg)



# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(predxg,y_test))
# Plotting the ROC Curve



prob_xgb= xgb.predict_proba(x_test)

auc_xgb = roc_auc_score(y_test,prob_xgb[:,1])

fprxgb,tprxgb,_= roc_curve(y_test,prob_xgb[:,1])

roc_auc_xgb=auc(fprxgb,tprxgb)

plt.plot(fprxgb,tprxgb,label = "AUC = %.2f" % auc_xgb)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for XGBoost")

plt.plot([0,1],[0,1],"--")

plt.legend()

plt.show()
# Accuracy values for all the models

print("1)  Logistic Regression    :",round(log, 2))

print("2)  KNN                    :",round(KNN, 2))

print("3)  Naive-Bayes            :",round(GNB, 2))

print("4)  SVM                    :",round(SVC, 2))

print("5)  Decision Tree          :",round(DTREE, 2))

print("6)  Gradient Boosting      :",round(gbc, 2))

print("7)  Random Forest          :",round(RFC, 2))

print("8)  AdaBoost               :",round(ada, 2))

print("9)  XGBoost                :",round(xg, 2))
# Area Under the Curve(AUC) of all the models

print('Area under the curve for Logistic Regression :',round(roc_auc, 2))

print('Area under the curve for KNN                 :',round(roc_auc_knn, 2))

print('Area under the curve for Naive-Bayes         :',round(roc_auc_gnb, 2))

print('Area under the curve for SVM                 :',round(roc_auc_svc, 2))

print('Area under the curve for Decision Tree       :',round(roc_auc_dt, 2))

print('Area under the curve for Gradient Boosting   :',round(roc_auc_GBC, 2))

print('Area under the curve for Random Forest       :',round(roc_auc_rfc, 2))

print('Area under the curve for AdaBoost            :',round(roc_auc_adaboost, 2))

print('Area under the curve for XGBoost             :',round(roc_auc_xgb, 2))
#ROC Curve for all models

plt.figure(figsize = (20,10))

plt.plot(fprlr,tprlr,label = "Logistic Regression")

plt.plot(fprknn,tprknn,label = "KNN")

plt.plot(fprgnb,tprgnb,label = "Naive-Bayes")

plt.plot(fprsvc,tprsvc,label = "SVM")

plt.plot(fprdt,tprdt,label = "Decision Tree")

plt.plot(fprGBC,tprGBC,label = "Gradient Boosting",color='black')

plt.plot(fprrfc,tprrfc,label = "Random Forest",color='yellow')

plt.plot(fpradaboost,tpradaboost,label = " AdaBoost")

plt.plot(fprxgb,tprxgb,label = "XGBoost")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.legend(loc="lower right", fontsize=10)

plt.grid(True)
# f1_score of all models

print("1)  Logistic Regression    :",round(f1_score(y_pred,y_test), 2))

print("2)  KNN                    :",round(f1_score(pred_knn,y_test), 2))

print("3)  Naive-Bayes            :",round(f1_score(pred_gnb,y_test), 2))

print("4)  SVM                    :",round(f1_score(pred_svc,y_test), 2))

print("5)  Decision Tree          :",round(f1_score(pred_dt,y_test), 2))

print("6)  Gradient Boosting      :",round(f1_score(Y_predict,y_test), 2))

print("7)  Random Forest          :",round(f1_score(pred_rf,y_test), 2))

print("8)  AdaBoost               :",round(f1_score(pred,y_test), 2))

print("9)  XGBoost                :",round(f1_score(predxg,y_test), 2))
#Accessing the False Positives of all models from their confusion Matrix

print("1)  Logistic Regression    :",cmlr[0][1])

print("2)  KNN                    :",cm_knn[0][1])

print("3)  Naive-Bayes            :",cm_gnb[0][1])

print("4)  SVM                    :",cm_svc[0][1])

print("5)  Decision Tree          :",cm_dt[0][1])

print("6)  Gradient Boosting      :",cm_gbc[0][1])

print("7)  Random Forest          :",cm_rf[0][1])

print("8)  AdaBoost               :",cm_ada[0][1])

print("9)  XGBoost                :",cm_xg[0][1])