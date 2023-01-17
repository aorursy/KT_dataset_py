import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Data = pd.read_csv("../input/heart.csv")

Data.head()
Data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
f,ax = plt.subplots(figsize=(13,10))

sns.heatmap(Data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)

plt.title("Correlation Matrix",fontsize=14)

plt.show()
plt.figure(figsize=(10,10))

g=sns.barplot(Data["sex"],Data["target"],hue=Data["cp"],palette="rainbow",edgecolor='yellow')

plt.title("Target-0(SAFE), Target-1(UNSAFE)",fontsize=14)

plt.xlabel("SEX",fontsize=14)

plt.ylabel("Target",fontsize=14)

plt.xticks(np.arange(2),["FEMALE","MALE"])

plt.show()
plt.figure(figsize=(10,10))

g=sns.scatterplot(Data["age"],Data["trestbps"],hue=Data["sex"],size=Data["target"],size_order=[1,0],palette="copper_r",s=400)

#g=sns.scatterplot(Data["age"],Data["chol"],hue=Data["target"],size=Data["sex"],palette="copper_r",ax=ax[1],s=200)

plt.title("Target-0(SAFE), Target-1(UNSAFE)\nSex-0(Female) Sex-1(Male)",fontsize=14)

plt.xlabel("AGE",fontsize=14)

plt.ylabel("resting blood pressure",fontsize=14)

plt.grid(True)

plt.show()
plt.figure(figsize=(15,10))

g=sns.scatterplot(Data["age"],Data["chol"],hue=Data["sex"],size=Data["target"],palette="plasma_r",size_order=[1,0],s=800)

plt.title("Target-0(SAFE), Target-1(UNSAFE)\nSex-0(Female) Sex-1(Male)",fontsize=14)

plt.xlabel("AGE",fontsize=14)

plt.ylabel("serum cholestoral in mg/dl ",fontsize=14)

plt.ylim([100,450])

plt.grid(True)

plt.show()
plt.figure(figsize=(50,50))

g=sns.catplot("target","age",col="sex",hue="restecg",data=Data,palette="magma_r")

#plt.title("Target-0(SAFE), Target-1(UNSAFE)",fontsize=14)

#plt.xlabel("AGE",fontsize=14)

#plt.ylabel("Target",fontsize=14)

plt.show()
plt.figure(figsize=(15,15))

g=sns.barplot("target","slope",hue="age",data=Data,palette="plasma_r",errwidth =0)

plt.xlabel("Target",fontsize=14)

plt.ylabel("Slope of ST segment",fontsize=14)

plt.xticks(np.arange(2),["SAFE","UNSAFE"])

plt.grid(True)

plt.show()
plt.figure(figsize=(15,15))

g=sns.barplot("age","cp",hue="target",data=Data,palette="plasma_r",errwidth =0)

plt.title("Target-0(SAFE), Target-1(UNSAFE)",fontsize=14)

plt.xlabel("AGE",fontsize=14)

plt.ylabel("(ca)number of major vessels (0-3) colored by flourosopy",fontsize=14)

plt.grid(True)

plt.show()
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score,recall_score,auc,roc_curve
y = Data['target']

Data.drop("target", axis=1, inplace=True)

X = Data
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
Model = GradientBoostingClassifier(verbose=1, learning_rate=0.5,warm_start=True)

Model.fit(x_train, y_train)
# feature importance

print(Model.feature_importances_)
plt.figure(figsize=(15,15))

plt.bar(range(len(Model.feature_importances_)), Model.feature_importances_)

plt.title("Feature Importance")

plt.xticks(np.arange(13), Data.columns)

plt.grid(True)

plt.show()
y_pred = Model.predict(x_test)
print("Accuracy(GradientBoostingClassifier)\t:"+str(accuracy_score(y_test,y_pred)))

print("Precision(GradientBoostingClassifier)\t:"+str(precision_score(y_test,y_pred)))

print("Recall(GradientBoostingClassifier)\t:"+str(recall_score(y_test,y_pred)))
Model_2 = RandomForestClassifier(verbose=1,n_estimators=200,n_jobs=-1,warm_start=True)

Model_2.fit(x_train, y_train)
y_pred_2 = Model_2.predict(x_test)
print("Accuracy(RandomForestClassifier)\t:"+str(accuracy_score(y_test,y_pred_2)))

print("Precision(RandomForestClassifier)\t:"+str(precision_score(y_test,y_pred_2)))

print("Recall(RandomForestClassifier)\t:"+str(recall_score(y_test,y_pred_2)))
from xgboost import XGBClassifier

Model_3 = XGBClassifier()

Model_3.fit(x_train, y_train)
y_pred_3 = Model_3.predict(x_test)
print("Accuracy(XGBClassifier)\t:"+str(accuracy_score(y_test,y_pred_3)))

print("Precision(XGBClassifier)\t:"+str(precision_score(y_test,y_pred_3)))

print("Recall(XGBClassifier)\t:"+str(recall_score(y_test,y_pred_3)))
prob_1=Model.predict_proba(x_test)

prob_1 = prob_1[:,1]# Probalility prediction for GradientBoosting classifier

prob_2=Model_2.predict_proba(x_test)

prob_2 = prob_2[:,1]# Probalility prediction for Rangomforest classifier

prob_3=Model_3.predict_proba(x_test)

prob_3 = prob_3[:,1]# Probalility prediction for XGBoost classifier
fpr1, tpr1, _ = roc_curve(y_test, prob_1)

fpr2, tpr2, _ = roc_curve(y_test, prob_2)

fpr3, tpr3, _ = roc_curve(y_test, prob_3)

plt.figure(figsize=(14,12))

plt.title('Receiver Operating Characteristic',fontsize=14)

plt.plot(fpr1, tpr1, label = 'AUC(GradientBoosting Classifier) = %0.3f' % auc(fpr1, tpr1))

plt.plot(fpr2, tpr2, label = 'AUC(Randomforest Classifier) = %0.3f' % auc(fpr2, tpr2))

plt.plot(fpr3, tpr3, label = 'AUC(XGBoost Classifier) = %0.3f' % auc(fpr3, tpr3))

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.ylabel('True Positive Rate',fontsize=14)

plt.xlabel('False Positive Rate',fontsize=14)

plt.show()