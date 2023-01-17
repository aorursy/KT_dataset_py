import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#libraries for data modelling

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#libraries for data preprocessing

from sklearn.preprocessing import StandardScaler
#model for Dimentionality Reduction

from sklearn.decomposition import PCA
#libraries for performance measures

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
#libraries For data splitting

from sklearn.model_selection import train_test_split
os.chdir("../input")
data_file = pd.read_csv("data.csv")
pd.options.display.max_columns = 200
data_file.head()
data_file.tail()
data_file.shape
data_file.info()
data_file.describe()
data_file.dtypes
data_file.isna().sum()
#dropping the unwanted coloumns

df=data_file.drop(['id','Unnamed: 32'],axis=1)
df.shape
df['diagnosis'].unique()
#Splitting the Features and Target Data into X and y

X =df.iloc[:,1:]

y=df.iloc[:,:1]
X.shape
y.shape
X.head()
y.head()
y=y.diagnosis.map({'M':1,'B':0})
y.head()
#Scale all numerical features in X  using sklearn's StandardScaler class

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_sc = scaler.fit_transform(X)
sca_x=pd.DataFrame(X_sc)
sca_x.head()
#Perform PCA on numeric features

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca=PCA(.95)
X=pca.fit_transform(sca_x)
X.shape
#Split and shuffle data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2,shuffle=True)
#Create default classifiers

dtc=DecisionTreeClassifier()

knc=KNeighborsClassifier()

xgbc=XGBClassifier()

etc=ExtraTreesClassifier()

gbc=GradientBoostingClassifier()

rfc=RandomForestClassifier()
# Train data

dtc_train=dtc.fit(X_train,y_train)
knc_train=knc.fit(X_train,y_train)
xgbc_train=xgbc.fit(X_train,y_train)
etc_train=etc.fit(X_train,y_train)
gbc_train=gbc.fit(X_train,y_train)
rfc_train=rfc.fit(X_train,y_train)
# Make predictions

y_pred_dtc=dtc_train.predict(X_test)
y_pred_etc=etc_train.predict(X_test)
y_pred_rfc=rfc_train.predict(X_test)
y_pred_gbc=gbc_train.predict(X_test)
y_pred_xgbc=xgbc_train.predict(X_test)
y_pred_knc=knc_train.predict(X_test)
#Get probability values

y_pred_dtc_prob = dtc_train.predict_proba(X_test)

y_pred_rfc_prob = rfc_train.predict_proba(X_test)

y_pred_etc_prob = etc_train.predict_proba(X_test)

y_pred_knc_prob = knc_train.predict_proba(X_test)

y_pred_xgbc_prob = xgbc_train.predict_proba(X_test)

y_pred_gbc_prob= gbc_train.predict_proba(X_test)
#xi) Compare the performance of each of these models by calculating metrics as follows:: 

         #a) accuracy,

         #b) Precision & Recall,

         #c) F1 score,

         #d) AUC

        

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support
# Calculate accuracy

accuracy_score(y_test,y_pred_dtc)
accuracy_score(y_test,y_pred_etc)
accuracy_score(y_test,y_pred_rfc)
accuracy_score(y_test,y_pred_gbc)
accuracy_score(y_test,y_pred_xgbc)
accuracy_score(y_test,y_pred_knc)
#Calculate Confusion Matrix

print("DecisionTreeClassifier: ")

confusion_matrix(y_test,y_pred_dtc)
print("RandomForestClassifier: ")

confusion_matrix(y_test,y_pred_rfc)
print("ExtraTreesClassifier: ")

confusion_matrix(y_test,y_pred_etc)
print("GradientBoostingClassifier: ")

confusion_matrix(y_test,y_pred_gbc)
print("KNeighborsClassifier: ")

confusion_matrix(y_test,y_pred_knc)
print("XGBClassifier: ")

confusion_matrix(y_test,y_pred_xgbc)
#Get probability values

y_pred_dtc_prob = dtc_train.predict_proba(X_test)

y_pred_rfc_prob = rfc_train.predict_proba(X_test)

y_pred_etc_prob = etc_train.predict_proba(X_test)

y_pred_knc_prob = knc_train.predict_proba(X_test)

y_pred_xgbc_prob = xgbc_train.predict_proba(X_test)

y_pred_gbc_prob= gbc_train.predict_proba(X_test)
fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dtc_prob[: , 1], pos_label= 1)

fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rfc_prob[: , 1], pos_label= 1)

fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_rfc_prob[: , 1], pos_label= 1)

fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_rfc_prob[: , 1], pos_label= 1)

fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xgbc_prob[: , 1], pos_label= 1)

fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test, y_pred_gbc_prob[: , 1], pos_label= 1)
print("DecisionTreeClassifier")

auc(fpr_dt,tpr_dt)
print("RandomForestClassifier")

auc(fpr_rf,tpr_rf)
print("ExtraTreesClassifier")

auc(fpr_etc,tpr_etc)
print("GradientBoostingClassifier")

auc(fpr_gbm,tpr_gbm)



print("KNeighborsClassifier")

auc(fpr_knc,tpr_knc)



print("XGBClassifier")

auc(fpr_xg,tpr_xg)
print("DecisionTreeClassifier: ")

precision_recall_fscore_support(y_test,y_pred_dtc)
print("RandomForestClassifier: ")

precision_recall_fscore_support(y_test,y_pred_rfc)
print("ExtraTreesClassifier: ")

precision_recall_fscore_support(y_test,y_pred_etc)
print("GradientBoostingClassifier: ")

precision_recall_fscore_support(y_test,y_pred_gbc)
print("KNeighborsClassifier: ")

precision_recall_fscore_support(y_test,y_pred_knc)
print("XGBClassifier: ")

precision_recall_fscore_support(y_test,y_pred_xgbc)
#Plot ROC curve now

fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111)



# Connect diagonals

ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line



# Labels etc

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')



# Set graph limits

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])



# Plot each graph now

ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

ax.plot(fpr_etc, tpr_etc, label = "etc")

ax.plot(fpr_knc, tpr_knc, label = "knc")

ax.plot(fpr_xg, tpr_xg, label = "xg")

ax.plot(fpr_gbm, tpr_gbm, label = "gbm")



# Set legend and show plot

ax.legend(loc="lower right")

plt.show()