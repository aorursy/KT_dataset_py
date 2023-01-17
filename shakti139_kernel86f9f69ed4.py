import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline
cancer_df=pd.read_csv('../input/data.csv')
cancer_df.head()
cancer_df.columns
# check column types in the dataframe



cancer_df.dtypes
#check null values on each columns

cancer_df.isna().sum()
#drop unwanted columns

df=cancer_df.drop(['id','Unnamed: 32'],axis=1)
df.head()
df.dtypes
df['diagnosis'].unique()
#deviding the dataset into predictor and target sets

y=df.iloc[:,:1]

X=df.iloc[:,1:]
X.head()
y.head()
y.diagnosis=y.diagnosis.map({'M':1,'B':0})
y.head()
#Scale all numerical features in X  using sklearn's StandardScaler class

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaled_X_array=scaler.fit_transform(X)
scaled_X=pd.DataFrame(scaled_X_array)
scaled_X.head()
#Perform PCA on numeric features, X. Use sklearn's PCA class. Only retain as many principal components (PCs) as explain 95% variance.

from sklearn.decomposition import PCA
pca=PCA(.95)
final_X=pca.fit_transform(scaled_X)

final_X.shape
#viii) Split X,y into train and test datasets in the ratio of 80:20 using sklearn's train_test_split function. 

#You get: X_train, X_test, y_train, y_test.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(final_X,y,test_size=.2,shuffle=True)
# ix) Perform modeling on (X_train,y_train) using above listed algorithms (six).



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
#Create default classifiers



dt=DecisionTreeClassifier()

kn=KNeighborsClassifier()

xgb=XGBClassifier()

et=ExtraTreesClassifier()

gb=GradientBoostingClassifier()

rf=RandomForestClassifier()
# Train data



dt_train=dt.fit(X_train,y_train)

kn_train=kn.fit(X_train,y_train)

xgb_train=xgb.fit(X_train,y_train)

et_train=et.fit(X_train,y_train)

gb_train=gb.fit(X_train,y_train)

rf_train=rf.fit(X_train,y_train)
# Make predictions

y_pred_dt=dt_train.predict(X_test)
y_pred_et=et_train.predict(X_test)
y_pred_rf=rf_train.predict(X_test)
y_pred_gb=gb_train.predict(X_test)
y_pred_xgb=xgb_train.predict(X_test)
y_pred_kn=kn_train.predict(X_test)
#xi) Compare the performance of each of these models by calculating metrics as follows:: 

         #a) accuracy,

         #b) Precision & Recall,

         #c) F1 score,

         #d) AUC

        

# For performance measures

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support
# Calculate accuracy
accuracy_score(y_test,y_pred_dt)
accuracy_score(y_test,y_pred_et)
accuracy_score(y_test,y_pred_rf)
accuracy_score(y_test,y_pred_gb)
accuracy_score(y_test,y_pred_xgb)
accuracy_score(y_test,y_pred_kn)
# best accuracy score is for XGBoost model.
# calculating Precision,Recall and F1 score for each model. 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_et))
print(classification_report(y_test,y_pred_gb))
print(classification_report(y_test,y_pred_xgb))
print(classification_report(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_kn))
# creating confusion matrics for each model

confusion_matrix(y_test,y_pred_dt)



confusion_matrix(y_test,y_pred_et)
confusion_matrix(y_test,y_pred_gb)
confusion_matrix(y_test,y_pred_xgb)
confusion_matrix(y_test,y_pred_rf)
confusion_matrix(y_test,y_pred_kn)
# Again XGB model has the best results as per the F1 score as well and this has the best confusion matrix too.
# calculating the AUC values for each model
# decission tree model
fpr_dt, tpr_dt, thresholds = roc_curve(y_test,

                                 dt_train.predict_proba(X_test)[: , 1],

                                 pos_label= 1

                                 )
dt_auc=auc(fpr_dt,tpr_dt)
# ExtraTreesClassifier model
fpr_et, tpr_et, thresholds = roc_curve(y_test,

                                 et_train.predict_proba(X_test)[: , 1],

                                 pos_label= 1

                                 )

et_auc=auc(fpr_et,tpr_et)
# random forest classifier model
fpr_rf, tpr_rf, thresholds = roc_curve(y_test,

                                 rf_train.predict_proba(X_test)[: , 1],

                                 pos_label= 1

                                 )

rf_auc=auc(fpr_rf,tpr_rf)
fpr_gb, tpr_gb, thresholds = roc_curve(y_test,

                                 gb_train.predict_proba(X_test)[: , 1],

                                 pos_label= 1

                                 )

gb_auc=auc(fpr_gb,tpr_gb)
fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test,

                                 xgb_train.predict_proba(X_test)[: , 1],

                                 pos_label= 1

                                 )

xgb_auc=auc(fpr_xgb,tpr_xgb)
fpr_kn, tpr_kn, thresholds = roc_curve(y_test,

                                 kn_train.predict_proba(X_test)[: , 1],

                                 pos_label= 1

                                 )

kn_auc=auc(fpr_kn,tpr_kn)
auc_dict={'dt_auc':dt_auc,'et_auc':et_auc,'rf_auc':rf_auc,'gb_auc':gb_auc,'xgb_auc':xgb_auc,'kn_auc':kn_auc}
max(auc_dict)
#again the AUC specifies the XGB is the best model for this prediction.
#xii) Also draw ROC curve for each
fig = plt.figure(figsize=(20,10))          

ax = fig.add_subplot(111)   





ax.plot([0, 1], [0, 1], ls="--")  



ax.set_xlabel('False Positive Rate')  

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')





ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])





ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

ax.plot(fpr_gb, tpr_gb, label = "gbm")

ax.plot(fpr_xgb, tpr_xgb, label = "xgb")

ax.plot(fpr_et, tpr_et, label = "et")

ax.plot(fpr_kn, tpr_kn, label = "kn")



ax.legend()

plt.show()
## From all metrices the XGB model gives the best results.
# Dtata Explorationa and Visualization:

cancer_df.head()
cancer_df.diagnosis.value_counts()
# Here we will use Seaborn to create a heat map of the correlations between the features.

features_mean= list(cancer_df.columns[1:11])

plt.figure(figsize=(25,15))

sns.heatmap(cancer_df[features_mean].corr(), annot=True, square=True, cmap='coolwarm')

plt.show()
color_dic = {'M':'red', 'B':'blue'}

colors = cancer_df['diagnosis'].map(lambda x: color_dic.get(x))



sm = pd.scatter_matrix(cancer_df[features_mean], c=colors, alpha=0.4, figsize=((15,15)));



plt.show()
# plotting the distribution of each type of diagnosis for each of the mean features.



bins = 12

plt.figure(figsize=(15,15))

rows = int(len(features_mean)/2)

features_mean = features_mean[1:]

for i, feature in enumerate(features_mean):

    

    plt.subplot(rows, 2, i+1)

    

    sns.distplot(cancer_df[cancer_df['diagnosis']=='M'][feature], bins=bins, color='r', label='M');

    sns.distplot(cancer_df[cancer_df['diagnosis']=='B'][feature], bins=bins, color='b', label='B');

    



    plt.legend(loc='upper right')

    

plt.tight_layout()

plt.show()
bins = 12

plt.figure(figsize=(10,8))





sns.distplot(cancer_df[cancer_df['diagnosis']=='M']['radius_mean'], bins=bins, color='r', label='M');

sns.distplot(cancer_df[cancer_df['diagnosis']=='B']['radius_mean'], bins=bins, color='b', label='B');
rows = int(len(features_mean)/2)

rows
plt.figure(figsize=(15,15))

features_mean = features_mean[1:]

rows = int(len(features_mean)/2)

for i, feature in enumerate(features_mean):

    

    plt.subplot(rows, 2, i+1)

    

    sns.boxplot(x='diagnosis', y=feature, data=cancer_df, palette="Set1")



plt.tight_layout()

plt.show()
plt.figure(figsize=(10,8))



sns.boxplot(x='diagnosis',y='texture_mean',data=cancer_df,palette="Set1")
for i, feature in enumerate(features_mean):

    rows = int(len(features_mean)/2)

    print(i,rows,feature)