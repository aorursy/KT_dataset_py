import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



plt.style.use('ggplot')

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

print('Shape of dataset : ',df.shape)

df.head()
#check for null values

df.isna().sum()
print("Class counts : ")

df.DEATH_EVENT.value_counts()
plt.figure(figsize=(10,6))

plt.title("Correlation Heatmap")

sns.heatmap(abs(df.corr()), annot=True, cmap='coolwarm')

#plt.tight_layout()

plt.show()
# Scaling numeric features

col_num=['age', 'creatinine_phosphokinase','ejection_fraction', 'platelets',

         'serum_creatinine', 'serum_sodium','time']

scalar=MinMaxScaler()

for col in col_num:

    df[col]=scalar.fit_transform(np.array(df[col]).reshape(-1,1))



# Converting categorical features from type object to category

cols_cat=['anaemia', 'diabetes','high_blood_pressure', 'sex', 'smoking','DEATH_EVENT']

for col in cols_cat:

    df[col]=df[col].astype('category')

df.head()
X=df.drop('DEATH_EVENT', axis=1)

y=df.DEATH_EVENT



X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=1)
model=RandomForestClassifier(class_weight='balanced', random_state=0)

model.fit(X_train,y_train)



y_pred=model.predict(X_test)

y_proba=model.predict_proba(X_test)

y_proba=[p[1] for p in y_proba]



print("Accuracy Score : {}".format(accuracy_score(y_test,y_pred)))

print("Confusion Matrix : \n", confusion_matrix(y_test,y_pred))

print("ROC AUC Score : {}".format(roc_auc_score(y_test,y_proba)))

print(classification_report(y_test,y_pred))
df_feat=pd.DataFrame({'Feature':X.columns,

                      'Importance':model.feature_importances_})

df_feat.sort_values(by='Importance', ascending=False, inplace=True)

df_feat.reset_index(inplace=True)



sns.barplot(x='Importance',y='Feature', data=df_feat, orient='h')

plt.show()
acc=[]

rocauc=[]

feat=[]

for i in range(1,13):

    cols=df_feat.Feature[:i]

    X_tr=X_train[cols]

    X_ts=X_test[cols]

    

    model=RandomForestClassifier(class_weight='balanced', random_state=0)

    model.fit(X_tr,y_train)



    y_pred=model.predict(X_ts)

    y_proba=model.predict_proba(X_ts)

    y_proba=[p[1] for p in y_proba]

    

    print("Number of Columns : ",i)

    print(cols)

    print("Accuracy Score : {}".format(accuracy_score(y_test,y_pred)))

    print("Confusion Matrix : \n", confusion_matrix(y_test,y_pred))

    print("ROC AUC Score : {}".format(roc_auc_score(y_test,y_proba)))

    print(classification_report(y_test,y_pred))

    feat.append(i)

    acc.append(accuracy_score(y_test,y_pred))

    rocauc.append(roc_auc_score(y_test,y_proba))
sns.lineplot(feat,acc, label='Accuracy')    

sns.lineplot(feat,rocauc, label='ROC AUC Score')

#plt.ylim(0,1)

plt.legend()

plt.show()
# Columns used for final model

print("Features considered for final prediction")

x_cols=df_feat.Feature[:7]

x_cols
X=df[x_cols]

y=df.DEATH_EVENT



X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=1)



model=RandomForestClassifier(class_weight='balanced', random_state=0)

model.fit(X_train,y_train)



y_pred=model.predict(X_test)

y_proba=model.predict_proba(X_test)

y_proba=[p[1] for p in y_proba]



print("Accuracy Score : {}".format(accuracy_score(y_test,y_pred)))

print("Confusion Matrix : \n", confusion_matrix(y_test,y_pred))

print("ROC AUC Score : {}".format(roc_auc_score(y_test,y_proba)))

print(classification_report(y_test,y_pred))