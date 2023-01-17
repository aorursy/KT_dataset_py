import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option("display.max_columns", 50)
dataset = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df = dataset.copy()
df.head()
gender_label = df.gender.value_counts().index

gender_size = df.gender.value_counts().values



partner_label = df['Partner'].value_counts().index

partner_size = df['Partner'].value_counts().values



plt.figure(figsize=(10,10))

plt.subplot(1,2,1)

plt.pie(gender_size, labels=gender_label, colors=['cyan', 'pink'], autopct='%1.1f%%', shadow=True, startangle=45, textprops={'fontsize':25})

plt.title('Gender Distribution of the Customers', color='navy', fontsize=15)



plt.subplot(1,2,2)

plt.pie(partner_size, labels=partner_label, colors=['cyan', 'pink'], autopct='%1.1f%%', shadow=True, startangle=45, textprops={'fontsize':25})

plt.title('Maritial Status of the Customers', color='navy', fontsize=15);
plt.figure(figsize=(8,5))

sns.countplot(df['Partner'], hue=df['Dependents'], palette='hls')

plt.title('Maritial and Dependent Status of the Customers', color='navy', fontsize=15)

plt.xlabel('Having Partner')

plt.ylabel('No of Customers with Dependents');
churn_label = df['Churn'].value_counts().index

churn_color = ['cyan', 'red']

churn_size = df['Churn'].value_counts().values



plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.pie(churn_size, labels=churn_label, colors=churn_color, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize':25})

plt.title('Percentage of Churn', color='navy', fontsize=15)



plt.subplot(1,2,2)

sns.countplot(df['Churn'], palette={'No':'cyan', 'Yes':'red'})

plt.title('Counts of Churn and No Churn', color='navy', fontsize=15)

plt.ylabel('Counts of Churn');
#'Total Charges' feature should be float but it has an object type. So I will convert it into float type:

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')



df[['MonthlyCharges', 'TotalCharges', 'tenure']].describe()
plt.figure(figsize=(7,17))

plt.subplot(3,1,1)

sns.distplot(df['MonthlyCharges'])

plt.title('Distribution of Monthly Charges', color='navy', fontsize=15)



plt.subplot(3,1,2)

sns.distplot(df['TotalCharges'])

plt.title('Distribution of Total Charges', color='navy', fontsize=15)



plt.subplot(3,1,3)

sns.distplot(df['tenure'])

plt.title('Tenure (Months) of the Customers', color='navy', fontsize=15);
plt.figure(figsize=(15,15))

plt.subplot(3,2,1)

sns.countplot(df['Contract'], hue=df['Churn'], palette=['cyan', 'red'])



plt.subplot(3,2,2)

sns.countplot(df['PhoneService'], hue=df['Churn'], palette=['cyan', 'red'])



plt.subplot(3,2,3)

sns.countplot(df['InternetService'], hue=df['Churn'], palette=['cyan', 'red'])



plt.subplot(3,2,4)

sns.countplot(df['StreamingTV'], hue=df['Churn'], palette=['cyan', 'red'])



plt.subplot(3,2,5)

sns.countplot(df['StreamingMovies'], hue=df['Churn'], palette=['cyan', 'red'])



plt.subplot(3,2,6)

sns.countplot(df['PaymentMethod'], hue=df['Churn'], palette=['cyan', 'red'])

plt.xticks(rotation=45);
# Label encoding for 'Yes' and 'No' features:

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()



binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

df[binary_cols] = df[binary_cols].astype('category')



for each in binary_cols:

    df[each] = encoder.fit_transform(df[each])
# Features with multiple values:

print('MultipleLines Values   : ', df['MultipleLines'].unique())

print('Contract Values        : ', df['Contract'].unique())

print('InternetService Values : ', df['InternetService'].unique())

print('PaymentMethod Values   : ', df['PaymentMethod'].unique())

print('OnlineSecurity Values  : ', df['OnlineSecurity'].unique())

print('OnlineBackup Values    : ', df['OnlineBackup'].unique())

print('DeviceProtection Values: ', df['DeviceProtection'].unique())

print('TechSupport Values     : ', df['TechSupport'].unique())

print('StreamingTV Values     : ', df['TechSupport'].unique())

print('StreamingMovies Values : ', df['TechSupport'].unique())
cols_for_dummies = ['MultipleLines', 'Contract', 'InternetService', 

                    'PaymentMethod', 'OnlineSecurity', 'OnlineBackup', 

                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

df_ = df.drop(cols_for_dummies, axis=1)

dms = pd.get_dummies(df[cols_for_dummies])

df = pd.concat([df_, dms], axis=1)
df.columns = df.columns.str.lower()
df.isnull().sum()
df.dropna(inplace=True)

df = df.drop('customerid', axis=1)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

df[['tenure', 'monthlycharges', 'totalcharges']] = scaler.fit_transform(df[['tenure', 'monthlycharges', 'totalcharges']])
df.head(10)
corrmatrix = pd.DataFrame(df.corr()['churn'].sort_values(ascending=True)).rename(columns={'churn':'correlation'})



stronger_corr = corrmatrix[(corrmatrix.correlation > 0.20) | (corrmatrix.correlation < -0.20)].drop('churn')



plt.figure(figsize=(10,5))

sns.barplot(x=stronger_corr.correlation, y=stronger_corr.index)

plt.title('Features and Churn Correlation', color='navy', fontsize=15);
y = df['churn']

X = df.drop('churn', axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)



print('X_train shape : ', X_train.shape)

print('y_train shape : ', y_train.shape)

print('X_test shape  : ', X_test.shape)

print('y_test shape  : ', y_test.shape)
from sklearn.linear_model import LogisticRegression



loj = LogisticRegression(solver='liblinear')



loj_model = loj.fit(X_train, y_train)



y_pred = loj_model.predict(X_test)

y_pred_prob = loj_model.predict_proba(X_test)
y_pred
y_pred_prob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve



print('Accuracy of the default model  : %.2f' % accuracy_score(y_test, y_pred))

print('Precision of the default model : %.2f' % precision_score(y_test, y_pred))

print('Recall of the default model    : %.2f' % recall_score(y_test, y_pred))

print('F1 Score of the default model  : %.2f' % f1_score(y_test, y_pred))



sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu')

plt.title('Confusion Matrix of the Default Model', color='navy', fontsize=15)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values');
logit_roc_auc = roc_auc_score(y_test, y_pred)



fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])



plt.figure(figsize=(7,7))

plt.plot(fpr, tpr, label='AUC (Area = %0.2f)' %logit_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Sensitivity : False Positive Ratio')

plt.ylabel('1 - Specifity : True Positive Ratio')

plt.title('ROC and AUC of the Default Model')

plt.legend()

plt.show()
# New predictions with 0.4 threshold:

y_pred_4 = loj_model.predict_proba(X_test)[:,1] >= 0.4



# New predictions with 0.3 threshold:

y_pred_3 = loj_model.predict_proba(X_test)[:,1] >= 0.3
print('Accuracy of the default model             : %.2f' % accuracy_score(y_test, y_pred))

print('Precision of the default model            : %.2f' % precision_score(y_test, y_pred))

print('Recall of the default model               : %.2f' % recall_score(y_test, y_pred))

print('F1 Score of the default model             : %.2f' % f1_score(y_test, y_pred))

print('--'*24)

print('Accuracy of the model with 0.4 threshold  : %.2f' % accuracy_score(y_test, y_pred_4))

print('Precision of the model with 0.4 threshold : %.2f' % precision_score(y_test, y_pred_4))

print('Recall of the model with 0.4 threshold    : %.2f' % recall_score(y_test, y_pred_4))

print('F1 Score of the model with 0.4 threshold  : %.2f' % f1_score(y_test, y_pred_4))

print('--'*24)

print('Accuracy of the model with 0.3 threshold  : %.2f' % accuracy_score(y_test, y_pred_3))

print('Precision of the model with 0.3 threshold : %.2f' % precision_score(y_test, y_pred_3))

print('Recall of the model with 0.3 threshold    : %.2f' % recall_score(y_test, y_pred_3))

print('F1 Score of the model with 0.3 threshold  : %.2f' % f1_score(y_test, y_pred_3))
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu')

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Default Model', color='navy', fontsize=15)



plt.subplot(1,3,2)

sns.heatmap(confusion_matrix(y_test, y_pred_4), annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu')

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Model with 0.4 Threshold', color='navy', fontsize=15)



plt.subplot(1,3,3)

sns.heatmap(confusion_matrix(y_test, y_pred_3), annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu')

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Model with 0.3 Threshold', color='navy', fontsize=15);