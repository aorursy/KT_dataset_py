# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas_profiling as pr

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
telco = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

telco.head()
telco.info()
#Checking missing values

telco.isnull().any().sum()
telco.describe()
pr.ProfileReport(telco)
dict(telco.dtypes)
# Converting Total Charges to a numerical data type.

telco.TotalCharges = pd.to_numeric(telco.TotalCharges, errors='coerce')

telco.isnull().sum()

telco.fillna(0,inplace=True)
telco.head()
telco_churn = pd.DataFrame(telco.Churn.value_counts())

telco_churn

sns.barplot(telco_churn.index, telco_churn.Churn)
#by gender



gender_count  = telco[['gender','Churn']].groupby(['gender','Churn']).size().reset_index()

gender_count.columns = ['gender','churn','count']

sns.catplot(x = 'gender',y='count',hue='churn',data = gender_count,kind='bar')
senior_citizen_count  = telco[['SeniorCitizen','Churn']].groupby(['SeniorCitizen','Churn']).size().reset_index()

senior_citizen_count.columns = ['SeniorCitizen','churn','count']

sns.catplot(x = 'SeniorCitizen',y='count',hue='churn',data = senior_citizen_count,kind='bar')
telco[['MonthlyCharges', 'TotalCharges']].plot.scatter(x = 'MonthlyCharges',

                                                              y='TotalCharges')
#Total charges and Churn

sns.distplot(telco.MonthlyCharges[(telco["Churn"] == 'No') ] ,color='r')

sns.distplot(telco.MonthlyCharges[(telco['Churn'] == 'Yes')],color='g')

#MonthlyCharges and Churn



sns.kdeplot(telco.TotalCharges[(telco["Churn"] == 'No') ] ,color='r',shade=True)

sns.kdeplot(telco.TotalCharges[(telco['Churn'] == 'Yes')],color='g',shade=True)

sns.boxplot(x='tenure',y='Churn',data=telco)
#payment method and churn



payment_method_count  = telco[['PaymentMethod','Churn']].groupby(['PaymentMethod','Churn']).size().reset_index()

payment_method_count.columns = ['paymentMethod','churn','count']

sns.catplot(x = 'count',y='paymentMethod',hue='churn',data = payment_method_count,kind='bar')
#divide numeric and categorical variables

numeric_var_names = [key for key in dict(telco.dtypes) if dict(telco.dtypes)[key] in ['int32','int64','float32','float64']]

cat_var_names = [key for key in dict(telco.dtypes) if dict(telco.dtypes)[key] in ['object','O']]

                     
telco_num = telco[numeric_var_names]

telco_cat = telco[cat_var_names]
#create data audit report



def var_summary(x):

    return pd.Series([x.count(),x.isnull().sum(),x.sum(),x.var(),x.std(),x.mean(),x.median(),x.min(),x.dropna().quantile(0.01),x.dropna().quantile(0.05),

              x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75),x.dropna().quantile(0.90),

              x.dropna().quantile(0.95),x.dropna().quantile(0.99)],index=['N','NMISS','SUM','VAR','STD','MEAN','MEDIAN','MIN','P1','P5','P10','P25','P50','P75','P90','P95','P99'])

num_summary = telco_num.apply(lambda x : var_summary(x)).T
num_summary

#As we can see that there are no outliers
sns.boxplot(telco_num.TotalCharges)
#Missing value treatment of TotalCharges



telco_num.TotalCharges.fillna(0,inplace=True)
def cat_summary(x):

    return pd.Series([x.count(),x.isnull().sum(),x.value_counts()],index=['N','NMISS','COUNT'])



cat_summary = telco_cat.apply(lambda x : cat_summary(x)).T

cat_summary
telco_cat.head()
#lets convert categorical variables to numeric by Label Encoding



telco_cat['gender'] = LabelEncoder().fit_transform(telco_cat['gender'])

telco_cat['Partner'] = LabelEncoder().fit_transform(telco_cat['Partner'])

telco_cat['Dependents'] = LabelEncoder().fit_transform(telco_cat['Dependents'])

telco_cat['PhoneService'] = LabelEncoder().fit_transform(telco_cat['PhoneService'])

telco_cat['MultipleLines'] = LabelEncoder().fit_transform(telco_cat['MultipleLines'])

telco_cat['InternetService'] = LabelEncoder().fit_transform(telco_cat['InternetService'])

telco_cat['OnlineSecurity'] = LabelEncoder().fit_transform(telco_cat['OnlineSecurity'])

telco_cat['OnlineBackup'] = LabelEncoder().fit_transform(telco_cat['OnlineBackup'])

telco_cat['DeviceProtection'] = LabelEncoder().fit_transform(telco_cat['DeviceProtection'])

telco_cat['TechSupport'] = LabelEncoder().fit_transform(telco_cat['TechSupport'])

telco_cat['StreamingTV'] = LabelEncoder().fit_transform(telco_cat['StreamingTV'])

telco_cat['StreamingMovies'] = LabelEncoder().fit_transform(telco_cat['StreamingMovies'])

telco_cat['Contract'] = LabelEncoder().fit_transform(telco_cat['Contract'])

telco_cat['PaperlessBilling'] = LabelEncoder().fit_transform(telco_cat['PaperlessBilling'])

telco_cat['PaymentMethod'] = LabelEncoder().fit_transform(telco_cat['PaymentMethod'])
telco_cat.drop(['customerID'],axis=1,inplace=True)
telco_cat['Churn'] = telco_cat['Churn'].map({'No':0,'Yes':1})
telco_df = pd.DataFrame(pd.concat([telco_num,telco_cat],axis=1))

telco_df.head()
#Correlation

plt.figure(figsize=(20,10))

sns.heatmap(telco_df.corr(),annot=True)
#Predictive Modelling



feature_columns = telco_df.columns.difference(['Churn'])

feature_columns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
train_x, test_x, train_y, test_y = train_test_split(telco_df[feature_columns], telco_df['Churn'], test_size=0.2,random_state=42)

print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
#Building Model

logreg = LogisticRegression()

logreg.fit(train_x,train_y)
logreg.coef_
list(zip(feature_columns, logreg.coef_[0]))
logreg.predict_proba(test_x)
pred_y = pd.DataFrame({'actual': test_y,'predicted':logreg.predict(test_x)})

pred_y.reset_index()
#confusion Matrix

cm = metrics.confusion_matrix(pred_y.actual,pred_y.predicted,[1,0])

cm
sns.heatmap(cm, annot=True, xticklabels=['churn','not_churn'], yticklabels = ['churn','not_churn'], fmt='.1f')

plt.xlabel('Predicted')

plt.ylabel('Actual')
#Accuracy Score

print("Accuracy " +str(metrics.accuracy_score(pred_y.actual,pred_y.predicted)))

print("Precision " +str(metrics.precision_score(pred_y.actual,pred_y.predicted)))

print("Recall " +str(metrics.recall_score(pred_y.actual,pred_y.predicted)))

print("ROC AUC " +str(metrics.roc_auc_score(pred_y.actual,pred_y.predicted)))

print("f1 score " +str(metrics.f1_score(pred_y.actual,pred_y.predicted)))

#the Recall score is not that much great. As we see that the data is quite imbalanced lets try to balance the data and then check the accuracy and recall score.
#Building model by rebalancing the data

logreg1 = LogisticRegression(class_weight='balanced')

logreg1.fit(train_x,train_y)
list(zip(feature_columns, logreg1.coef_[0]))
pred_y = pd.DataFrame({'actual': test_y,'predicted':logreg1.predict(test_x)})

pred_y = pred_y.reset_index()

pred_y.head()
#confusion Matrix

cm = metrics.confusion_matrix(pred_y.actual,pred_y.predicted,[1,0])

cm
sns.heatmap(cm, annot=True, xticklabels=['churn','not_churn'], yticklabels = ['churn','not_churn'], fmt='.1f')

plt.xlabel('Predicted')

plt.ylabel('Actual')
#Different parameter checking

print("Accuracy " +str(metrics.accuracy_score(pred_y.actual,pred_y.predicted)))

print("Precision " +str(metrics.precision_score(pred_y.actual,pred_y.predicted)))

print("Recall " +str(metrics.recall_score(pred_y.actual,pred_y.predicted)))

print("ROC AUC " +str(metrics.roc_auc_score(pred_y.actual,pred_y.predicted)))

print("f1 score " +str(metrics.f1_score(pred_y.actual,pred_y.predicted)))



#As we see that Recall score increases rapidly as we balanced the data.
#How good the model is



pred_prob_y = pd.DataFrame(logreg1.predict_proba(test_x))

pred_prob_y.columns = ['Not Churn','Churn']

pred_prob_y.head()
predicted = pd.concat([pred_y,pred_prob_y],axis=1)

predicted
#let's Plot graph



sns.distplot(predicted[predicted.actual==0]['Churn'],color='g')

sns.distplot(predicted[predicted.actual==1]['Churn'],color='r')
auc_score = metrics.roc_auc_score(pred_y.actual,pred_y.predicted)

auc_score
#Finding the appropriate cutoff probability

fpr, tpr, thresholds = metrics.roc_curve( predicted.actual,

                                     predicted.Churn,

                                     drop_intermediate = False )



plt.figure(figsize=(6, 4))

plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
print(thresholds[0:10])

print(fpr[0:5])

print(tpr[0:10])
cutoff_prob = thresholds[(np.abs(tpr-0.75)).argmin()]

cutoff_prob
predicted['new_label'] = predicted.Churn.apply(lambda x : 0 if x<0.57 else 1 )
predicted
#confusion Matrix

cm = metrics.confusion_matrix(predicted.actual,predicted.new_label,[1,0])

cm

sns.heatmap(cm, annot=True, xticklabels=['churn','not_churn'], yticklabels = ['churn','not_churn'], fmt='.1f')

plt.xlabel('Predicted')

plt.ylabel('Actual')