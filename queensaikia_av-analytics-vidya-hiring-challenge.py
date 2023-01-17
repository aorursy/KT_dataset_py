import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
train_df=pd.read_csv("../input/av-datascience/Train_gQ1XM0h.csv")
test_df=pd.read_csv("../input/av-datascience/Test_wyCirpO_aS3XPkc.csv")
train_df.shape
test_df.shape
train_df.head()
test_df.head()
train_df=train_df.drop('ID',axis=1)
test_id=test_df['ID']
test_df=test_df.drop('ID',axis=1)
import matplotlib.pyplot as plt

train_df.hist(bins = 50 , figsize = (20,15))
plt.show()
corr_matrix = train_df.corr()
corr_matrix
import seaborn as sns
sns.heatmap(corr_matrix, annot = True )
corr_matrix['Business_Sourced']
corr_matrix['Business_Sourced'].sort_values(ascending = False)
def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms
missingdata(train_df)
missingdata(test_df)
train_df

train_df['Applicant_City_PIN']=train_df['Applicant_City_PIN'].fillna(train_df['Applicant_City_PIN'].mode()[0])
train_df['Applicant_Gender']=train_df['Applicant_Gender'].fillna(train_df['Applicant_Gender'].mode()[0])
train_df['Applicant_BirthDate']=train_df['Applicant_BirthDate'].fillna(train_df['Applicant_BirthDate'].mode()[0])
train_df['Applicant_Marital_Status']=train_df['Applicant_Marital_Status'].fillna(train_df['Applicant_Marital_Status'].mode()[0])
train_df['Applicant_Occupation']=train_df['Applicant_Occupation'].fillna(train_df['Applicant_Occupation'].mode()[0])
train_df['Applicant_Qualification']=train_df['Applicant_Qualification'].fillna(train_df['Applicant_Qualification'].mode()[0])
train_df['Manager_DOJ']=train_df['Manager_DOJ'].fillna(train_df['Manager_DOJ'].mode()[0])
train_df['Manager_Joining_Designation']=train_df['Manager_Joining_Designation'].fillna(train_df['Manager_Joining_Designation'].mode()[0])
train_df['Manager_Current_Designation']=train_df['Manager_Current_Designation'].fillna(train_df['Manager_Current_Designation'].mode()[0])
train_df['Manager_Grade']=train_df['Manager_Grade'].fillna(train_df['Manager_Grade'].mode()[0])
train_df['Manager_Status']=train_df['Manager_Status'].fillna(train_df['Manager_Status'].mode()[0])
train_df['Manager_Gender']=train_df['Manager_Gender'].fillna(train_df['Manager_Gender'].mode()[0])
train_df['Manager_DoB']=train_df['Manager_DoB'].fillna(train_df['Manager_DoB'].mode()[0])
train_df['Manager_Num_Application']=train_df['Manager_Num_Application'].fillna(train_df['Manager_Num_Application'].mean())
train_df['Manager_Num_Coded']=train_df['Manager_Num_Coded'].fillna(train_df['Manager_Num_Coded'].mean())
train_df['Manager_Business']=train_df['Manager_Business'].fillna(train_df['Manager_Business'].mean())
train_df['Manager_Num_Products']=train_df['Manager_Num_Products'].fillna(train_df['Manager_Num_Products'].mean())
train_df['Manager_Business2']=train_df['Manager_Business2'].fillna(train_df['Manager_Business2'].mean())
train_df['Manager_Num_Products2']=train_df['Manager_Num_Products2'].fillna(train_df['Manager_Num_Products2'].mean())

test_df['Applicant_City_PIN']=test_df['Applicant_City_PIN'].fillna(test_df['Applicant_City_PIN'].mode()[0])
test_df['Applicant_Gender']=test_df['Applicant_Gender'].fillna(test_df['Applicant_Gender'].mode()[0])
test_df['Applicant_BirthDate']=test_df['Applicant_BirthDate'].fillna(test_df['Applicant_BirthDate'].mode()[0])
test_df['Applicant_Marital_Status']=test_df['Applicant_Marital_Status'].fillna(test_df['Applicant_Marital_Status'].mode()[0])
test_df['Applicant_Occupation']=test_df['Applicant_Occupation'].fillna(test_df['Applicant_Occupation'].mode()[0])
test_df['Applicant_Qualification']=test_df['Applicant_Qualification'].fillna(test_df['Applicant_Qualification'].mode()[0])
test_df['Manager_DOJ']=test_df['Manager_DOJ'].fillna(test_df['Manager_DOJ'].mode()[0])
test_df['Manager_Joining_Designation']=test_df['Manager_Joining_Designation'].fillna(test_df['Manager_Joining_Designation'].mode()[0])
test_df['Manager_Current_Designation']=test_df['Manager_Current_Designation'].fillna(test_df['Manager_Current_Designation'].mode()[0])
test_df['Manager_Grade']=test_df['Manager_Grade'].fillna(test_df['Manager_Grade'].mode()[0])
test_df['Manager_Status']=test_df['Manager_Status'].fillna(test_df['Manager_Status'].mode()[0])
test_df['Manager_Gender']=test_df['Manager_Gender'].fillna(test_df['Manager_Gender'].mode()[0])
test_df['Manager_DoB']=test_df['Manager_DoB'].fillna(test_df['Manager_DoB'].mode()[0])
test_df['Manager_Num_Application']=test_df['Manager_Num_Application'].fillna(test_df['Manager_Num_Application'].mean())
test_df['Manager_Num_Coded']=test_df['Manager_Num_Coded'].fillna(test_df['Manager_Num_Coded'].mean())
test_df['Manager_Business']=test_df['Manager_Business'].fillna(test_df['Manager_Business'].mean())
test_df['Manager_Num_Products']=test_df['Manager_Num_Products'].fillna(test_df['Manager_Num_Products'].mean())
test_df['Manager_Business2']=test_df['Manager_Business2'].fillna(test_df['Manager_Business2'].mean())
test_df['Manager_Num_Products2']=test_df['Manager_Num_Products2'].fillna(test_df['Manager_Num_Products2'].mean())

test_df.isnull().sum()
test_df
train_df

# train_df=train_df.drop('ID',axis=1)
train_df=train_df.drop('Application_Receipt_Date',axis=1)
train_df=train_df.drop('Applicant_BirthDate',axis=1)
train_df=train_df.drop('Manager_DOJ',axis=1)
train_df=train_df.drop('Manager_DoB',axis=1)
train_df=train_df.drop('Manager_Num_Products2',axis=1)
train_df=train_df.drop('Manager_Business2',axis=1)
train_df=train_df.drop('Manager_Current_Designation',axis=1)
train_df=train_df.drop('Manager_Joining_Designation',axis=1)

train_df
# test_df=test_df.drop('ID',axis=1)
test_df=test_df.drop('Application_Receipt_Date',axis=1)
test_df=test_df.drop('Applicant_BirthDate',axis=1)
test_df=test_df.drop('Manager_DOJ',axis=1)
test_df=test_df.drop('Manager_DoB',axis=1)
test_df=test_df.drop('Manager_Num_Products2',axis=1)
test_df=test_df.drop('Manager_Business2',axis=1)
test_df=test_df.drop('Manager_Current_Designation',axis=1)
test_df=test_df.drop('Manager_Joining_Designation',axis=1)
test_df
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


train_df['Applicant_Gender'] = le.fit_transform(train_df['Applicant_Gender'])
train_df['Applicant_Marital_Status'] = le.fit_transform(train_df['Applicant_Marital_Status'])
train_df['Applicant_Occupation'] = le.fit_transform(train_df['Applicant_Occupation'])
train_df['Applicant_Qualification'] = le.fit_transform(train_df['Applicant_Qualification'])
train_df['Manager_Status'] = le.fit_transform(train_df['Manager_Status'])
train_df['Manager_Gender'] = le.fit_transform(train_df['Manager_Gender'])


test_df['Applicant_Gender'] = le.fit_transform(test_df['Applicant_Gender'])
test_df['Applicant_Marital_Status'] = le.fit_transform(test_df['Applicant_Marital_Status'])
test_df['Applicant_Occupation'] = le.fit_transform(test_df['Applicant_Occupation'])
test_df['Applicant_Qualification'] = le.fit_transform(test_df['Applicant_Qualification'])
test_df['Manager_Status'] = le.fit_transform(test_df['Manager_Status'])
test_df['Manager_Gender'] = le.fit_transform(test_df['Manager_Gender'])

train_df
test_df
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(train_df.drop('Business_Sourced',axis=1),train_df['Business_Sourced'] , test_size=0.2, random_state=0,shuffle=True)
y_test.head()
y_train
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
cat_model=CatBoostClassifier(
  learning_rate=0.01,depth=7,iterations=2500, loss_function='CrossEntropy',eval_metric='AUC')
cat_model.fit(X_train,y_train)
preds=cat_model.predict(X_test)


print('ROC: ', roc_auc_score(y_test, preds)*100, '%')
def plot_roc_curve( roc_auc_test):
    plt.title('Receiver Operating Characteristic')
#     plt.plot(fpr_tr, tpr_tr, 'g', label = 'Training AUC = %0.2f' % roc_auc_train)
    plt.plot(fpr_ts, tpr_ts, 'b', label = 'Testing AUC = %0.2f' % roc_auc_test)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,preds))

submission = pd.DataFrame()
submission['ID'] = test_id
submission['Business_Sourced']=cat_model.predict(test_df)
submission.head()
submission.to_csv("my_submission.csv",index=False)