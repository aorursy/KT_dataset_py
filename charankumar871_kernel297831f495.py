import numpy as np
import pandas as pd
credit_fraud=pd.read_excel("../input/credit-card/CREDIT_CARD_FRAUD (1).xlsx")
credit_fraud.head()
credit_fraud.shape
credit_fraud.columns
credit_fraud.shape
credit_fraud.isnull().sum()
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
credit_fraud.Customer_City.value_counts().sort_values(ascending=False).plot.bar()
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
credit_fraud.Customer_State.value_counts().sort_values(ascending=False).plot.bar()
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
credit_fraud.Tansaction_Mode.value_counts().sort_values(ascending=False).plot.bar()
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
credit_fraud.Transaction_type.value_counts().sort_values(ascending=False).plot.bar()
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
credit_fraud.Gender.value_counts().sort_values(ascending=False).plot.bar()
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
credit_fraud.Credit_Fraud.value_counts().sort_values(ascending=False).plot.bar()
credit_fraud.drop(['Customer_Id','Customer_Name','Time','Amount','Pancard'],inplace=True,axis=1)
credit_fraud["Customer_City"]=credit_fraud["Customer_City"].map({'AGRA': 1, 'AMRITSAR': 2, 'BANGLORE': 3, 'BHOPAL': 4, 
                                                                 'CHENNAI': 5, 'DEHERADUN': 6, 'GUNTUR': 7, 'HYDERABAD': 8, 
                                                                 'INDORE': 9, 'JABALPUR': 10, 'JAIPUR': 11, 'JODHPUR': 12, 
                                                                 'KAMMAM': 13, 'KANPUR': 14, 'KARIMNAGAR': 15, 'KOCHI': 16, 
                                                                 'KOLKATA': 17, 'LUCKNOW': 18, 'MADHURI': 19, 'MANGLORE': 20, 
                                                                 'MUMBAI': 21, 'MYSURU': 22, 'NAKPUR': 23, 'NASHIK': 24, 
                                                                 'NIZAMBAD': 25, 'NODIA': 26, 'PATNA': 27, 'PUNE': 28, 
                                                                 'RAIPUR': 29, 'RAJKOT': 30, 'RANCHI': 31, 'SHIMLA': 32, 
                                                                 'SURYAPET': 33, 'THANE': 34, 'THIRUVANANTHPURAM': 35, 
                                                                 'TIRIPATHI': 36, 'UDAIPUR': 37, 'VAIJAWADA': 38, 
                                                                 'VARANASI': 39, 'VIKERABAD': 40, 'VISHAKAPATANAM': 41, 
                                                                 'WARANGAL': 42})
credit_fraud["Customer_State"]=credit_fraud["Customer_State"].map({'ANDHRA': 1, 'BIHAR': 2, 'CHENNAI': 3, 'Chhattisgarh': 4, 
                                                                   'GUJARAT': 5, 'HIMACHAL PRADESH': 6, 'JHARKHAND': 7, 
                                                                   'KARNATAKA': 8, 'KERALA': 9, 'MADHYA PRADESH': 10,
                                                                   'MAHARASHTRA': 11, 'PUNJAB': 12, 'RAJASTHAN': 13, 
                                                                   'TAMIL NADU': 14, 'TELANGANA': 15, 'UTTAR PRADESH': 16, 
                                                                   'UTTARAKHAND': 17, 'WEST BENGAL': 18})
credit_fraud["Tansaction_Mode"]=credit_fraud["Tansaction_Mode"].map({'OFFLINE': 1, 'ONLINE': 2})
credit_fraud["Transaction_type"]=credit_fraud["Transaction_type"].map({'CASH_WITHDRAWL': 1, 'CHEQUE': 2, 'NET_BANKING': 3, 'POS': 4})
credit_fraud["Gender"]=credit_fraud["Gender"].map({"MALE": 0, "FEMALE":1})
credit_fraud["Self_Employment"]=credit_fraud["Self_Employment"].map({"NO": 0, "YES":1})
credit_fraud["Education"]=credit_fraud["Education"].map({"GRADUATE": 0, "NON_GRADUATE":1})
credit_fraud["Existing_Credit_Card"]=credit_fraud["Existing_Credit_Card"].map({'ANDRA_BANK': 1, 'AXIS': 2, 'HSBC': 3, 'ICICI': 4, 'KOTAK': 5, 'PUNJAB_NATIONAL_BANK': 6, 'SBH': 7, 'SBI': 8})
#from sklearn.metrics import plot_confusion_matrix
credit_fraud["Housing"]=credit_fraud["Housing"].map({"RENT": 0, "OWN":1})
credit_fraud["Property_Area"]=credit_fraud["Property_Area"].map({"URBAN": 0, "SEMI_URBAN":1,"RURAL":2})
credit_fraud["Present_residence"]=credit_fraud["Present_residence"].map({"SHIMLA": 0, "BANGLORE":1,"VIKARABAD":2,"NAKPUR":3,
                                                                         "SECUNDRABAD":4,"MUMBAI":5,"CHENNAI":6,"PUNE":7,
                                                                        "AGRA":8,"KOLKOTA":9,"RANCHI":10,"NIZAMBAD":11,
                                                                         "WARANGAL":12,"HYDERABAD":13,"MYSORE":14,
                                                                         "NALGONDA":15,"BUPAL":16,"INDORE":17})
credit_fraud["Married"]=credit_fraud["Married"].map({"NO": 0, "YES":1})
credit_fraud["Credit_Fraud"]=credit_fraud["Credit_Fraud"].map({"NO": 0, "YES":1})
credit_fraud1=credit_fraud.copy()
credit_fraud1.shape
y=credit_fraud["Credit_Fraud"].values
y
credit_fraud.drop(["Credit_Fraud"],axis=1,inplace=True)
x=credit_fraud.values
x
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x=scaler.fit_transform(x)
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(max_depth=5,n_estimators=100)
RF.fit(x_train,y_train)
RF.score(x_train,y_train)
RF.score(x_test,y_test)
RF.score(x,y)
y_predict=RF.predict(x_test)
y_predict#predicted values
y_test#actual values
from sklearn.metrics import confusion_matrix, classification_report

cm_df = pd.DataFrame(confusion_matrix(y_test, y_predict).T, index=RF.classes_,columns=RF.classes_)
cm_df.index.name = 'Predicted'
cm_df.columns.name = 'True'
print(cm_df)
print(classification_report(y_test, y_predict))
from pylab import rcParams
import warnings
import seaborn as sns
%matplotlib inline
np.random.seed(27)
rcParams['figure.figsize'] = 10, 6
warnings.filterwarnings('ignore')
sns.set(style="darkgrid")
unique, count = np.unique(y_train, return_counts=True)
y_train_dict_value_count = { k:v for (k,v) in zip(unique, count)}
y_train_dict_value_count
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
sm = SMOTE(random_state=12)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())
print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 

unique, count = np.unique(y_train_res, return_counts=True)
y_train_smote_value_count = { k:v for (k,v) in zip(unique, count)}
y_train_smote_value_count
clf = RandomForestClassifier().fit(x_train_res, y_train_res)
Y_Test_Pred = clf.predict(x_test)
pd.crosstab(Y_Test_Pred, y_test, rownames=['Predicted'], colnames=['Actual'])
def generate_model_report(y_actual, y_predicted):
    print("Accuracy = " , accuracy_score(y_actual, y_predicted))
    print("Precision = " ,precision_score(y_actual, y_predicted))
    print("Recall = " ,recall_score(y_actual, y_predicted))
    print("F1 Score = " ,f1_score(y_actual, y_predicted))
    pass
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve



generate_model_report(y_test, Y_Test_Pred)
def generate_auc_roc_curve(clf, x_test):
    y_pred_proba = clf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
generate_auc_roc_curve(clf, x_test)
minority_class_len = len(credit_fraud1[credit_fraud1['Credit_Fraud'] == 1])
print(minority_class_len)
majority_class_indices = credit_fraud1[credit_fraud1['Credit_Fraud'] == 0].index
print(majority_class_indices)
random_majority_indices = np.random.choice(majority_class_indices,
                                           minority_class_len, 
                                           replace=False)
print(len(random_majority_indices))
minority_class_indices = credit_fraud1[credit_fraud1['Credit_Fraud'] == 1].index
print(minority_class_indices)
under_sample_indices = np.concatenate([minority_class_indices,random_majority_indices])
 
under_sample = credit_fraud1.loc[under_sample_indices]
under_sample
under_sample.columns
sns.countplot(x=under_sample['Credit_Fraud'], data=under_sample)
under_sample.columns!='Credit_Fraud'
X = under_sample.loc[:, under_sample.columns!='Credit_Fraud']
Y = under_sample.loc[:, under_sample.columns=='Credit_Fraud']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
clf = RandomForestClassifier().fit(X_train, Y_train)
Y_Test_Pred = clf.predict(X_test)
generate_model_report(Y_test, Y_Test_Pred)
clf.predict(X_test)
def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test,  y_pred_proba)
    auc = roc_auc_score(Y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
generate_auc_roc_curve(clf, X_test)
