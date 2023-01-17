import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Read the data

data = pd.read_csv(r"../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.shape
data['TotalCharges'] = data["TotalCharges"].replace(" ",np.nan)
data.dropna(inplace = True)
data.isnull().sum()
data.nunique()
data.head()
# Variables and their unique values for analysis

for item in data.columns:

    print(item," : ", data[item].unique())
data['MultipleLines'].replace(to_replace = 'No phone service',value = 'No',inplace = True)



replace_columns = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingMovies','StreamingTV']

for i in replace_columns:

    data[i].replace(to_replace = 'No internet service',value = 'No',inplace = True)
data.nunique()
def plot_feature(feature):

    ax = sns.countplot(x = feature,data = data)

    total = len(data)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total * 100),

            ha="center")     

    plt.title("Customer Distribution by {}".format(feature)) 

    plt.show()
plot_feature('Churn')

plot_feature('gender')

plot_feature('Partner')

plot_feature('SeniorCitizen')

plot_feature('PhoneService')

plot_feature('MultipleLines')

plot_feature('InternetService')

plot_feature('OnlineSecurity')

plot_feature('Contract')
def plot_bar(d,var1,var2):

    grp = d.groupby(var1)[var2].value_counts()

    grp.unstack().plot(kind = 'bar')

    plt.xlabel(var1)

    plt.ylabel("Count of Churn Customers")

    plt.title("Churn Customer Distribution by {}".format(var1)) 
plot_bar(data,'gender','Churn');

plot_bar(data,'SeniorCitizen','Churn');

plot_bar(data,'Partner','Churn');

plot_bar(data,'Dependents','Churn');

plot_bar(data,'PhoneService','Churn');

plot_bar(data,'MultipleLines','Churn');

plot_bar(data,'InternetService','Churn');

plot_bar(data,'OnlineSecurity','Churn');

plot_bar(data,'OnlineBackup','Churn');

plot_bar(data,'DeviceProtection','Churn');

plot_bar(data,'TechSupport','Churn');

plot_bar(data,'StreamingTV','Churn');

plot_bar(data,'StreamingMovies','Churn');

plot_bar(data,'Contract','Churn');

plot_bar(data,'PaperlessBilling','Churn');

plot_bar(data,'PaymentMethod','Churn');
ax = sns.distplot(data['tenure'], hist=True, kde=False, 

             bins=int(200/6), 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax.set_ylabel('Number of Customers')

ax.set_xlabel('Tenure in months')

ax.set_title('Distribution of Customers by tenure')
def tenure_count(row):

    if row['tenure'] <= 12 :

        return 'tenure_0-12'

    elif (row['tenure'] > 12 and row['tenure'] <= 24):

        return 'tenure_12-24'

    elif (row['tenure'] > 42 and row['tenure'] <= 36):

        return 'tenure_24-36'

    elif (row['tenure'] > 36 and row['tenure'] <= 48):

        return 'tenure_36-48'

    elif (row['tenure'] > 48 and row['tenure'] <= 60):

        return 'tenure_48-60'

    else:

        return 'tenure_60+'

    

data['grp_tenure'] = data.apply(tenure_count,axis = 1)

data['grp_tenure'].value_counts()
plot_bar(data,'grp_tenure','Churn')
data['TotalCharges'] = data['TotalCharges'].astype(float)
plt.scatter(x = data['MonthlyCharges'],y = data['TotalCharges'],c = 'green')

plt.xlabel("Monthly Charges")

plt.ylabel("Total Charges Charges")

plt.title("Ralation between Monthly and Total Charges")
ax = sns.kdeplot(data[data['Churn'] == 'No']['MonthlyCharges'])

ax = sns.kdeplot(data[data['Churn'] == 'Yes']['MonthlyCharges'],color = 'Red')

ax.set_xlabel('Monthly Charges')

ax.set_ylabel('Density')

ax.set_title('Distribution of Monthly charges by churn')
ax = sns.kdeplot(data[data['Churn'] == 'No']['TotalCharges'])

ax = sns.kdeplot(data[data['Churn'] == 'Yes']['TotalCharges'],color = 'Red')

ax.set_xlabel('Total Charges')

ax.set_ylabel('Density')

ax.set_title('Distribution of Total charges by churn')
data.drop(['customerID'],axis = 1,inplace = True)
objList = data.select_dtypes(include = "object").columns

print (objList)
#Label Encoding for object to numeric conversion

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for feat in objList:

    data[feat] = le.fit_transform(data[feat].astype(str))



print (data.info())
y = data['Churn']

data.drop(['Churn'],axis = 1, inplace = True)

Train_x = data
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X=sc.fit_transform(Train_x)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from lightgbm import *

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
LR = LogisticRegression(solver = 'liblinear')

LR.fit(X_train,y_train)

y_pred_LR = LR.predict(X_test)

LR_Score = accuracy_score(y_pred_LR,y_test)

print("Accuracy Using LR : ", LR_Score)
#Weights of the Variables

pd.Series(LR.coef_[0],index = Train_x.columns.values)
RF = RandomForestClassifier(n_estimators=600,max_features=15,

                            n_jobs = -1,random_state=0,

                            min_samples_leaf=50,oob_score=True,

                            max_leaf_nodes=30 )

RF.fit(X_train,y_train)

y_pred_RF = RF.predict(X_test)

RF_Score = accuracy_score(y_pred_RF,y_test)

print("Accuracy Using RF  : ", RF_Score)
imp_features = pd.Series(RF.feature_importances_,index = Train_x.columns.values)

imp_features.sort_values()[-5:].plot(kind = 'bar')
SVM = SVC(kernel='rbf',C =1) 

SVM.fit(X_train,y_train)

y_pred_SVM = SVM.predict(X_test)

SVM_Score = accuracy_score(y_pred_SVM,y_test)

print("Accuracy Using SVM  : ", SVM_Score)
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred_GB = gaussian.predict(X_test)

GB_Score = accuracy_score(y_pred_GB,y_test)

print("Accuracy Using Gaussian Algorithm : ", GB_Score)
XGB = XGBClassifier(n_estimators=100,learning_rate = 0.1,max_depth = 4)

XGB.fit(X_train, y_train)

y_pred_XGB = XGB.predict(X_test)

XGB_Score = accuracy_score(y_test, y_pred_XGB)

print("Accuracy Using XGBoost : ", XGB_Score)
lgbm = LGBMClassifier(n_estimators=100,learning_rate = 0.1,max_depth = 5)

lgbm.fit(X_train, y_train)

y_pred_LGBM = lgbm.predict(X_test)

LGBM_Score = accuracy_score(y_test,y_pred_LGBM )

print("Accuracy Using LIGTH GBM Classifier : ", LGBM_Score)
labels = ['Churn', 'Not-Churn']

cm = confusion_matrix(y_test, y_pred_LGBM)

print(cm)
ax= plt.subplot()

sns.heatmap(cm,annot=True, ax = ax); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['Not-Churn', 'Churn']); ax.yaxis.set_ticklabels(['Not-Churn', 'Churn']);
Results = pd.DataFrame({'Model': ['Logistic Regression','Gaussian Naive Bayes','SVM','Random Forest','XG_Boost','LightGBM'],

                        'Accuracy Score' : [LR_Score,GB_Score,SVM_Score,RF_Score,XGB_Score,LGBM_Score]})
Final_Results = Results.sort_values(by = 'Accuracy Score', ascending=False)

Final_Results = Final_Results.set_index('Model')

print(Final_Results)