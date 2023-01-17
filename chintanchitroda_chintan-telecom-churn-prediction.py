# By :-Chintan Chitroda
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from sklearn import metrics
trainds = pd.read_csv("/kaggle/input/predict-the-churn-for-customer-dataset/Train File.csv")

testds = pd.read_csv("/kaggle/input/predict-the-churn-for-customer-dataset/Test File.csv")



trainds.head(3)
testds.head(3)
print('Train Dataset Infomarion')

print ("Rows     : " ,trainds.shape[0])

print ("Columns  : " ,trainds.shape[1])

print ("\nFeatures : \n" ,trainds.columns.tolist())

print ("\nMissing values :  ", trainds.isnull().sum().values.sum())

print ("\nUnique values :  \n",trainds.nunique())
plt.subplots(figsize=(10, 6))

plt.title('Cooralation Matrix', size=30)

sns.heatmap(trainds.corr(),annot=True,linewidths=0.5)
trainds.loc[trainds['TotalCharges'].isnull()] #NUll values Present
trainds['TotalCharges'] = trainds['TotalCharges'].fillna(trainds['TotalCharges'].median()) #

#trainds = trainds[trainds["TotalCharges"].notnull()]
CustomerIDS = testds['customerID']

trainds.drop('customerID', axis=1,inplace =True)

testds.drop('customerID', axis=1,inplace =True)
trainds.columns
testds.describe()
testds['TotalCharges'] = testds['TotalCharges'].fillna(testds['TotalCharges'].median())
trainds["InternetService"]=trainds["InternetService"].astype('str')

testds["InternetService"]=testds["InternetService"].astype('str')
trainds["TotalCharges"] = trainds["TotalCharges"].astype(float)

trainds["MonthlyCharges"] = trainds["MonthlyCharges"].astype(float)



testds["TotalCharges"] = testds["TotalCharges"].astype(float)

testds["MonthlyCharges"] = testds["MonthlyCharges"].astype(float)
replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']

for i in replace_cols : 

    trainds[i]  = trainds[i].replace({'No internet service' : 'No'})

    testds[i]  = testds[i].replace({'No internet service' : 'No'})
replace_cols = ['MultipleLines']

for i in replace_cols : 

    trainds[i]  = trainds[i].replace({'No phone service' : 'No'})

    testds[i]  = testds[i].replace({'No phone service' : 'No'})
def customercountplot(x):

    z = "Customer Count wrt "+ x

    plt.title(z,size=20)

    sns.countplot(trainds[x])
def churnratio():

    import plotly.offline as py

    import plotly.graph_objs as go

    val = trainds["Churn"].value_counts().values.tolist()



    trace = go.Pie(labels = ["Not Churned","Churned"] ,

                   values = val ,

                   marker = dict(colors =  [ 'royalblue' ,'lime']), hole = .5)

    layout = go.Layout(dict(title = "Train Dataset Customers"))

    data = [trace]

    fig = go.Figure(data = data,layout = layout)

    py.iplot(fig)
def churnrate():

    features = ['PhoneService','MultipleLines','InternetService',

                'TechSupport','StreamingTV','StreamingMovies','Contract']

    for i, item in enumerate(features):

        if i < 3:

            fig1 = pd.crosstab(trainds[item],trainds.Churn,margins=True)

            fig1.drop('All',inplace=True)

            fig1.drop('All',axis=1, inplace=True)

            fig1.plot.bar()

            z= 'Customer Churned wrt ' + item

            plt.title(z,size=20)

        elif i >=3 and i < 6:

            fig1 = pd.crosstab(trainds[item],trainds.Churn,margins=True)

            fig1.drop('All',inplace=True)

            fig1.drop('All',axis=1, inplace=True)

            fig1.plot.bar()

            z= 'Customer Churned wrt ' + item

            plt.title(z,size=20)

        elif i < 9:

            fig1 = pd.crosstab(trainds[item],trainds.Churn,margins=True)

            fig1.drop('All',inplace=True)

            fig1.drop('All',axis=1, inplace=True)

            fig1.plot.bar()

            z= 'Customer Churned wrt ' + item

            plt.title(z,size=20)
churnratio()
customercountplot('Churn')
customercountplot('gender')
customercountplot('Contract')
customercountplot('Partner')
customercountplot('PhoneService')
customercountplot('MultipleLines')
customercountplot('StreamingTV')
tempdf = trainds.copy()

bins=[0,12,24,48,60,100]

tempdf['tenure_group']=pd.cut(tempdf['tenure'],bins,labels=['0-12','12-24','24-48','48-60','>60'])

plt.title('Customer Count wrt to tenure',size=20)

sns.countplot(tempdf['tenure_group'])
plt.title("Distribution Plot For Montly Charges",size=20)

sns.distplot(trainds['MonthlyCharges'],hist_kws={'edgecolor':'black','alpha':.5})
plt.title("Distribution Plot For TotalCharges",size=20)

sns.distplot(trainds['TotalCharges'],hist_kws={'edgecolor':'black','alpha':.5})
churnrate()
train = trainds.copy()

test = testds.copy()

train
train.columns
train = pd.get_dummies(train, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',

                                       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',

                                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',

                                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'])
test = pd.get_dummies(test, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',

                                       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',

                                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',

                                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'])
train.head(3)
train["Churn"] = train["Churn"].replace({'Yes':1,'No':0})
# For writing solution to file

def writetofile(solution,filename):

    with open(filename,'w') as file:

        file.write('customerID,Churn\n')

        for (a, b) in zip(CustomerIDS, solution):

            c=""

            if b==0:

                c="No"

            else:

                c='Yes'

            file.write(str(a)+','+str(c)+'\n')
X = train.drop('Churn', axis=1)

y = train['Churn']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.metrics import f1_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
sol2=logreg.predict(test)

sol2
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
import collections, numpy

collections.Counter(sol2)
pds = pd.DataFrame(columns=['CustomerID','Churn'])

pds['CustomerID'] = CustomerIDS

pds['Churn']=sol2

pds
#writetofile(Prediction ,'filename you want to save')

writetofile(sol2,'Prediction-Solution')
from sklearn import tree



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)

dt = dt.fit(X_train,y_train)



y_pred = dt.predict(X_test)

sol4=dt.predict(test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print(sol4)
#writetofile(Prediction ,'filename you want to save')

#writetofile(sol4,'Prediction-Solution')
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, average_precision_score

from xgboost import XGBClassifier

import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV



param_test = {

    

    'gamma': [0.5, 1, 1.5, 2, 5],

    'max_depth': [3, 4, 5]

  

}



clf = GridSearchCV(estimator = 

XGBClassifier(learning_rate =0.1,

              objective= 'binary:logistic',

              nthread=4,

              seed=27), 

              param_grid = param_test,

              scoring= 'accuracy',

              n_jobs=4,

              iid=False,

              verbose=10)
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)

print(y_pred)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
sol3= clf.predict(test)

print(y_pred)
import collections, numpy

collections.Counter(sol3)
#writetofile(Prediction ,'filename you want to save')

#writetofile(sol3,'Prediction-Solution')
from sklearn.ensemble import RandomForestClassifier



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

rf = RandomForestClassifier(n_estimators = 50, random_state = 42)

rf.fit(X_train,y_train)
y_pred = rf.predict(X_train)
y_pred= clf.predict(X_test)

print(y_pred)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
sol3 = rf.predict(test)
#import collections, numpy

#collections.Counter(sol3)
#writetofile(Prediction ,'filename you want to save')

#writetofile(sol3,'Prediction-Solution')