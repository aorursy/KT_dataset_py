from IPython.display import Image

import os

#!ls ../input/

Image("../input/baggingimage/Bagging_Main.PNG")
# Loading Libraries

import pandas as pd # for data analysis

import numpy as np # for scientific calculation

import seaborn as sns # for statistical plotting

import datetime # for working with date fields

import matplotlib.pyplot as plt # for plotting

%matplotlib inline

import math # for mathematical calculation
#Reading Loan Payment given Data Set.

import os

for dirname, _, filenames in os.walk('/kaggle/input/loan-payment/Loan_payments_data.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading Loan Payment .csv file.

loanpayment = pd.read_csv("../input/loan-payment/Loan_payments_data.csv")                             # Reading data using simple Pandas
# Describe method is used to view some basic statistical details like percentile, mean, std etc. of a data frame of numeric values.

loanpayment.describe()
#Checking shape of data

loanpayment.shape
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# Verifying top 3 sample records of data.

loanpayment.head(3)

# Checking Null Values : We can see there are No Null Values 

loanpayment.isnull().sum()

# Checking the data information.

# Observation: No missing values

loanpayment.info()
loanpayment['loan_status'].value_counts()
loanpayment['education'].value_counts()
loanpayment['Gender'].value_counts()
loanpayment['terms'].value_counts()
loanpayment.groupby(by=['Gender','education','loan_status'])['loan_status'].count()
print(np.min(loanpayment.age))

print(np.max(loanpayment.age))
loanpayment['age_bins'] = pd.cut(x=loanpayment['age'], bins=[18, 20, 30, 40, 50, 60])
plt.rcParams['figure.figsize'] = (20.0, 10.0)

plt.rcParams['font.family'] = "serif"

fig, ax =plt.subplots(3,2)

sns.countplot(loanpayment['Gender'], ax=ax[0,0])

sns.countplot(loanpayment['education'], ax=ax[0,1])

sns.countplot(loanpayment['loan_status'], ax=ax[1,0])

sns.countplot(loanpayment['Principal'], ax=ax[1,1])

sns.countplot(loanpayment['terms'], ax=ax[2,0])

sns.countplot(loanpayment['age_bins'], ax=ax[2,1])

fig.show();
import plotly.express as px



fig = px.histogram(loanpayment, x="terms", y="Principal", color = 'Gender',

                   marginal="rug", # or violin, rug,

                   hover_data=loanpayment.columns,

                   color_discrete_sequence=['indianred','lightblue'],

                   )



fig.update_layout(

    title="Gender wise segregation of loan terms and principal",

    xaxis_title="Loan Term",

    yaxis_title="Principal Count/Gender Segregation",

)

fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))



fig.show();
fig = px.scatter_3d(loanpayment,z="age",x="Principal",y="terms",

    color = 'Gender', size_max = 18,

    #color_discrete_sequence=['indianred','lightblue'] 

                    symbol='Gender', opacity=0.7

    )

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

loanpayment.head(1)
for dataset in [loanpayment]:

    dataset.loc[dataset['age'] <= 20,'age']=0

    dataset.loc[(dataset['age']>20) & (dataset['age']<=25),'age']=1

    dataset.loc[(dataset['age']>25) & (dataset['age']<=30),'age']=2

    dataset.loc[(dataset['age']>30) & (dataset['age']<=35),'age']=3

    dataset.loc[(dataset['age']>35) & (dataset['age']<=40),'age']=4

    dataset.loc[(dataset['age']>40) & (dataset['age']<=45),'age']=5

    dataset.loc[(dataset['age']>45) & (dataset['age']<=50),'age']=6

    dataset.loc[(dataset['age']>50) & (dataset['age']<=55),'age']=7
loanpayment.head(1)
# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

loanpayment_fe=loanpayment.copy()

# Encode labels in column 'education'. 

loanpayment_fe['education_label']= label_encoder.fit_transform(loanpayment_fe['education']) 

#loanpayment_fe['education_label'].unique() 

# Encode labels in column 'loan_status'. 

loanpayment_fe['loan_status_label']= label_encoder.fit_transform(loanpayment_fe['loan_status']) 

#loanpayment_fe['loan_status_label'].unique() 

# Encode labels in column 'Gender'. 

loanpayment_fe['gender_label']= label_encoder.fit_transform(loanpayment_fe['Gender']) 

#loanpayment_fe['gender_label'].unique() 
loanpayment_fe.head(1)
loanpayment_fe=loanpayment_fe.drop(['Loan_ID','loan_status','education','Gender','age_bins'],axis=1)

loanpayment_fe.head(1)
loanpayment_fe['effective_date']= pd.to_datetime(loanpayment_fe['effective_date']) 

loanpayment_fe['due_date']= pd.to_datetime(loanpayment_fe['due_date']) 

loanpayment_fe['paid_off_time']= pd.to_datetime(loanpayment_fe['paid_off_time']) 

loanpayment_fe.info()
loanpayment_fe['actual_tenure_days'] = loanpayment_fe['due_date'] - loanpayment_fe['effective_date']

loanpayment_fe['actual_tenure_days']=loanpayment_fe['actual_tenure_days']/np.timedelta64(1,'D')
print(np.min(loanpayment_fe['actual_tenure_days']))

print(np.max(loanpayment_fe['actual_tenure_days']))
loanpayment_fe['paidoff_tenure_days'] = loanpayment_fe['paid_off_time'] - loanpayment_fe['effective_date']

loanpayment_fe['paidoff_tenure_days']=loanpayment_fe['paidoff_tenure_days']/np.timedelta64(1,'D')
print(np.min(loanpayment_fe['paidoff_tenure_days']))

print(np.max(loanpayment_fe['paidoff_tenure_days']))
loanpayment_fe.isnull().sum()
loanpayment_fe.describe().T
loanpayment_fe=loanpayment_fe.drop(['past_due_days'],axis=1)
null_data = loanpayment_fe[loanpayment_fe.isnull().any(axis=1)]

null_data.head(2)
notnull_data = loanpayment_fe[loanpayment_fe.notnull().any(axis=1)]

notnull_data.head(2)
loanpayment_fe['paidoff_tenure_days'] = loanpayment_fe.groupby(['Principal','terms','age','education_label','gender_label'])['paidoff_tenure_days'].transform(lambda x:x.fillna(x.mean()))

loanpayment_fe['paidoff_tenure_days'] = loanpayment_fe['paidoff_tenure_days'].fillna(0)


loanpayment_fe.isnull().sum()
loanpayment_fe=loanpayment_fe.drop(['effective_date','due_date','paid_off_time'],axis=1)

loanpayment_fe.head(1)
loanpayment_fe['paidoff_tenure_days']=loanpayment_fe['paidoff_tenure_days'].round().astype(int)
loanpayment_fe['paidoff_tenure_days'].value_counts().head(2)
loanpayment_fe.info()
loanpayment_fe.head(2)
loanpayment_fe['loan_status_label'].value_counts().plot.bar();
X2=loanpayment_fe.drop(['loan_status_label'],axis=1)

X1=preprocessing.scale(X2)

X=pd.DataFrame(X1)

y=loanpayment_fe['loan_status_label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=0.25,stratify=y)
print(len(X_train[X_train==0]), len(X_train[X_train==1]), len(X_train[X_train==2]))

print(len(X_test[X_test==0]), len(X_test[X_test==1]), len(X_test[X_test==2]))

print(len(y_train[y_train==0]), len(y_train[y_train==1]), len(y_train[y_train==2]))

print(len(y_test[y_test==0]), len(y_test[y_test==1]), len(y_test[y_test==2]))

#!pip install imblearn
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

oversampler = SMOTE(random_state=0)

X_train ,y_train = oversampler.fit_sample(X_train, y_train)
print(len(X_train[X_train==0]), len(X_train[X_train==1]), len(X_train[X_train==2]))

print(len(X_test[X_test==0]), len(X_test[X_test==1]), len(X_test[X_test==2]))

print(len(y_train[y_train==0]), len(y_train[y_train==1]), len(y_train[y_train==2]))

print(len(y_test[y_test==0]), len(y_test[y_test==1]), len(y_test[y_test==2]))
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from IPython.display import Image

import os

#!ls ../input/

Image("../input/baggingimage/Bagging_Flow.PNG")
# Get some classifiers to evaluate

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings("ignore")







seed = 123

kfold = model_selection.KFold(n_splits=5, random_state=seed,shuffle=True)

dt = DecisionTreeClassifier()

num_trees = 100

model_dt = BaggingClassifier(base_estimator=dt, n_estimators=num_trees, random_state=seed).fit(X_train,y_train)

results = model_selection.cross_val_score(model_dt, X_train, y_train, cv=kfold,n_jobs=1)



predicted = model_selection.cross_val_predict(model_dt,X_test,y_test,cv=kfold) 

test_scores = accuracy_score(y_test, predicted)    

print("Train Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()))

print("\nTest Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()))

print("\nTest classification_report \n", classification_report(y_test, predicted))
seed = 123

num_trees = 100

kfold = model_selection.KFold(n_splits=5, random_state=seed,shuffle=True)

model_rf = RandomForestClassifier() 

bagging_clf = BaggingClassifier(model_rf,n_estimators=num_trees, random_state=seed).fit(X_train,y_train) 

results = model_selection.cross_val_score(bagging_clf, X_train, y_train, cv=kfold,n_jobs=1)



predicted = model_selection.cross_val_predict(bagging_clf,X_test,y_test,cv=kfold) 

test_scores = accuracy_score(y_test, predicted)    

print("Train Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()))

print("\nTest Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()))

print ("\nTest classification_report \n", classification_report(y_test, predicted))
seed = 123

num_trees = 100

kfold = model_selection.KFold(n_splits=5, random_state=seed,shuffle=True)

model_etc = ExtraTreesClassifier() 

bagging_clf = BaggingClassifier(model_etc,n_estimators=num_trees, random_state=seed).fit(X_train,y_train) 

results = model_selection.cross_val_score(model_etc, X_train, y_train, cv=kfold,n_jobs=1)



predicted = model_selection.cross_val_predict(bagging_clf,X_test,y_test,cv=kfold) 

test_scores = accuracy_score(y_test, predicted)    

print("Train Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()))

print("\nTest Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()))

print ("\nTest classification_report \n", classification_report(y_test, predicted))
from IPython.display import Image

import os

#!ls ../input/

Image("../input/baggingimages/Ensemble_Bagging_Final_Flow.PNG")
from sklearn.model_selection import cross_val_score



seed = 123





# Create classifiers

dtc = DecisionTreeClassifier()

rf = RandomForestClassifier()

et = ExtraTreesClassifier()

knn = KNeighborsClassifier()

svc = SVC()

rg = RidgeClassifier()



clf_array = [dtc, rf, et, knn, svc, rg]



for clf in clf_array:

    vanilla_scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)

    bagging_clf = BaggingClassifier(base_estimator=clf,n_estimators=num_trees, random_state=seed).fit(X_train,y_train) 

    bagging_scores = model_selection.cross_val_score(bagging_clf, X_train, y_train, cv=10, n_jobs=1)#, error_score='raise')

    print ("Mean of: {1:.3f}, std: (+/-) {2:.3f} [PlainVanilla {0}]".format(clf.__class__.__name__,vanilla_scores.mean(), vanilla_scores.std()))

    print ("Mean of: {1:.3f}, std: (+/-) {2:.3f} [Bagging {0}]\n".format(clf.__class__.__name__,bagging_scores.mean(), bagging_scores.std()))

    predicted = model_selection.cross_val_predict(bagging_clf,X_test,y_test,cv=kfold) 

    test_scores = accuracy_score(y_test, predicted)    

    print("Test Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()))

    print ("Test classification_report \n", classification_report(y_test, predicted))
from IPython.display import Image

import os

#!ls ../input/

Image("../input/baggingimage/Voting.PNG")


clf = [rf, et, knn, svc, rg]

eclf = VotingClassifier(estimators=[('Random Forests', rf), ('Extra Trees', et), ('KNeighbors', knn), ('SVC', svc), ('Ridge Classifier', rg)], voting='hard')

for clf, label in zip([rf, et, knn, svc, rg, eclf], ['Random Forest', 'Extra Trees', 'KNeighbors', 'SVC', 'Ridge Classifier', 'Ensemble']):

    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
