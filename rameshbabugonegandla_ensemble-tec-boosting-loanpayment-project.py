from IPython.display import Image

import os

#!ls ../input/

Image("../input/boostingimage/Boosting_Main.PNG")
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

                   marginal="violin", # or violin, rug,

                   color_discrete_sequence=['indianred','lightblue'],

                   )



fig.update_layout(

    title="Gender wise segregation of loan terms and principal",

    xaxis_title="Loan Term",

    yaxis_title="Principal Count/Gender Segregation",

)

fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))



fig.show()
fig = px.scatter_3d(loanpayment,z="age",x="Principal",y="terms",

    color = 'Gender', size_max = 18,

    color_discrete_sequence=['indianred','lightblue']

    )



fig.show()
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

Image("../input/boostingimage/Boosting_DataFlow.PNG")
# Import Models

import xgboost

import lightgbm as lgb

import xgboost as xgb



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from catboost import CatBoostClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier





from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.ensemble import VotingClassifier

from mlxtend.classifier import EnsembleVoteClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix



import warnings

warnings.filterwarnings("ignore")



learning_rate = 1

seed=123


model = AdaBoostClassifier(random_state=seed)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

test_scores = accuracy_score(y_test, predictions)  

print("Training Accuracy score (training): {0:.3f}".format(model.score(X_train, y_train)));

print("Test Accuracy Score: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()));



print("Test Confusion Matrix:")

print(confusion_matrix(y_test, predictions));



print("Test Classification Report")

print(classification_report(y_test, predictions));
kfold = model_selection.KFold(n_splits=5, random_state=seed,shuffle=True)



bdt_real = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=600,learning_rate=1)



bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=600,learning_rate=1.5,algorithm="SAMME")



# For real samples.

bdt_real.fit(X_train, y_train)

results = model_selection.cross_val_score(bdt_real, X_train, y_train, cv=kfold,n_jobs=1)

predicted = model_selection.cross_val_predict(bdt_real,X_test,y_test,cv=kfold) 

test_scores = accuracy_score(y_test, predicted)    

print("Real Samples Train Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()));

print("Real Samples Test Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()));

print("Real Samples Test classification_report \n", classification_report(y_test, predicted));



# For Discrete samples

bdt_discrete.fit(X_train, y_train)

results = model_selection.cross_val_score(bdt_discrete, X_train, y_train, cv=kfold,n_jobs=1)

predicted = model_selection.cross_val_predict(bdt_discrete,X_test,y_test,cv=kfold) 

test_scores = accuracy_score(y_test, predicted)    

print("Discrete Samples Train Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()));

print("Discrete Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()));

print("Discrete classification_report \n", classification_report(y_test, predicted));





real_test_errors = []

discrete_test_errors = []



for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):

    real_test_errors.append(1. - accuracy_score(real_test_predict, y_test))

    discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, y_test))



n_trees_discrete = len(bdt_discrete)

n_trees_real = len(bdt_real)



# Boosting might terminate early, but the following arrays are always

# n_estimators long. We crop them to the actual number of trees here:

discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]

real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]

discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]



plt.figure(figsize=(15, 5))



plt.subplot(131)

plt.plot(range(1, n_trees_discrete + 1), discrete_test_errors, c='black', label='SAMME')

plt.plot(range(1, n_trees_real + 1),real_test_errors, c='black',linestyle='dashed', label='SAMME.R')

plt.legend()

plt.ylim(0.18, 0.62)

plt.ylabel('Test Error')

plt.xlabel('Number of Trees')



plt.subplot(132);

plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,"b", label='SAMME', alpha=.5)

plt.plot(range(1, n_trees_real + 1), real_estimator_errors,"r", label='SAMME.R', alpha=.5)

plt.legend()

plt.ylabel('Error')

plt.xlabel('Number of Trees')

plt.ylim((.2,max(real_estimator_errors.max(),discrete_estimator_errors.max()) * 1.2))

plt.xlim((-20, len(bdt_discrete) + 20))



plt.subplot(133);

plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,"b", label='SAMME')

plt.legend()

plt.ylabel('Weight')

plt.xlabel('Number of Trees')

plt.ylim((0, discrete_estimator_weights.max() * 1.2))

plt.xlim((-20, n_trees_discrete + 20))



# prevent overlapping y-axis labels

plt.subplots_adjust(wspace=0.25)

plt.show();
kfold = model_selection.KFold(n_splits=5, random_state=seed,shuffle=True)



lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]



for learning_rate in lr_list:

    gb_clf = GradientBoostingClassifier(n_estimators=600, learning_rate=learning_rate,random_state=seed) #max_features=2, max_depth=2

    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate);

    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)));

    print("Accuracy score (testing): {0:.3f}".format(gb_clf.score(X_test, y_test)));
gb_clf2 = GradientBoostingClassifier(n_estimators=600, learning_rate=0.1, random_state=seed)

gb_clf2.fit(X_train, y_train)

predictions = gb_clf2.predict(X_test)

test_scores = accuracy_score(y_test, predictions)  

print("Learning rate: ", learning_rate);

print("Training Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)));

print("Test Accuracy Score: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()));



print("Test Confusion Matrix:");

print(confusion_matrix(y_test, predictions));



print("Test Classification Report");

print(classification_report(y_test, predictions));


model=xgb.XGBClassifier(n_estimators=600, learning_rate=learning_rate,random_state=seed,gamma=0.2,colsample_bytree=0.75)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

test_scores = accuracy_score(y_test, predictions)  

print("Learning rate: ", learning_rate);

print("Training Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)));

print("Test Accuracy Score: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()));



print("Test Confusion Matrix:");

print(confusion_matrix(y_test, predictions));



print("Test Classification Report");

print(classification_report(y_test, predictions));      

      
model=CatBoostClassifier(iterations=50,random_seed=seed,learning_rate=learning_rate,custom_loss=['AUC', 'Accuracy']);

model.fit(X_train,y_train,eval_set=(X_test, y_test));

predictions = model.predict(X_test);

test_scores = accuracy_score(y_test, predictions);

print("Learning rate: ", learning_rate);

print("Training Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)));

print("Test Accuracy Score: %0.2f (+/- %0.2f)" % (test_scores.mean(),test_scores.std()));



print("Test Confusion Matrix:");

print(confusion_matrix(y_test, predictions));



print("Test Classification Report");

print(classification_report(y_test, predictions));
#Converting the dataset in proper LGB format

d_train=lgb.Dataset(X_train, label=y_train)



from sklearn.metrics import precision_score

# Parameters setting

params={}

params['learning_rate']=1

params['boosting_type']='gbdt' 

params['objective']='multiclass' 

params['metric']='multi_logloss' 

params['n_estimators']=600

params['num_class']=3 



#training the model on 100 epocs

clf=lgb.train(params,d_train,100)  

predictions=clf.predict(X_test)



#argmax() method for preditions

predictions = [np.argmax(line) for line in predictions]



#using precision score for error metrics

print(predictions);

print(precision_score(predictions,y_test,average=None).mean());

ada_boost = AdaBoostClassifier()

grad_boost = GradientBoostingClassifier()

xgb_boost = XGBClassifier()



ada_boost_real = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=600,learning_rate=learning_rate,random_state=seed)

ada_boost_discrete = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=600,learning_rate=learning_rate,random_state=seed,algorithm="SAMME")

grad_boost = GradientBoostingClassifier(n_estimators=600, learning_rate=learning_rate,random_state=seed)

xgb_boost=xgb.XGBClassifier(n_estimators=600, learning_rate=0.05,random_state=seed,gamma=0.2,colsample_bytree=0.75)





eclf = EnsembleVoteClassifier(clfs=[ada_boost_real,ada_boost_discrete, grad_boost, xgb_boost], voting='hard')

labels = ['Ada Boost Real','Ada Boost Discrete', 'Grad Boost', 'XG Boost','Ensemble']



for clf, label in zip([ada_boost_real,ada_boost_discrete, grad_boost, xgb_boost,eclf], labels):

    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy');

    print("Hard Voting Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(), scores.std(), label));

    
