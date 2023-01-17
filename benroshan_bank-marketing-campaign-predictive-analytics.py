## Database Phase

import pandas as pd

import numpy as np



# Machine Learning Phase

import sklearn 

from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split



#Metrics Phase

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score



#Visualization Phase

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl

import matplotlib.pylab as pylab

%matplotlib inline

pd.set_option('display.max_columns', 500)

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
bank=pd.read_csv("../input/bank-marketing-campaigns-dataset/bank-additional-full.csv",sep=';')

bank_copy=bank.copy()



## print shape of dataset with rows and columns and information 

print ("The shape of the  data is (row, column):"+ str(bank_copy.shape))

print (bank_copy.info())
bank_copy.head()
bank_copy.dtypes
#Checking out the statistical parameters

bank_copy.describe()
#Checking out the categories and their respective counts in each feature

print("Job:",bank_copy.job.value_counts(),sep = '\n')

print("-"*40)

print("Marital:",bank_copy.marital.value_counts(),sep = '\n')

print("-"*40)

print("Education:",bank_copy.education.value_counts(),sep = '\n')

print("-"*40)

print("Default:",bank_copy.default.value_counts(),sep = '\n')

print("-"*40)

print("Housing loan:",bank_copy.housing.value_counts(),sep = '\n')

print("-"*40)

print("Personal loan:",bank_copy.loan.value_counts(),sep = '\n')

print("-"*40)

print("Contact:",bank_copy.contact.value_counts(),sep = '\n')

print("-"*40)

print("Month:",bank_copy.month.value_counts(),sep = '\n')

print("-"*40)

print("Day:",bank_copy.day_of_week.value_counts(),sep = '\n')

print("-"*40)

print("Previous outcome:",bank_copy.poutcome.value_counts(),sep = '\n')

print("-"*40)

print("Outcome of this campaign:",bank_copy.y.value_counts(),sep = '\n')

print("-"*40)
import missingno as msno 

msno.matrix(bank_copy)
print('Data columns with null values:',bank_copy.isnull().sum(), sep = '\n')
import plotly.express as px



fig = px.box(bank_copy, x="job", y="duration", color="y")

fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default

fig.show()
fig = px.scatter(bank_copy, x="campaign", y="duration", color="y")

fig.show()
plt.bar(bank_copy['month'], bank_copy['campaign'])
plt.subplot(231)

sns.distplot(bank_copy['emp.var.rate'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(232)

sns.distplot(bank_copy['cons.price.idx'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(233)

sns.distplot(bank_copy['cons.conf.idx'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(234)

sns.distplot(bank_copy['euribor3m'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(235)

sns.distplot(bank_copy['nr.employed'])

fig = plt.gcf()

fig.set_size_inches(10,10)
sns.violinplot( y=bank_copy["marital"], x=bank_copy["cons.price.idx"] )
bank_yes = bank_copy[bank_copy['y']=='yes']





df1 = pd.crosstab(index = bank_yes["marital"],columns="count")    

df2 = pd.crosstab(index = bank_yes["month"],columns="count")  

df3= pd.crosstab(index = bank_yes["job"],columns="count") 

df4=pd.crosstab(index = bank_yes["education"],columns="count")



fig, axes = plt.subplots(nrows=2, ncols=2)

df1.plot.bar(ax=axes[0,0])

df2.plot.bar(ax=axes[0,1])

df3.plot.bar(ax=axes[1,0])

df4.plot.bar(ax=axes[1,1])       
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(bank_copy.corr(),annot=True,linewidths=0.5,linecolor="black",fmt=".1f",ax=ax)

plt.show()
plt.figure(figsize = (15, 30))

plt.style.use('seaborn-white')

ax=plt.subplot(521)

plt.boxplot(bank_copy['age'])

ax.set_title('age')

ax=plt.subplot(522)

plt.boxplot(bank_copy['duration'])

ax.set_title('duration')

ax=plt.subplot(523)

plt.boxplot(bank_copy['campaign'])

ax.set_title('campaign')

ax=plt.subplot(524)

plt.boxplot(bank_copy['pdays'])

ax.set_title('pdays')

ax=plt.subplot(525)

plt.boxplot(bank_copy['previous'])

ax.set_title('previous')

ax=plt.subplot(526)

plt.boxplot(bank_copy['emp.var.rate'])

ax.set_title('Employee variation rate')

ax=plt.subplot(527)

plt.boxplot(bank_copy['cons.price.idx'])

ax.set_title('Consumer price index')

ax=plt.subplot(528)

plt.boxplot(bank_copy['cons.conf.idx'])

ax.set_title('Consumer confidence index')

ax=plt.subplot(529)

plt.boxplot(bank_copy['euribor3m'])

ax.set_title('euribor3m')

ax=plt.subplot(5,2,10)

plt.boxplot(bank_copy['nr.employed'])

ax.set_title('No of employees')

numerical_features=['age','campaign','duration']

for cols in numerical_features:

    Q1 = bank_copy[cols].quantile(0.25)

    Q3 = bank_copy[cols].quantile(0.75)

    IQR = Q3 - Q1     



    filter = (bank_copy[cols] >= Q1 - 1.5 * IQR) & (bank_copy[cols] <= Q3 + 1.5 *IQR)

    bank_copy=bank_copy.loc[filter]
plt.figure(figsize = (15, 10))

plt.style.use('seaborn-white')

ax=plt.subplot(221)

plt.boxplot(bank_copy['age'])

ax.set_title('age')

ax=plt.subplot(222)

plt.boxplot(bank_copy['duration'])

ax.set_title('duration')

ax=plt.subplot(223)

plt.boxplot(bank_copy['campaign'])

ax.set_title('campaign')
bank_features=bank_copy.copy()

lst=['basic.9y','basic.6y','basic.4y']

for i in lst:

    bank_features.loc[bank_features['education'] == i, 'education'] = "middle.school"



bank_features['education'].value_counts()
month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}

bank_features['month']= bank_features['month'].map(month_dict) 



day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}

bank_features['day_of_week']= bank_features['day_of_week'].map(day_dict) 
bank_features.loc[:, ['month', 'day_of_week']].head()
bank_features.loc[bank_features['pdays'] == 999, 'pdays'] = 0
bank_features['pdays'].value_counts()
dictionary={'yes':1,'no':0,'unknown':-1}

bank_features['housing']=bank_features['housing'].map(dictionary)

bank_features['default']=bank_features['default'].map(dictionary)

bank_features['loan']=bank_features['loan'].map(dictionary)
dictionary1={'no':0,'yes':1}

bank_features['y']=bank_features['y'].map(dictionary1)
bank_features.loc[:,['housing','default','loan','y']].head()
dummy_contact=pd.get_dummies(bank_features['contact'], prefix='dummy',drop_first=True)

dummy_outcome=pd.get_dummies(bank_features['poutcome'], prefix='dummy',drop_first=True)

bank_features = pd.concat([bank_features,dummy_contact,dummy_outcome],axis=1)

bank_features.drop(['contact','poutcome'],axis=1, inplace=True)
bank_features.loc[:,['dummy_telephone','dummy_nonexistent','dummy_success']].head()
bank_job=bank_features['job'].value_counts().to_dict()

bank_ed=bank_features['education'].value_counts().to_dict()
bank_features['job']=bank_features['job'].map(bank_job)

bank_features['education']=bank_features['education'].map(bank_ed)

bank_features.loc[:,['job','education']].head()
bank_features.groupby(['marital'])['y'].mean()
ordinal_labels=bank_features.groupby(['marital'])['y'].mean().sort_values().index

ordinal_labels
ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}

ordinal_labels2
bank_features['marital_ordinal']=bank_features['marital'].map(ordinal_labels2)

bank_features.drop(['marital'], axis=1,inplace=True)
bank_features.marital_ordinal.value_counts()
bank_scale=bank_features.copy()

Categorical_variables=['job', 'education', 'default', 'housing', 'loan', 'month',

       'day_of_week','y', 'dummy_telephone', 'dummy_nonexistent',

       'dummy_success', 'marital_ordinal']





feature_scale=[feature for feature in bank_scale.columns if feature not in Categorical_variables]





scaler=StandardScaler()

scaler.fit(bank_scale[feature_scale])
scaled_data = pd.concat([bank_scale[['job', 'education', 'default', 'housing', 'loan', 'month',

       'day_of_week','y', 'dummy_telephone', 'dummy_nonexistent',

       'dummy_success', 'marital_ordinal']].reset_index(drop=True),

                    pd.DataFrame(scaler.transform(bank_scale[feature_scale]), columns=feature_scale)],

                    axis=1)

scaled_data.head()
X=scaled_data.drop(['y'],axis=1)

y=scaled_data.y



model = ExtraTreesClassifier()

model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(17).plot(kind='barh')

plt.show()
X=scaled_data.drop(['pdays','month','cons.price.idx','loan','housing','emp.var.rate','y'],axis=1)

y=scaled_data.y



X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=1)

print("Input Training:",X_train.shape)

print("Input Test:",X_test.shape)

print("Output Training:",y_train.shape)

print("Output Test:",y_test.shape)
#creating the objects

logreg_cv = LogisticRegression(random_state=0)

dt_cv=DecisionTreeClassifier()

knn_cv=KNeighborsClassifier()

svc_cv=SVC()

nb_cv=BernoulliNB()

cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree',2:'KNN',3:'SVC',4:'Naive Bayes'}

cv_models=[logreg_cv,dt_cv,knn_cv,svc_cv,nb_cv]





for i,model in enumerate(cv_models):

    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, X, y, cv=10, scoring ='accuracy').mean()))
param_grid = {'C': np.logspace(-4, 4, 50),

             'penalty':['l1', 'l2']}

clf = GridSearchCV(LogisticRegression(random_state=0), param_grid,cv=5, verbose=0,n_jobs=-1)

best_model = clf.fit(X_train,y_train)

print(best_model.best_estimator_)

print("The mean accuracy of the model is:",best_model.score(X_test,y_test))
logreg = LogisticRegression(C=0.18420699693267145, random_state=0)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n",confusion_matrix)

print("Classification Report:\n",classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
svc_classifier = SVC(random_state = 0)

svc_classifier.fit(X_train,y_train)

y_pred=svc_classifier.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Classification Report:\n",classification_report(y_test,y_pred))