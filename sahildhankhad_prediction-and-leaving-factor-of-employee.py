%matplotlib inline

#Import basic packages

import numpy as np 

import pandas as pd

import matplotlib as matplot

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation

from IPython.display import display

from sklearn.metrics import classification_report,confusion_matrix

mydataset = pd.read_csv('../input/HR_comma_sep.csv')
mydataset.shape
mydataset.head()
mydataset.describe()
mydataset.dtypes
fig = plt.figure(figsize=(10,10))

corr = mydataset.corr()

sns.heatmap(corr, vmax=1, square=True,annot=True)

sns.plt.title('Correlation Matrix Heatmap')
mydataset.mean()
# Now we are going to use mean value and find the number of people against it.

print('there are {} employees evaluated more than 0.7'.format(len(mydataset[mydataset['last_evaluation']>0.7])))

print('there are {} employees evaluated less than 0.7'.format(len(mydataset[mydataset['last_evaluation']<=0.7])))

print('there are {} employees satisfication level more than 0.6'.format(len(mydataset[mydataset['satisfaction_level']>0.6])))

print('there are {} employees satisfication level less than 0.6'.format(len(mydataset[mydataset['satisfaction_level']<=0.6])))

print('there are {} employees have project more than 3.80'.format(len(mydataset[mydataset['number_project']>3.80])))

print('there are {} employees have project less than 3.80'.format(len(mydataset[mydataset['number_project']<=3.80])))

print('there are {} employees that spend average monthly hours more than 201.050337'.format(len(mydataset[mydataset['average_montly_hours']>201.050337])))

print('there are {} employees that spend average monthly hours less than 201.050337'.format(len(mydataset[mydataset['average_montly_hours']<=201.050337])))

mydataset.groupby(["sales","left"]).mean()
sns.factorplot("sales", col="salary", col_wrap=3, data=mydataset, kind="count", size=15, aspect=.4)

mydataset.hist(figsize =(15,15))

plt.show()
left = mydataset[mydataset.left == 1]

stay = mydataset[mydataset.left ==0 ]

print('number of people  stay ='+str(len(stay)))

print('Number of people left='+str(len(left)))

sns.countplot(x="sales", data=mydataset)

plt.title('Distribution of employess across departments')

plt.xticks(rotation=90);
sns.countplot(x="sales",hue='left', data=mydataset)

plt.title('Number of people left from particular department')

plt.xticks(rotation=90);

temp3 = pd.crosstab(mydataset['sales'], mydataset['salary'])

temp3.plot(kind='bar', stacked=True, color=['red','blue','Green'], grid=False)
sns.countplot(x="number_project",hue='left', data=mydataset)

plt.title('Number of People Doing project and still they left')
sns.countplot(x="number_project",hue='salary', data=mydataset)

plt.title('Number of project Vs Salary')
sns.factorplot('number_project','average_montly_hours',hue='left',data=mydataset)
sns.factorplot('number_project','satisfaction_level',hue='left',data=mydataset)

plt.title('Number of project vs Satisfaction level based on that people leaving and staying')
sns.factorplot('number_project','last_evaluation',hue='left',data=mydataset)

plt.title('Number of project vs last_evaluation based on that people leaving and staying')
sns.countplot(x="Work_accident",hue='left', data=mydataset)

plt.title('number of people have accident at work they Stay or left')
sns.countplot(x="Work_accident",hue='salary', data=mydataset)

plt.title('number of people have accident at work,their salary')
sns.countplot(x="time_spend_company",hue='salary', data=mydataset)

plt.title('Salary based on the time spend in company')
sns.countplot(x="time_spend_company",hue='left', data=mydataset)

plt.title('the time spend in company')
sns.factorplot('time_spend_company','average_montly_hours',hue='left',data=mydataset)

plt.title('Time spend at company vs Avarge montly hours based on that who stay or left')
sns.barplot(x = 'time_spend_company', y = 'satisfaction_level', data = mydataset)

sns.plt.title('Satisfaction based on the time spend at company')
sns.barplot('time_spend_company','average_montly_hours',data=mydataset)

plt.title('Time spend at company vs average montly hours ')
sns.countplot(x="salary",hue='left', data=mydataset)

plt.title('People left Company based on their salary')
plt.figure(figsize=(10,6))

sns.kdeplot(data=mydataset[mydataset["left"]==0]["satisfaction_level"],color='b',shade=True,label='Stay')

sns.kdeplot(data=mydataset[mydataset["left"]==1]["satisfaction_level"],color='r',shade=True,label='Left')

plt.title('satisfaction_level vs left')

plt.xlabel('satisfaction_level')

plt.ylabel('count')
plt.figure(figsize=(10,6))

sns.kdeplot(data=mydataset[mydataset["left"]==0]["last_evaluation"],color='b',shade=True,label='Stay')

sns.kdeplot(data=mydataset[mydataset["left"]==1]["last_evaluation"],color='r',shade=True,label='Left')

plt.title('last_evaluation')

plt.xlabel('last_evaluation')

plt.ylabel('count')
plt.figure(figsize=(10,10))

sns.kdeplot(data=mydataset[mydataset["left"]==0]["average_montly_hours"],color='b',shade=True,label='Stay')

sns.kdeplot(data=mydataset[mydataset["left"]==1]["average_montly_hours"],color='r',shade=True,label='Left')

plt.title('average_montly_hours')

plt.xlabel('average_montly_hours')

plt.ylabel('count')
plt.figure(figsize=(10,10))

sns.kdeplot(data=mydataset[mydataset["left"]==0]["time_spend_company"],color='b',shade=True,label='Stay')

sns.kdeplot(data=mydataset[mydataset["left"]==1]["time_spend_company"],color='r',shade=True,label='Left')

plt.title('time_spend_company')

plt.xlabel('time_spend_company')

plt.ylabel('count')
sns.countplot(x="promotion_last_5years",hue='left', data=mydataset)

plt.title('Number of employess Who got promotion or not in last 5 years they stay or left')
plt.figure(figsize=(20,20))

p = pd.crosstab(mydataset['average_montly_hours'], mydataset['salary'])

p.plot(kind='bar', stacked=True, color=['red','blue','yellow'], grid=False)

plt.tight_layout()
plt.figure(figsize=(30,8))

sns.barplot(x=mydataset['satisfaction_level'],y=mydataset['average_montly_hours'])

plt.tight_layout()
sns.barplot('promotion_last_5years', 'salary', data = mydataset)

sns.plt.title('Salary over promotion_last_5years ')
# now i'm converting string value into Binary form; so we can use them in our proejct

mydataset['salary'].replace({'low':1,'medium':2,'high':3},inplace=True)

mydataset['sales'].replace({'IT':11,'RandD':12,'accounting':13,'hr':14,'management':15,'marketing':16,'product_mng':17,'sales':18,'support':19,'technical':20},inplace=True)
# now i spliting my Traning and testing data

from sklearn.model_selection import train_test_split

panel = mydataset[['satisfaction_level','average_montly_hours','promotion_last_5years','salary','number_project','last_evaluation','time_spend_company']]

x=panel # i tried to use pop command, but itn't let me take value more than 2.

y=mydataset['left']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,test_size=0.25)

print('Training set volume:', x_train.shape[0])

print('Test set volume:', x_test.shape[0])
x_train.head()
x_test.head()
#Logistic Regression

lr=LogisticRegression()

lr.fit(x_train,y_train)

testscoreLR=accuracy_score(y_test,lr.predict(x_test))

print('logistic regression accuracy score:'+str(testscoreLR))

print(confusion_matrix(y_test,lr.predict(x_test)))

print(classification_report(y_test,lr.predict(x_test)))
#Decision tree

from sklearn import tree

dt = tree.DecisionTreeClassifier(max_depth=3)

dt.fit(x_train, y_train)

testscoreDT=accuracy_score(y_test,dt.predict(x_test))

print("decision tree accuracy Rate is:"+str(testscoreDT))

print(confusion_matrix(y_test,dt.predict(x_test)))

print(classification_report(y_test,dt.predict(x_test)))
#Decision tree

from sklearn import tree

dt = tree.DecisionTreeClassifier(max_depth=8)

dt.fit(x_train, y_train)

testscoreDT=accuracy_score(y_test,dt.predict(x_test))

print("decision tree accuracy Rate is:"+str(testscoreDT))

print(confusion_matrix(y_test,dt.predict(x_test)))

print(classification_report(y_test,dt.predict(x_test)))

#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=100)

rf.fit(x_train, y_train)

testscoreRF=accuracy_score(y_test,dt.predict(x_test)).mean()

print("Random Tree accuracy Rate is:"+str(testscoreRF))

print(confusion_matrix(y_test,rf.predict(x_test)))

print(classification_report(y_test,rf.predict(x_test)))
#Gaussian navie bayers

nb=GaussianNB()

nb.fit(x_train,y_train)

testscoreNB=accuracy_score(y_test,nb.predict(x_test))

print('GaussianNB accuracy score:'+str(testscoreNB))

print(confusion_matrix(y_test,nb.predict(x_test)))

print(classification_report(y_test,nb.predict(x_test)))
#KNN

kn=KNeighborsClassifier()

kn.fit(x_train,y_train)

testscoreKN=accuracy_score(y_test,kn.predict(x_test))

print('KNeighborsClassifier accuracy score:'+str(testscoreKN))

print(confusion_matrix(y_test,kn.predict(x_test)))

print(classification_report(y_test,kn.predict(x_test)))
#Arrrange the model according tp there accuracy score

models = pd.DataFrame({'Model' : [ 'random Forest', 'Decision Tree', 'Logistic Regression', 'KNN Regression','Gaussian Naive Bays'],'Testing_Score' : [ testscoreRF, testscoreDT, testscoreLR, testscoreKN, testscoreNB],})

models.sort_values(by='Testing_Score', ascending=False)
#using Random Forest

importances=rf.feature_importances_

f=np.argsort(importances)[::-1]

print ('feature ranking:')

for i in range(x.shape[1]):

     print ("feature no. {}: {} ({})".format(i+1,x.columns[f[i]],importances[f[i]]))

importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(rf.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=True).set_index('feature')

importances.plot.bar()
#Decision Tree

importances=dt.feature_importances_

f=np.argsort(importances)[::-1]

print ('feature ranking:')

for i in range(x.shape[1]):

     print ("feature no. {}: {} ({})".format(i+1,x.columns[f[i]],importances[f[i]]))
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(dt.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=True).set_index('feature')

importances.plot.bar()
y_pred = lr.predict(x_test)

y_prob = lr.predict_proba(x_test)
y1_pred = dt.predict(x_test)

y1_prob = dt.predict_proba(x_test)

y2_prob = rf.predict_proba(x_test)

y3_prob = nb.predict_proba(x_test)

y4_prob = kn.predict_proba(x_test)
from sklearn.metrics import roc_curve, auc

# area under the curve

Falsepositive, truepositive,_ = (roc_curve(y_test,y_prob[:,1]))

FalsepositiveDT, truepositiveDT,_ = (roc_curve(y_test,y1_prob[:,1]))

FalsepositiveRF, truepositiveRF,_ = (roc_curve(y_test,y2_prob[:,1]))

FalsepositiveNB, truepositiveNB,_ = (roc_curve(y_test,y3_prob[:,1]))

FalsepositiveKN, truepositiveKN,_ = (roc_curve(y_test,y4_prob[:,1]))

ROC_AUCLR = auc(Falsepositive, truepositive)

ROC_AUCDT = auc(FalsepositiveDT, truepositiveDT)

ROC_AUCRF = auc(FalsepositiveDT, truepositiveDT)

ROC_AUCNB = auc(FalsepositiveNB, truepositiveNB)

ROC_AUCKN = auc(FalsepositiveKN, truepositiveKN)

#Plottig

plt.plot(Falsepositive, truepositive, label='Logistic Regression (area = %0.2f)' % ROC_AUCLR, linewidth=4)

plt.plot(FalsepositiveDT, truepositiveDT, label='Dicision Tree (area = %0.2f)' % ROC_AUCDT, linewidth=3)

plt.plot(FalsepositiveRF, truepositiveRF, label='Random Forest (area = %0.2f)' % ROC_AUCRF, linewidth=1)

plt.plot(FalsepositiveNB, truepositiveNB, label='Navie bayes(area = %0.2f)' % ROC_AUCNB, linewidth=2)

plt.plot(FalsepositiveKN, truepositiveKN, label='KN(area = %0.2f)' % ROC_AUCNB, linewidth=2)

plt.plot([0, 1], [0, 1], 'r--', linewidth=2)

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('False Positive Rate', fontsize=14)

plt.ylabel('True Positive Rate', fontsize=14)

plt.title('ROC curve for Employee Leaving factor', fontsize=16)

plt.legend(loc="lower right")

plt.show()
test_predict = x_test.iloc[0:10,:] # you can change the 10 to any number you want till 14999

realpr = dt.predict(test_predict) # we are using real prediction value

for i in realpr:

    print (i)
y_test.iloc[0:10] # It randmly select 10 employee.
# Load our actual value for prediction

dataf = pd.get_dummies(mydataset)
# we need to provide information of 

left = dataf[dataf['left'] == 1]

left1 = pd.get_dummies(left)

            

c = left1

a = c['left'].values

c = c.drop(['left'],axis=1)

b= c.values

pred = dt.predict_proba(b[:, :7]) # we have 8 model input so, i selected 7 input

## i used dicison tree for my data

# number of employees that definitely are leaving

sum(pred[:,1]==1)
left['Will leave the job'] = pred[:,1]

# you can change this leaving prob,but i select 45%.

left[left['Will leave the job']>=0.45]
#from sklearn.model_selection import train_test_split

panel12= mydataset[['satisfaction_level','average_montly_hours','number_project','time_spend_company']]

x1=panel12 # i tried to use pop command, but itn't let me take value more than 2.

y1=mydataset['left']

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=42,test_size=0.25)

print('Training set volume:', x1_train.shape[0])

print('Test set volume:', x1_test.shape[0])
#Logistic Regression

lr1=LogisticRegression()

lr1.fit(x1_train,y1_train)

testscoreLR1=accuracy_score(y1_test,lr1.predict(x1_test))

print('logistic regression accuracy score:'+str(testscoreLR1))

print(confusion_matrix(y1_test,lr1.predict(x1_test)))

print(classification_report(y1_test,lr1.predict(x1_test)))
#Decision tree

from sklearn import tree

dt1 = tree.DecisionTreeClassifier(max_depth=8)

dt1.fit(x1_train, y1_train)

testscoreDT1=accuracy_score(y1_test,dt1.predict(x1_test))

print("decision tree accuracy Rate is:"+str(testscoreDT1))

print(confusion_matrix(y1_test,dt1.predict(x1_test)))

print(classification_report(y1_test,dt1.predict(x1_test)))



#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf1= RandomForestClassifier(n_estimators=100)

rf1.fit(x1_train, y1_train)

testscoreRF1=accuracy_score(y1_test,dt1.predict(x1_test)).mean()

print("decision tree accuracy Rate is:"+str(testscoreRF1))

print(confusion_matrix(y1_test,rf1.predict(x1_test)))

print(classification_report(y1_test,rf1.predict(x1_test)))



nb1=GaussianNB()

nb1.fit(x1_train,y1_train)

testscoreNB1=accuracy_score(y1_test,nb1.predict(x1_test))

print('GaussianNB accuracy score:'+str(testscoreNB1))

print(confusion_matrix(y1_test,nb1.predict(x1_test)))

print(classification_report(y1_test,nb1.predict(x1_test)))