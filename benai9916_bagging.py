import numpy as np 
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import precision_score, accuracy_score,  confusion_matrix, classification_report, auc, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns
add = pd.read_csv('../input/advertsuccess/Train.csv')
# print the size of the data
add.shape
# checking the first 5 row of data set
add.head()
# get the brief information about dataset
add.info()
# get the aggregation
add.describe()
# make id as index
add.set_index('id', inplace=True)

# sort index
add.sort_index(ascending=True, inplace=True)
# check missing values
add.isnull().sum()
# check the number of unique value in each column
for cols in add.columns:
    print(cols, '--> ',add[cols].nunique())
#Understanding the target variable netgain.
# We can see that the count of unsuccessful ads are higher.

sns.countplot('netgain',data = add)
plt.show()
# method to add percentage on the plot
def add_percentage(ax, total):
    for i in ax.patches:
        height = i.get_height()
        ax.text(i.get_x()+i.get_width()/2.,height + 5,'{:1.2f}'.format(height/total*100) + '%')
# Around 76% of the ad campaigns are not successful.
# 0 - False = ad campaign not successful.
# 1 - True = ad campaign successful.

total = float(len(add))
plt.title('Netgain sucess and failure percentage')
ax = sns.countplot(x="netgain", data=add)

add_percentage(ax, total)
for cols in add.columns:
    if add[cols].dtype == 'O':
        total = float(len(add))
        plt.figure(figsize=(15,4.5))
        plt.title('Netgain success based on '+ cols, fontsize=24)
        ax = sns.countplot(x=cols, data=add, hue='netgain')

        add_percentage(ax, total)
        plt.show()
# As suspected above, yes Pharma industry dominates the other sectors and has the highest count of more than 10000 observations
# realted to it.Around 40% of the industry sector is contributed to Pharma industry and it also has high count of successful ads.

total = float(len(add))
plt.figure(figsize=(15,4.5))
plt.title('Netgain sucess and failure percentage')
ax = sns.countplot(x="industry", data=add)

add_percentage(ax, total)
# We can see that the average run time of the ads per week is around 40 mins 
# i.e the ads were aired around 40 mins per week.

plt.figure(figsize=(15,6))
sns.distplot(add['average_runtime(minutes_per_week)'])
plt.show()
plt.figure(figsize=(25,20))
sns.factorplot(data=add,x='netgain',y='ratings',hue='genre')
plt.show()
# Daytime ads are run more amount of time compared to the other airtimes. 

sns.catplot(x='airtime', y='average_runtime(minutes_per_week)', data=add, kind='boxen', aspect=2)
plt.title('Boxen Plot', weight='bold', fontsize=16)
plt.show()
# Ads from pharma industry are aired more compared to others.

plt.figure(figsize=(200,400))
sns.factorplot(data=add,x='industry',y='average_runtime(minutes_per_week)')
plt.title('Factor Plot', weight='bold', fontsize=16)
plt.show()
# Splitting the independent and target variables.
x = add.iloc[:, :-1]
y = add.iloc[:, -1]
# convert categorical variable to numerical
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)
x = x.apply(LabelEncoder().fit_transform)
# print the size of x and y
y.shape, x.shape
# train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

print('X train size: ', x_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', x_test.shape)
print('y test size: ', y_test.shape)
# Decision tree classifier with grid seacrh CV and model evaluation using accuracy score, precision score and AUC/ROC curve.

parm = {'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 'max_depth': [2,3,4,5,6,7,8,9],'min_samples_leaf':[1,10,15,20,25,100],'random_state':[14]}

dtc_grid = GridSearchCV(DecisionTreeClassifier(), parm, cv=5, scoring='roc_auc')
dtc_grid.fit(x_train, y_train)
print('The best parameters are: ', dtc_grid.best_params_)
print('best mean cross-validated score (auc) : ', dtc_grid.best_score_)
y_pred = dtc_grid.predict(x_test)
precision_score_DT_test =  precision_score(y_test, y_pred)
accuracy_score_DT_test = accuracy_score(y_test, y_pred)
print('The precision score of decision tree on TEST is : ',round(precision_score_DT_test * 100,2), '%')
print('The accuracy score of decision tree on TEST is : ',round(accuracy_score_DT_test * 100,2), '%')
adsu = dtc_grid.predict_proba(x_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr, label="ROC Curve")
x = np.linspace(0,1,num=50)
plt.plot(x,x,linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1,4)
plt.ylim(0,1,4)
plt.show()

AUC_DT = auc(fpr,tpr)
print('DT AUC is: ', round(AUC_DT * 100,2), '%')
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier

param = {'n_estimators':[700],'n_jobs':[-1], 'max_features': [0.5,0.7,0.9], 'max_depth': [3,5,7],'min_samples_leaf':[1,10],'random_state':[14]}

rfc_grid = GridSearchCV(RandomForestClassifier(), param, cv=5, scoring='roc_auc')
rfc_grid.fit(x_train, y_train)
print('The best parameters are: ', rfc_grid.best_params_)
print('best mean cross-validated score (auc) : ', rfc_grid.best_score_)
y_pred = rfc_grid.predict(x_test)
precision_score_RF_test =  precision_score(y_test, y_pred)
accuracy_score_RF_test = accuracy_score(y_test, y_pred)
print('The precision score on TEST is : ',round(precision_score_RF_test * 100,2), '%')
print('The accuracy score on TEST is : ',round(accuracy_score_RF_test * 100,2), '%')
# Now let's plot the ROC curve and calculate AUC on the test set

adsu = rfc_grid.predict_proba(x_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr, label='ROC Curve')
x = np.linspace(0,1,num=50)
plt.plot(x,x,linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

AUC_RF = auc(fpr,tpr)
print('RF AUC is: ', round(AUC_RF * 100,2), '%')
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_neighbors=3)

knc_bc = BaggingClassifier(base_estimator=knc, n_estimators=30, max_samples=0.8, max_features=0.8)

knc_bc.fit(x_train, y_train)
y_pred = knc_bc.predict(x_test)
precision_score_RF_test =  precision_score(y_test, y_pred)
accuracy_score_RF_test = accuracy_score(y_test, y_pred)
print('The precision score  on TEST is : ',round(precision_score_RF_test * 100,2), '%')
print('The accuracy score  on TEST is : ',round(accuracy_score_RF_test * 100,2), '%')
# Now let's plot the ROC curve and calculate AUC on the test set

adsu = rfc_grid.predict_proba(x_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr, label='ROC Curve')
x = np.linspace(0,1,num=50)
plt.plot(x,x,linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.show()

AUC_RF = auc(fpr,tpr)
print('RF AUC is: ', round(AUC_RF * 100,2), '%')
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

rfClf = RandomForestClassifier(n_estimators=500, random_state=0) # 500 trees. 
svmClf = SVC(probability=True, random_state=0) # probability calculation
logClf = LogisticRegression(random_state=0)

vc = VotingClassifier(estimators= [('rf',rfClf), ('svm',svmClf), ('log', logClf)], voting='soft')

vc.fit(x_train, y_train)
y_pred = vc.predict(x_test)
precision_score_RF_test =  precision_score(y_test, y_pred)
accuracy_score_RF_test = accuracy_score(y_test, y_pred)
print('The precision score  on TEST is : ',round(precision_score_RF_test * 100,2), '%')
print('The accuracy score  on TEST is : ',round(accuracy_score_RF_test * 100,2), '%')
# Now let's plot the ROC curve and calculate AUC on the test set

adsu = vc.predict_proba(x_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr, label='ROC Curve')
x = np.linspace(0,1,num=50)
plt.plot(x,x,linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.show()

AUC_RF = auc(fpr,tpr)
print('RF AUC is: ', round(AUC_RF * 100,2), '%')
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))