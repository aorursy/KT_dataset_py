#IMPORT THE REQUIRED LIBRARIES 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

%matplotlib inline
%pylab inline
#Read in all three datasets

recpies = pd.read_csv('../epi_r.csv')
bikes = pd.read_csv('../nyc-east-river-bicycle-counts.csv')
weather = pd.read_csv('../weatherHistory.csv')
#We will first analyze the recpies dataset
recpies.head(5)
plt.scatter(x=recpies['calories'],y=recpies['dessert'])
recpies['calories'].describe()
print(' Max Value in Calories : ',recpies['calories'].max())
print(' Min Value in Calories : ',recpies['calories'].min())
print(' Mean Value in Calories :', recpies['calories'].mean())
print('Total Records Count :',len(recpies['calories']))

print(' Count of Records with Calories >10000 :',len(recpies[recpies['calories']>1000]))
print(' Count of Records with Calories >10000 :',len(recpies[pd.notnull(recpies['calories'])]))

plt.figure(1)
plt.subplot(211)
plt.hist(recpies[['calories']])

plt.subplot(212)
plt.plot(recpies[['calories']])
plt.show()
# Removing the NA Value holding records ( in calories column )
oldLength = len(recpies)
recpies = recpies[pd.notnull(recpies['calories'])]
newLength = len(recpies)

print('Removed ',(oldLength-newLength),' No. of Records From the Data Set')
print('Total No. of Records in the Revised Data Set :',len(recpies))
# Removing the Records having Calory value more than 10000
oldLength = len(recpies)

print('No. of Records Having the Calory value More than 10000 : ',len(recpies[recpies['calories']>10000]))
display('\nPrint the Outliers : ',recpies[['calories']][recpies['calories']>10000])
# Removing the outlier Records

recpies = recpies[recpies['calories']<=10000]
print(' Current Length of the Data Set : ',len(recpies))

plt.figure(1)
plt.subplot(211)
plt.hist(recpies[['calories']])

plt.subplot(212)
plt.plot(recpies[['calories']])
plt.show()
print(' Data Type of calories : ',recpies['calories'].dtype)
plt.scatter(x=recpies['calories'],y=recpies['dessert'])
from sklearn import linear_model
from statsmodels.formula.api import ols
import statsmodels.api as sm

linear_model = ols('dessert ~ calories',data = recpies).fit()
linear_model.summary()

import seaborn as sns
import statsmodels.api as sm

fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(linear_model, 'calories',fig = fig)

#BELOW GIVEN GRAPHS are for Continuous Type - Just for Show casing purpose
logit = sm.Logit(recpies['dessert'],recpies['calories'])
result = logit.fit()

fig = plt.figure(figsize=(15,8))


#BELOW GIVEN GRAPHS are for Continuous Type - Just for Show casing purpose
sns.countplot(x='dessert',data=recpies)
plt.show()
#FEATURE SELECTION - JUST TO DEMONSTRATE
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg,18)


rfe = rfe.fit(recpies[['dessert']],recpies['calories'])

print(rfe.support_)
print(rfe.ranking_)
#IMPLEMENTING THE MODEL

import statsmodels.formula.api as sm

model = sm.Logit(recpies['dessert'],recpies['calories'])
result = model.fit()

result.summary()
#MODEL FITTING

from sklearn.cross_validation import train_test_split

# GENERATE TRAINING AND TEST DATA SETS FROM THE GIVEN DATA SETS 

X_train, X_test, Y_train, Y_test = train_test_split(recpies[['calories']],
                                                    recpies['dessert'],test_size=0.3,random_state=0)
print("Length of Training Data Set :",len(X_train))
print("Length of Testing Data Set   :",len(X_test))
print("Length of Total Data Set    :",len(recpies))
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
#PREDICTING THE TEST RESULTS AND CALCULATING THE ACCURACY
Y_pred = logreg.predict(X_test)
print(' Accuracy of Logistic Regression Classifier on test set : {:.2f}',format(logreg.score(X_test,Y_test)))
#CROSS VALIDATION
#This attempts to avoid overfitting

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10,random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV,X_train,Y_train,cv=kfold,scoring=scoring)
print("10-fold corss validation average accuracy : %.3f"%results.mean())
#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
# COMPUTE PRECISION, RECALL, F - MEASURE AND SUPPORT
from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))
#PL IGNORE - JUST FOR REFERENCE 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(Y_test,logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)'%logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(loc = "lower right")
plt.show()
sns.set(style='darkgrid')
g = sns.regplot(x='calories',y='dessert',data = recpies,logistic=True)
g.figure.set_size_inches(8,8)
sns.set(style='darkgrid')
g = sns.regplot(x='calories',y='dessert',data = recpies)
g.figure.set_size_inches(8,8)
#ANALYZING THE BIKE DATASET

bikes.head(10)
summ = bikes['Total'].sum()
countt = bikes['Total'].count()
average = summ  /  countt
print(' average      : ',average)
print(' bikes.mean() : ',bikes['Total'].mean())
print('\nAs per the given dataset ',average,' No. of cycles cross the Manhattan per day\n')
plt.scatter(y = bikes['Total'],x = bikes['Low Temp (°F)'],marker='.')
sns.pairplot(bikes,vars=['Total','Low Temp (°F)','High Temp (°F)'],markers='.',size=5)
print(' We could see high temperature and low temperature, are linearly corelated with the Total cycle count')
bridges = ['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge']

highLoadHavingTopBridge = ""
highLoad = 0

for bridge in bridges:
    if bikes[bridge].mean()>highLoad:
        highLoadHavingTopBridge = bridge
        highLoad = bikes[bridge].mean()
        
print(' High Load Having Top Bridge :',highLoadHavingTopBridge)
print(' High Load                   :',highLoad)
    
sns.regplot(x=bikes['High Temp (°F)'],y=bikes['Total'],label='high')
sns.regplot(x=bikes['Low Temp (°F)'],y=bikes['Total'],label='low')
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(bikes['Low Temp (°F)'], bikes['High Temp (°F)'], bikes[['Total']], rstride=1, cstride=1, cmap='hot')
weather.head(3)
from subprocess import check_output
import scipy.stats
import datetime
from pylab import rcParams
weather.info()
#Categorical Variables
categorical = weather.select_dtypes(include=['object']).keys()
print(categorical)
#Quantitative Variables
quantitative = weather.select_dtypes(include=['int64','float64']).keys()
print(quantitative)
weather['Date'] = pd.to_datetime(weather['Formatted Date'])
weather['year'] = weather['Date'].dt.year
weather['month']= weather['Date'].dt.month
weather['day']= weather['Date'].dt.day
weather['hour']= weather['Date'].dt.hour
weather.head(3)
weather.info()
weather[quantitative].describe()
weather.shape
rcParams['figure.figsize']=15,15
weather[quantitative].hist()
#Since Loud Cover Takes Value ' 0 ' - we drop it
#weather = weather.drop('Loud Cover',axis=1)
#quantitative.drop('Loud Cover')
sns.pairplot(weather[quantitative])
weather = weather.drop('Loud Cover',axis=1)
#Pressure (millibars)
#Some observations are zero.
pressure_median = weather['Pressure (millibars)'].median()
print(pressure_median)
def pressuree(x):
    if x<200:
        return pressure_median
    else:
        return x

weather['Pressure (millibars)'] = weather.apply( lambda row : pressuree(row['Pressure (millibars)']),axis = 1)

rcParams['figure.figsize'] = 5,3
weather['Pressure (millibars)'].hist()


rcParams['figure.figsize']=8,5
sns.countplot(y = weather['Summary'])
len(weather['Summary'].unique())
summaryValues = weather['Summary'].unique()
print(summaryValues)
print("\n")

summary_freq = pd.crosstab(index = weather['Summary'],columns='count')
summary_freq_rel = summary_freq / summary_freq.sum()
summary_freq_rel.sort_values('count',ascending=False)
def cloud_categorizer(row):
    row = str(row).lower()
    category=""
    if "foggy" in row:
        category=5
    elif "overcast" in row:
        category=4
    elif "mostly cloudy" in row:
        category = 3
    elif "partly cloudly" in row:
        category = 2
    elif "clear" in row:
        category = 1
    else:
        category = 0
        
    return category

weather['cloud (summary)'] = weather.apply(lambda row : cloud_categorizer(row['Summary']),axis=1)
weather.head(3)
rcParams['figure.figsize'] = 5,5
sns.countplot(weather['cloud (summary)'])
sns.boxplot(x=weather['cloud (summary)'],y=weather['Visibility (km)'])
