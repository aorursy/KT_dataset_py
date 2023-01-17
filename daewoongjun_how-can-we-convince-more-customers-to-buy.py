# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
    
    

data = pd.read_csv("../input/online-shoppers-intention/online_shoppers_intention.csv")
data.head(10)
data.describe()
null_table = pd.DataFrame(data.isnull().sum().values.reshape(1,-1), columns = data.isnull().sum().index)
null_table = null_table.rename(index = {0:'Total Null Values'})
null_pct = null_table.iloc[0,:]/12330 *100
null_pct = pd.DataFrame(null_pct.values.reshape(1,-1), columns = null_pct.index)
null_pct = null_pct.rename(index = {0 : 'Null %'})
null_table = null_table.append(null_pct)
null_table
# Cleaning
data = data.dropna()
data
neg_ad_dur = data[data['Administrative_Duration'] < 0]
neg_info_dur = data[data['Informational_Duration'] < 0]
neg_prd_dur = data[data['ProductRelated_Duration'] < 0]
print(" The length of each durations are {} ,{} , {}".format(len(neg_ad_dur), len(neg_info_dur),len(neg_prd_dur)))
# Dropping the negative Durations
data = data.drop(data[data['Administrative_Duration'] < 0].index)
data = data.drop(data[data['Informational_Duration'] < 0].index)
data = data.drop(data[data['ProductRelated_Duration'] < 0].index)
#Checking , no negative values
data.describe()
# Countplots on 
plt.style.use('fivethirtyeight')
fig,ax = plt.subplots(nrows = 2, ncols = 4,figsize = (17,7))
fig.tight_layout()

#fig.suptitle('Countplots Of Some Features')    Main TItle
ax[0,0].bar(data['Administrative'].value_counts().index,data['Administrative'].value_counts().values,color = 'red')
ax[0,0].set_title('Administrative',size=13)
ax[0,0].set_xlim(0,20)

ax[0,1].bar(data['Informational'].value_counts().index,data['Informational'].value_counts().values,color = 'blue')
ax[0,1].set_title('Informational',size=13)
ax[0,1].set_xlim(0,10)

ax[0,2].bar(data['ProductRelated'].value_counts().index,data['ProductRelated'].value_counts().values,color = 'purple')
ax[0,2].set_title('ProductRelated',size=13)
ax[0,2].set_xlim(0,200)

ax[0,3].bar(data['OperatingSystems'].value_counts().index,data['OperatingSystems'].value_counts().values,color = 'yellow')
ax[0,3].set_title('OperatingSystems',size=13)

ax[1,0].bar(data['Browser'].value_counts().index,data['Browser'].value_counts().values,color = 'orange')
ax[1,0].set_title('Browser',size=13)

ax[1,1].bar(data['Region'].value_counts().index,data['Region'].value_counts().values,color = 'black')
ax[1,1].set_title('Region',size=13)


ax[1,2].bar(data['TrafficType'].value_counts().index,data['TrafficType'].value_counts().values,color = 'navy')
ax[1,2].set_title('TrafficType',size=13)

fig.delaxes(ax[1,3])   # since it is a odd number plots, delete last subplot
# Lets see other categorical features with pie chart
plt.style.use('ggplot')
fig,ax = plt.subplots(3,2,figsize=(10,6))
fig.set_figheight(12)
fig.set_figwidth(12)
plt.tight_layout()
# Revenue
ax[0,0].pie(data['Revenue'].value_counts().values,labels = ['False','True'],explode = [0.1,0.1],shadow = True,autopct='%1.1f%%')
ax[0,0].set_title('Ratio of Revenue Availability')


# Month() No Jan and April
ax[0,1].pie(data['Month'].value_counts().values,labels= data['Month'].value_counts().index[:],shadow = True, explode = [0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.3],autopct='%1.0f%%')
ax[0,1].set_title('Ratio of Each Month')

# Visitor Types
ax[1,0].pie(data['VisitorType'].value_counts().values,labels= data['VisitorType'].value_counts().index[:],explode = [0.1,0.1,0.3],shadow = True,autopct='%1.1f%%',colors=['green','red','blue'])
ax[1,0].set_title('Ratio of Visitor Types')

# Weekend
ax[1,1].pie(data['Weekend'].value_counts().values,labels= data['Weekend'].value_counts().index[:],explode = [0.1,0.1],shadow = True,autopct='%1.1f%%')
ax[1,1].set_title('Ratio of Weekends')

# Special Days
ax[2,0].pie(data['SpecialDay'].value_counts().values,labels= data['SpecialDay'].value_counts().index[:],explode = [0.1,0.2,0.2,0.3,0.3,0.4],shadow = True,autopct='%1.0f%%')
ax[2,0].set_title('Ratio of Special Days')

fig.delaxes(ax[2,1])
# Lets see the Ratio of Revenue in each types
plt.style.use('fivethirtyeight')
fig,ax = plt.subplots(nrows = 2, ncols = 4,figsize = (17,10))
fig.tight_layout(pad = 3)


adm_rev = data[['Administrative','Revenue']]
rev_p1 = pd.DataFrame(data.groupby('Revenue')['Administrative'].sum()).T
rev_p1.plot.bar(stacked=True,ax=ax[0,0])
ax[0,0].set_xticklabels(['Administrative'], rotation=360)
plt.legend(loc='best')


info_rev = data[['Informational','Revenue']]
rev_p2 = pd.DataFrame(data.groupby('Revenue')['Informational'].sum()).T
rev_p2.plot.bar(stacked=True,ax = ax[0,1],color = ['black','white'])
ax[0,1].set_xticklabels(['Informational'], rotation=360)
plt.legend(loc='best')

info_rev = data[['ProductRelated','Revenue']]
rev_p2 = pd.DataFrame(data.groupby('Revenue')['ProductRelated'].sum()).T
rev_p2.plot.bar(stacked=True,ax = ax[0,2],color = ['purple','red'])
ax[0,2].set_xticklabels(['ProductRelated'], rotation=360)
plt.legend(loc='best')



info_rev = data[['OperatingSystems','Revenue']]
rev_p2 = pd.DataFrame(data.groupby('Revenue')['OperatingSystems'].sum()).T
rev_p2.plot.bar(stacked=True,ax = ax[0,3],color = ['green','yellow'])
ax[0,3].set_xticklabels(['OperatingSystems'], rotation=360)


info_rev = data[['Browser','Revenue']]
rev_p2 = pd.DataFrame(data.groupby('Revenue')['Browser'].sum()).T
rev_p2.plot.bar(stacked=True,ax = ax[1,0],color = ['navy','orange'])
ax[1,0].set_xticklabels(['Browser'], rotation=360)


info_rev = data[['Region','Revenue']]
rev_p2 = pd.DataFrame(data.groupby('Revenue')['Region'].sum()).T
rev_p2.plot.bar(stacked=True,ax = ax[1,1],color = ['black','red'])
ax[1,1].set_xticklabels(['Region'], rotation=360)


info_rev = data[['TrafficType','Revenue']]
rev_p2 = pd.DataFrame(data.groupby('Revenue')['TrafficType'].sum()).T
rev_p2.plot.bar(stacked=True,ax = ax[1,2],color = ['blue','green'])
ax[1,2].set_xticklabels(['TrafficType'], rotation=360)

fig.delaxes(ax[1,3])
# Lets see the Ratio of Revenue in each categorical features
fig, ax = plt.subplots(2,2,figsize = (14,6))
plt.tight_layout(pad= 3)

month_rev = data[['Month','Revenue']]
vis_rev = data[['VisitorType','Revenue']]
weekends_rev = data[['Weekend','Revenue']]
spd_rev = data[['SpecialDay','Revenue']]

sns.countplot(x = month_rev['Month'],hue = month_rev['Revenue'],ax =ax[0,0]).set_title('Ratio of Revenue Of Each Months')
sns.countplot(x = vis_rev['VisitorType'],hue = vis_rev['Revenue'],ax =ax[0,1]).set_title('Ratio of Revenue Of Each Visitor Types')
sns.countplot(x = weekends_rev['Weekend'],hue = weekends_rev['Revenue'],ax =ax[1,0]).set_title('Ratio of Revenue In Weekdays Or Weekends')
sns.countplot(x = spd_rev['SpecialDay'],hue = spd_rev['Revenue'],ax =ax[1,1]).set_title('Ratio of Revenue Of Each Special Days')
# lETS SEE SOME DATA RELATED TO THE BOUNCE AND EXIT RATE
sns.boxplot(x=data['Revenue'],y=data['BounceRates']).set_title('Distribution of BounceRates Depending on Availability of Making Revenue')

sns.boxplot(x=data['Revenue'],y=data['ExitRates']).set_title('Distribution of ExitRates Depending on Availability of Making Revenue')

sns.boxenplot(x=data['Revenue'],y=data['PageValues']).set_title('Page Values depending on Availability of Making Revenue')
sns.scatterplot(data['BounceRates'],data['ExitRates'],hue = data['VisitorType'],palette='Set3_r').set_title('ExitRates VS BounceRates')
# Exit/Bounce Rates VS Page Values, Is higher the page value less Exit Rates?

sns.scatterplot(x= data['PageValues'],y=data['ExitRates'],hue = data['Revenue']).set_title('Will the Exit Rates descrease as the Page Values Increase?')
sns.scatterplot(x= data['PageValues'],y=data['BounceRates'],hue = data['Revenue']).set_title('Will the Bounce Rates descreases as the Page Value Increases')
sns.scatterplot(x=data['Administrative_Duration'],y=data['PageValues'],hue = data['Revenue']).set_title('Page Values VS Administrative_Duration ')
sns.scatterplot(x=data['Informational_Duration'],y=data['PageValues'],hue = data['Revenue']).set_title('Page Values VS Informational_Duration ')
sns.scatterplot(x=data['ProductRelated_Duration'],y=data['PageValues'],hue = data['Revenue']).set_title('Page Values VS ProductRelated_Duration ')
# Durations
admin_df = pd.DataFrame(data['Administrative_Duration'])
admin_df.rename(columns={'Administrative_Duration' :'Duration'},inplace = True)
admin_df['type'] = admin_df['Duration'].apply(lambda x:'Administrative')

inform_df= pd.DataFrame(data['Informational_Duration'])
inform_df.rename(columns={'Informational_Duration' :'Duration'},inplace =True)
inform_df['type'] = inform_df['Duration'].apply(lambda x:'Informational')

prod_df= pd.DataFrame(data['ProductRelated_Duration'])
prod_df.rename(columns={'ProductRelated_Duration' :'Duration'},inplace =True)
prod_df['type'] = prod_df['Duration'].apply(lambda x:'ProductRelated')



dur_df = pd.concat([admin_df,inform_df,prod_df])
sns.boxenplot(dur_df['type'], dur_df['Duration']).set_title('Each Types of Durations Distribution')


# Wow, the gap is too big, so lets see them 
dur_df12 = pd.concat([admin_df,inform_df])
#plt.ylim(0,500)
sns.boxplot(dur_df12['type'],dur_df12['Duration']).set_title('Administrative and Informational Durations Distribution')
pd.DataFrame(data['Administrative_Duration'].describe())
pd.DataFrame(data['Informational_Duration'].describe())
# outliers of each columns:
# Administrative :  High : 93.95 + 93.95 (IQR)
#                   Low  : 0- 93.95
# Outliers of Administrative is larger than 187.9 and lower than - 93.95
ad_zero = data[data['Administrative_Duration'] == 0]
outlier_len = len(data[data['Administrative_Duration'] > 187.9]) 
outlier_prt = outlier_len / len(data['Administrative_Duration']) *100    # 12%
print("The % of Outliers of Administrative Duration Columns are ",outlier_prt)
ad_zero_prt = len(ad_zero)/len(data['Administrative_Duration']) *100     # 47%
print("The % of 0 Values of Administrative Duration Columns are ",ad_zero_prt)

# Informational :  High : 0
#                   Low  : 0
# Outliers of Administrative is larger than 187.9 and lower than - 93.95
info_zero = data[data['Informational_Duration'] == 0]
outlier_len2 = len(data[data['Informational_Duration'] > 0]) 
outlier_prt2 = outlier_len2 / len(data['Informational_Duration']) *100    
print("The % of Outliers of Informational Duration Column is ",outlier_prt2)
info_zero_prt = len(info_zero)/len(data['Informational_Duration']) *100     
print("The % of 0 Values of Informational Duration Column is ",info_zero_prt)


prddur_rev = data.groupby('Revenue')['ProductRelated_Duration'].mean()
prddur_rev = pd.DataFrame(prddur_rev)
prddur_rev
# WHich MOnth will have the most transactions near the SpecialDays ( =0 if close to the special days)?
spcday_0 = data[data['SpecialDay'] == 0]
data_month = spcday_0.groupby('Month')['SpecialDay'].count()
plt.figure(figsize=(13,6))
plt.title('Number of Special Days In Each Months')
plt.pie(data_month.values,labels = data_month.index,shadow =True, explode=[0.2,0.2,0.2,0.1,0.1,0.12,0.13,0.1,0.3,0.3],autopct='%1.0f%%')
fig , ax = plt.subplots(2,3,figsize = (14,7))
plt.tight_layout(pad = 2)
sns.boxplot(data['Month'],data['ExitRates'],ax=ax[0,0])
sns.violinplot(data['Month'],data['BounceRates'],ax=ax[0,1])
sns.violinplot(data['Month'],data['PageValues'],ax=ax[0,2])
sns.boxenplot(data['Month'],data['Administrative_Duration'],ax = ax[1,0])
sns.boxenplot(data['Month'],data['Informational_Duration'],ax = ax[1,1])
sns.boxenplot(data['Month'],data['ProductRelated_Duration'],ax = ax[1,2])
# Customer Types
fig , ax = plt.subplots(1,3,figsize = (20,5))
ax.flatten()
plt.tight_layout(pad =3)


sns.violinplot(data['VisitorType'],data['ExitRates'],ax = ax[0],hue = data['Revenue'])

sns.violinplot(data['VisitorType'],data['BounceRates'],ax = ax[1],hue = data['Revenue'])

sns.violinplot(data['VisitorType'],data['PageValues'],ax = ax[2],hue = data['Revenue'])

fig , ax = plt.subplots(1,4,figsize = (20,5))
ax.flatten()
plt.tight_layout(pad =3)

sns.violinplot(data['OperatingSystems'],data['ExitRates'],ax = ax[0],split=True,hue = data['Revenue'])
sns.violinplot(data['Browser'],data['ExitRates'],ax = ax[1],split=True,hue = data['Revenue'])
sns.violinplot(data['Region'],data['ExitRates'],ax = ax[2],split=True,hue = data['Revenue'])
sns.violinplot(data['TrafficType'],data['ExitRates'],ax = ax[3],split=True,hue = data['Revenue'])

fig , ax = plt.subplots(1,4,figsize = (20,5))
ax.flatten()
plt.tight_layout(pad =3)

sns.violinplot(data['OperatingSystems'],data['BounceRates'],ax = ax[0],split=True,hue = data['Revenue'])
sns.violinplot(data['Browser'],data['BounceRates'],ax = ax[1],split=True,hue = data['Revenue'])
sns.violinplot(data['Region'],data['BounceRates'],ax = ax[2],split=True,hue = data['Revenue'])
sns.violinplot(data['TrafficType'],data['BounceRates'],ax = ax[3],split=True,hue = data['Revenue'])

fig , ax = plt.subplots(1,4,figsize = (20,5))
ax.flatten()
plt.tight_layout(pad =3)

sns.boxenplot(data['OperatingSystems'],data['PageValues'],ax = ax[0],hue = data['Revenue'])
sns.boxenplot(data['Browser'],data['PageValues'],ax = ax[1],hue = data['Revenue'])
sns.boxenplot(data['Region'],data['PageValues'],ax = ax[2],hue = data['Revenue'])
sns.boxenplot(data['TrafficType'],data['PageValues'],ax = ax[3],hue = data['Revenue'])
# Correlation with Revenue
data_corr = data.corr()['Revenue'] 
sns.barplot(data_corr[0:-1].index,data_corr[0:-1].values).set_title('Correlation with the Revenue')
plt.xticks(rotation = 90)
plt.figure(figsize=(15,5))
ax = sns.heatmap(data.corr(),cmap='Blues',annot=True)
ax.set_title('The Correlation Heatmap')
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+1,top-1)

# Lets Cluster the Group 
X = data[['ProductRelated_Duration','BounceRates']]
# THe Elbow Method

inertia =[]
for i in range(1,11):
    kms = KMeans(n_clusters=i,max_iter = 100 , n_init = 10,init = 'k-means++',random_state=100).fit(X)
    inertia.append(kms.inertia_)
plt.plot(range(1,11),inertia,marker ='X')
# 3 Cluseters
kms = KMeans(n_clusters=3,max_iter = 100 , n_init = 10,init = 'k-means++',random_state=100).fit(X)
pred = kms.fit_predict(X)
#plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=pred, s=50, cmap='viridis')
plt.scatter(X.iloc[pred == 0,0],X.iloc[pred == 0,1],label='First',color ='green')
plt.scatter(X.iloc[pred == 1,0],X.iloc[pred == 1,1],label ='Second',color='blue')
plt.scatter(X.iloc[pred == 2,0],X.iloc[pred == 2,1],label='Third',color='yellow')
centers = kms.cluster_centers_
plt.scatter(centers[:, 0],centers[:,1],s=100,color='red',label='Centroids')

plt.legend(loc='best')
print(data[pred == 0]['Revenue'].value_counts())
print("The Percentage of first cluster groups who made a transactions is : " ,1341/8758*100,"%")
print(data[pred == 1]['Revenue'].value_counts())
print("The Percentage of second cluster groups who made a transactions is : " ,71/139*100,"%")
print(data[pred == 2]['Revenue'].value_counts())
print("The Percentage of third cluster groups who made a transactions is : " ,496/1478*100,"%")
# Administrative The Elbow Method
inertia = []
Y = data[['Administrative_Duration','BounceRates']]
for i in range(1,11):
    kms = KMeans(n_clusters = i,max_iter=500,n_init = 10,init = 'k-means++',random_state = 100).fit(Y)
    inertia.append(kms.inertia_)
plt.plot(range(1,11),inertia,marker='X')
#3 Clusters

kms = KMeans(n_clusters = 3,max_iter=500,n_init = 10,init = 'k-means++',random_state = 100).fit(Y)
pred_a = kms.predict(Y)
#plt.scatter(x=Y.iloc[:,0],y=Y.iloc[:,1],c=pred_a,cmap='viridis')
plt.scatter(Y.iloc[pred_a == 0,0],Y.iloc[pred_a == 0,1],label='First',color ='green')
plt.scatter(Y.iloc[pred_a == 1,0],Y.iloc[pred_a == 1,1],label ='Second',color='blue')
plt.scatter(Y.iloc[pred_a == 2,0],Y.iloc[pred_a == 2,1],label='Third',color='yellow')
centers = kms.cluster_centers_
plt.scatter(x=kms.cluster_centers_[:,0],y=kms.cluster_centers_[:,1],s=100,label='Centroids',color = 'red')
plt.legend(loc ='best')
#first group
print(data[pred_a == 0]['Revenue'].value_counts())
print("The Percentage of first cluster groups who made a transactions is : " ,1493/9062*100,"%")
# second group
print(data[pred_a == 1]['Revenue'].value_counts())
print("The Percentage of first cluster groups who made a transactions is : " ,382/1193*100,"%")
# third group
print(data[pred_a == 2]['Revenue'].value_counts())
print("The Percentage of first cluster groups who made a transactions is : " ,33/120*100,"%")
# Informational The Elbow Method
inertia = []
Z= data[['Informational_Duration','BounceRates']]
for i in range(1,11):
    kms = KMeans(n_clusters = i,max_iter=500,n_init = 10,init = 'k-means++',random_state = 100).fit(Z)
    inertia.append(kms.inertia_)
plt.plot(range(1,11),inertia,marker='X')
#3 Clusters

kms = KMeans(n_clusters = 3,max_iter=500,n_init = 10,init = 'k-means++',random_state = 100).fit(Z)
pred_i = kms.predict(Z)
#plt.scatter(x=Z.iloc[:,0],y=Z.iloc[:,1],c=pred_i,cmap='viridis')
plt.scatter(Z.iloc[pred_i == 0,0],Z.iloc[pred_i == 0,1],label='First',color ='green')
plt.scatter(Z.iloc[pred_i == 1,0],Z.iloc[pred_i == 1,1],label ='Second',color='blue')
plt.scatter(Z.iloc[pred_i == 2,0],Z.iloc[pred_i == 2,1],label='Third',color='yellow')
centers = kms.cluster_centers_
plt.scatter(x=kms.cluster_centers_[:,0],y=kms.cluster_centers_[:,1],s=100,label='Centroids',color = 'red')
plt.legend(loc ='best')
#first group
print(data[pred_i == 0]['Revenue'].value_counts())
print("The Percentage of first cluster groups who made a transactions is : " ,1719/9896*100,"%")
# second group
print(data[pred_i == 1]['Revenue'].value_counts())
print("The Percentage of first cluster groups who made a transactions is : " ,158/394*100,"%")

# third group
print(data[pred_i == 2]['Revenue'].value_counts())
print("The Percentage of first cluster groups who made a transactions is : " ,31/85*100,"%")
data_true = data[data['Revenue'] == True][['ExitRates','BounceRates','Administrative_Duration','Informational_Duration','ProductRelated_Duration']]
data_true.describe()
data_true = data[data['Revenue'] == False][['ExitRates','BounceRates','Administrative_Duration','Informational_Duration','ProductRelated_Duration']]
data_true.describe()
le = LabelEncoder()
data['Month'] = le.fit_transform(data['Month'])
data['VisitorType'] = le.fit_transform(data['VisitorType'])
data['Weekend'] = le.fit_transform(data['Weekend'])
data['Revenue'] = le.fit_transform(data['Revenue'])
data
X = data.drop('Revenue',axis=1)
y = data['Revenue']

# Lets make a classification model to classify whether the customer will make a transaction or not
# RandomClassification Classifier


X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.3,random_state = 101)
rfclf = RandomForestClassifier(n_estimators = 30,max_depth = 10,random_state = 101)


rfclf.fit(X_train,y_train)
pred = rfclf.predict(X_test)
print(classification_report(y_test,pred))
# Lets Optimize the Random Forest Classifier using GridSearch

param_grid = {
    'n_estimators' : [80,100,120,150],
    'max_depth' : [7,10,15,20],
    'min_samples_leaf' : [1,2,3,4],
    'min_samples_split': [2,4,6,8]
}

gridsearch = GridSearchCV(estimator=rfclf,param_grid=param_grid,verbose = 1)
gridsearch.fit(X_train,y_train)

gridsearch.best_params_
rfclf = RandomForestClassifier(n_estimators = 150,max_depth = 7,min_samples_leaf = 4, min_samples_split = 2,random_state = 101)
rfclf.fit(X_train,y_train)
pred = rfclf.predict(X_test)
print(classification_report(y_test,pred))

# 0 is False, 1 is True, the precision of detecting True has increased
from sklearn.metrics import accuracy_score
rfacc = accuracy_score(y_test,pred)
ax = sns.heatmap(confusion_matrix(y_test,pred),annot = True,cmap='Blues')
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
plt.title('Confusion Matrix of RandomForestClassifier')
#Logistic Regression

lregression = LogisticRegression(max_iter = 100)
lregression.fit(X_train,y_train)
pred_i = lregression.predict(X_test)
print(classification_report(y_test,pred_i))
logregacc = accuracy_score(y_test,pred_i)
ax = sns.heatmap(confusion_matrix(y_test,pred_i),annot = True,cmap='YlGnBu')
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
plt.title('Confusion Matrix of Logistic Regression')
# SVM 


param_grid = {'C':[0.1,1,10,100,1000],
              'kernel':['rbf'],
              'gamma' : [0.1,1,10,100,1000]}
gridsearch = GridSearchCV(SVC(),param_grid = param_grid,verbose = 1)
gridsearch.fit(X_train,y_train)

svm = SVC(C=100,kernel = 'rbf',gamma = 100)
svm.fit(X_train,y_train)
pred_s = svm.predict(X_test)

print(classification_report(y_test,pred_s))
svmacc = accuracy_score(y_test,pred_s)

ax = sns.heatmap(confusion_matrix(y_test,pred_s),annot = True,cmap='Blues_r')
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
plt.title('Confusion Matrix of SupportVectorMachine')
# KNN

error_rate = []
for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_k = knn.predict(X_test)
    error_rate.append(np.mean(pred_k!= y_test))
plt.plot(range(1,15),error_rate,marker='X',linestyle='dashed',markerfacecolor='red', markersize=10)
knn = KNeighborsClassifier(n_neighbors = 6 )
knn.fit(X_train,y_train)
pred_k = knn.predict(X_test)
print(classification_report(y_test,pred_k))
knnacc = accuracy_score(y_test,pred_k)
ax = sns.heatmap(confusion_matrix(y_test,pred_k),annot = True,cmap='Blues')
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
ax.set_ylim(bottom+0.5,top-0.5)
plt.title('Confusion Matrix of Logistic Regression')
accuracy_df = pd.DataFrame.from_dict({'Accuracy_Score' : [rfacc,logregacc,svmacc,knnacc] }) 

accuracy_df.rename(index = {0:'RandomForestClassifier',1:'Logistic Regression', 2: 'SVM',3:'KNN'})
