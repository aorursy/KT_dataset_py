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



data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#LabelEncoder for transforming the categorical data.

from sklearn.preprocessing import LabelEncoder



#Train test split

from sklearn.model_selection import train_test_split

#Confusion matrix and ROC/AUC for comparing the models

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc

#Models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
#Checking whats the shape of our data

data.shape
#Checking the general info of our data

data.info()
# Checking if we have missing data in the dataframe

data.isnull().sum()
#Dropping agent and company columns and filling missing data for children and country with the mode of those columns

data=data.drop(['agent','company'],axis=1)

data.country=data.country.fillna(data.country.mode()[0])

data.children=data.children.fillna(data.children.mean())
#Checking if we have more missing data:

data.isnull().sum()
#Getting a grasp of what the data contains and start planning what questions we want to answer in our exploratory data analysis

data.head()
# Finding out the unique values for the categorical variable in the dataset:

print('Unique values for hotel:\n', data.hotel.unique())

print('Unique values for arrival_date_month:\n', data.arrival_date_month.unique())

print('Unique values for customer_type:\n', data.customer_type.unique())

print('Unique values for reservation_status:\n', data.reservation_status.unique())

print('Unique values for deposit_type:\n',data.deposit_type.unique())

print('Unique values for reserved_room_type :\n',data.reserved_room_type .unique())

print('Unique values for assigned_room_type :\n',data.assigned_room_type .unique())

print('Unique values for distribution_channel :\n',data.distribution_channel .unique())

print('Unique values for market_segment :\n',data.market_segment.unique())

print('Unique values for meal :\n',data.meal.unique())
# Transforming arrival date month names to numbers 1-12 for easier use and visualisation

# I did this manually originally and then I saw 



data.arrival_date_month=data.arrival_date_month.map({'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12})



# transforming resort hotel and city hotel to 0 and 1 for easier use



data.hotel=data.hotel.map({'Resort Hotel':0 ,'City Hotel':1})
data.head()
#creating a column that defines whether the customers are family or not using the adults/children/babies columns

def family(data):

    if (data.adults>0 and data.babies>0 and data.children>0):

        val=1

    else:

        val=0

    return val

data['family']=data.apply(family,axis=1)
#creating a column that calculates the tot number of people booking:

data['n_people']=data.adults + data.babies + data.children



#creating a column that includes the total number of nights the customer booked for:

data['n_nights']=data['stays_in_weekend_nights']+data['stays_in_week_nights']
# 1. Whats the most popular month for bookings for both resort hotel and and city hotel

plt.figure(figsize=(16,10))

legend_data={'Resort Hotel':0 ,'City Hotel':1}

ax=sns.countplot(x='arrival_date_month',data=data,hue='hotel')

ax.set(xlabel='Months',ylabel='Number of bookings')

ax.legend(legend_data)

plt.show()
#How does cancellation looks like for both hotels.



ax=sns.barplot(x='hotel',y='lead_time',hue='is_canceled', data=data.groupby(["hotel","is_canceled"]).lead_time.count().reset_index())

#ax.legend(legend_data)

plt.show()

data.groupby(["is_canceled"]).mean().reset_index()
country_data=data.country.value_counts()

# Adding the bookings for all countries outside of top 10 and naming that row other.

other_vals=pd.DataFrame({0:country_data[10:].sum()},index=['other'])

#Adding that new row to our data

country_data=country_data.append(other_vals)

# Sorting the country data in descending order

country_data=country_data.sort_values(by=[0],ascending=False)





# Creating the pie chart and displaying only the top 10 sources

plt.figure(figsize=(10,6))

plt.title('Top 10 sources per gross bookings volume')

fig=plt.pie(country_data[0:9],labels=country_data.index[0:9],autopct='%1.1f%%')

plt.show()

#Are there any major differences in lead time for both hotels.



data[['hotel','lead_time']].groupby(["hotel"]).mean().reset_index()
plt.figure(figsize=(10,6))

ax=sns.barplot(x='hotel',y='lead_time', data=data[['hotel','lead_time']].groupby(["hotel"]).mean().reset_index())

#ax.legend(legend_data)

plt.title('Booking lead times by hotel type')

plt.show()
#Whats the ADR behaviour across different months

data_line_plot=data[['arrival_date_year','arrival_date_month','lead_time']].groupby(['arrival_date_year',"arrival_date_month"]).mean().reset_index()

data_line_plot
#plotting a graph with lead time per year for all the data.

plt.figure(figsize=(10,6))

plt.title('Yearly Lead time trends for both hotels at the same time')

sns.lineplot(x='arrival_date_month',y='lead_time',data=data_line_plot,hue='arrival_date_year',marker='o',palette=['red','blue','green'])

plt.show()
#Whats the proportion between booked and cancelled(what's the cancellation rate for both hotels)

lineplot2=data[['is_canceled','hotel','lead_time']].groupby(['is_canceled','hotel']).count().reset_index()

lineplot2
plt.figure(figsize=(10,6))

ax=sns.barplot(x='hotel',y='lead_time', data=lineplot2,hue='is_canceled')

#ax.legend(['Resort hotel is 0','City hotel is 1'])

plt.title('Booking lead times by hotel type')

plt.show()
# Calculating the cancelation rate for both hotels:

print('Calcelation rate for resort hotel:',round(lineplot2['lead_time'][2]/(lineplot2['lead_time'][0]+lineplot2['lead_time'][2]),2))

print('Calcelation rate for city hotel:',round(lineplot2['lead_time'][3]/(lineplot2['lead_time'][1]+lineplot2['lead_time'][3]),2))
#Is there a link between number of guests and hotel booked

#Pivoting the number of hotels and seeing what the mean of n_people is:

data[['hotel','n_people']].groupby(['hotel']).mean().reset_index()
#Are cancellation spiking in a particular month or year

data_line_plot=data[['arrival_date_year','arrival_date_month','is_canceled','hotel']].groupby(['arrival_date_year',"arrival_date_month",'hotel']).mean().reset_index()

data_line_plot
#plotting a graph with lead time per year for all the data.

plt.figure(figsize=(10,6))

plt.title('Yearly cancelation trends for both hotels')

sns.lineplot(x='arrival_date_month',y='is_canceled',data=data_line_plot,hue='hotel',marker='o',palette=['red','blue'])

plt.show()
# Calling the LabelEncoder. Also I am duplicating data into data1 where I will encode all the categorical values.

# After that I will see the correlations of all the variables and particularly the correllation with is_cancelled.

#That way if some of the variables have very low correlation I can safely drop from the models.

le = LabelEncoder()

data1=data

data1.customer_type=le.fit_transform(data1.customer_type)

# changing the country strings to numerical data

data1.country=le.fit_transform(data1.country)

#Transforming all the data to numerical values so that we can use those variables for correlation analysis and the model building

data1.deposit_type=le.fit_transform(data1.deposit_type)

data1.reserved_room_type=le.fit_transform(data1.reserved_room_type)

data1.assigned_room_type=le.fit_transform(data1.assigned_room_type)

data1.distribution_channel=le.fit_transform(data1.distribution_channel)

data1.market_segment=le.fit_transform(data1.market_segment)

data1.meal=le.fit_transform(data1.meal)
data1.corr()
#Determining how how are all x values correlated to is_canceled

data1.corr()['is_canceled']
#Correlation heatmap

plt.figure(figsize=(40,25))

sns.heatmap(data1.corr(),annot=True,annot_kws={'size':18})

plt.xticks(fontsize=25)

plt.yticks(fontsize=25)

plt.show()
#droping all the abovementioned columns

data1=data1.drop(['reservation_status','stays_in_weekend_nights','arrival_date_week_number','stays_in_weekend_nights','arrival_date_day_of_month','meal','babies','children','n_people'],axis=1)

data1=data1.drop(['n_nights'],axis=1)

data1=data1.drop(['reservation_status_date'],axis=1)

data1.head()
# Firstly move the is canceled column to y as thats what we want to predict

y=data1['is_canceled']



#Then drop the is_canceled column from our features.

x=data1.drop(['is_canceled'],axis=1)



# Create the test/train split for our models. I have arbitrary chosen 80-20 split.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
#logistic regression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

y.prob=logreg.decision_function(x_test)

acc_log = round(logreg.score(x_test, y_test) * 100, 2)

print('Score:',acc_log)

print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))
#Creating roc and aoc for logreg

logreg_fpr,logreg_tpr,threshold=roc_curve(y_test,y_pred)

auc_logreg=auc(logreg_fpr,logreg_tpr)
#Naive bayes classifier

nbc = GaussianNB()

nbc.fit(x_train, y_train)

y_pred = nbc.predict(x_test)

acc_nbc = round(nbc.score(x_test, y_test) * 100, 2)

print('Score:',acc_nbc)

print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))
#Creating roc and aoc for GaussianNB

nbc_fpr,nbc_tpr,threshold=roc_curve(y_test,y_pred)

auc_nbc=auc(nbc_fpr,nbc_tpr)
#SVC

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_test, y_test) * 100, 2)

print('Score:',acc_svc)

print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))
#Creating roc and aoc for SVC

svc_fpr,svc_tpr,threshold=roc_curve(y_test,y_pred)

auc_svc=auc(svc_fpr,svc_tpr)
#Random Forest

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

acc_rf = round(rf.score(x_test, y_test) * 100, 2)

print('Score:',acc_rf)

print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))
#Creating roc and aoc for Random Forest

rf_fpr,rf_tpr,threshold=roc_curve(y_test,y_pred)

auc_rf=auc(rf_fpr,rf_tpr)
#knn neghbours

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(knn.score(x_test, y_test) * 100, 2)

print('Score:',acc_knn)

print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))
#Creating roc and aoc for knn neighbours

knn_fpr,knn_tpr,threshold=roc_curve(y_test,y_pred)

auc_knn=auc(knn_fpr,knn_tpr)
model_names=['Logistic Regression','Naive Bayes Classifier','SVC(Support Vector Classification)','Random Forest','k-Nearest Neighbors']

accuracy=[acc_log,acc_nbc,acc_svc,acc_rf,acc_knn]

auc=[auc_logreg,auc_nbc,auc_svc,auc_rf,auc_knn]

results=pd.DataFrame({'Model':model_names,'Accuracy':accuracy,'AUC':auc})

results
#plotting the AUC curves to visualise that random forest model is the best in this situation

plt.figure(figsize=(10,10))

plt.plot(logreg_fpr,logreg_tpr,label='Logistic Regrassion (auc=%0.3f)' %auc_logreg)

plt.plot(nbc_fpr,nbc_tpr,label='Naive Bayes Classifier (auc=%0.3f)' %auc_nbc)

plt.plot(svc_fpr,svc_tpr, label='Support Vector Classifier (auc=%0.3f)' %auc_svc)

plt.plot(rf_fpr,rf_tpr, label='Random Forest (auc=%0.3f)' %auc_rf)

plt.plot(knn_fpr,knn_tpr, label='KNN neighbours (auc=%0.3f)' %auc_knn)

plt.title('AUC for all 5 models')

plt.xlabel('False Positives')

plt.ylabel('True Postives')

plt.legend(loc=4)

plt.show()