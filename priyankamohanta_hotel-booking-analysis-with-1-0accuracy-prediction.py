# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#visualization

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/bankingdata/hotel_bookings.csv")
data
data.head()
data.tail()
data.shape
data.dtypes
data.info()
data.describe(include='all')
data.isnull().sum()
#company column has high number of null value it can't replace with any one so i can

#drop it and drop the unrelevent column

data.drop(['company','agent','country','children','stays_in_weekend_nights',

           'arrival_date_day_of_month', 'arrival_date_week_number','reservation_status_date'],axis=1,inplace=True)
data.drop(['days_in_waiting_list'],axis=1,inplace=True)
#after cleaning the null vale check the null value again

data.isnull().sum()
corr=data.corr()

plt.figure(figsize=(12,5))

sns.heatmap(corr,annot=True,cmap="coolwarm")

plt.show()
data
data["is_canceled"].value_counts()
sns.countplot(data.is_canceled)

plt.show()
data["hotel"].value_counts()
sns.countplot(data.hotel,hue=data.is_canceled)

plt.show()
data['lead_time'].value_counts()
sns.distplot(data.lead_time[data.is_canceled==0])

sns.distplot(data.lead_time[data.is_canceled==1])

plt.show()
data['arrival_date_year'].value_counts()
sns.countplot(data.arrival_date_year,hue=data.is_canceled)

plt.show()
data['arrival_date_month'].value_counts()
sns.catplot(y='arrival_date_month',x='is_canceled',kind='bar',data=data)

plt.show()
data['stays_in_week_nights'].value_counts()
sns.distplot(data.stays_in_week_nights[data.is_canceled==0])

sns.distplot(data.stays_in_week_nights[data.is_canceled==1])

plt.show()
data['adults'].value_counts()
sns.countplot(data.adults,hue=data.is_canceled)

plt.show()
data['babies'].value_counts()
sns.countplot(data.babies,hue=data.is_canceled)

plt.show()
data['meal'].value_counts()
sns.countplot(data.meal,hue=data.is_canceled)

plt.show()
data['market_segment'].value_counts()
sns.catplot(y='market_segment',x='is_canceled',kind='bar',data=data)

plt.show()
data['distribution_channel'].value_counts()
sns.countplot(data.distribution_channel,hue=data.is_canceled)

plt.show()
data['is_repeated_guest'].value_counts()
sns.countplot(data.is_repeated_guest,hue=data.is_canceled)

plt.show()
data['previous_cancellations'].value_counts()
sns.set(rc={'figure.figsize':(15,10)})

sns.lineplot(x="previous_cancellations",y="is_canceled",data=data,color="g")

plt.show()
data['previous_bookings_not_canceled'].value_counts()
sns.set(rc={'figure.figsize':(15,10)})

sns.lineplot(x="previous_bookings_not_canceled",y="is_canceled",data=data,color="r")

plt.show()
data['reserved_room_type'].value_counts()
sns.countplot(data.reserved_room_type,hue=data.is_canceled)

plt.show()
data['assigned_room_type'].value_counts()
sns.countplot(data.assigned_room_type,hue=data.is_canceled)

plt.show()
data['booking_changes'].value_counts()
sns.catplot(y='booking_changes',x='is_canceled',kind='bar',data=data)

plt.show()
data['deposit_type'].value_counts()
sns.countplot(data.deposit_type,hue=data.is_canceled)

plt.show()
data['customer_type'].value_counts()
sns.countplot(data.deposit_type,hue=data.is_canceled)

plt.show()
data['adr'].value_counts()
sns.distplot(data.adr[data.is_canceled==0])

sns.distplot(data.adr[data.is_canceled==1])

plt.show()
data['required_car_parking_spaces'].value_counts()
sns.countplot(data.required_car_parking_spaces,hue=data.is_canceled)

plt.show()
data['total_of_special_requests'].value_counts()
sns.countplot(data.total_of_special_requests,hue=data.is_canceled)

plt.show()
data['reservation_status'].value_counts()
sns.countplot(data.reservation_status,hue=data.is_canceled)

plt.show()
#Label Encoding

from sklearn.preprocessing import LabelEncoder

lb1=LabelEncoder()

data.hotel=lb1.fit_transform(data.hotel)



lb2=LabelEncoder()

data.arrival_date_month=lb2.fit_transform(data.arrival_date_month)



lb3=LabelEncoder()

data.meal=lb3.fit_transform(data.meal)



lb4=LabelEncoder()

data.market_segment=lb4.fit_transform(data.market_segment)



lb5=LabelEncoder()

data.reserved_room_type=lb5.fit_transform(data.reserved_room_type)



lb6=LabelEncoder()

data.distribution_channel=lb6.fit_transform(data.distribution_channel)



lb7=LabelEncoder()

data.assigned_room_type=lb7.fit_transform(data.assigned_room_type)



lb8=LabelEncoder()

data.deposit_type=lb8.fit_transform(data.deposit_type)



lb9=LabelEncoder()

data.customer_type=lb9.fit_transform(data.customer_type)



lb10=LabelEncoder()

data.reservation_status=lb10.fit_transform(data.reservation_status)

data
ip=data.drop(['is_canceled'],axis=1)

op=data['is_canceled']
from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.3)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)
from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()

clf.fit(xtr,ytr)

yp1=clf.predict(xts)



yp1=clf.predict(xts)
from sklearn import metrics

accuracy=metrics.accuracy_score(yts,yp1)

print(accuracy)
recall=metrics.recall_score(yts,yp1)

print(recall)
#find the the number of neighbors in KNN



from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.4,random_state=42)



from sklearn.neighbors import KNeighborsClassifier



neighbors=np.arange(1,9)

train_accuracy=np.empty(len(neighbors))

test_accuracy=np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(xtr,ytr)

    train_accuracy[i]=knn.score(xtr,ytr)

    test_accuracy[i]=knn.score(xts,yts)



plt.xlabel('neighbors of number')

plt.ylabel('accuracy')

plt.title('k-NN Varying number of neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.legend()

plt.show()
from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.4)
knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(xtr,ytr)

yp2=knn.predict(xts)
from sklearn import metrics

accuracy=metrics.accuracy_score(yts,yp2)

print(accuracy)
from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.1)



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)
from sklearn import svm

alg=svm.SVC(C=20,gamma=0.03)



#train the algorithm with training data

alg.fit(xtr,ytr)

yp3=alg.predict(xts)
from sklearn import metrics

accuracy=metrics.accuracy_score(yts,yp3)

print(accuracy)



recall = metrics.recall_score(yts,yp3)

print(recall)