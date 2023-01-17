# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

import statsmodels.api as sm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
hotel_booking=pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
hotel_booking.describe()
hotel_booking.info()
hotel_booking.isnull().sum()
hotel_booking.drop("company",axis=1,inplace=True)
print(hotel_booking["agent"].nunique())
hotel_booking.drop("agent",axis=1,inplace=True)
pd.crosstab(index=hotel_booking["lead_time"],columns=hotel_booking["is_canceled"],normalize="index")
pd.crosstab(index=hotel_booking["arrival_date_year"],columns=hotel_booking["is_canceled"],normalize="index")
hotel_booking.drop("arrival_date_year",axis=1,inplace=True)
ax = sns.countplot(x="arrival_date_month", data=hotel_booking, hue="is_canceled")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.tight_layout()

plt.show()
pd.crosstab(index=[hotel_booking["arrival_date_month"],hotel_booking["arrival_date_week_number"]],columns=hotel_booking["is_canceled"],normalize="index")
hotel_booking.drop("arrival_date_week_number",axis=1,inplace=True)

hotel_booking.drop("arrival_date_day_of_month",axis=1,inplace=True)
pd.crosstab(columns=hotel_booking["hotel"],index=hotel_booking["stays_in_weekend_nights"])
print(sum(hotel_booking["stays_in_weekend_nights"]>4))

hotel_booking.drop(hotel_booking[hotel_booking["stays_in_weekend_nights"]>4].index,inplace=True)
pd.crosstab(columns=hotel_booking["hotel"],index=hotel_booking["stays_in_week_nights"])
print(sum(hotel_booking["stays_in_week_nights"]>10))

hotel_booking.drop(hotel_booking[hotel_booking["stays_in_week_nights"]>10].index,inplace=True)
pd.crosstab(columns=hotel_booking["hotel"],index=hotel_booking["adults"])
print(sum(hotel_booking["adults"]>4))

hotel_booking.drop(hotel_booking[hotel_booking["adults"]>3].index,inplace=True)
pd.crosstab(columns=hotel_booking["hotel"],index=hotel_booking["children"])
hotel_booking["children"].replace(np.nan,0,inplace=True)

hotel_booking.isnull().sum()
hotel_booking.drop(hotel_booking[hotel_booking["children"]>2].index,inplace=True)
pd.crosstab(columns=hotel_booking["hotel"],index=hotel_booking["babies"])
print(sum(hotel_booking["babies"]>1))

hotel_booking.drop(hotel_booking[hotel_booking["babies"]>1].index,inplace=True)
pd.crosstab(columns=hotel_booking["hotel"],index=hotel_booking["meal"])
mod=hotel_booking["country"].mode()

print(mod)

hotel_booking["country"].replace(np.nan,"PRT",inplace=True)
pd.crosstab(columns=hotel_booking["hotel"],index=hotel_booking["country"])
print(hotel_booking["market_segment"].value_counts())
ax=sns.countplot("market_segment",data=hotel_booking,hue="is_canceled")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.tight_layout()

plt.show()
value_counts=hotel_booking["market_segment"].value_counts()



to_remove=value_counts[value_counts<=200].index

hotel_booking.replace(to_remove,np.nan,inplace=True)

hotel_booking.dropna(axis=0,how="any",inplace=True)
print(hotel_booking["distribution_channel"].value_counts())
ax=sns.countplot("distribution_channel",data=hotel_booking,hue="is_canceled")

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.tight_layout()

plt.show()
print(hotel_booking["is_repeated_guest"].value_counts())
sns.countplot("is_repeated_guest",data=hotel_booking,hue="is_canceled")
print(hotel_booking["previous_cancellations"].value_counts())
sns.countplot("previous_cancellations",data=hotel_booking,hue="hotel")
pd.crosstab(index="counts",columns=hotel_booking["previous_cancellations"],normalize="index")

print(sum(hotel_booking["previous_cancellations"]>=2))

hotel_booking.drop(hotel_booking[hotel_booking["previous_cancellations"]>=2].index,axis=0,inplace=True)
print(hotel_booking["previous_bookings_not_canceled"].value_counts())
print(hotel_booking["previous_bookings_not_canceled"].unique())
print(sum(hotel_booking["previous_bookings_not_canceled"]>10))
hotel_booking.drop(hotel_booking[hotel_booking["previous_bookings_not_canceled"]>10].index,axis=0,inplace=True)
print(hotel_booking["reserved_room_type"].value_counts())
sns.countplot("reserved_room_type",data=hotel_booking,hue="hotel")
sns.countplot("reserved_room_type",data=hotel_booking,hue="is_canceled")
value_counts=hotel_booking["reserved_room_type"].value_counts()

threshold=100

to_remove=value_counts[value_counts<=threshold].index

hotel_booking["reserved_room_type"].replace(to_remove,np.nan,inplace=True)

hotel_booking.dropna(axis=0,how="any",inplace=True)
print(hotel_booking["assigned_room_type"].value_counts())
sns.countplot("assigned_room_type",data=hotel_booking,hue="hotel")
sns.countplot("assigned_room_type",data=hotel_booking,hue="is_canceled")
print(hotel_booking["booking_changes"].value_counts())
print(sum(hotel_booking["booking_changes"]>4))
sns.countplot("booking_changes",data=hotel_booking,hue="hotel")
sns.countplot("booking_changes",data=hotel_booking,hue="is_canceled")
hotel_booking.drop(hotel_booking[hotel_booking["booking_changes"]>4].index,axis=0,inplace=True)
print(hotel_booking["deposit_type"].value_counts())
sns.countplot("deposit_type",data=hotel_booking,hue="is_canceled")
sns.countplot("deposit_type",data=hotel_booking,hue="hotel")
print(hotel_booking["customer_type"].value_counts())
sns.countplot("customer_type",data=hotel_booking,hue="hotel")
sns.countplot("customer_type",data=hotel_booking,hue="is_canceled")
print(hotel_booking["adr"].value_counts())
print(hotel_booking["required_car_parking_spaces"].value_counts())
sns.countplot("required_car_parking_spaces",data=hotel_booking,hue="hotel")
sns.countplot("required_car_parking_spaces",data=hotel_booking,hue="is_canceled")
hotel_booking.drop(hotel_booking[hotel_booking["required_car_parking_spaces"]>1].index,axis=0,inplace=True)
print(hotel_booking["total_of_special_requests"].value_counts())
sns.countplot("total_of_special_requests",data=hotel_booking,hue="hotel")
sns.countplot("total_of_special_requests",data=hotel_booking,hue="is_canceled")
pd.crosstab(index=hotel_booking["total_of_special_requests"],columns=hotel_booking["is_canceled"])
hotel_booking.drop(hotel_booking[hotel_booking["total_of_special_requests"]>4].index,inplace=True)
hotel_booking.drop("reservation_status_date",axis=1,inplace=True)
data1=pd.get_dummies(hotel_booking,drop_first=True)

x=data1.drop("is_canceled",axis=1,inplace=False)

y=data1["is_canceled"].values

x=x.values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

LR=LogisticRegression(solver="lbfgs",max_iter=10000,class_weight="balanced")

fit_model=LR.fit(x_train,y_train)

prediction=LR.predict(x_test)

print(accuracy_score(y_test,prediction))

print(confusion_matrix(y_test,prediction))
prediction1=LR.predict(x)

print(accuracy_score(y,prediction1))

print(confusion_matrix(y,prediction1))



col1=hotel_booking["hotel"].values



data_pred=pd.DataFrame({'hotel':col1,'prediction':prediction1})
print(data_pred)