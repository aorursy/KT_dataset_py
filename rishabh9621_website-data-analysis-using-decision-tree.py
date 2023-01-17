import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix
#Importing Data Set

user_data=pd.read_csv("../input/user_data.csv")
user_data.head()
#Dropping the Column as Data in the column is not available

user_data=user_data.drop(["device_mobileDeviceModel"], axis=1)
user_data.isnull().sum()
#We use Value_Count function to see the data count of a column so that we can fill the missing data accordingly

a=user_data["totals_bounces"].value_counts()

b=user_data["totals_pageviews"].value_counts()

c=user_data["totals_timeOnSite"].value_counts()

print(a)

print(b)

print(c)
#Filling the missing data as per their values

user_data["totals_bounces"].fillna(0, inplace=True)

user_data["totals_pageviews"].fillna(user_data["totals_pageviews"].mean(), inplace=True)

user_data["totals_timeOnSite"].fillna(user_data["totals_timeOnSite"].mean(), inplace=True)

user_data["totals_totalTransactionRevenue"].fillna(0, inplace=True)

user_data["totals_transactions"].fillna(0, inplace=True)
one=LabelEncoder()
#Converting Categorical Data to Numeric Values

user_data["trafficSource_source"]=one.fit_transform(user_data["trafficSource_source"])

user_data["trafficSource_medium"]=one.fit_transform(user_data["trafficSource_medium"])

user_data["trafficSource_campaign"]=one.fit_transform(user_data["trafficSource_campaign"])

user_data["device_deviceCategory"]=one.fit_transform(user_data["device_deviceCategory"])

user_data["device_operatingSystem"]=one.fit_transform(user_data["device_operatingSystem"])

user_data["geoNetwork_city"]=one.fit_transform(user_data["geoNetwork_city"])

user_data["channelGrouping"]=one.fit_transform(user_data["channelGrouping"])
#Spliting the Dataset

x=user_data.drop("totals_transactions", axis=1)

y=user_data["totals_transactions"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
DT=DecisionTreeClassifier()

DT.fit(x_train,y_train)

DT.score(x_train,y_train)
prediction=DT.predict(x_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))