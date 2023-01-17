import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import statsmodels.api as sm

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

sns.set()
raw_data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

raw_data.head()
#The data has 119,390 entries across 32 columns

raw_data.shape
raw_data.info()
raw_data.isnull().sum()
# Company column will be dropped due to the high number of missing data

raw_data.drop(['company'], axis=1, inplace=True)
# The data will be split into two for each hotel, this will include all entries for each respective hotel

cityhotel = raw_data.loc[(raw_data["hotel"] == "City Hotel")]

resorthotel = raw_data.loc[(raw_data["hotel"] == "Resort Hotel")]
#booking cancelation rate for each hotel

cancellation_rate = raw_data[['hotel','is_canceled']] 

cancel = cancellation_rate.groupby(['hotel'], as_index=False).sum()

pd.options.display.max_rows = 40

cancel_sorted = cancel.sort_values(by=['is_canceled'], ascending = False)

cancel_sorted
# booking reservation status for each hotel by numbers

raw_data.groupby("hotel")["reservation_status"].value_counts()
sns.countplot(x="reservation_status", data=raw_data)

plt.show()
#The best month to book a hotel based on the lowest average daily rate(adr) for each months

booking_decision = raw_data[['arrival_date_month','adr']] 

purchase = booking_decision.groupby(['arrival_date_month'], as_index=False).mean()

pd.options.display.max_rows = 100

adr_sort = purchase.sort_values(by=['adr'], ascending = True)

adr_sort
# Graph showing the average daily rate for each month

adr_sort.plot('arrival_date_month','adr',kind='barh')



plt.xlabel('adr',fontsize=20)

plt.ylabel('arrival_month',fontsize=20)

plt.title('adr by month')

plt.show()
plt.figure(figsize = (13,10))

sns.set(style = "darkgrid")

plt.title("Countplot Distrubiton of Segment by Deposit Type", fontdict = {'fontsize':20})

ax = sns.countplot(x = "market_segment", hue = 'deposit_type', data = raw_data)
#whtether or not a hotel is was likely to receive a disproportionately high number of special request

#Special Requests for City Hotel

special_request_city = cityhotel[['arrival_date_month','total_of_special_requests']] 

purchase = special_request_city.groupby(['arrival_date_month']).sum()

pd.options.display.max_rows = 100

city_request_sort = purchase.sort_values(by=['total_of_special_requests'], ascending = True)

city_request_sort
#whtether or not a hotel is was likely to receive a disproportionately high number of special request

#Special Requests for Resort Hotel

special_request_resort = resorthotel[['arrival_date_month','total_of_special_requests']] 

purchase = special_request_resort.groupby(['arrival_date_month']).sum()

pd.options.display.max_rows = 100

resort_request_sort = purchase.sort_values(by=['total_of_special_requests'], ascending = True)

resort_request_sort
# plotting the intended dependent column i.e 'is_canceled' to determine its distribution

#The graph below shows canceled bookings(1) and the non-canceled bookings(1)

# since there are only two possible outcomes, a logistic regression will be used for the model that will predict whether a booking was canceled or not

raw_data['is_canceled'].plot(kind='hist')
#Turning Categorical data into numerical data #preparing to model

raw_data_with_dummies = pd.get_dummies(raw_data, drop_first=True) 

raw_data_with_dummies
# Predicting Cancellation

y = raw_data['is_canceled']

x1 = raw_data_with_dummies[['lead_time','total_of_special_requests','required_car_parking_spaces','booking_changes','previous_cancellations','adr']]

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(x_test, y_test)))
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):

    width = 12

    height = 10

    plt.figure(figsize=(width, height))



    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)

    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)



    plt.title(Title)

    plt.xlabel('Cancellations')

    plt.ylabel('Hotel_bookings')



    plt.show()

    plt.close()
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test,y_pred,"Actual Values (Test)","Predicted Values (Test)",Title)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
log_reg.coef_