import pandas as pd
import numpy as np
import time
from datetime import datetime, date
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE#import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
#Read in data
data = pd.read_csv("../input/KaggleV2-May-2016.csv", header=0)
data.head()
data['DAY_OF_WEEK'] = pd.to_datetime(data['AppointmentDay']).dt.weekday
data.head()
data.DAY_OF_WEEK.hist()
plt.title('Days of Week Histogram')
plt.xlabel('Day of Week')
plt.ylabel('Freq')
plt.show()
data['WEEKEND'] = data['DAY_OF_WEEK'].apply(lambda x: 1 if (x== 5 or x==6) else 0)
data['DAY_OF_MONTH'] = pd.to_datetime(data['AppointmentDay']).dt.day
data.DAY_OF_MONTH.hist()
plt.title('Days of Month Histogram')
plt.xlabel('Day of Month')
plt.ylabel('Freq')
plt.show()
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay']).dt.date
data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay']).dt.date
#Calculate distance from when appointment scheduled to appointment day
data['APP_DELTA'] = abs(data['ScheduledDay'] - data['AppointmentDay']).astype('int64')
#Sanity check
data[['ScheduledDay','AppointmentDay','APP_DELTA']].head()
data['APP_DELTA'].mean()
data.APP_DELTA.hist()
plt.title('APP_DELTA Histogram')
plt.xlabel('APP_DELTA')
plt.ylabel('Freq')
plt.show()
dr = pd.date_range(start='2016-04-28', end='2016-06-08')
df = pd.DataFrame()
df['date'] = dr
cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())
holidays=pd.to_datetime(holidays).date
#Sanity check : https://www.timeanddate.com/holidays/us/2016
holidays
def time_delta(holidays, pivot):
    nearest=min(holidays, key=lambda x: abs(x-pivot))
    timedelta=abs(nearest-pivot).days
    return timedelta
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])
data['HOLIDAY_DELTA'] = data['AppointmentDay'].dt.date.apply(lambda x: time_delta(holidays,x))
data.HOLIDAY_DELTA.hist()
plt.title('HOLIDAY_DELTA Histogram')
plt.xlabel('HOLIDAY_DELTA')
plt.ylabel('Freq')
plt.show()
data['NAT_HOLIDAY'] = data['AppointmentDay'].dt.date.astype('datetime64[ns]').isin(holidays)*1
data['Gender'] = data['Gender'].apply(lambda x: 1 if x=='F' else 0)
data['AGE_B'] = data.Age.apply(lambda x: 4 if x>49 else( 3 if x >34 else(2 if x >23 else(1))))
#Sanity check
data.dropna(how='any', inplace=True)
data[['Age','AGE_B']].head()
data.Age.hist()
plt.title('Age Histogram')
plt.xlabel('Age')
plt.ylabel('Freq')
plt.show()
data.AGE_B.hist()
plt.title('Age Histogram (Bucketized)')
plt.xlabel('Age')
plt.ylabel('Freq')
plt.show()
data['No-show'] = data['No-show'].apply(lambda x: 1 if x=='Yes' else 0)
#Sanity check
data['No-show'].sum()
#Sanity check
data.groupby('No-show').count()
data_in = data.drop(['ScheduledDay','AppointmentDay','PatientId', 'AppointmentID','Neighbourhood', 'Age'], axis=1)
#from sklearn import datasets
from sklearn.feature_selection import RFE
#data_in = data_in.dropna(how='any')
data_in = data_in.drop(['No-show'], axis=1)
names = data_in.columns.values
logreg = LogisticRegression()
rfe=RFE(logreg)
rfe=rfe.fit(data_in, data['No-show'])
print("Features sorted by rank:")
print(sorted(zip(map(lambda x: round(x,4), rfe.ranking_),names)))
X = data_in[['AGE_B','APP_DELTA','DAY_OF_MONTH','DAY_OF_WEEK','Gender','HOLIDAY_DELTA','Hipertension','SMS_received','Scholarship','Diabetes','Alcoholism','Handcap','NAT_HOLIDAY','WEEKEND']]
y = data['No-show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predict = logreg.predict_proba(X_test)
my_list = map(lambda x: x[0], y_predict)
y_pred = pd.Series(my_list).round()

print('Accuracy: {:.2f}'.format(logreg.score(X_test, y_test)))
kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'
results = model_selection.cross_val_score(logreg, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation\n   average accuracy: %.3f" % (results.mean()))
print("   max accuracy: %.3f" %(results.max()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="upper left")
plt.show()
print(logit_roc_auc)
