"""Importing required libraries"""

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/KaggleV2-May-2016.csv')
data.head()
#Renaming Columns
data = data.rename(columns={'PatientId':'patient_id','AppointmentID':'appointment_id','Gender': 'sex', 'ScheduledDay': 'scheduled_day', 'AppointmentDay': 'appointment_day', 
                                    'Age': 'age', 'Neighbourhood': 'neighbourhood', 'Scholarship': 'scholarship', 
                                    'Hipertension': 'hypertension', 'Diabetes': 'diabetic', 'Alcoholism': 'alcoholic', 
                                    'Handcap': 'handicap', 'No-show': 'no_show'})


#binarizing columns
data['no_show'] = data['no_show'].map({'No': 1, 'Yes': 0})
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
data['handicap'] = data['handicap'].apply(lambda x: 2 if x > 2 else x)
#Converting the AppointmentDay and ScheduledDay into a date and time format 
data['scheduled_day'] = pd.to_datetime(data['scheduled_day'], infer_datetime_format=True)
data['appointment_day'] = pd.to_datetime(data['appointment_day'], infer_datetime_format=True)
data.describe()
data.drop(data[data['age'] <= 0].index, inplace=True)
data.drop(data[data['age'] >100].index, inplace=True)
encoder_neighbourhood = LabelEncoder()
data['neighbourhood_enc'] = encoder_neighbourhood.fit_transform(data['neighbourhood'])
data['waiting_time'] = list(map(lambda x: x.days+1 , data['appointment_day'] - data['scheduled_day']))
data.drop(data[data['waiting_time'] <= -1].index, inplace=True)

"""We are adding the days of week column in order to find out the days when the patient is not likely to show up for the appointment. 
For example: A patient might not show up because it is a weekend."""
data['appointment_dayofWeek'] = data['appointment_day'].map(lambda x: x.dayofweek)
data['no_of_noshows'] = data.groupby('patient_id')[['no_show']].transform('sum')
data['total_appointment'] = data.groupby('patient_id')[['no_show']].transform('count')

data['risk_score'] = data.no_of_noshows / data.total_appointment
sns.countplot(x='sex', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='handicap', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='alcoholic', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='hypertension', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='diabetic', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='scholarship', hue='no_show', data=data, palette='RdBu')
plt.show();
#Removing columns that are not necessary for prediction
data.drop(['scheduled_day','appointment_day','neighbourhood','patient_id','appointment_id'], axis=1, inplace=True)
#Splitting the data into training and testing sets.
X = data.drop(['no_show'], axis=1)
y = data['no_show']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y)


#Modeling : Random Forest
clf = RandomForestClassifier(n_estimators=300)
clf.fit(X_train, y_train)

#Performance Check
print("Mean Accuracy:")
print(clf.score(X_test, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, clf.predict(X_test)))
print("Classification Report:")
print(classification_report(y_test, clf.predict(X_test)))

#Plotting feature importance
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')
plt.show()