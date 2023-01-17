import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
data = pd.read_csv('../input/KaggleV2-May-2016.csv')

data.head()
#Rename columns and remove errors like Hipertension
data = data.rename(columns={'PatientId':'patient_id','AppointmentID':'appointment_id','Gender': 'sex', 'ScheduledDay': 'scheduled_day', 'AppointmentDay': 'appointment_day', 
                                    'Age': 'age', 'Neighbourhood': 'neighbourhood', 'Scholarship': 'scholarship', 
                                    'Hipertension': 'hypertension', 'Diabetes': 'diabetic', 'Alcoholism': 'alcoholic', 
                                    'Handcap': 'handicap', 'No-show': 'no_show'})
data['no_show'] = data['no_show'].map({'No': 1, 'Yes': 0})
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
#handicap values are taken from (0,1,2) instead of (0,1,2,3,4) as 3,4 are not significantly high (no of records) 
data['handicap'] = data['handicap'].apply(lambda x: 2 if x > 2 else x)
#get data and time from scheduled_day and appointment_day
data['scheduled_day'] = pd.to_datetime(data['scheduled_day'], infer_datetime_format=True)
data['appointment_day'] = pd.to_datetime(data['appointment_day'], infer_datetime_format=True)
data.describe()
data.drop(data[data['age'] <= 0].index, inplace=True)
data.drop(data[data['age'] >100].index, inplace=True)
encoder_neighbourhood = LabelEncoder()
data['neighbourhood_enc'] = encoder_neighbourhood.fit_transform(data['neighbourhood'])
#calculating waiting time. How long a patient has to wait before they see the doctor
data['waiting_time'] = list(map(lambda x: x.days+1 , data['appointment_day'] - data['scheduled_day']))
#removing incorrect entries where appointment day was before the scheduled day which is not possible
data.drop(data[data['waiting_time'] <= -1].index, inplace=True)
# from waiting time created a column range according to the length of waiting time.(<1 month,>1 and <2 month...etc)
data['waiting_time_range'] = data['waiting_time'].apply(lambda x: 1 if x>=0 and x<=30 else 
                                                          2 if x>30 and x<=60 else 
                                                          3 if x>60 and x<=90 else 
                                                          4 if x>90 and x<=120 else 
                                                          5 if x>120 and x<=150 else
                                                          6 if x>150 and x<=180 else
                                                          7)
#created age into different group to see if one particular group misses the appointment most
data['age_group'] = data['age'].apply(lambda x: 1 if x>0 and x<19 else 
                                                            2 if x>18 and x<38 else 
                                                            3 if x>37 and x<56 else 
                                                            4 if x>55 and x<76 else 5)
#there are two types insurance for people (medicare and medicaid).<65 years and >65 years
data['insurance_age'] = data['age'].apply(lambda x: 1 if x >= 65 else 0)
#added days of appointment to see if ppl misses out on particular day
data['appointment_dayofWeek'] = data['appointment_day'].map(lambda x: x.dayofweek)
#created risk score column by calculating (no of appointments they didn't show up/ total appointment booked)
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
sns.countplot(x='insurance_age', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='age_group', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='appointment_dayofWeek', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='waiting_time_range', hue='no_show', data=data, palette='RdBu')
plt.show();

#drop columns which are not necessary (patient id and appointment id, neighbourhood, schedule and appointment day)
#drop columns which are not contributing the prediction(insurance_age,age_group,waiting_time_range)
data.drop(['scheduled_day','appointment_day','neighbourhood','patient_id','appointment_id','insurance_age','age_group','waiting_time_range'], axis=1, inplace=True)
X = data.drop(['no_show'], axis=1)
y = data['no_show']
X.head()

y.head()

Counter(y)

sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_sample(X, y)
Counter(y_res)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=101)
model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)
print("Accuracy:")
model.score(X_test, y_test)
print("confusion matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))
print("classification report:")
print(classification_report(y_test, model.predict(X_test)))
#we can see which are features that influenced the most in prediction 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')
plt.show()