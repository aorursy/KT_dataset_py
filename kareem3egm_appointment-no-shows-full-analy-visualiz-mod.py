#  import libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

df= pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')

df.head()
#exploration data shape 

df.shape
#the basic information about the data 

df.info()
df.duplicated().any()
#Find out the null values in the data

df.isnull().sum()
# Converting the date information in string to datetime type:



df.AppointmentDay=pd.to_datetime(df.AppointmentDay)

df.ScheduledDay=pd.to_datetime(df.ScheduledDay)

df.info()
df.drop(['AppointmentDay', 'ScheduledDay'], axis=1, inplace=True)

df.head()
df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)

df.head()
df= df.rename(columns={'No-show': 'Noshow'})
df['Noshow'].value_counts()
plt.style.use('bmh')


ax = sns.countplot(x=df.Noshow, data=df)

ax.set_title("Show/NoShow Patients")

plt.show()
print("Unique Values in `Gender` ",  df.Gender.unique())
df['Gender'].value_counts()
ax = sns.countplot(x=df.Gender, hue=df.Noshow, data=df)

ax.set_title("Show/NoShow for Females and Males")

x_ticks_labels=['Female', 'Male']

ax.set_xticklabels(x_ticks_labels)

plt.show()
np.sort(df.Age.unique())
df[df.Age == -1].shape[0]
df = df[df.Age >= 0]
df.shape
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.countplot(x=df.Age, hue=df.Noshow)

ax.set_title("Show/NoShow of Appointments by Age")

plt.show()
#Below we will see the patients count for each Neighbourhood.



plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.countplot(x=df.Neighbourhood, hue=df.Noshow)

ax.set_title("Show/NoShow by Neighbourhood")

plt.show()



df['Scholarship'].value_counts()

x = sns.countplot(x=df.Scholarship, hue=df.Noshow, data=df)

ax.set_title("Show/NoShow for Scholarship")

x_ticks_labels=['No Scholarship', 'Scholarship']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df['Hipertension'].value_counts()

ax = sns.countplot(x=df.Hipertension, hue=df.Noshow, data=df)

ax.set_title("Show/NoShow for Hipertension")

x_ticks_labels=['No Hipertension', 'Hipertension']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df['Diabetes'].value_counts()

ax = sns.countplot(x=df.Diabetes, hue=df.Noshow, data=df)

ax.set_title("Show/NoShow for Diabetes")

x_ticks_labels=['No Diabetes', 'Diabetes']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df['Alcoholism'].value_counts()

ax = sns.countplot(x=df.Alcoholism, hue=df.Noshow, data=df)

ax.set_title("Show/NoShow for Alcoholism")

x_ticks_labels=['No Alcoholism', 'Alcoholism']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df['Handcap'].value_counts()

#Handicap عائق
ax = sns.countplot(x=df.Handcap, hue=df.Noshow, data=df)

ax.set_title("Show/NoShow for Handcap")

plt.show()
df['SMS_received'].value_counts()

ax = sns.countplot(x=df.SMS_received, hue=df.Noshow, data=df)

ax.set_title("Show/NoShow for SMS_received")

x_ticks_labels=['No SMS_received', 'SMS_received']

ax.set_xticklabels(x_ticks_labels)

plt.show()
le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['Neighbourhood'] = le.fit_transform(df['Neighbourhood'])

df['Noshow'] = le.fit_transform(df['Noshow'])

df.head()
from pandas_profiling import ProfileReport 



profile = ProfileReport( df, title='Pandas profiling report ' , html={'style':{'full_width':True}})



profile.to_notebook_iframe()
df.shape
X = df.drop(['Noshow'], axis=1)

y = df['Noshow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#Applying RandomForestClassifier Model 



'''

ensemble.RandomForestClassifier(n_estimators='warn’, criterion=’gini’, max_depth=None,

                                min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0,

                                max_features='auto’,max_leaf_nodes=None,min_impurity_decrease=0.0,

                                min_impurity_split=None, bootstrap=True,oob_score=False, n_jobs=None,

                                random_state=None, verbose=0,warm_start=False, class_weight=None)

'''



RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=2,random_state=33) #criterion can be also : entropy 

RandomForestClassifierModel.fit(X_train, y_train)





#Calculating Details

print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))

print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))

print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)




#Calculating Prediction

y_pred = RandomForestClassifierModel.predict(X_test)

y_pred_prob = RandomForestClassifierModel.predict_proba(X_test)

print('Predicted Value for RandomForestClassifierModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for RandomForestClassifierModel is : ' , y_pred_prob[:10])


#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

AccScore = accuracy_score(y_test, y_pred, normalize=False)

print('Accuracy Score is : ', AccScore)