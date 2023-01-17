import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import date

import warnings

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
data = pd.read_csv('../input/electric_faults_data.csv')

data.head()
print("The shape of the data is :",data.shape)
data.describe()
plt.figure(figsize=(12,7))

f = sns.heatmap(data.isnull(), cbar = False, cmap = 'viridis')

f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})

plt.title("Heatmap of Missing Values", fontsize = 15)

plt.show()
data['tripping_reason'].value_counts()
data['tripping_reason'].fillna(value = 'transient fault', inplace = True)
data['other_circuit'].value_counts()
data.drop('other_line_status', inplace = True, axis  =1)

data.drop('observation', inplace = True, axis  =1)

data.drop('other_circuit', inplace = True, axis  =1)
data['repair_carried'].value_counts()
data['repair_carried'].fillna(value = 'nil', inplace= True)
#Separating year

data['trip_year'] = pd.to_datetime(data['date_of_trip'], dayfirst= True ).dt.year

data['restore_year'] = pd.to_datetime(data['date of restoration'], dayfirst= True ).dt.year



#Separating month

data['trip_month'] = pd.to_datetime(data['date_of_trip'], dayfirst= True ).dt.month

data['restore_month'] = pd.to_datetime(data['date of restoration'], dayfirst= True ).dt.month





#separating hours

data['trip_hour'] = pd.to_datetime(data['time_of_trip']).dt.hour

data['restore_hour'] = pd.to_datetime(data['time_of_restoration']).dt.hour
data.head(10)
data['trip_month'] = data['trip_month'].map({1:'January', 2:'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July',

                               8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})



data['restore_month'] = data['restore_month'].map({1:'January', 2:'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July',

                               8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})

data['weekday'] = pd.to_datetime(data['date_of_trip']).dt.weekday
data['weekday'] = data['weekday'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
temp = data['weekday'].value_counts().reset_index()



plt.figure(figsize= (12,7))

plt.title('Trips on Weekdays',fontsize = 15)

f = sns.barplot(x = temp['index'], y = temp['weekday'], palette = 'hls')

f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})

plt.xlabel('Weekdays', fontsize = 16)

plt.yticks(list(range(max(temp['weekday']))))

plt.show()
temp = data['trip_year'].value_counts()



plt.figure(figsize= (12,7))

plt.title('Trips in Years',fontsize = 15)

f = sns.barplot(x = temp.index, y = temp.values, palette = 'Set2')

f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})

plt.xlabel('Years', fontsize = 16)

plt.yticks(list(range(max(temp))))

plt.show()
temp = data['trip_month'].value_counts()



plt.figure(figsize= (12,7))

plt.title('Trips in Months',fontsize = 15)

f = sns.barplot(x = temp.index[::-1], y = temp.values[::-1], palette = 'rainbow')

f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})

plt.xlabel('Months', fontsize = 16)

plt.yticks(list(range(max(temp))))

plt.show()
temp = data['weather'].value_counts()



plt.figure(figsize= (12,7))

plt.title('Trips in Weather',fontsize = 15)

f = sns.barplot(x = temp.index, y = temp.values, palette = 'inferno')

f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})

plt.xlabel('Weather', fontsize = 16)

plt.yticks(list(range(0,max(temp)+2)))

plt.show()
temp = data['line_trip'].value_counts().reset_index()



plt.figure(figsize=(9,9))

f = plt.pie(x = temp['line_trip'],labels = ['Yes','No'], colors=('lightblue','orange'), autopct= "%1.1f%%")

plt.title('Line Trips at other End', fontsize  = 15)

plt.show()
temp = data['tripping_reason'].value_counts()



plt.figure(figsize= (12,7))

plt.title('Trips Reasons',fontsize = 15)

f = sns.barplot(x = temp.index, y = temp.values, palette = 'autumn')

f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13}, rotation = 30)

plt.xlabel('Reasons', fontsize = 16)

plt.yticks(range(max(temp)+2))

plt.show()
plt.figure(figsize= (15,10))

#plt.suptitle("Distributions of Different Features", fontsize = 20)

#Histograms

plt.subplot(3,3,1)

sns.distplot(data['voltage'], rug = True, kde = False)

plt.xlabel('Voltage in KiloVolts', fontsize = 12)

plt.title('Distribution of Voltage',fontsize = 15)



plt.subplot(3,3,2)

sns.distplot(data['load_of_line'], color= 'green',rug = True, kde = False)

plt.title('Distribution of Load of Line',fontsize = 15)

plt.xlabel('Load on line in Amperes', fontsize = 12)



plt.subplot(3,3,3)

sns.distplot(data['frequency'], rug= True, color= 'orange', kde = False)

plt.xlabel('Voltage in KiloVolts', fontsize = 12)

plt.title('Distribution of Frequency',fontsize = 15)





#Kde Plots

plt.subplot(3,3,4)

sns.kdeplot(data['voltage'], shade = True)

plt.xlabel('Voltage in KiloVolts', fontsize = 12)

plt.title('Distribution of Voltage',fontsize = 15)



plt.subplot(3,3,5)

sns.kdeplot(data['load_of_line'], shade = True, color = 'g')

plt.title('Distribution of Load of Line',fontsize = 15)

plt.xlabel('Load on line in Amperes', fontsize = 12)



plt.subplot(3,3,6)

sns.kdeplot(data['frequency'],shade= True, color = 'Orange')

plt.title('Distribution of Frequency',fontsize = 15)



#Box Plots

plt.subplot(3,3,7)

sns.boxplot(x = data['voltage'], orient = 'v',color= 'b', boxprops=dict(alpha=.5))

plt.subplot(3,3,8)

sns.boxplot(x = data['load_of_line'], orient = 'v', color= 'g', boxprops=dict(alpha=.5))

plt.subplot(3,3,9)

sns.boxplot(x = data['frequency'], orient = 'v', color= 'Orange', boxprops=dict(alpha=.5))



plt.tight_layout()

plt.show()
sns.jointplot(x = data['load_of_line'], y = data['voltage'], kind = 'reg', color= 'g')



plt.show()
sns.jointplot(x = data['load_of_line'], y = data['frequency'], kind = 'reg', color= 'darkorange')

plt.show()
sns.jointplot(x = data['voltage'], y = data['frequency'], kind = 'reg', color = 'blue')

plt.show()
temp = data['trip_hour'].value_counts()

plt.figure(figsize= (10,10))



plt.subplot(2,1,1)

sns.pointplot(x = temp.index, y = temp.values ,palette= 'Reds')

sns.pointplot(x = temp.index, y = temp.values ,join= True, color = 'r',markers = '')

plt.title('Trips on Hours',fontsize = 15)

plt.xlabel('Trip Hours of Day', fontsize = 12)

plt.ylabel('Number of Hours', fontsize = 12)

plt.yticks([0,1,2,3,4])



temp = data['restore_hour'].value_counts()

plt.subplot(2,1,2)

sns.pointplot(x = temp.index, y = temp.values ,palette= 'Greens')

sns.pointplot(x = temp.index, y = temp.values ,join= True, color='g', markers = '')

plt.title('Restoration on Hours',fontsize = 15)

plt.xlabel('Restore Hours of Day', fontsize = 12)

plt.ylabel('Number of Hours', fontsize = 12)

plt.yticks([0,1,2,3,4])

plt.show()
for i in range(0,len(data['repair_carried'])):

    if data['repair_carried'][i] == 'nil':

        data['repair_carried'][i] = 'None'

temp = data['repair_carried'].value_counts()

plt.figure(figsize= (12,7))

plt.title('Repairs Carried',fontsize = 15)

f = sns.barplot(x = temp.index, y = temp.values, palette = 'Set1')

f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})

plt.xlabel('Repair Types', fontsize = 15)

plt.yticks(range(max(temp)+2))

plt.show()
plt.figure(figsize= (9,7))

plt.title('Nature of the fault',fontsize = 15)

f = sns.countplot(data['nature'], palette= 'hls')

f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})

plt.xlabel("Types", fontsize= 14)

plt.show()
data['line_trip'] = data['line_trip'].map({'no':0, 'yes':1})
data['type_of_fault'] = data['type_of_fault'].map({'low':-1, 'medium':0, 'high': 1})
data.head()
X_full = data.iloc[:, [3,4,6]].values

y_full = data['type_of_fault'].values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_full)

X_full = sc.transform(X_full)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.25, random_state = 1)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_train, classifier.predict(X_train)))
from sklearn.metrics import classification_report

cr = classification_report(y_test, y_test_pred)

print(cr)
t = input("Enter Trip\t")

l = float(input("Enter load of line\t"))

f = float(input("Enter Frequency\t"))



if t =='yes':

    t = 1

elif t == 'no':

    t = 0

    

samp = np.array([[int(t), int(l), float(f)]])

samp = sc.transform(samp)

res = classifier.predict(samp)

print("\n------Output-----\n")

if res == -1:

    print("Low Fault")

elif res == 0:

    print("Medium Fault")

else:

    print("High Fault")
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver= 'newton-cg',multi_class= 'multinomial')

classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
print(classification_report(y_train, classifier.predict(X_train)))
from sklearn.metrics import classification_report

cr = classification_report(y_test, y_test_pred)

print(cr)
t = input("Enter Trip\t")

l = float(input("Enter load of line\t"))

f = float(input("Enter Frequency\t"))



if t =='yes':

    t = 1

elif t == 'no':

    t = 0

    

samp = np.array([[int(t), int(l), float(f)]])

samp = sc.transform(samp)

res = classifier.predict(samp)

print("\n------Output-----\n")

if res == -1:

    print("Low Fault")

elif res == 0:

    print("Medium Fault")

else:

    print("High Fault")