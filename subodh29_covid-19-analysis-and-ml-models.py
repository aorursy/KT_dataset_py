# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import datetime

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

missing_values = ["n/a", "na", "-","NaN"]
dataset = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',na_values = missing_values)
dataset.head()
cases_local = {} #dict for local cases
cases_foreign = {} #dict for foreign patients in india



for i in range(446):
    cases_local[dataset['State/UnionTerritory'][i]] = dataset['ConfirmedIndianNational'][i]
    cases_foreign[dataset['State/UnionTerritory'][i]] = dataset['ConfirmedForeignNational'][i]
print(cases_local.values())
p1 = plt.bar(np.arange(len(cases_local)),list(cases_local.values()), 0.7)
p2 = plt.bar(np.arange(len(cases_local)), list(cases_foreign.values()), 0.7,
             bottom=list(cases_local.values()))

plt.ylabel('Cases')
plt.title('Cases by states/UTs')
plt.xticks(np.arange(len(cases_local)), cases_local.keys(),rotation='vertical')
plt.yticks(np.arange(0, 200, 20))
plt.legend((p1[0], p2[0]), ('Local patient', 'Foreign patient'))

plt.show()
deaths = {} #dict for deaths
for i in range(len(dataset)):
    deaths[dataset['State/UnionTerritory'][i]] = int(dataset['Deaths'][i])

plt.bar(np.arange(len(deaths)),deaths.values(), 0.7)
plt.ylabel('Deaths')
plt.xticks(np.arange(len(deaths)), deaths.keys(),rotation='vertical')
plt.yticks(np.arange(0, 8, 1))
plt.title('Deaths in various states/UTs',bbox={'facecolor':'0.8', 'pad':5})
plt.show()
cured_per_cases = {}  #percentage of cured per confirmed cases
for i in range(len(dataset)):
    cured_per_cases[dataset['State/UnionTerritory'][i]] = (dataset['Cured'][i] / (dataset['Confirmed'][i]))*100
cured_per_cases
plt.pie([v for v in cured_per_cases.values()],labels = [k for k in cured_per_cases.keys()])
plt.show()
start = datetime.datetime.strptime(dataset['Date'][0], "%d/%m/%y")
end = datetime.datetime.strptime(dataset['Date'][len(dataset)-1], "%d/%m/%y")
date = pd.date_range(start,end)
cases_by_date = dataset['Date'].value_counts().to_dict()

cases = []
list1 = list(cases_by_date.values())
list1.sort()

for i in range(len(date)):
    if i==0:
        cases.append(list1[i])
        
    else:
        cases.append(cases[i-1] + list1[i])
    
fig, ax = plt.subplots()
ax.plot(date,cases)
ax.xaxis_date()     # interpret the x-axis values as dates
fig.autofmt_xdate()
plt.title('Cases progression by date')
plt.ylabel('Cases till date')
plt.show()
samples_tested =  pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv')
samples_tested.head()
individual_tested = {}
positive_cases = {}
for i in range(len(samples_tested)):
    individual_tested[samples_tested['DateTime'][i]] = samples_tested['TotalIndividualsTested'][i]
    positive_cases[samples_tested['DateTime'][i]] = samples_tested['TotalPositiveCases'][i]
    
p1 = plt.bar(np.arange(len(individual_tested)),list(individual_tested.values()), 0.7)
p2 = plt.bar(np.arange(len(positive_cases)), list(positive_cases.values()), 0.7,
             bottom=list(individual_tested.values()))

plt.ylabel('Samples')
plt.title('Samples tested')
plt.xticks(np.arange(len(individual_tested)), individual_tested.keys(),rotation='vertical')
#plt.yticks(np.arange(0, 200, 20))
plt.legend((p1[0], p2[0]), ('Individual tested', 'Positive Cases'))

plt.show()
AgeGroupDetails =  pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
AgeGroupDetails.head()

age_percent = {}
for i in range(len(AgeGroupDetails)):
    age_percent[AgeGroupDetails['AgeGroup'][i]] = AgeGroupDetails['Percentage'][i][0:-1]
age_percent
plt.pie(age_percent.values(), labels=age_percent.keys(), autopct='%1.1f%%')
        
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Cases by Age groups')
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()  
IndividualDetails =  pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv',na_values=missing_values)
IndividualDetails.head()


def fill_missing(column):
    col = column.value_counts().to_dict()
    prob = random.choices(list(col.keys()), weights = list(col.values()), k=100)
    null_val = column.isnull()
    for i in range(len(null_val)):
        if null_val[i]== True:
            column[i] = prob[i%100]
    return column
            
print(IndividualDetails.isnull().sum()) #before filling values

detected_state = IndividualDetails['detected_state']
age = IndividualDetails['age']
gender = IndividualDetails['gender']
status = IndividualDetails['current_status']
nationality = IndividualDetails['nationality']

fill_missing(detected_state)
fill_missing(age)
fill_missing(gender)
fill_missing(status)
fill_missing(nationality)


print(IndividualDetails.isnull().sum())  #after filling values
for i in range(len(IndividualDetails['age'])):
    IndividualDetails['age'][i] = int(IndividualDetails['age'][i][:2])

plt.pie(gender.value_counts().to_dict().values(),labels=gender.value_counts().to_dict().keys(),autopct='%1.1f%%')
plt.show()

from datetime import datetime
for i in range(len(IndividualDetails['diagnosed_date'])):
    IndividualDetails['diagnosed_date'][i] = datetime.strptime(IndividualDetails['diagnosed_date'][i], '%d/%m/%Y').date()
IndividualDetails['diagnosed_date']
weeks = []
for i in range(len(IndividualDetails['diagnosed_date'])):
    weeks.append(int(IndividualDetails['diagnosed_date'][i].strftime("%U")))
IndividualDetails['weeks'] = weeks
IndividualDetails.head()
X = IndividualDetails.loc[:,['gender','detected_state','nationality','age','weeks']].values
y = IndividualDetails.loc[:,['current_status']].values

X[:,0] = LabelEncoder().fit_transform(X[:,0])
X[:,1] = LabelEncoder().fit_transform(X[:,1])
X[:,2] = LabelEncoder().fit_transform(X[:,2])
y[:,0] = LabelEncoder().fit_transform(y[:,0])
y=y.astype('int')




x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
entropy = DecisionTreeClassifier(criterion='entropy')
entropy.fit(x_train,y_train)
y_pred = entropy.predict(x_test)
accuracy_score(y_test,y_pred)
clf = RandomForestClassifier(n_jobs=2,random_state=0)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test,y_pred)
