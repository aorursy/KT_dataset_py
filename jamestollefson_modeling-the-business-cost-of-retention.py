# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.head()
data.describe()
pd.options.mode.chained_assignment = None



data['RealHours'] = data['DailyRate'] / data['HourlyRate'] * 10

data['HoursDelta'] = data['RealHours'] - data['StandardHours']

print(data['HoursDelta'][:15])



data['PaidOverTime'] = data['HoursDelta'] - 80

print(data['PaidOverTime'][:15])

for row, value in enumerate(data['PaidOverTime']):

    if value < 0:

        data['PaidOverTime'][row] = 0

    if value > 0:

        data['PaidOverTime'][row] = data['PaidOverTime'][row] / 1.5

    if data['OverTime'][row] == 'No':

        data['PaidOverTime'][row] = 0

        

print(data['PaidOverTime'][:15])
data['OT'] = 0

data['OT'][data['OverTime'] == 'Yes'] = 1





_ = plt.scatter(data['MonthlyRate'], data['MonthlyIncome'], c=data['OT'])

_ = plt.xlabel('Monthly Rate')

_ = plt.ylabel('Monthly Income')

_ = plt.title('Monthly Rate vs. Monthly Income')

plt.show()

print(np.corrcoef(data['MonthlyRate'], data['MonthlyIncome']))



_ = plt.scatter(data['DailyRate'], data['MonthlyRate'], c=data['OT'])

_ = plt.xlabel('Daily Rate')

_ = plt.ylabel('Monthly Rate')

_ = plt.title('Daily Rate vs. Monthly Rate')

plt.show()

print(np.corrcoef(data['DailyRate'], data['MonthlyRate']))
_ = plt.scatter((data['MonthlyRate'] / data['DailyRate']), data['DailyRate'])

_ = plt.xlabel('Ratio of Monthly to Daily Rate')

_ = plt.ylabel('Daily Rate')

_ = plt.title('Monthly/Daily Rate Ratio vs. Daily Rate')

plt.show()



_ = plt.scatter((data['MonthlyRate'] / data['DailyRate']), data['DailyRate'])

_ = plt.semilogx()

_ = plt.xlabel('Logarithm of Monthly/Daily Rate Ratio')

_ = plt.semilogy()

_ = plt.ylabel('Logarithm of Daily Rate')

_ = plt.title('Logarithmic Monthly/Daily Rate Ratio vs. Log. Daily Rate')

plt.show()



data['lograteratio'] = np.log(data['MonthlyRate'] / data['DailyRate'])

_ = plt.hist(data['lograteratio'], bins=50)

_ = plt.xlabel('Logarithmic Monthly/Daily Rate Ratio')

_ = plt.ylabel('Count')

_ = plt.title('Histogram of Logarithmic Monthly/Daily Rate Ratio')

plt.show()
data['left'] = 0

data['left'][data['Attrition'] == 'Yes'] = 1

x = data['left']

print('Monthly Rate:', np.corrcoef(data['MonthlyRate'], x))

print('Daily Rate', np.corrcoef(data['DailyRate'], x))

print('Hourly Rate', np.corrcoef(data['HourlyRate'], x))

print('Monthly Income', np.corrcoef(data['MonthlyIncome'], x))

print('Log Rate Ratio', np.corrcoef(data['lograteratio'] ** 35, x))
del data['RealHours']

del data['HoursDelta']

del data['PaidOverTime']



data['lograteratio'] = data['lograteratio'] ** 35
data.head()
from bokeh.plotting import figure, ColumnDataSource

from bokeh.io import output_file, show



source = ColumnDataSource(data)

p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')



p.circle(fertility, female_literacy, source=source)



# Call the output_file() function and specify the name of the file

output_file('fert_lit.html')
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['YearsAtCompany']

y = data['Age']

z = data['YearsInCurrentRole']

s = data['WorkLifeBalance']

c = data['left']

cmap = plt.get_cmap('seismic')

_ = ax.scatter(xs=x, ys=y, zs=z, c=c, cmap=cmap)

_ = ax.set_xlabel('Years at Company')

_ = ax.set_ylabel('Age')

_ = ax.set_zlabel('Years in Current Role')

_ = plt.title('')

plt.show()
young = data[(data['Age'] < 30) & (data['YearsAtCompany'] <= 2) & (data['YearsInCurrentRole'] <= 1)]

data['young'] = 0

data['young'][(data['Age'] < 30) & (data['YearsAtCompany'] <= 2) & (data['YearsInCurrentRole'] <= 1)] = 1



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = young['YearsAtCompany']

y = young['Age']

z = young['YearsInCurrentRole']

s = young['WorkLifeBalance']

c = young['left']

cmap = plt.get_cmap('seismic')

_ = ax.scatter(xs=x, ys=y, zs=z, c=c, cmap=cmap)

_ = ax.set_xlabel('Years at Company')

_ = ax.set_ylabel('Age')

_ = ax.set_zlabel('Years in Current Role')

_ = plt.title('')

plt.show()



_ = sns.boxplot(young['left'], young['Age'])

plt.show()
print(np.corrcoef(young['left'], young['Age']))

print(np.count_nonzero(young['left']) / len(young['left']))

percent1 = np.round(np.count_nonzero(young['left']) / len(young['left']) * 100, decimals=2)

print('{}% of workers aged under 30 leaves the firm'.format(percent1))

print(np.corrcoef(data['left'], data['Age']))

percent = np.round(np.count_nonzero(data['left']) / len(data['left']) * 100, decimals=2)

print('{}% of the total population leaves the firm.'.format(percent))

corr = np.corrcoef(data['young'], data['left'])

for item in corr[1]:

    print(np.round(item * 100, decimals=2),'%')
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['YearsAtCompany']

y = data['Age']

z = data['YearsInCurrentRole']

s = data['WorkLifeBalance']

c = data['left']

cmap = plt.get_cmap('seismic')

_ = ax.scatter(xs=x, ys=y, zs=z, c=c, cmap=cmap, s=s ** 3)

_ = ax.set_xlabel('Years at Company')

_ = ax.set_ylabel('Age')

_ = ax.set_zlabel('Years in Current Role')

_ = plt.title('')

plt.show()
mid = data[(data['Age'] > 35) & (data['Age'] <= 40) & (data['YearsAtCompany'] <= 10) & (data['YearsAtCompany'] > 2) & (data['YearsInCurrentRole'] <= 7)]

data['mid'] = 0

data['mid'][(data['Age'] > 35) & (data['Age'] <= 40) & (data['YearsAtCompany'] <= 10) & (data['YearsAtCompany'] > 2) & (data['YearsInCurrentRole'] <= 7)] = 1



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = mid['YearsAtCompany']

y = mid['Age']

z = mid['YearsInCurrentRole']

s = mid['WorkLifeBalance']

c = mid['left']

cmap = plt.get_cmap('seismic')

_ = ax.scatter(xs=x, ys=y, zs=z, c=c, cmap=cmap)

_ = ax.set_xlabel('Years at Company')

_ = ax.set_ylabel('Age')

_ = ax.set_zlabel('Years in Current Role')

_ = plt.title('')

plt.show()



print(np.count_nonzero(mid['left']) / len(mid['left']))

percent1 = np.round(np.count_nonzero(mid['left']) / len(mid['left']) * 100, decimals=2)

print('{}% of my mid-career workers leave the firm'.format(percent1))

print('There are {} mid-career employees with exceptionally low average attrition in this firm.'.format(len(mid)))

corr = np.corrcoef(data['mid'], data['left'])

for item in corr[1]:

    print(np.round(item * 100, decimals=2),'%')
_ = sns.kdeplot(data = data['Age'], data2 = data['TotalWorkingYears'])

_ = plt.scatter(data['Age'], data['TotalWorkingYears'], alpha=.5, s=20, c=data['left'])

_ = plt.xlabel('Age')

_ = plt.ylabel('Tenure (in years)')

_ = plt.title('Age vs. Tenure')

plt.show()



_ = sns.kdeplot(data=data['MonthlyIncome'], data2=data['Age'])

_ = plt.scatter(data['MonthlyIncome'], data['Age'], alpha=0.5, s=20, c=data['left'])

_ = plt.xlabel('Monthly Income')

_ = plt.ylabel('Age')

_ = plt.title('Monthly Income vs. Age')

plt.show()
data['high_income'] = 0

data['high_income'][(data['Age'] >= 25) & (data['MonthlyIncome'] > 13000)] = 1



count = np.count_nonzero(data['high_income'])

print('There are {} highly paid employees with low average attrition'.format(count))

corr = np.corrcoef(data['high_income'], data['left'])

for item in corr[0]:

    l = []

    l.append(np.round(item * 100, decimals=2))

print('Correlation between this group and attrition is {}%'.format(l[0]))
from sklearn.preprocessing import LabelEncoder as LE



data['Attrition'] = LE().fit_transform(data['Attrition'])

data['Department'] = LE().fit_transform(data['Department'])

data['EducationField'] = LE().fit_transform(data['EducationField'])

data['Gender'] = LE().fit_transform(data['Gender'])

data['JobRole'] = LE().fit_transform(data['JobRole'])

data['MaritalStatus'] = LE().fit_transform(data['MaritalStatus'])

data['Over18'] = LE().fit_transform(data['Over18'])

data['OverTime'] = LE().fit_transform(data['OverTime'])

data['BusinessTravel'] = LE().fit_transform(data['BusinessTravel'])

del data['left']

del data['OT']

del data['EmployeeNumber']

del data['EmployeeCount']
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans



X = data

y = data['Attrition']

del X['Attrition']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)





cluster = KMeans(n_clusters=80, random_state=42).fit_predict(X_train)

X_train['cluster'] = cluster

X_train['cluster'].plot(kind='hist', bins=80)

_ = plt.xlabel('Cluster')

_ = plt.ylabel('Count')

_ = plt.title('Histogram of Clusters')

plt.show()



_ = plt.scatter(x=X_train['Age'], y=X_train['DailyRate'], c=X_train['cluster'], cmap='Blues')

_ = sns.kdeplot(data=X_train['Age'], data2=X_train['DailyRate'])

_ = plt.xlabel('Age')

_ = plt.ylabel('Daily Rate')

_ = plt.title('Clusters within Age/Daily Rate')

plt.show()



x = np.corrcoef(X_train['cluster'], y_train)

print(x)
cluster = KMeans(n_clusters=80, random_state=42).fit_predict(X_test)

X_test['cluster'] = cluster

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.metrics import accuracy_score



fmodel = RFC(n_estimators=100, random_state=42, max_depth=11, max_features=11).fit(X_train, y_train)

prediction = fmodel.predict(X_test)

score = accuracy_score(y_test, prediction)

print(score)
from sklearn.svm import SVC



model = SVC(random_state=42).fit(X_train, y_train)

prediction = model.predict(X_test)

score = accuracy_score(y_test, prediction)

print(score)
from sklearn.ensemble import AdaBoostClassifier as ABC



model = ABC(n_estimators=100, random_state=42, learning_rate=.80).fit(X_train, y_train)

prediction = model.predict(X_test)

score = accuracy_score(y_test, prediction)

print(score)
from sklearn.ensemble import BaggingClassifier as BC



model = BC(n_estimators=100, random_state=42).fit(X_train, y_train)

prediction = model.predict(X_test)

score = accuracy_score(y_test, prediction)

print(score)
from sklearn.ensemble import ExtraTreesClassifier as XTC



model = XTC(n_estimators=100, random_state=42, criterion='entropy', max_depth=20).fit(X_train, y_train)

prediction = model.predict(X_test)

score = accuracy_score(y_test, prediction)

print(score)