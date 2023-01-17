import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/HR-Employee-Attrition.csv', header=0)

data = data.dropna()

print(data.shape)
print(list(data.columns))
data.head()
data.drop(columns='Attrition').dtypes
print(data['Attrition'].dtype)
data.isna().sum()
data.duplicated().sum()
# cat_col = data.select_dtypes(exclude=np.number)

# fig, ax = plt.subplots(3, 3, figsize=(30, 40))

# for variable, subplot in zip(cat_col, ax.flatten()):

#     cp=sns.countplot(data[variable], ax=subplot,order = data[variable].value_counts().index,hue=data['Attrition'])

#     cp.set_title(variable,fontsize=40)

#     cp.legend(fontsize=30)

#     for label in subplot.get_xticklabels():

#         label.set_rotation(90)

#         label.set_fontsize(36)                

#     for label in subplot.get_yticklabels():

#         label.set_fontsize(36)        

#         cp.set_ylabel('Count',fontsize=40)    

# plt.tight_layout()
data['Attrition'].replace({'No':0,'Yes':1},inplace=True)
num_cols = data.select_dtypes(include = np.number)
a = num_cols[num_cols.columns].hist(bins=15, figsize=(15,35), layout=(9,3),color = 'red',alpha=0.6)
cat_col = data.select_dtypes(exclude=np.number)
cat_col.columns
fig, ax = plt.subplots(4, 2, figsize=(15, 15))

for variable, subplot in zip(cat_col, ax.flatten()):

    sns.countplot(data[variable], ax=subplot,palette = 'plasma')

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

plt.tight_layout()
data[['StandardHours','EmployeeCount']].describe()
data[['StandardHours','EmployeeCount']].corr()
corr = data.drop(columns=['StandardHours','EmployeeCount']).corr()

corr.style.background_gradient(cmap='YlGnBu')
# Age - TotalWorkingYears - JobLevel - MonthlyIncome

# JobLevel - TotalWorkingYears - 
hm = data.drop(columns=['StandardHours','EmployeeCount'],axis=1)
hm
cols = ['Age', 'BusinessTravel', 'Department',

       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',

        'EnvironmentSatisfaction', 'Gender', 

       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',

       'MaritalStatus', 'NumCompaniesWorked',

       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',

       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',

       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']

for col in cols:

    pd.crosstab(data[col],data.Attrition).plot(kind='bar',color = ('blue','red'),figsize=(10,5))
# Age Vs Attrition - From data, it appears that attrition is more at age group 18-23

# % of attrition is more among people who travel frequently

# % of attrition is more in sales department

# %of attrition is more during 0-1 years of working in company

# People in job role of Sales Representative tend to have more attrition %

# From given data, overtime population has more attrition
data.columns.shape
cat_col.columns.shape
num_cols.columns.shape
cat_col_encoded = pd.get_dummies(cat_col)
cat_col_encoded
df = pd.concat([num_cols,cat_col_encoded],sort=False,axis=1)
df.head()
X = df.drop(columns='Attrition')
y = df['Attrition']
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
k = 5

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

neigh
yhat = neigh.predict(X_test)

yhat[0:5]
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
metrics.confusion_matrix(y_train, neigh.predict(X_train))
metrics.confusion_matrix(y_test, yhat)
Ks = 800

import numpy as np

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 