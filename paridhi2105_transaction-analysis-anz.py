# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_excel("../input/anz-synthesised-transaction-datasetxlsx/ANZ synthesised transaction dataset.xlsx")

tmp=data.head()



#We assume that a customer uses all the 'balance' he has over the 3 month period.The total 'amount' used by each customer can be approximated as his salary for three month period.
corr_age=list()

corr_gender=list()
salary_data=data[['amount','customer_id']]
salary={}

for key in list(salary_data['customer_id'].unique()):

    bal=0

    for i in list(data.index[data['customer_id'] == key]):

        bal+=salary_data.iloc[i,0]

    corr_age.append(data.loc[i,'age'])

    corr_gender.append(data.loc[i,'gender'])

    salary[key]=bal 
#plotting salary vs cust_id for the dataset (3 months)

fig = plt.figure() 

ax = fig.add_axes([0,0,1,1])

ax.set_title('Salary info for 3 months')

ax.set_xlabel('Customer_id')

ax.set_ylabel('Salary')

ax.bar(salary.keys(),salary.values())

ax.set_xticks(np.asarray([i for i in range(len(salary.keys()))]))

ax.set_xticklabels(salary.keys(),rotation=90)

plt.show()
salary_annual=salary.copy()

for key,val in salary_annual.items():

    val=(val/3)*12

    salary_annual[key]=val

    

#salary_annual contains the annual salary of each customer
y=list(salary_annual.values())
#checking against age

x_age=corr_age 

x_gender=corr_gender

plt.title('Relation between age and salary')

plt.xlabel('Age')

plt.ylabel('Annual Salary')

plt.scatter(x_age,y)

plt.show()
#checking against gender

female_total=0

male_total=0

for index in range(len(x_gender)):

    if x_gender[index]=='F': female_total+=y[index]

    else : male_total+=y[index]

plt.title('Ratio of salary on the basis of gender')

plt.pie([female_total,male_total],labels=['Female','Male'],colors=['pink','blue'],startangle=0)

plt.show()
modified_dataset=pd.DataFrame(np.asarray(x_age),columns=['age'])

modified_dataset['gender']=x_gender

modified_dataset['salary']=y

modified_dataset.describe()

modified_dataset['age']=(modified_dataset['age']-modified_dataset['age'].mean())/modified_dataset['age'].std()

modified_dataset['salary']=(modified_dataset['salary']-modified_dataset['salary'].mean())/modified_dataset['salary'].std()

features_original=modified_dataset[['age','gender']]

label=modified_dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
col_transformer=ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')

features=np.array(col_transformer.fit_transform(features_original),dtype=np.float32)

features=features[:,1:]

features=pd.DataFrame(features,columns=['gender','age'])

features['gender']=(features['gender']-features['age'].mean())/features['gender'].std()

#features=np.array(features)      

features_train,features_test,label_train,label_test=train_test_split(features,label,test_size=0.2)
model=LinearRegression()

model.fit(features_train,label_train)
label_train_pred=model.predict(features_train)

accuracy_train=mean_squared_error(label_train,label_train_pred)

print('The training accuracy (MSE) is {}'.format(accuracy_train))
label_test_pred=model.predict(features_test)

accuracy_test=mean_squared_error(label_test,label_test_pred)

print('The test accuracy (MSE) is {}'.format(accuracy_test))