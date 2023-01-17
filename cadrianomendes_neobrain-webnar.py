print("hello")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
yoe_path = '../input/salary/Salary.csv'

yoe_data = pd.read_csv(yoe_path) 

print("Count lines: ",  len(yoe_data))

yoe_data
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
sns.regplot(x="YearsExperience", y="Salary", data=yoe_data);
X = yoe_data.iloc[:, :-1].values #get a copy of dataset exclude last column

y = yoe_data.iloc[:, 1].values #get array of dataset in column 1st
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
regressor.save("")



from sklearn.linear_model import LinearRegression

regressor.load("")

# Visualizing the Training set results

viz_train = plt

viz_train.scatter(X_train, y_train, color='red')

viz_train.plot(X_train, regressor.predict(X_train), color='blue')

viz_train.title('Salary VS Experience (Training set)')

viz_train.xlabel('Year of Experience')

viz_train.ylabel('Salary')

viz_train.show()



# Visualizing the Test set results

viz_test = plt

viz_test.scatter(X_test, y_test, color='red')

viz_test.plot(X_train, regressor.predict(X_train), color='blue')

viz_test.title('Salary VS Experience (Test set)')

viz_test.xlabel('Year of Experience')

viz_test.ylabel('Salary')

viz_test.show()
regressor.predict([[20]])
#REAL CASE
baltimore_file_path = '../input/baltimore/baltimore-city-employee-salaries-fy2019-1.csv'

baltimore_data = pd.read_csv(baltimore_file_path) 

baltimore_data.columns
baltimore_data
print('Count Jobs', len(baltimore_data.JOBTITLE.unique()))

print('Count Departments', len(baltimore_data.DEPTID.unique()))
from dateutil.relativedelta import relativedelta, MO

import time

import datetime
def get_years_of_experience(str_date):

    try:

        s = str_date[:-12]

        datetime1 = datetime.datetime.strptime(s, "%m/%d/%Y")

        datetime2 = datetime.datetime(2020, 6, 1)



        time_difference = relativedelta(datetime2, datetime1)

        difference_in_years = time_difference.years



        return difference_in_years

    except Exception as e:

        print(e)

        return 0

    

get_years_of_experience("08/27/2010 12:00:00 AM")  
baltimore_data['years_of_experience'] = [get_years_of_experience(date) for date in baltimore_data['HIRE_DT']] 

baltimore_data.head()
sns.regplot(x="years_of_experience", y="Gross", data=baltimore_data);
!pip install gender_guesser
import gender_guesser.detector as gender

detector = gender.Detector()
def parse_name(full_name):

    name = ''

    

    try:

        name = full_name.split(',')[1]

    except:

        pass



    try:

        name = name.lstrip().split(' ')[0]

    except:

        pass

    

    return name



def get_gender(full_name):

    return detector.get_gender(parse_name(full_name))



full_name_1 = "Aaron, Patricia G"

print(get_gender(full_name_1))



full_name_3 = "Abbeduto,Mack"

print(get_gender(full_name_3))



full_name_2 = "Rocha, Tiago a"

print(get_gender(full_name_2))
baltimore_data['gender'] = [get_gender(name) for name in baltimore_data['NAME']]
baltimore_data
sns.catplot(x="gender", y="Gross", kind="boxen", data=baltimore_data).set_xticklabels(rotation=30)
y = baltimore_data['Gross']

y = y.fillna(0)

baltimore_features = ['gender', 'years_of_experience', 'JOBTITLE', 'DEPTID']

X = pd.get_dummies(baltimore_data[baltimore_features])

X = X.fillna(0)
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Define model

melbourne_model = DecisionTreeRegressor()

# Fit model

melbourne_model.fit(train_X, train_y)



# get predicted prices on validation data

val_predictions = melbourne_model.predict(val_X)

print('MAE', mean_absolute_error(val_y, val_predictions))

print('RMSE', mean_squared_error(val_y, val_predictions))
# initialize list of lists 

df = val_X[:3]



predictions = melbourne_model.predict(df)

val_X[:3].index

baltimore_data.loc[4493]
predictions[0]
baltimore_data.loc[6945]
predictions[1]
baltimore_data.loc[9043]
predictions[2]
baltimore_data[baltimore_data['JOBTITLE'].str.contains("Director")]
baltimore_data[baltimore_data['DESCR'].str.contains("HR")]
for column in live_example.columns:

    if column.startswith('JOBTITLE'):

        print(column.split('_')[1])
yoe = 15

jobtitle = 'Executive Assistant'

gender = 'unknown'

depid = 'A91011'



live_example = X[:1]

for column in live_example.columns:

    live_example[column] = 0



live_example['years_of_experience'][0] = yoe

live_example['gender_'+gender][0] = 1

live_example['JOBTITLE_'+jobtitle][0] = 1

live_example['DEPTID_'+depid][0] = 1



#live_example



print(melbourne_model.predict(live_example))