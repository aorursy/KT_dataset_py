# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# loading the data

data = pd.read_csv('/kaggle/input/glassdoor_jobs.csv')

data.head(3)
data.info()
data.columns
data.isna().sum()
# take a copy of data and remove unnecessary attributes

emp_data = data.copy(deep= True)

emp_data.drop(columns= ['Unnamed: 0'], inplace = True)

emp_data.head()
emp_data.columns
emp_data['Job Title'].unique()


# job title cleaning



def jobtitle_cleaner(title):

    if 'data scientist' in title.lower():

        return 'D-sci'

    elif 'data engineer' in title.lower():

        return 'D-eng'

    elif 'analyst' in title.lower():

        return 'analyst'

    elif 'machine learning' in title.lower():

        return 'ML'

    elif 'manager' in title.lower():

        return 'manager'

    elif 'director' in title.lower():

        return 'director'

    elif 'research' in title.lower():

        return 'R&D'

    else:

        return 'na'
emp_data['JobTitles'] = emp_data['Job Title'].apply(jobtitle_cleaner)
emp_data['Job Title'].unique()
emp_data['JobTitles'].unique()


emp_data['JobTitles'].value_counts()
senior_list = ['sr','sr.','senior','principal', 'research', 'lead', 'R&D','II', 'III']

junior_list = ['jr','jr.','junior']





def jobseniority(title):

    for i in senior_list:

        if i in title.lower():

            return 'Senior Prof'

            

    for j in junior_list:

        if j in title.lower():

            return 'Junior Prof'

        else:

            return 'No Desc'


emp_data['Job Seniority'] = emp_data['Job Title'].apply(jobseniority)


emp_data['Job Seniority'].unique()
emp_data['Job Seniority'].value_counts()
# job descriptions

jobs_list = ['python', 'excel','r studio', 'spark','aws']



for i in jobs_list:

    emp_data[i+'_'+'job'] = emp_data['Job Description'].apply(lambda x : 1 if i in x.lower() else 0)



for i in jobs_list:

    print(emp_data[i+'_'+'job'].value_counts())
emp_data['Company Name'].unique()
emp_data['Company Name'][0].split('\n')[0]
# remove numbers from company name

emp_data['Company Name'] = emp_data['Company Name'].apply(lambda x : x.split("\n")[0])

emp_data['Company Name'].value_counts()
emp_data['Headquarters'].unique()


emp_data['Hquarters'] = emp_data['Headquarters'].str.split(',').str[1]

emp_data['Hquarters'].value_counts().head()
emp_data['Location'].unique()


emp_data['loaction spots'] = emp_data['Location'].str.split(',').str[1]

emp_data['loaction spots'].value_counts().head()
emp_data['Competitors'].unique()
emp_data['compitator company'] = emp_data['Competitors'].str.split(',').str[0].replace('-1', 'no compitator')


emp_data['compitator company'].value_counts()
emp_data['Type of ownership'].unique()


emp_data['Ownership'] = emp_data['Type of ownership'].str.split('-').str[1].replace(np.NaN, 'others')

emp_data['Ownership'].value_counts()
emp_data['Revenue'].unique()
emp_data['Revenue'] = emp_data['Revenue'].str.replace('-1','others')
emp_data['Revenue'].value_counts()
emp_data['Size'].unique()
emp_data['Size'] = emp_data['Size'].str.replace('-1','others')

emp_data['Size'].value_counts()
emp_data["Salary Estimate"].unique()


emp_data['min_sal'] = emp_data['Salary Estimate'].str.split(",").str[0].str.replace('(Glassdoor est.)','')
emp_data['min_sal'] = emp_data['min_sal'].str.replace('(Glassdoor est.)','').str.split('-').str[0].str.replace('$','').str.replace('K','')
emp_data['min_sal'].unique()
emp_data['min_sal'] = emp_data['min_sal'].str.replace('Employer Provided Salary:','')

emp_data['min_sal'].unique()
emp_data['max_sal'] = emp_data['Salary Estimate'].str.split(",").str[0].str.replace('(Glassdoor est.)','')

emp_data['max_sal']
emp_data['max_sal'] = emp_data['max_sal'].str.replace('(Glassdoor est.)','').str.split('-').str[1].str.replace('$','').str.replace('K','')


emp_data['max_sal'] = emp_data['max_sal'].str.replace('(Employer est.)','')
emp_data['max_sal'] = emp_data['max_sal'].str.split().str[0].str.replace('(','').str.replace(')','')
emp_data['max_sal'].unique()
emp_data['min_sal'] = pd.to_numeric(emp_data['min_sal'], errors='coerce')

type(emp_data['min_sal'])
emp_data['min_sal'].isna().sum()
emp_data['min_sal'].hist()

plt.show()
emp_data['max_sal'].isna().sum()


emp_data['min_sal'] = emp_data['min_sal'].replace(np.nan, emp_data['min_sal'].mean())
emp_data['min_sal'].isna().sum()


emp_data['max_sal'] = pd.to_numeric(emp_data['max_sal'], errors='coerce')

type(emp_data['max_sal'])
emp_data['max_sal'].isnull().sum()


emp_data['max_sal'].hist()

plt.show()


emp_data['avg.salary'] = (emp_data['min_sal'] + emp_data['max_sal'])/ 2
emp_data['avg.salary'].hist()

plt.show()


emp_data.head()
final_data = emp_data[['Rating',

       'Company Name', 'Size',

       'Type of ownership','Sector', 'Revenue',

       'JobTitles', 'Job Seniority', 'python_job', 'excel_job', 'r studio_job',

       'spark_job', 'aws_job', 'Hquarters', 'loaction spots',

       'compitator company', 'Ownership','avg.salary']]

final_data.head()
final_data = pd.get_dummies(data = final_data, columns = ['Company Name', 'Size', 'Type of ownership', 'Sector',

       'Revenue', 'JobTitles', 'Job Seniority','Hquarters', 'loaction spots',

       'compitator company', 'Ownership'])


final_data.head()
from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

final_data[['Rating', 'avg.salary']] = ms.fit_transform(final_data[['Rating', 'avg.salary']])
final_data.head()
# split the data into attributes and lable

X = final_data.drop(columns= 'avg.salary').values

y = final_data.iloc[:, 6].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Using GridSearchCV to find the best algorithm for this problem

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR



def find_best_model(X, y):

    models = {

        'linear_regression': {

            'model': LinearRegression(),

            'parameters': {

                'n_jobs': [-1]

            }

            

        },

        

        'decision_tree': {

            'model': DecisionTreeRegressor(criterion='mse', random_state= 0),

            'parameters': {

                'max_depth': [5,10]

            }

        },

        

        'random_forest': {

            'model': RandomForestRegressor(criterion='mse', random_state= 0),

            'parameters': {

                'n_estimators': [10,15,20,50,100,200]

            }

        },

        

        'svm': {

            'model': SVR(gamma='auto'),

            'parameters': {

                'C': [1,10,20],

                'kernel': ['rbf','linear']

            }

        }



    }

    

    scores = [] 

    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)

        

    for model_name, model_params in models.items():

        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)

        gs.fit(X, y)

        scores.append({

            'model': model_name,

            'best_parameters': gs.best_params_,

            'Test score': gs.best_score_

        })

        

    return pd.DataFrame(scores, columns=['model','best_parameters','Test score'])



find_best_model(X_train, y_train)
# Creating linear regression model

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

# Fitting the dataset to the model

lr_model.fit(X_train, y_train)

print("Accuracy of the Linear Regression Model on Training set is : {}% and on Test set is {}%".format(round(lr_model.score(X_train, y_train),4)*100, round(lr_model.score(X_test, y_test),4)*100))
# Creating decision tree regression model

from sklearn.tree import DecisionTreeRegressor

decision_model = DecisionTreeRegressor(criterion='mse', max_depth=10, random_state=0)

# Fitting the dataset to the model

decision_model.fit(X_train, y_train)

print("Accuracy of the Decision Tree Regression Model on Training set is : {}% and on Test set is {}%".format(round(decision_model.score(X_train, y_train),4)*100, round(decision_model.score(X_test, y_test),4)*100))
# Creating random forest regression model

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=0)

# Fitting the dataset to the model

forest_model.fit(X_train, y_train)

print("Accuracy of the Random Forest Regression Model on Training set is : {}% and on Test set is {}%".format(round(forest_model.score(X_train, y_train),4)*100, round(forest_model.score(X_test, y_test),4)*100))
# Creating AdaBoost regression model

from sklearn.ensemble import AdaBoostRegressor

adb_model = AdaBoostRegressor(base_estimator=decision_model, n_estimators=250, learning_rate=1, random_state=0)

# Fitting the dataset to the model

adb_model.fit(X_train, y_train)

print("Accuracy of the AdaBoost Regression Model on Training set is : {}% and on Test set is {}%".format(round(adb_model.score(X_train, y_train),4)*100, round(adb_model.score(X_test, y_test),4)*100))