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
base = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
#Importing Libraries



import seaborn as sns

import matplotlib.pyplot as plt
#Verifying if there is any missing values



base.isna().sum()
base.rename(columns = {'race/ethnicity': 'race', 'parental level of education': 'parents_level_education', 'test preparation course': 'course', 'math score': 'math', 'reading score': 'reading', 'writing score': 'writing'}, inplace = True)
#Creating new variables



base['total_score'] = base['math'] + base['reading'] + base['writing']

base['mean_score'] = base['total_score'] / 3
base.head()
sns.countplot(base.gender)

plt.xlabel('GENDER')

plt.ylabel('QUANTITY')
mean_score_female = base[(base.gender == 'female')]

mean_score_male = base[(base.gender == 'male')]
plt.figure(figsize = (12,8))

sns.barplot(x = mean_score_female.course, y= mean_score_female.mean_score, hue = mean_score_female.race)

plt.xlabel('PREPARATION COURSE')

plt.ylabel('MEAN')
plt.figure(figsize = (12,8))

sns.barplot(x = mean_score_male.course, y= mean_score_male.mean_score, hue = mean_score_male.race)

plt.xlabel('PREPARATION COURSE')

plt.ylabel('MEAN')

base[(base.gender == 'female')].mean().plot(color = 'red')

base[(base.gender == 'male')].mean().plot(color = 'black')
plt.figure(figsize = (12,8))

mean_score_female.describe().plot()

mean_score_female.describe()
plt.figure(figsize = (12,8))

mean_score_male.describe().plot()

mean_score_male.describe()
plt.figure(figsize = (10,8))

sns.countplot(base.parents_level_education)

base.parents_level_education.value_counts()

plt.xlabel('PARENTS EDUCATION')

plt.ylabel('QUANTITY')
parents_masters = base[(base.parents_level_education == "master's degree")]

parents_bach = base[(base.parents_level_education == "bachelor's degree")]

parents_some_college = base[(base.parents_level_education == "some college")]

parents_associate = base[(base.parents_level_education == "associate's degree")]

parents_high_school = base[(base.parents_level_education == "high school")]

parents_some_high_school = base[(base.parents_level_education == "some high school")]
parents_masters.describe().plot()

parents_masters.describe()
parents_bach.describe().plot()

parents_bach.describe()
parents_some_college.describe().plot()

parents_some_college.describe()
parents_associate.describe().plot()

parents_associate.describe()
parents_high_school.describe().plot()

parents_high_school.describe()
parents_some_high_school.describe().plot()

parents_some_high_school.describe()
base.corr()
plt.figure(figsize = (10,8))

sns.scatterplot(x = base.parents_level_education, y = base.total_score, hue = base.course)

plt.xlabel('PARENTS EDUCATION')

plt.ylabel('TOTAL SCORE')

plt.figure(figsize = (10,8))

sns.scatterplot(x = base.parents_level_education, y = base.writing, color = 'red')

sns.scatterplot(x = base.parents_level_education, y = base.reading, color = 'black')

sns.scatterplot(x = base.parents_level_education, y = base.math, color = 'blue')

plt.xlabel('PARENTS EDUCATION')

plt.ylabel('GRADES')
plt.figure(figsize = (10,8))

sns.scatterplot(x = base.lunch, y = base.total_score)

plt.xlabel('LUNCH')

plt.ylabel('TOTAL SCORE')
base.drop('mean_score', axis = 1, inplace = True)
base['total_score'] = base['total_score'] /3

base.head()
#Importing libraries



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
#Creating variables



X = base.iloc[:,0:8].values

Y = base.iloc[:,8].values
#LabelEncoder - transforming cateroric variables



label_encoder_X = LabelEncoder()

X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

X[:, 1] = label_encoder_X.fit_transform(X[:, 1])

X[:, 2] = label_encoder_X.fit_transform(X[:, 2])

X[:, 3] = label_encoder_X.fit_transform(X[:, 3])

X[:, 4] = label_encoder_X.fit_transform(X[:, 4])
#Scaler



scaler = StandardScaler()

X = scaler.fit_transform(X)
#Train and test



train_x,test_x,train_y,test_y = train_test_split(X, Y, test_size = 0.05)
#Decision Tree Regression



regressor = DecisionTreeRegressor()

regressor.fit(train_x, train_y)



score = regressor.score(train_x, train_y)



predicts = regressor.predict(test_x)

mae = mean_absolute_error(test_y, predicts)

print(score, mae)
#Random Forest Regressor



regressor = RandomForestRegressor(n_estimators = 10)

regressor.fit(train_x, train_y)



score = regressor.score(train_x, train_y)



predicts = regressor.predict(test_x)

mae = mean_absolute_error(test_y, predicts)



print(score, mae)
#SVR



regressor = SVR(kernel = 'linear')

regressor.fit(train_x, train_y)



score = regressor.score(train_x, train_y)



predicts = regressor.predict(test_x)

mae = mean_absolute_error(test_y, predicts)



print(score, mae)