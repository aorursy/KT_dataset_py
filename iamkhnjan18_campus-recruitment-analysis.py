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
#Importing Liabraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Reading the DataSet
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
pd.set_option('Display.max_columns' , 15)
df.head(10)
#Shape Of DataSet
df.shape
#Getting the null values
df.isna().sum()
# Removing null values
df.dropna(axis = 'index' , how  = 'any' , inplace = True)
# The DataTypes of the all attributes
df.dtypes
# About Gender
# Graphical view of attribute Gender
plt.style.use('fivethirtyeight')
plt.title('Gender Counting')
sns.countplot(df['gender'] , color = '#000066',edgecolor = 'k' ,saturation = 1 )
plt.tight_layout()
# About SSC Board
# Graphical view of attribute ssc_b
plt.title('SSC Board')
labels = ['Others' , 'Central']
colors = ['#B0E0E6' , '#4682B4']
plt.pie(df['ssc_b'].value_counts() ,labels = labels , startangle = 90 ,
        colors = colors ,counterclock = False, wedgeprops = {'edgecolor' : 'k' , 'linestyle' : 'solid' ,},
        autopct = '%1.1f%%' , shadow = True)
plt.tight_layout()
# About SSC Board Percentage
# Graphical view of attribute ssc_p
plt.title('SSC Percentage')
bins = [40,50,60,70,80,90]
plt.hist(df['ssc_p'] , bins = bins , color = '#006666', edgecolor = 'k')
plt.tight_layout()
plt.xlabel('Range')
plt.ylabel('Percentage')
plt.show()
# About HSC Board
# Graphical view of attribute hsc_b

plt.title('HSC Board')
labels = ['Others' , 'Central']
colors = ['#778899' , '#E6E6FA']
plt.pie(df['hsc_b'].value_counts() ,labels = labels , startangle = 90 ,
        colors = colors ,counterclock = False, wedgeprops = {'edgecolor' : 'k' , 'linestyle' : 'solid' ,},
        autopct = '%1.1f%%' , shadow = True)
plt.tight_layout()
# About HSC Board Percentage
# Graphical view of attribute hsc_p

plt.title('SSC Board Percentage')
plt.hist(df['ssc_p'] ,color = '#8B0000' ,edgecolor = 'k')
plt.tight_layout()
plt.show()
# About HSC Stream
# Graphical view of attribute hsc_s

plt.title('HSC Stream')
sns.countplot(df['hsc_s'] , color = '#B0C4DE' , edgecolor = 'k' , saturation = 1)
plt.tight_layout()
# About Degree
# Graphical view of attribute degree_t

plt.title('Degree')
labels = ['Sci&Tech' , 'Comm&Mgm' , 'Others']
colors = ['#8FBC8F' , '#00FF7F' , '#66CDAA']
plt.pie(df['degree_t'].value_counts() ,labels = labels , startangle = 90 ,
        colors = colors ,counterclock = False, wedgeprops = {'edgecolor' : 'k' , 'linestyle' : 'solid' ,},
        autopct = '%1.1f%%' , shadow = True)
plt.tight_layout()
# About Degree Percentage
# Graphical view of attribute degree_p

plt.title('Degree Percentage')
plt.hist(df['degree_p' ] ,color = '#AFEEEE' ,edgecolor = 'k' , alpha = 0.7)
plt.tight_layout()
plt.show()
# About Work Experience
# Graphical view of attribute workex

plt.title('Work Experience')
sns.countplot(df['workex'] , color = '#660066' , edgecolor = 'k'  ,saturation = 1 )
plt.tight_layout()
# About eTest
# Graphical view of attribute etest_p

plt.title('eTest Mark')
bins = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110]
plt.hist(df['etest_p'] ,bins = bins ,color = '#FFFF99' ,  edgecolor = 'k')
plt.tight_layout()
plt.show()
# About Specialisation Course
# Graphical view of attribute specialisation

plt.title('Specialisation Course Counting')
sns.countplot(df['specialisation'] ,color = '#006600' ,edgecolor = 'k' , saturation = 1)
plt.tight_layout()
# About MBA Percentages
# Graphical view of attribute mba_p

plt.title('MBA Percentages')
bins = [50,55,60,65,70,75,80]
plt.hist(df['mba_p'] ,bins = bins,color = '#0080FF', edgecolor = 'k')
plt.show()
# About Salaries
# Graphical view of attribute salary

plt.title('Salaries')
bins = [2,4,6,8]
plt.hist(df['salary']/100000 , color = '#A0522D' , bins = bins)
plt.tight_layout()
plt.show()
# Machine Learning
# Importing LabelEncoder to convert graphical data into numerical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['gender'] = labelencoder.fit_transform(df['gender'])
df['ssc_b'] = labelencoder.fit_transform(df['ssc_b'])
df['hsc_b'] = labelencoder.fit_transform(df['hsc_b'])
df['hsc_s'] = labelencoder.fit_transform(df['hsc_s'])
df['degree_t'] = labelencoder.fit_transform(df['degree_t'])
df['workex'] = labelencoder.fit_transform(df['workex'])
df['specialisation'] = labelencoder.fit_transform(df['specialisation'])
df['status'] = labelencoder.fit_transform(df['status'])
X = df[['gender' , 'ssc_b' , 'ssc_p' ,'hsc_b' , 'hsc_p' , 'hsc_s', 'degree_t' , 'degree_p' , 'workex',
        'etest_p' , 'specialisation', 'mba_p' , 'salary']].values
Y = df.iloc[ : , -2].values
# Importing train_test_split method to split data into train and test part
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size = 0.2 , random_state = 1)
# Importing tree for DecisionTree Algorithm
from sklearn import tree
dec_tree = tree.DecisionTreeClassifier()
# Fitting the trainning data into algorithm
dec_tree.fit(X_train , Y_train)
# Predicting the testing part
prediction = dec_tree.predict(X_test)
# Getting the accuracy score
dec_tree.score(X,Y)
#Example
dec_tree.predict([[1,1,88,1,88,1,0,88,1,86,1,88,120000]])