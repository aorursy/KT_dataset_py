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
# reading the data and also checking the computation time

test = pd.read_csv('/kaggle/input/employee-promotion-datasets/test (3).csv')

train = pd.read_csv('/kaggle/input/employee-promotion-datasets/train.csv')





# lets also check the shape of the dataset

print(test.shape)
test.head()
train.head()
# lets import all the required libraries



# for mathematical operations

import numpy as np

# for dataframe operations

import pandas as pd



# for data visualizations

import seaborn as sns

import matplotlib.pyplot as plt



# setting up the size of the figures

plt.rcParams['figure.figsize'] = (16, 5)

# setting up the style of the plot

plt.style.use('fivethirtyeight')



# for interactivity

import ipywidgets as widgets

from ipywidgets import interact

from ipywidgets import interact_manual







# for machine learning

import sklearn

import imblearn
train.info()
# lets check the Target Class Balance



plt.rcParams['figure.figsize'] = (15, 5)

plt.style.use('fivethirtyeight')



plt.subplot(1, 2, 1)

sns.countplot(train['is_promoted'],)



plt.xlabel('Promoted or Not?', fontsize = 10)



plt.subplot(1, 2, 2)

train['is_promoted'].value_counts().plot(kind = 'pie', explode = [0, 0.1], autopct = '%.2f%%', startangle = 90,

                                       labels = ['1','0'], shadow = True, pctdistance = 0.5)

plt.axis('off')



plt.suptitle('Target Class Balance', fontsize = 15)

plt.show()
train.iloc[:,1:].describe().style.background_gradient(cmap = 'copper')
train.describe(include = 'object')
# Lets make an interactive function to check the statistics of these numerical columns at a time



@interact

def check(column = list(train.select_dtypes('number').columns[1:8])):

    print("Maximum Value :", train[column].max())

    print("Minimum Value :", train[column].min())

    print("Mean : {0:.2f}".format(train[column].mean()))

    print("Median :", train[column].median())

    print("Standard Deviation :  {0:.2f}".format(train[column].std()))
# missing values in training data set



# lets calculate the total missing values in the dataset

train_total = train.isnull().sum()



# lets calculate the percentage of missing values in the dataset

train_percent = ((train.isnull().sum()/train.shape[0])*100).round(2)



# lets calculate the total missing values in the dataset

test_total = test.isnull().sum()



# lets calculate the percentage of missing values in the dataset

test_percent = ((test.isnull().sum()/test.shape[0])*100).round(2)



# lets make a dataset consisting of total no. of missing values and percentage of missing values in the dataset

train_missing_data = pd.concat([train_total, train_percent, test_total, test_percent],

                                axis=1, 

                                keys=['Train_Total', 'Train_Percent %','Test_Total', 'Test_Percent %'],

                                sort = True)



# lets check the head

train_missing_data.style.bar(color = ['gold'])
# checking datatype of columns in the data

train.dtypes[train.isnull().any()]
# lets impute the missing values in the Training Data



train['education'] = train['education'].fillna(train['education'].mode()[0])

train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].mode()[0])



# lets check whether the Null values are still present or not?

print("Number of Missing Values Left in the Training Data :", train.isnull().sum().sum())
# lets impute the missing values in the Testing Data



test['education'] = test['education'].fillna(test['education'].mode()[0])

test['previous_year_rating'] = test['previous_year_rating'].fillna(test['previous_year_rating'].mode()[0])



# lets check whether the Null values are still present or not?

print("Number of Missing Values Left in the Training Data :", test.isnull().sum().sum())
train.select_dtypes('number').head()
# lets check the boxplots for the columns where we suspect for outliers

plt.rcParams['figure.figsize'] = (15, 5)

plt.style.use('fivethirtyeight')



# Box plot for average training score

plt.subplot(1, 2, 1)

sns.boxplot(train['avg_training_score'], color = 'red')

plt.xlabel('Average Training Score', fontsize = 12)

plt.ylabel('Range', fontsize = 12)



# Box plot for length of service

plt.subplot(1, 2, 2)

sns.boxplot(train['length_of_service'], color = 'red')

plt.xlabel('Length of Service', fontsize = 12)

plt.ylabel('Range', fontsize = 12)



plt.suptitle('Box Plot', fontsize = 20)

plt.show()
# Lets check the distribution for the columns for which we suspect for the outliers

plt.rcParams['figure.figsize'] = (15, 5)

plt.style.use('fivethirtyeight')



# Distribution plot for Average training score

plt.subplot(1, 2, 1)

sns.distplot(train['avg_training_score'], color = 'darkblue')

plt.xlabel('Average Training Score', fontsize = 12)

plt.ylabel('Range', fontsize = 12)



# Distribution plot for Length of Service

plt.subplot(1, 2, 2)

sns.distplot(train['length_of_service'], color = 'darkblue')

plt.xlabel('Length of Service', fontsize = 12)

plt.ylabel('Range', fontsize = 12)



plt.suptitle('Distribution Plot', fontsize = 20)

plt.show()
# lets plot pie chart for the columns where we have very few categories

plt.rcParams['figure.figsize'] = (16,5)

plt.style.use('fivethirtyeight')



# plotting a pie chart to represent share of Previous year Rating of the Employees

plt.subplot(1, 3, 1)

labels = ['0','1']

sizes = train['KPIs_met >80%'].value_counts()

colors = plt.cm.Wistia(np.linspace(0, 1, 5))

explode = [0, 0]



plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('KPIs Met > 80%', fontsize = 20)



# plotting a pie chart to represent share of Previous year Rating of the Employees

plt.subplot(1, 3, 2)

labels = ['1', '2', '3', '4', '5']

sizes = train['previous_year_rating'].value_counts()

colors = plt.cm.Wistia(np.linspace(0, 1, 5))

explode = [0, 0, 0, 0, 0.1]



plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Previous year Ratings', fontsize = 20)



# plotting a pie chart to represent share of Previous year Rating of the Employees

plt.subplot(1, 3, 3)

labels = ['0', '1']

sizes = train['awards_won?'].value_counts()

colors = plt.cm.Wistia(np.linspace(0, 1, 5))

explode = [0,0.1]



plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Awards Won?', fontsize = 20)





plt.legend()

plt.show()
# lets check the distribution of trainings undertaken by the employees



plt.rcParams['figure.figsize'] = (17, 4)

sns.countplot(train['no_of_trainings'], palette = 'spring')

plt.xlabel(' ', fontsize = 14)

plt.title('Distribution of Trainings undertaken by the Employees')

plt.show()
# lets check the Age of the Employees



plt.rcParams['figure.figsize'] = (8, 4)

sns.distplot(train['age'], color = 'black')

plt.title('Distribution of Age among the Employees', fontsize = 15)

plt.xlabel('Age of the Employees')

plt.grid()

plt.show()
train.select_dtypes('object').head()
# lets check different Departments



plt.rcParams['figure.figsize'] = (8, 5)

sns.countplot(y = train['department'], palette = 'cividis', orient = 'v')

plt.xlabel('')

plt.ylabel('Department Name')

plt.title('Distribution of Employees in Different Departments', fontsize = 15)



plt.show()
# lets check distribution of different Regions



plt.rcParams['figure.figsize'] = (8,15)

sns.countplot(y = train['region'], palette = 'copper', orient = 'v')

plt.xlabel('')

plt.ylabel('Region')

plt.title('Different Regions', fontsize = 15)

plt.xticks(rotation = 90)



plt.show()
# lets plot pie chart for the columns where we have very few categories

plt.rcParams['figure.figsize'] = (16,5)



# plotting a pie chart to represent share of Previous year Rating of the Employees

plt.subplot(1, 3, 1)

labels = train['education'].value_counts().index

sizes = train['education'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))

explode = [0, 0, 0.1]



plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Education', fontsize = 20)



# plotting a pie chart to represent share of Previous year Rating of the Employees

plt.subplot(1, 3, 2)

labels = train['gender'].value_counts().index

sizes = train['gender'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))

explode = [0, 0]



plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Gender', fontsize = 20)



# plotting a pie chart to represent share of Previous year Rating of the Employees

plt.subplot(1, 3, 3)

labels = train['recruitment_channel'].value_counts().index

sizes = train['recruitment_channel'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))

explode = [0,0,0.1]



plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Recruitment Channel', fontsize = 20)



plt.show()
# interactive function for plotting univariate charts for categorical data



plt.rcParams['figure.figsize'] = (15, 4)

@interact_manual

def check(column = list(train.select_dtypes('object').columns),

          palette = ['cividis','copper','spring','Reds','Blues']):

    sns.countplot(train[column], palette = palette)

   

    plt.show()
# Lets compare the Gender Gap in the promotion



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 3)

x = pd.crosstab(train['gender'], train['is_promoted'])

colors = plt.cm.Wistia(np.linspace(0, 1, 5))

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = False, color = colors)

plt.title('Effect of Gender on Promotion', fontsize = 15)

plt.xlabel(' ')

plt.show()
# lets compare the effect of different Departments and Promotion



plt.rcParams['figure.figsize'] = (12,4)

x = pd.crosstab(train['department'], train['is_promoted'])

colors = plt.cm.copper(np.linspace(0, 1, 3))

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = False, color = colors)

plt.title('Effect of Department on Promotion', fontsize = 15)

plt.xlabel(' ')

plt.show()
# lets compare the effect of Education and Promotion



plt.rcParams['figure.figsize'] = (12,4)

x = pd.crosstab(train['education'], train['is_promoted'])

colors = plt.cm.bone(np.linspace(0, 1, 3))

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = False, color = colors)

plt.title('Effect of Education on Promotion', fontsize = 15)

plt.xlabel(' ')

plt.show()
# checking the effect of number of trainings on promotion



sns.violinplot(train['no_of_trainings'], train['is_promoted'])

plt.title('Effect of Trainings on Promtions', fontsize = 15)

plt.xlabel('Number of Trainings', fontsize = 10)

plt.ylabel('Employee Promoted or not?', fontsize = 10)

plt.show()
# Effect of Age on the Promotion



sns.boxenplot(train['is_promoted'], train['age'], palette = 'PuRd')

plt.title('Effect of Age on Promotion', fontsize = 15)

plt.xlabel('Is the Employee Promoted?', fontsize = 10)

plt.ylabel('Age of the Employee', fontsize = 10)

plt.show()
# lets check relation between number of trainings and average training score



sns.boxplot(train['no_of_trainings'], train['avg_training_score'])

plt.title('Relation between No. of Trainings and Average Training Score', fontsize = 15)

plt.xlabel('Number of Trainings', fontsize = 10)

plt.ylabel('Average Training Score', fontsize = 10)

plt.show()
# lets check the relation between the length of service and the average training score



sns.stripplot(train['avg_training_score'], train['length_of_service'], palette = 'Set3')

plt.title('Average Training Score vs Length of Service', fontsize = 15)

plt.xlabel('Average Training Score', fontsize = 10)

plt.ylabel('Length of Service', fontsize = 10)

plt.show()
## lets check the relation between KPIs Met and Promotion



x = pd.crosstab(train['KPIs_met >80%'], train['is_promoted'])

x.style.background_gradient(cmap = 'bone')
x = pd.crosstab(train['awards_won?'], train['is_promoted'])

x.style.background_gradient(cmap = 'Blues')
# lets make an Interactive Function for Bivariate Analysis



plt.rcParams['figure.figsize'] = (15, 4)

@interact_manual

def bivariate_plot(column1 = list(train.select_dtypes('object').columns),

                   column2 = list(train.select_dtypes('number').columns[1:])):

    sns.boxplot(train[column1], train[column2])
# lets create some extra features from existing features to improve our Model



# creating a Metric of Sum

train['sum_metric'] = train['awards_won?']+train['KPIs_met >80%'] + train['previous_year_rating']

test['sum_metric'] = test['awards_won?']+test['KPIs_met >80%'] + test['previous_year_rating']



# creating a total score column

train['total_score'] = train['avg_training_score'] * train['no_of_trainings']

test['total_score'] = test['avg_training_score'] * test['no_of_trainings']
# lets remove some of the columns which are not very useful for predicting the promotion.



# we already know that the recruitment channel is very least related to promotion of an employee, so lets remove this column

# even the region seems to contribute very less, when it comes to promotion, so lets remove it too.

# also the employee id is not useful so lets remove it.



train = train.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)

test = test.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)



# lets check the columns in train and test data set after feature engineering

train.columns
# lets group the employees based on their Education



train[['education','is_promoted']].groupby(['education']).agg(['count','sum'])
## lets use the interactive function to make it more reusable



@interact

def group_operations(column = list(train.select_dtypes('object').columns)):

    return train[[column, 'is_promoted']].groupby([column]).agg('count').style.background_gradient(cmap = 'Wistia')
# lets get the names of all the employees who have taken trainings more than 7 Times



@interact

def check(column = 'no_of_trainings', x = 5):

    y = train[train['no_of_trainings'] > x]

    return y['is_promoted'].value_counts()
# lets remove the above two columns as they have a huge negative effect on our training data



# lets check shape of the train data before deleting two rows

print("Before Deleting the above two rows :", train.shape)



train = train.drop(train[(train['KPIs_met >80%'] == 0) & (train['previous_year_rating'] == 1.0) & 

      (train['awards_won?'] == 0) & (train['avg_training_score'] < 40)].index)



# lets check the shape of the train data after deleting the two rows

print("After Deletion of the above two rows :", train.shape)
# lets check how many of the employees have greater than 30 years of service and still do not get promotion



@interact

def check_promotion(x = 20):

    x = train[(train['length_of_service'] > x)]

    return x['is_promoted'].value_counts()

   
# lets start encoding these categorical columns to convert them into numerical columns



# lets encode the education in their degree of importance 

train['education'] = train['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),

                                                (3, 2, 1))

test['education'] = test['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),

                                                (3, 2, 1))



# lets use Label Encoding for Gender and Department to convert them into Numerical

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train['department'] = le.fit_transform(train['department'])

test['department'] = le.fit_transform(test['department'])

train['gender'] = le.fit_transform(train['gender'])

test['gender'] = le.fit_transform(test['gender'])



# lets check whether we still have any categorical columns left after encoding

print(train.select_dtypes('object').columns)

print(test.select_dtypes('object').columns)
# lets check the data after encoding

train.head(3)
# lets split the target data from the train data



y = train['is_promoted']

x = train.drop(['is_promoted'], axis = 1)

x_test = test



# lets print the shapes of these newly formed data sets

print("Shape of the x :", x.shape)

print("Shape of the y :", y.shape)

print("Shape of the x Test :", x_test.shape)
# It is very important to resample the data, as the Target class is Highly imbalanced.

# Here We are going to use Over Sampling Technique to resample the data.

# lets import the SMOTE algorithm to do the same.



from imblearn.over_sampling import SMOTE



x_resample, y_resample  = SMOTE().fit_sample(x, y.values.ravel())



# lets print the shape of x and y after resampling it

print(x_resample.shape)

print(y_resample.shape)
# lets also check the value counts of our target variable4



print("Before Resampling :")

print(y.value_counts())



print("After Resampling :")

y_resample = pd.DataFrame(y_resample)

print(y_resample[0].value_counts())
# lets create a validation set from the training data so that we can check whether the model that we have created is good enough

# lets import the train_test_split library from sklearn to do that



from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)



# lets print the shapes again 

print("Shape of the x Train :", x_train.shape)

print("Shape of the y Train :", y_train.shape)

print("Shape of the x Valid :", x_valid.shape)

print("Shape of the y Valid :", y_valid.shape)

print("Shape of the x Test :", x_test.shape)
# lets split the target data from the train data



y = train['is_promoted']

x = train.drop(['is_promoted'], axis = 1)

x_test = test



# It is very import to scale all the features of the dataset into the same scale

# Here, we are going to use the standardization method, which is very commonly used.



# lets import the standard scaler library from sklearn to do that

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_valid = sc.transform(x_valid)

x_test = sc.transform(x_test)# lets print the shapes of these newly formed data sets

print("Shape of the x :", x.shape)

print("Shape of the y :", y.shape)

print("Shape of the x Test :", x_test.shape)






# Lets use Logistic Regression to classify the data

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report



model = DecisionTreeClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_valid)



print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_valid, y_valid))



cm = confusion_matrix(y_valid, y_pred)

plt.rcParams['figure.figsize'] = (3, 3)

sns.heatmap(cm, annot = True, cmap = 'Wistia', fmt = '.8g')

plt.show()
# lets take a look at the Classification Report



cr = classification_report(y_valid, y_pred)

print(cr)
prediction = model.predict(np.array([[2, #department code

                                      3, #masters degree

                                      1, #male

                                      1, #1 training

                                      30, #30 years old

                                      5, #previous year rating

                                      10, #length of service

                                      1, #KPIs met >80%

                                      1, #awards won

                                      95, #avg training score

                                      7, #sum of metric 

                                      700 #total score

                                     ]]))



print("Whether the Employee should get a Promotion : 1-> Promotion, and 0-> No Promotion :", prediction)