# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



#load packages

import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format(pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}". format(matplotlib.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



import sklearn #collection of machine learning algorithms

print("scikit-learn version: {}". format(sklearn.__version__))



#misc libraries

import random

import time







#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import autocorrelation_plot



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
#Loading the single csv file to a variable named 'data'

HR=pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
#Lets look at a glimpse of table

HR.head()
#Lets look at no.of columns and information about its factors

print ("The shape of the  data is (row, column):"+ str(HR.shape))

print (HR.info())
#Looking at the datatypes of each factor

HR.dtypes
import missingno as msno 

msno.matrix(HR);
print('Data columns with null values:',HR.isnull().sum(), sep = '\n')
print("Gender classification:",HR.Gender.value_counts(),sep = '\n')

print("-"*40)

print("Business Travel:",HR.BusinessTravel.value_counts(),sep = '\n')

print("-"*40)

print("Departments:",HR.Department.value_counts(),sep = '\n')

print("-"*40)

print("Educational Field:",HR.EducationField.value_counts(),sep = '\n')

print("-"*40)

print("Job Roles:",HR.JobRole.value_counts(),sep = '\n')

print("-"*40)
plt.figure(figsize = (15, 7))

plt.style.use('seaborn-white')

plt.subplot(331)

label = LabelEncoder()

HR['EducationField'] = label.fit_transform(HR['EducationField'])

sns.countplot(HR['EducationField'],)

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(332)

sns.countplot(HR['Gender'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(333)

sns.countplot(HR['JobInvolvement'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(334)

sns.countplot(HR.JobSatisfaction)

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(335)

sns.countplot(HR.EnvironmentSatisfaction)

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(336)

sns.countplot(HR.RelationshipSatisfaction)

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(337)

sns.countplot(HR.OverTime)

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(338)

sns.countplot(HR.WorkLifeBalance)

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(339)

sns.countplot(HR.StockOptionLevel)

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.figure(figsize = (15, 7))

plt.style.use('seaborn-white')

plt.subplot(331)

sns.distplot(HR['Age'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(332)

sns.distplot(HR['DistanceFromHome'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(333)

sns.distplot(HR['DailyRate'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(334)

sns.distplot(HR['YearsAtCompany'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(335)

sns.distplot(HR['TotalWorkingYears'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(336)

sns.distplot(HR['NumCompaniesWorked'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(337)

sns.distplot(HR['MonthlyRate'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(338)

sns.distplot(HR['MonthlyIncome'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(339)

sns.distplot(HR['PercentSalaryHike'])

fig = plt.gcf()

fig.set_size_inches(10,10)
plt.figure(figsize=(20,5))

plt.hist(HR.Age,bins=20)

plt.xlabel("Age")

plt.ylabel("Counts")

plt.title("Age Counts")

plt.show()
# Code forked from- https://www.kaggle.com/roshansharma/fifa-data-visualization

labels = ['R&D', 'Sales', 'HR']

sizes = HR['Department'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))

explode = [0.1, 0.1, 0.2]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('Department Distribution', fontsize = 20)

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.boxplot(x = HR['JobRole'], y =HR['MonthlyIncome'], data = HR, palette = 'inferno')

ax.set_xlabel(xlabel = 'Names of Job Roles', fontsize = 20)

ax.set_ylabel(ylabel = 'Monthly Income', fontsize = 20)

ax.set_title(label = 'Distribution of Salary across Job Roles', fontsize = 30)

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize=(10,5))

plt.style.use('ggplot')

sns.jointplot(x='TotalWorkingYears', y='MonthlyIncome', data=HR)
sns.lmplot(x = 'TotalWorkingYears', y = 'MonthlyIncome', data = HR, col = 'Attrition')

plt.show()
plt.figure(figsize = (15, 7))

plt.style.use('fivethirtyeight')

plt.subplot(131)

sns.swarmplot(x="Attrition", y="Age", data=HR)

plt.subplot(132)

sns.swarmplot(x="Attrition", y="MonthlyIncome", data=HR)

plt.subplot(133)

sns.swarmplot(x="Attrition", y="YearsAtCompany", data=HR)
fig,ax = plt.subplots(figsize=(10,7))

sns.violinplot(x='Gender', y='MonthlyIncome',hue='Attrition',split=True,data=HR)
plt.style.use('ggplot')

g = sns.pairplot(HR, vars=["MonthlyIncome", "Age"],hue="Attrition",size=5)
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(HR.corr(),annot=True,linewidths=0.5,linecolor="green",fmt=".1f",ax=ax)

plt.show()
s = (HR.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_data = HR.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_data[col] = label_encoder.fit_transform(HR[col])
s = (label_data.dtypes == 'object')

print(list(s[s].index))
label_data.head()
data_features=['Age','EnvironmentSatisfaction',

               'Gender','JobInvolvement', 'JobLevel', 'JobRole',

               'JobSatisfaction','MonthlyIncome','PerformanceRating',

               'TotalWorkingYears','YearsAtCompany','OverTime']

X=label_data[data_features]

y=label_data.Attrition
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline



my_pipeline = Pipeline(steps=[('model', RandomForestRegressor(n_estimators=50,

                                                              random_state=0))])
from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("MAE scores:\n", scores)

print("Average MAE score (across experiments):",scores.mean())
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)

my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))

cv_scores = cross_val_score(my_pipeline, X, y, 

                            cv=5,

                            scoring='accuracy')



print("Cross-validation accuracy: %f" % cv_scores.mean())
# Drop leaky predictors from dataset

potential_leaks = ['EnvironmentSatisfaction', 'JobSatisfaction', 'PerformanceRating', 'JobInvolvement']

X2 = X.drop(potential_leaks, axis=1)



# Evaluate the model with leaky predictors removed

cv_scores = cross_val_score(my_pipeline, X2, y, 

                            cv=5,

                            scoring='accuracy')



print("Cross-val accuracy: %f" % cv_scores.mean())
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
val_y.head()
#Code forked from -https://www.kaggle.com/vanshjatana/applied-machine-learning

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 500000)

model.fit(train_X, train_y)

y_pred = model.predict(val_X)

accuracy = model.score(val_X, val_y)

print(accuracy)
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
# compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    return mean_absolute_error(val_y, preds)
print("Mean Absolute error of the Model:")

print(score_dataset(train_X, val_X, train_y, val_y))
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(train_X, train_y, 

             early_stopping_rounds=5, 

             eval_set=[(val_X, val_y)], 

             verbose=False)

predictions = my_model.predict(val_X)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(val_y, predictions.round())

conf_matrix
plt.subplots(figsize=(5,5))

group_names = ["True Neg","False Pos","False Neg","True Pos"]

group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]

group_percentages = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.show()