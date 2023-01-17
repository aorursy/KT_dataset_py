# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# We use dataset 'Students Performance in Exams', you can download it from https://www.kaggle.com/spscientist/students-performance-in-exams
# This data set consists of the marks secured by the students in various subjects.
df_train = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df_train.info()
df_train.columns
# Descriptive statistics summary
df_train.describe()
# Let's look at sample data
df_train.head()
df_train.tail()
# Function that groups only categorical variables according to the size by categories
def group_by(pdf):
    for col in pdf.iloc[:,:]:
        if str(pdf[col].dtypes) == 'object':            
            display(pdf.groupby(col).size().reset_index())
group_by(df_train)
# Function that groups only categorical variables based on numerical variables 
def group_by1(pdf):
    for col in pdf.iloc[:,:]:
        if str(pdf[col].dtypes) == 'object':            
            display(pdf.groupby(col).mean())
            
group_by1(df_train)
# Function that groups only numerical variables
def group_numeric(pdf):
    list(pdf.select_dtypes([np.int64,np.float64]).columns)
    display(pdf[list(pdf)].describe())
    
group_numeric(df_train)
# Function to get distribution plots for numerical variables
def plot_num(pdf):
    for i, col in enumerate(pdf.select_dtypes([np.int64,np.float64]).columns):
        plt.figure(i)
        sns.countplot(x=col, data=pdf)
        sns.distplot(pdf[col])
plot_num(df_train)        
# Plot distributions of numeric variables
def plot_dist(pdf):
    for i, col in enumerate(pdf.select_dtypes([np.int64,np.float64]).columns):
        plt.figure(i)
        sns.distplot(df_train[col])
        
plot_dist(df_train) 
# Plot distributions of numeric variables
def plot_violine(pdf):
    for i, col in enumerate(pdf.select_dtypes([np.int64,np.float64]).columns):
        plt.figure(i)
        sns.violinplot(data=pdf, x=col, y="race/ethnicity", hue="gender",
               split=True, inner="quart", linewidth=1, palette={"female": "b", "male": ".85"})
        sns.despine(left=True)
        
plot_violine(df_train) 
# Plot the joint distribution using kernel density estimation
def plot_joint_dist(pdf):
    for i, item in enumerate(combinations(list(df_train.select_dtypes([np.int64,np.float64]).columns), 2)):
        plt.figure(i)
        sns.jointplot(data=df_train, x=item[0], y=item[1], kind="kde")            
        
plot_joint_dist(df_train) 