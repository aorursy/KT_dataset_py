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
#Loading the single csv file to a variable named 'customer'
customer=pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
#Lets look at a glimpse of table
customer.head()
print ("The shape of the  data is (row, column):"+ str(customer.shape))
print (customer.info())
#Looking at the datatypes of each factor
customer.dtypes
import missingno as msno 
msno.matrix(customer)
print('Data columns with null values:',customer.isnull().sum(), sep = '\n')
#Donut Chart
labels = ['Male','Female']
sizes = customer['Gender'].value_counts()
colors = plt.cm.magma(np.linspace(0, 1, 5))


plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(sizes, labels = labels, colors = colors, shadow = True,autopct='%1.0f%%', 
        pctdistance=1.1,labeldistance=1.2,startangle=90)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

#Lets reveal
plt.title('Gender Distribution', fontsize = 20)
plt.legend()
plt.show()
sns.distplot(customer['Age'])
fig,ax = plt.subplots(figsize=(10,7))
sns.violinplot(x='Gender', y='Annual Income (k$)',split=True,data=customer)
sns.pairplot(customer,vars = ['Spending Score (1-100)', 'Annual Income (k$)', 'Age'], hue="Gender")
sns.set(style="ticks")
y= customer['Spending Score (1-100)']
x = customer['Annual Income (k$)']
sns.jointplot(x, y, kind="hex", color="#4CB391")
cust_new = customer.drop('CustomerID', 1)
sns.heatmap(cust_new.corr(),annot=True,fmt='.1g',cmap='Greys')
plt.scatter(customer['Annual Income (k$)'],customer['Spending Score (1-100)'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
from sklearn.cluster import KMeans
#Creating a copy of the dataset
x=customer.drop(customer.loc[:,'CustomerID':'Age'].columns, axis = 1) 
# Create an object (which we would call kmeans)
# The number in the brackets is K, or the number of clusters we are aiming for, lets divide into half first
kmeans = KMeans(2)
# Fit the data
kmeans.fit(x)
# Create a copy of the input data
clusters = x.copy()
# predicted clusters 
clusters['cluster_pred']=kmeans.fit_predict(x)
plt.style.use('default')
plt.scatter(clusters['Annual Income (k$)'],clusters['Spending Score (1-100)'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
# Import a library which can do that easily
from sklearn import preprocessing
# Scale the inputs
# preprocessing.scale scales each variable (column in x) with respect to itself
# The new result is an array
x_scaled = preprocessing.scale(x)
x_scaled
# Createa an empty list
wcss =[]

# Create all possible cluster solutions with a loop
# We have chosen to get solutions from 1 to 9 clusters; you can ammend that if you wish
for i in range(1,10):
    # Clsuter solution with i clusters
    kmeans = KMeans(i)
    # Fit the STANDARDIZED data
    kmeans.fit(x_scaled)
    # Append the WCSS for the iteration
    wcss.append(kmeans.inertia_)
    
# Check the result
wcss
# Plot the number of clusters vs WCSS
plt.plot(range(1,10),wcss)
# Name your axes
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# Fiddle with K (the number of clusters)
kmeans_new = KMeans(5)
# Fit the data
kmeans_new.fit(x_scaled)
# Create a new data frame with the predicted clusters
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
# Check if everything seems right
clusters_new
# Final Segments
fig, ax = plt.subplots()
scatter=ax.scatter(clusters_new['Annual Income (k$)'],clusters_new['Spending Score (1-100)'],c=clusters_new['cluster_pred'],cmap='rainbow')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="top right", title="Segments")
ax.add_artist(legend1)
plt.xlabel('Annual Income ')
plt.ylabel('Spending Score')