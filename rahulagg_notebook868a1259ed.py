# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import time

from __future__ import division

from IPython.display import display # Allows the use of display() for DataFrames

import matplotlib.pyplot as plt

import matplotlib as matplot

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


# Pretty display for notebooks

%matplotlib inline



sns.set_style('whitegrid')

# Extract the Data

# Load the Census dataset



data = pd.read_csv("../input/HR_comma_sep.csv")



# Success - Display the first record

display(data.head(n=5))

# check the number of data point and the number of features

data.shape
# Identify Predictor and Target variable then type of variable and category of variable

# left is Target variable

# satisfaction_level,last_evaluation,number_project,average_monthly_hours, time_spend_company,work_accident

#promotion_last_5 years sales and salary are predictor



#Variable Category

#Categorical : sales, salary

#Continous : Rest all



print("Sales categories are:",data["sales"].unique())

print("Salary categories are:",data["salary"].unique())
#Stastical analysis of Numerical Data

print(data.describe())
# Analysis of univariate variables

#Plots in matplotlib reside within a figure object, use plt.figure to create new figure

fig=plt.figure()

#Create one or more subplots using add_subplot, because you can't create blank figure

ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,3)

ax4 = fig.add_subplot(2,2,4)

#Variable

ax1.hist(data['average_montly_hours'],bins = 20)

ax2.hist(data['time_spend_company'])

ax3.hist(data['last_evaluation'])

ax4.hist(data['satisfaction_level'])

#Labels and Tit

plt.title('Univariate analysis')

plt.ylabel('#Employee')

plt.show()

sns.distplot(data['average_montly_hours'])
sns.distplot(data['satisfaction_level'])
#sb.boxplot(x='am', y='mpg', data=cars, palette='hls')

fig=plt.figure(figsize=(18, 18))

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.boxplot(data['average_montly_hours'],ax=ax1)

sns.boxplot(data['satisfaction_level'],ax=ax2)
# for categorical variables check the count% under each category and visualise using bar plot or pie plot



print(data["sales"].value_counts()/len(data)*100);

print(data["salary"].value_counts()/len(data)*100);
#plot the categorical variables

fig, (ax1, ax2) = plt.subplots(nrows=2, sharey=True)

fig.set_size_inches(11.7, 8.27)

sns.countplot(data["salary"],ax=ax1)

sns.countplot(data["sales"],ax=ax2)
# Bivariate analysis

#Relationship between continous and continous using scatterplot

fig, (ax1, ax2) = plt.subplots(nrows=2)

fig.set_size_inches(11.7, 8.27)

sns.regplot(x="average_montly_hours", y="left", data=data,ax=ax1)

sns.regplot(x="satisfaction_level", y="left", data=data,ax=ax2)
# To Visualise the relationship plot for all the variables use pair plot

#df = data[["average_montly_hours","satisfaction_level"]]

#df= data.filter(["average_montly_hours","satisfaction_level"],axis=1)

sns.pairplot(data)
# To check the correlation between variables we use pearsonr correaltions for finding strength between linear relationships

# Pearson relationship assume that data is normally distributed that is the sample data is choosed randomly from normal population distribution

from scipy import stats



from scipy.stats.stats import pearsonr

from scipy.stats.stats import spearmanr

corr = data.corr()

sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,square=True, annot=True,

                     cmap='RdBu', fmt='+.2f')

#plt.figure(figsize=(10,10))

# to check the significance we should calculate p value and if it is less than significance level then the correlation is significant

pearsonr_coefficient, p_value = pearsonr(data["number_project"], data["last_evaluation"]) 

print('The pearson coefficent is {} and pvalue is {}'.format(pearsonr_coefficient,p_value))

#Moderate Positively Correlated Features:



#projectCount vs evaluation: 0.349333

#projectCount vs averageMonthlyHours: 0.417211

#averageMonthlyHours vs evaluation: 0.339742

#Moderate Negatively Correlated Feature:



#satisfaction vs turnover: -0.388375

#Stop and Think:



#What features affect our target variable the most (turnover)?

#What features have strong correlations with each other?

#Can we do a more in depth examination of these features?
corr = data.corr(method='spearman')

sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,square=True, annot=True, cmap='RdBu', fmt='+.2f') 

from scipy import stats

from scipy.stats.stats import spearmanr

spearmanr_coefficient, p_value = spearmanr (data["number_project"], data["last_evaluation"]) 

print('Spearman Rank Correlation Coefficient {}'.format(spearmanr_coefficient))

#analysis of the numerical variables with the label

# Overview of summary (Left vs Notleft)

left_Summary = data.groupby('left')

left_Summary.mean()
# analyzing the left variable with sales and salary

left_class = pd.crosstab(index=data["left"],columns=[data["sales"],data["salary"]],margins=True)

(left_class/left_class.loc["All"])*100
#plot the categorical variables

fig, (ax1, ax2) = plt.subplots(nrows=2)

fig.set_size_inches(11.7, 8.27)

sns.countplot(y="salary", hue='left', data=data,ax=ax1).set_title('Employee Salary Turnover Distribution')

sns.countplot(y="sales", hue='left', data=data,ax=ax2).set_title('Employee Salary Turnover Distribution')
# As it shows above that there employees from Sales, Technical and Support department has mostly left the company.

# Not management,product_mng , so we need to analyse more for each department.



# Kernel Density Plot

fig = plt.figure(figsize=(15,4),)

ax=sns.kdeplot(data.loc[(data['left'] == 0),'last_evaluation'] , color='b',shade=True,label='not left')

ax=sns.kdeplot(data.loc[(data['left'] == 1),'last_evaluation'] , color='r',shade=True, label='left')

ax.set(xlabel='Employee Evaluation', ylabel='Frequency')

plt.title('Employee Evaluation Distribution - Left V.S. Not Left')





# it shows that employees with low performace and high performance leaves the company more.


# Kernel Density Plot

fig = plt.figure(figsize=(15,4),)

ax=sns.kdeplot(data.loc[(data['left'] == 0),'average_montly_hours'] , color='b',shade=True,label='not left')

ax=sns.kdeplot(data.loc[(data['left'] == 1),'average_montly_hours'] , color='r',shade=True, label='left')

ax.set(xlabel='Employee working Hours', ylabel='Frequency')

plt.title('Employee Working Hours Distribution - Left V.S. Not Left')





# it shows that employees which are morking more or very less are leaving the company
#Check which feature data has missing values and how many

print(data.isnull().sum())

#As there is no null data so no cleanup is required
fig, (ax1) = plt.subplots(nrows=1)

fig.set_size_inches(11.7, 8.27)

sns.countplot(y="number_project", hue='left', data=data,ax=ax1).set_title('Employee Project count Distribution')



# This shows that employee working on 2, 6,7 projects are leaving more.(again too less or too high)
# Import KMeans Model

from sklearn.cluster import KMeans



# Graph and create 3 clusters of Employee Turnover

kmeans = KMeans(n_clusters=3,random_state=2)

kmeans.fit(data[data.left==1][["satisfaction_level","last_evaluation"]])



kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]



fig = plt.figure(figsize=(10, 6))

plt.scatter(x="satisfaction_level",y="last_evaluation", data=data[data.left==1],

            alpha=0.25,color = kmeans_colors)

plt.xlabel("Satisfaction")

plt.ylabel("Evaluation")

plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)

plt.title("Clusters of Employee Left")

plt.show()

table = pd.pivot_table(data, index=['left'], aggfunc=np.mean)

table

#data.describe()

data["sales"] = data["sales"].astype("category")

data["sales"] = data["sales"].cat.codes

data["salary"] = data["salary"].astype("category")

data["salary"] = data["salary"].cat.codes

#data.dtypes

display(data.head(n=3))
target_label = data["left"]

features = data.drop("left",axis=1)

#display(features.head(n=3))



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(features, target_label, test_size=0.2, random_state=20)

print (format(X_train.shape))

print (format(X_test.shape[0]))
# Find if any data is missing

#No Missing Data

data.isnull().any()
# For each feature find the data points with extreme high or low values

from collections import Counter

outliers_counter = Counter()

outliers_scores = None 



for feature in data.keys():

    

    # TODO: Calculate Q1 (25th percentile of the data) for the given feature

    Q1 = np.percentile(data[feature],25)

    

    # TODO: Calculate Q3 (75th percentile of the data) for the given feature

    Q3 = np.percentile(data[feature],75)

    

    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

    step = 1.5 * (Q3-Q1)

    

    # Display the outliers

    print(format(feature))

    current_outliers = data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))]

    display(current_outliers)

    outliers_counter.update(current_outliers.index.values)

    



# OPTIONAL: Select the indices for data points you wish to remove

#outliers  = [65, 66, 75, 154,128]



outliers = [x[0] for x in outliers_counter.items() if x[1] >= 2]

print(format(outliers))



# Remove the outliers, if any were specified

#good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score, fbeta_score



from sklearn.metrics import classification_report



clf = AdaBoostClassifier(n_estimators=20,random_state=20)

clf.fit(X_train,y_train)

pred=clf.predict(X_test)

print("Accuracy=",accuracy_score(y_test,pred))

print("F-score=",fbeta_score(y_test,pred,beta=0.5))

print(classification_report(y_test, pred))

importances = clf.feature_importances_
feat_names = data.drop(['left'],axis=1).columns



indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))

plt.title("Feature importances by AdaBoostClassifier")

plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")

plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)

plt.xlim([-1, len(indices)])

plt.show()
# Predicting the new