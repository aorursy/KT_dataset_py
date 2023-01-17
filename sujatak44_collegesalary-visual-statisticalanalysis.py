# DataSet - https://www.kaggle.com/wsj/college-salaries
# Project Description: Where it Pays to Attend College
# Salaries by college, region, and academic major
!pip3 install pandas_ml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns 
import statsmodels.api as sm

from sklearn import model_selection
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix

from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from scipy.cluster import hierarchy
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# Read all the csv file

# degrees-that-pay-back.csv - Undegrad Major degrees and salary information
df_degPayback = pd.read_csv('/kaggle/input/degrees-that-pay-back.csv')    

# salaries-by-college-type.csv - College Name, School Type and salary information
df_salCollege = pd.read_csv('/kaggle/input/salaries-by-college-type.csv') 

# salaries-by-region.csv - College Name, Region and salary information
df_SalRegion = pd.read_csv('/kaggle/input/salaries-by-region.csv')     

df_degPayback.head() # 50 rows X 8 Columns

df_salCollege.head() # 269 rows × 8 columns

df_SalRegion.head() #320 rows × 8 columns
# Data PreProcessing
# Removing and imputing missing values from the dataset
# Getting categorical data into shape for machine learning algorithms
# Selecting relevant features for the model construction

df_degPayback.isnull().sum()

df_salCollege.isnull().sum() # 38 Nan Value

# drop Mid-Career 10th Percentile Salary & Mid-Career 90th Percentile Salary columns with highest Nan Value
df_salCollege = df_salCollege.drop(['Mid-Career 10th Percentile Salary','Mid-Career 90th Percentile Salary'], axis=1) 
df_salCollege.isnull().sum()
df_SalRegion.isnull().sum() # 47 missing value
# drop Mid-Career 10th Percentile Salary & Mid-Career 90th Percentile Salary column with highest Nan Value
df_SalRegion = df_SalRegion.drop(['Mid-Career 10th Percentile Salary','Mid-Career 90th Percentile Salary'], axis=1) 
df_SalRegion.isnull().sum()
#Convert str to numbers by removing Dollar Sign first
str_cols = ['Starting Median Salary','Mid-Career Median Salary','Mid-Career 25th Percentile Salary','Mid-Career 75th Percentile Salary']

for x in str_cols:
    df_degPayback[x] = df_degPayback[x].str.replace("$","")
    df_degPayback[x] = df_degPayback[x].str.replace(",","")
    df_degPayback[x] = pd.to_numeric(df_degPayback[x])

print(df_degPayback.info())

#Convert str to numbers
str_cols = ['Starting Median Salary','Mid-Career Median Salary','Mid-Career 25th Percentile Salary','Mid-Career 75th Percentile Salary']

for x in str_cols:
    df_salCollege[x] = df_salCollege[x].str.replace("$","")
    df_salCollege[x] = df_salCollege[x].str.replace(",","")
    df_salCollege[x] = pd.to_numeric(df_salCollege[x])

print(df_salCollege.info())



#Convert str to numbers
str_cols = ['Starting Median Salary','Mid-Career Median Salary','Mid-Career 25th Percentile Salary','Mid-Career 75th Percentile Salary']

for x in str_cols:
    df_SalRegion[x] = df_SalRegion[x].str.replace("$","")
    df_SalRegion[x] = df_SalRegion[x].str.replace(",","")
    df_SalRegion[x] = pd.to_numeric(df_SalRegion[x])

print(df_SalRegion.info())



#Visualization

# Degree Payback
df_degPayback.head()
df_degPayback.sort_values(by = 'Starting Median Salary', ascending = False, inplace=True)
df_degPayback.head()

# Undergraduate Major vs Starting median Salary 
df_degPayback.sort_values(by = 'Starting Median Salary', ascending = False, inplace=True)
f, ax = plt.subplots(figsize=(8, 15)) 
ax.set_yticklabels(df_degPayback['Undergraduate Major'], rotation='horizontal', fontsize='medium')
g = sns.barplot(y = df_degPayback['Undergraduate Major'], x= df_degPayback['Starting Median Salary'])
plt.show()

# Highest Starting Salary Major - Physician Assistant, Chemical Engineering, Computer Engineering and so on

#Undergraduate Major vs Mid-Career median Salary
df_degPayback.sort_values(by = 'Mid-Career Median Salary', ascending = False, inplace=True)
f, ax = plt.subplots(figsize=(8,15)) 
ax.set_yticklabels(df_degPayback['Undergraduate Major'], rotation='horizontal', fontsize='medium') 
g = sns.barplot(y = df_degPayback['Undergraduate Major'], x= df_degPayback['Mid-Career Median Salary']) 
plt.show()

# Highest Mid-Career Salary Major - PChemical Engineering, Computer Engineering, Electric Engineering and so on


# Top 20 Fields ensuring higher returns and faster growth. 
Top_degree = df_degPayback.sort_values('Percent change from Starting to Mid-Career Salary', ascending=False).head(20)
f, ax = plt.subplots(figsize=(8, 10)) 
g = sns.barplot(y = Top_degree['Undergraduate Major'], x= Top_degree['Percent change from Starting to Mid-Career Salary'])
plt.show()

# Philosophy, Math, International Relations, Economics are some of the majors with higher growth rates
#set(df_SalRegion['Region'])
# Salary Information By Region

f, ax = plt.subplots(figsize=(10,5)) 
ax.set_xticklabels(df_SalRegion['Region'], rotation='horizontal', fontsize='medium') 
g = sns.countplot(x = 'Region',data = df_SalRegion)

plt.show()

# California got his owm region maybe because of UCs and Standford 

#Top 20 schools with highest 'Starting Median Salary' by region

Top_20School = df_SalRegion.sort_values('Starting Median Salary', ascending=False).head(20)
f, ax = plt.subplots(figsize=(10, 10)) 
g = sns.barplot(y = Top_20School['School Name'], 
                x= Top_20School['Starting Median Salary'], 
                hue=Top_20School['Region'])
plt.setp(ax.patches, linewidth=0)
plt.show()

# Yes, California Institute of Technology got the first place for highest starting Salary and then MIT
#Top 20 schools with highest 'Mid-Career Median Salary' by region
Top_20School = df_SalRegion.sort_values('Mid-Career Median Salary', ascending=False).head(20)
f, ax = plt.subplots(figsize=(10, 10)) 
g = sns.barplot(y = Top_20School['School Name'], 
                x= Top_20School['Mid-Career Median Salary'], 
                hue=Top_20School['Region'])
plt.setp(ax.patches, linewidth=0)
plt.show()


df_salCollege.head()
# Salary Infomation By College Type - Engineering, Party School, Libral Arts, Ivy League, State Colleges

# No. of Type of schools in the whole dataset
f, ax = plt.subplots(figsize=(10,5)) 
ax.set_xticklabels(df_salCollege['School Type'], rotation='horizontal', fontsize='medium') 
g = sns.countplot(x = 'School Type',data = df_salCollege)

plt.show()
# Salary Information Per School Type
df_salCollege.groupby('School Type').mean().plot(kind='bar',figsize=(10,7)) 
plt.show() 
# Starting Median Salary Vs Mid-career Median Salary
df_salCollege[['Starting Median Salary', 'Mid-Career Median Salary']].hist(figsize=(15, 6), 
                                                                  edgecolor='black', 
                                                                  linewidth=1.2,
                                                                  bins=30, 
                                                                  grid=False)
plt.show()
# School Type Pie Chart

schooltype = df_salCollege.groupby(['School Type']).size()
# Data to plot
labels = schooltype.index
sizes =  schooltype.values
#colors = ['red', 'yellowgreen', 'lightcoral', 'lightskyblue','gold']
 
#patches, texts = plt.pie(sizes, labels=labels ,autopct='%1.1f%%',shadow=True, startangle=100)

explode = (0, 0, 0, 0,0.1)  # explode 1st slice
 
pie = plt.pie(sizes, explode=explode,labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle= 50,labeldistance=1.1)
plt.title('School Types') 
plt.axis('equal')
plt.legend(pie[0], labels, loc="upper right")
plt.show()

'''Following are general analysis we can make about post-college salaries

1. By major

* if you want the highest starting salary, look into Engineering or Physician’s Assistant
* if highest mid-career salary is what you’re after, consider Engineering
* if you want to see the most percent growth in salary from start to mid-career look into Philosophy, 
Math and International Relation

2. By college type
* most colleges are State schools
* ivy league and engineering schools have the best long term mid-career salary potential (followed by liberal arts)

3. By region

* Northeastern Region has most colleges
* CIT and MIT are best for starting salaries
* Dartmouth and Prinston are best for Mid-career Salaries
* Type and region, California is good for an engineering, party, or state school
'''
# # Build the Statistical Model - On Salary based on School Type
# # salaries-by-college-type.csv - College Name, School Type and salary information

df_StatModel = df_salCollege.copy()
df_StatModel = df_StatModel.drop(['School Name'],axis=1)
df_StatModel.head()

# create design matrix X and target vector y


X = np.array(df_StatModel.ix[:,1:4])
y = np.array(df_StatModel['School Type']) 

# split into train and test - 90%-10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)




'''Let’s evaluate 6 different algorithms:

Logistic Regression (LR)
Linear Discriminant Analysis (LDA)
K-Nearest Neighbors (KNN).
Classification and Regression Trees (CART).
Gaussian Naive Bayes (NB).
Support Vector Machines (SVM).
'''
# Spot Check Algorithms
# Test options and evaluation metric

seed = 42
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# We now have 6 models and accuracy estimations for each. 
# We need to compare the models to each other and select the most accurate.
# The LDA and KNN algorithm was the most accurate model that we tested. 
# Now we want to get an idea of the accuracy of the KNN model on our validation set.

# Make predictions on validation dataset
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)  
cm = ConfusionMatrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred)) # 81.5 % accuracy
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
cm.plot()
plt.set_cmap('Blues')
plt.show()
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 15):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 15), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 
plt.show()    
# Mean Error is lowest when K=7
#Applying kmeans to the dataset / Creating the kmeans classifier

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 100, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize=(10,20))
cut4 = hierarchy.dendrogram(hierarchy.complete(X),
                            labels=y, orientation='right', color_threshold=140, leaf_font_size=10)
plt.vlines(140,0,plt.gca().yaxis.get_data_interval()[1], colors='r', linestyles='dashed');
plt.show()
# In Conclusion: I implemented Supervised Learning (6 Algorithms) 
# and Unsupervised Learning (Clustering Kmeans) 
# on one Dataset (Salaries-by-college-type.csv - College Name, School Type and salary information) 
# We can build the same Statistical Models on other 2 Dataset for some more analysis.
