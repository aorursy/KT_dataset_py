# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Using the pandas library for data analysis and the scikit-learn library for machine learning



#Build a model to predict wheather a new cust will churn or not
#Predicting Customer Churn



#Churn is when a customer stops doing business or ends a relationship with a company.



#It’s a common problem across a variety of industries, from telecommunications to cable TV etc

#company that can predict churn can take proactive action to retain valuable customers and get ahead of the competition
import pandas as pd

import numpy as np

import seaborn as sns # used for plot interactive graph. 

import matplotlib.pyplot as plt
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head() # checking first 5 rows of dataset
#How many churners does the dataset have, and how many non-churners

df['Churn'].value_counts()
df.info() # totalcharges is object so change in to float
#ChangeTotalcharges in to numeric variable 



df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
print(df.groupby(['Churn']).mean())



# churning cust having are having less tunure and high monthly charges 

#checking missing value 

df.isnull().sum() 
# remove missing values

df.dropna(inplace = True)
# check dimensions of dataset

df.shape
# EDA

# Important to understand how your variables are distributed
# visualize distribution of classes # after droping missing values 



plt.figure(figsize=(8, 4))

sns.countplot(df['Churn'], palette='RdBu')



# count number of obvs in each class

No,Yes = df['Churn'].value_counts()

print('Number of cells labeled yes: ', Yes)

print('Number of cells labeled no : ', No)

print('')

print('% of cells labeled churn', round(Yes / len(df) * 100, 2), '%')

print('% of cells labeled no churn', round(No / len(df) * 100, 2), '%')



import matplotlib.pyplot as plt  # distribution is skewed & not normally distibuted of feature totalcharges 

import seaborn as sns

sns.distplot(df['TotalCharges'])

plt.show()
sns.distplot(df['tenure']) # not normally distributed # not a proper bell shaped curve

plt.show()
sns.distplot(df['MonthlyCharges'])

plt.show()
#Differences in Monthlycharges of churn and non churn customer using boxplot

sns.boxplot(x ='Churn',y ='MonthlyCharges',data = df)

plt.show()

# there is much of a difference in monthly charges between churners and non-churners.

# mean monthly charges of cust those are churn are higher 
#Differences in tenure of churn and non churn customer using boxplot 

#There is a very noticeable difference here between churners and non-churners



sns.boxplot(x ='Churn',y ='tenure',data = df)

plt.show() 



#  there is much of a difference in tenure lengths between churners and non-churners.

# mean length of tenure of cust those are churn cust is less as compared to No churn cust 
# there is noticeable diff in length of tenure ,those who streaming movies have long tenure

sns.boxplot(x ='StreamingMovies',y ='tenure',data = df)

plt.show()

# cust those have Fibre optic are more consist of churn cust as compare to services like DSL and No services

sns.catplot(y="InternetService", hue="Churn", kind="count",

            palette="pastel", edgecolor=".6",

            data=df);



# In month to Month contract the percentage of churn cust is more as comapre to one year and Two year , very less no cust churn in 2 yr contract

sns.catplot(y="Contract", hue="Churn", kind="count",

            palette="pastel", edgecolor=".6",

            data=df);
# Add "internet services" as a third variable # for removing outlier can specify the additional parameter sym=""

sns.boxplot(x = 'Churn',

            y = 'tenure',

            data = df,

            sym = "",

            hue = "InternetService")



# Display the plot

plt.show()
#obtaining correlation matrix



#Tenure and Monthlycharges had highest correlation with TotalCharges 



corr = df.corr().round(2)



# Mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set figure size

f, ax = plt.subplots(figsize=(20, 20))



# Define custom colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.tight_layout()
#Remove customer IDs from the data set # unneccessary variable 



# Drop the unnecessary features 



df2 = df.drop(df[['SeniorCitizen','customerID']], axis=1)
# Verify if features dropped

print(df2.columns)
# convert Churn yes and no with 1,0 value 

# Replace 'no' with 0 and 'yes' with 1 in 'Churn'

df2['Churn'] = df2['Churn'].replace({'No':0 , 'Yes':1})
#Let's convert all the categorical variables into dummy variables  

# feature can be encoded numerically using the technique of one hot encoding:



df_new = pd.get_dummies(df2)

df_new.head()
#Feature scaling # different scales of the of tenure and charges and plots shown above not normally distributed 

#Centers the distribution around the mean

#the number of standard deviations away from the mean each point is



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



df_new[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(df_new[['tenure','MonthlyCharges','TotalCharges']].to_numpy())
# verifying first 5 rows 

df_new.head()
#Fit logistic regression model



#Tunning parameter of c (Inverse of regularization strength;) and use of regularization technique l2 as penalty
# target varaiable is y 

y = df_new['Churn'].values



# predictor matrix is x

x = df_new.drop(columns = ['Churn']) 
# Using Skicit-learn to split data into training and testing sets

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

train_x, test_x, train_y, test_y= train_test_split(x, y, test_size = 0.30, random_state = 200)

# checking the dimension of train and test dataset

train_x.shape[0] 

train_y.shape[0]
# Import required library for modelling 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import accuracy_score
# Different value of paramter c in list and looping tuning parameter # getting different metrics 

C = [1, .5, .25, .1, .05, .025, .01, .005, .0025]

l1_metrics = np.zeros((len(C), 5))

l1_metrics[:,0] = C

for index in range(0, len(C)):  

    logreg = LogisticRegression(penalty='l1', C=C[index], solver='liblinear')

    logreg.fit(train_x, train_y)

    pred_test_y = logreg.predict(test_x)

    l1_metrics[index,1] = np.count_nonzero(logreg.coef_)

    l1_metrics[index,2] = accuracy_score(test_y, pred_test_y)

    l1_metrics[index,3] = precision_score(test_y, pred_test_y)

    l1_metrics[index,4] = recall_score(test_y, pred_test_y) 

col_names = ['C','Non-Zero Coeffs','Accuracy','Precision','Recall']

print(pd.DataFrame(l1_metrics, columns=col_names)) 
# using support vector machine for modeling 

from sklearn.svm import SVC

classifier = SVC()

# tuning paramter and kernel 'rbf'

param_grid = {'C':[0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV # using gridsearch to find best parameter 



grid = GridSearchCV(SVC(), param_grid, verbose= 4, refit=True)

grid.fit(train_x, train_y)
# Best paramter 

grid.best_params_
# getting prediction 

optimized_preds = grid.predict(test_x)
#Evaluate the gridsearch SVM # Import accuracy_score

from sklearn.metrics import accuracy_score



# Compute test set accuracy  

acc = accuracy_score(test_y,optimized_preds)

print("Test set accuracy: {:.2f}".format(acc)) # test data accuracy is 80% 
# obtained different model metrices

from sklearn.metrics import  classification_report

print(classification_report(test_y, optimized_preds))