# Importing the neccesary libraries we are going to need

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

sns.set()
path = '/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv'



df = pd.read_csv(path)
df.head(5)
df.info()

#From the result we see that the dataset is clean i.e no misssing values
grade = [] #Declaring a new list

for i in df['quality']: 

    if i > 6.5:

        i = 1

        grade.append(i)

    else:

        i = 0

        grade.append(i)

df['grade'] = grade # A new column to hold our already categoried quality 
df.head(10)
df.drop('quality', axis = 1, inplace = True) #Dropping the quality column since we won't be needing it anymore
df.describe() #shows describption for only numerical variables
sns.distplot(df['fixed acidity']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['fixed acidity'].quantile(0.99)

df = df[df['fixed acidity'] < q]



sns.distplot(df['fixed acidity'])
sns.distplot(df['volatile acidity']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['volatile acidity'].quantile(0.99)

df = df[df['volatile acidity'] < q]



sns.distplot(df['volatile acidity'])
sns.distplot(df['citric acid']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['citric acid'].quantile(0.99)

df = df[df['citric acid'] < q]



sns.distplot(df['citric acid'])
sns.distplot(df['residual sugar']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['residual sugar'].quantile(0.99)

df = df[df['residual sugar'] < q]



sns.distplot(df['residual sugar'])
sns.distplot(df['chlorides']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['chlorides'].quantile(0.99)

df = df[df['chlorides'] < q]



sns.distplot(df['chlorides'])
sns.distplot(df['free sulfur dioxide']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['free sulfur dioxide'].quantile(0.99)

df = df[df['free sulfur dioxide'] < q]



sns.distplot(df['free sulfur dioxide'])
sns.distplot(df['total sulfur dioxide']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['total sulfur dioxide'].quantile(0.99)

df = df[df['total sulfur dioxide'] < q]



sns.distplot(df['total sulfur dioxide'])
sns.distplot(df['density']) #we can see those few outliers shown by the longer left tail of the distribution
#Removing the bottom 1% of the observation will help us to deal with the outliers

q = df['density'].quantile(0.01)

df = df[df['density'] > q]



sns.distplot(df['density'])
sns.distplot(df['pH']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['pH'].quantile(0.99)

df = df[df['pH'] < q]



sns.distplot(df['pH'])
sns.distplot(df['sulphates']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['sulphates'].quantile(0.99)

df = df[df['sulphates'] < q]



sns.distplot(df['sulphates'])
sns.distplot(df['alcohol']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['alcohol'].quantile(0.99)

df = df[df['alcohol'] < q]



sns.distplot(df['alcohol'])
df.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant



# the target column (in this case 'grade') should not be included in variables

#Categorical variables may or maynot be added if any

variables = df[['fixed acidity', 'volatile acidity', 'citric acid',

       'residual sugar', 'chlorides', 'free sulfur dioxide',

       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',]]

x = add_constant(variables)

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(x.values,i) for i in range (x.shape[1])]

vif['features'] = x.columns

vif



#Using 10 as the minimum vif values i.e any independent variable 10 and above will have to be dropped

#From the results all independent variable are below 10
#Declaring independent variable i.e x

#Declaring Target variable i.e y

x = df.drop('grade', axis =1 )

y = df['grade']
scaler = StandardScaler()

scaler.fit(x)

scaled_x = scaler.transform(x)
#Splitting our data into train and test dataset

x_train, x_test, y_train, y_test = train_test_split(scaled_x, y , test_size = 0.2, random_state  = 365)
reg = LogisticRegression() #select the algorithm

reg.fit(x_train,y_train) # we fit the algorithm with the training data and the training output
y_hat = reg.predict(x_test) # y_hat holding the prediction made with the algorithm using x_test
acc = metrics.accuracy_score(y_hat,y_test)# To know the accuracy

acc
reg.intercept_ # Intercept of the regression
reg.coef_ # coefficients of the variables / features 
result = pd.DataFrame(data = x.columns, columns = ['Features'])

result['weight'] = np.transpose(reg.coef_)

result['odds'] = np.exp(np.transpose(reg.coef_))

result
cm = confusion_matrix(y_hat,y_test)

cm
# Format for easier understanding

cm_df = pd.DataFrame(cm)

cm_df.columns = ['Predicted 0','Predicted 1']

cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})

cm_df
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
dd = DecisionTreeClassifier()

dd.fit(x_train,y_train)

y_1 = dd.predict(x_test)

acc_1 = metrics.accuracy_score(y_1,y_test)

acc_1
sv = svm.SVC() #select the algorithm

sv.fit(x_train,y_train) # we train the algorithm with the training data and the training output

y_2 = sv.predict(x_test) #now we pass the testing data to the trained algorithm

acc_2 = metrics.accuracy_score(y_2,y_test)

acc_2
knc = KNeighborsClassifier()

knc.fit(x_train,y_train)

y_3 = knc.predict(x_test)

acc_3 = metrics.accuracy_score(y_3,y_test)

acc_3