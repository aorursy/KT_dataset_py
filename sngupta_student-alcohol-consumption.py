# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #used for the visualization
import seaborn as sns #for visualization
%matplotlib inline

sns.set_style('whitegrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/student-mat.csv') #read the data set using pandas.read_csv()
df.head() #from this we retrieve the top five row of the dataset
df.info() #gives the info about data null values and many more
df.describe() #give the aggregrate of the data
#another way to show the null values or missing values in data is using the heat map
plt.figure(figsize = (15,10))
sns.heatmap(df.isnull(), cmap = 'viridis')
plt.show()
#it doesn't show any sign in the whole heatmap thus it hasn't any missing value
#create heatmap of corelation of the data
plt.figure(figsize = (15,10))
sns.heatmap(df.corr(), cmap = 'viridis', annot = True)
plt.show()
#the value with 1 is shows the corr()>0.7
plt.figure(figsize = (15,10))
sns.heatmap(df.corr()>0.75, cmap = 'viridis', annot = True)
plt.show()
#countplot of all the columns in the df dataframe, shows the result in the form of counting
for col in df.columns:
    sns.countplot(col, data = df)
    plt.show()
df.columns
sns.jointplot(x = 'G3', y = 'Dalc', data = df, color = 'red')
sns.jointplot(x = 'G3', y = 'Walc', data = df, color = 'blue')
plt.show()
#pairplot of the selected columns of the dataframe
df1 = pd.DataFrame(columns = ['G1', 'G2', 'G3', 'Dalc', 'Walc'], data = df)
sns.pairplot(df1)
#get dummy value for the categorial values
df2 = pd.get_dummies(df)
df2.head()
#check for the null values there is no null values in the dataframe
df2.isnull().count()
#that are the dataset set for the machine learning algorithm
X1 = df2.drop(['G1', 'G2', 'G3'], axis = 1) #without any grading value 
X2 = df2.drop(['G3'], axis = 1) #withot final grade
y1 = df2['G1']
y2 = df2['G2']
y3 = df2['G3']
from sklearn.model_selection import train_test_split
#use to analyse the G1 grading
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.3, random_state = 42)
#here the linearregression model is used
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
print(predictions[:5])
plt.scatter(predictions, y_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('The score of the grade G1')
print(lm.score(X_test, y_test))
from sklearn.model_selection import train_test_split
#use to analyse the G2 grading
X_train, X_test, y_train, y_test = train_test_split(X1, y2, test_size = 0.3, random_state = 42)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
print(predictions[:5])
plt.scatter(predictions, y_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('The score of the grade G2')
print(lm.score(X_test, y_test))
from sklearn.model_selection import train_test_split
#use to analyse the G3 grading
X_train, X_test, y_train, y_test = train_test_split(X1, y3, test_size = 0.3, random_state = 42)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
print(predictions[:5])
plt.scatter(predictions, y_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('The score of the grade G3')
print(lm.score(X_test, y_test))
from sklearn.model_selection import train_test_split
#use to analyse the G3 grading using the grade G1 and G2 also
X_train, X_test, y_train, y_test = train_test_split(X2, y3, test_size = 0.3, random_state = 42)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
print(predictions[:5])
plt.plot(predictions, y_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('The score of the grade G3 including G1 an G2')
print(lm.score(X_test, y_test))
print('Shows that the grade G1 and G2 not affected the result.')
#Now using the logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.3, random_state = 42)

logM = LogisticRegression()
logM.fit(X_train, y_train)
predictions = logM.predict(X_train)
print(predictions[:5])
plt.plot(predictions)

#print('The confusion matrix for G1:')
#print(confusion_matrix(predictions, y_test))
#print('The classification_report of G1:')
#print(classification_report(predictions, y_test))

print(logM.score(X_test, y_test))


