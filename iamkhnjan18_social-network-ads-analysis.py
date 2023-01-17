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
# Importing Liabraries And Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv') #importing data by pandas
pd.set_option('Display.max_rows',400) #Display all the rows
df.head()#DataFrame
# Data Wrangling
df.shape #Shape of our DataFrame
df.info() #information about our Data
df.describe() #Getting various statistical data of our DataFrame
df.isnull().sum() #Fortunately , there is no null values
df.duplicated().sum() #there is no duplicate value also
# About Age
print(df['Age'].median()) #median of the ages
#Graphical Distribution of our data age
plt.style.use('fivethirtyeight')
plt.hist(df['Age'] ,color = 'black' , edgecolor = 'cyan')
plt.xlabel('Ranges')
plt.ylabel('Ages')
plt.title('Age Distribution')
plt.tight_layout()
plt.show()
# About EstimatedSalary
print(df['EstimatedSalary'].mean()) #mean of estimated salary
#Graphical Distribution of our data Estimatedsalary
plt.hist(df['EstimatedSalary'] , color = 'cyan' , edgecolor = 'k'  )
plt.xlabel('Ranges')
plt.ylabel('EstimatedSalary')
plt.title('Est.Salary Distribution')
plt.tight_layout()
plt.show()
# Machine Learning Section  
#importing liabraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#Getting the heatmap
sns.heatmap(df.corr())
#having X and Y for Ml model 
X = df[['Age','EstimatedSalary']]
Y = df['Purchased']
#Using train_test_split method to splitting the data into training part and testing part for Ml
#Here test_size = 0.2 means Ml model will take only 80% of the data to train and 20% data to test
#Here random_state means it will not change the trainning and testing part after running this many times
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2 , random_state = 1)
log_reg = LogisticRegression() #having the logisticregreesion as variable
log_reg.fit(X_train , Y_train) #fitting the trainning data to use the Ml-LogisticRegression
predictions = log_reg.predict(X_test) #Predicting the testing data
confusion_matrix(Y_test,predictions)
#Having a confusion matrix to know connection between actual value and Predicting value
log_reg.score(X_test,Y_test)
#Getting the score(r-squared) value..if the score is close to 1 which means our prediction is good
log_reg.coef_ #Getting the coefficient
log_reg.intercept_ #getting the intercept
# I have got the Ml model(LogisticRegression) to predict the Purchase...Thank You 
