import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
ad_df = pd.read_csv('../input/advertising.csv')
ad_df.head()
# Histogram of the age
sns.set_style('whitegrid')
sns.distplot(ad_df[('Age')],bins = 30)
#Joint plot of area income vs age
sns.set_style('whitegrid')
sns.jointplot('Age','Area Income', data=ad_df)
#Joint plot of kde distribution of time spent vs age
sns.set_style('whitegrid')
sns.jointplot('Age','Daily Time Spent on Site', data=ad_df, kind = 'kde')
#Joint plot of 'Daily Time Spent on Site vs Daily Internet Usage
sns.set_style('whitegrid')
sns.jointplot('Daily Time Spent on Site','Daily Internet Usage', data=ad_df)
sns.set_style('whitegrid')
sns.pairplot(ad_df , hue = 'Clicked on Ad')
City = pd.get_dummies(ad_df['City'],drop_first=True)
Country = pd.get_dummies(ad_df['Country'],drop_first=True)
ad_df.drop(['City','Country','Timestamp','Ad Topic Line'],axis=1,inplace=True)
ad_df = pd.concat([ad_df,City,Country] , axis =1)

x_train, x_test, y_train, y_test = train_test_split(ad_df.drop('Clicked on Ad',axis=1), 
                                                    ad_df['Clicked on Ad'], test_size=0.4)
#Creating a logistic regression model
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)

#Displaying results
print(classification_report(y_test,predictions))
