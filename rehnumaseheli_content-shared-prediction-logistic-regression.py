#This model is building a logistic regression model to predict if the content was shared or not.

#Installing necessary libraries





import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline



#Loading the dataset

Popularity = pd.read_csv("../input/popularity-csv/NewsPopularity_Logistics_BB2.csv")

#Showing first few rows of the dataset



Popularity.head()
#Showing last few rows of the dataset



Popularity.tail()
#Observing column type of the dataset



Popularity.info()
#Showing number of rows and columns of the dataset



Popularity.shape
#Observing if there is any missing data on the file. No missing value found.



Popularity.isnull().sum()
#There were two categorical columns ('onWeekend' and 'wasShared'). Making them binary.





Popularity['onWeekend'] = Popularity['onWeekend'].map({'Yes' : 1, 'No' : 0})

Popularity['wasShared'] = Popularity['wasShared'].map({'Yes' : 1, 'NO' : 0})
#Again showing the first few rows of the file to see if those categorical columns are now integer or not.



Popularity.head()
#Observing the column type of the data file.



Popularity.info()
#showing the percentage of time the content was shared.



wasShared = (sum(Popularity['wasShared'])/len(Popularity['wasShared'].index))*100
wasShared
#Importing sklearn library and train_test_split for spliting the data file into test and train



from sklearn.model_selection import train_test_split
#Declaring depending and independing variables



x = Popularity.drop(['wasShared'], axis = 1)



y = Popularity['wasShared']
#Showing all independent variables



x.head()
#Showing dependent variable



y.head()
#Spliting dependent and independent variables into 70% and 30% train and test categories.



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, test_size = 0.3, random_state = 100)
import statsmodels.api as sm
#Buidling and Showing Linear Regression model



logml = sm.GLM(y_train,(sm.add_constant(x_train)), family = sm.families.Binomial())

logml.fit().summary()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Creating a Heatmap of the data file.



plt.figure(figsize = (20,10))

sns.heatmap(Popularity.corr(), annot= True)
#Removing 'positive_words' as it has high p-value and less impact on a content being shared.



x_test2 = x_test.drop(['positive_words'],axis =1)

x_train2 = x_train.drop(['positive_words'], axis = 1)
#After deleting 'Positive_words' creating the Heatmap again.



plt.figure(figsize = (20,10))

sns.heatmap(x_train2.corr(), annot= True)
#After deleting 'Positive_words' building and showing the result of Linear Regression model.



logm2 = sm.GLM(y_train,(sm.add_constant(x_train2)), family = sm.families.Binomial())

logm2.fit().summary()
#Building Logistic Regression model with train dataset of dependent and independent variables.



from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logsk = LogisticRegression()

logsk.fit(x_train,y_train)
logsk
#Creating a vaiable to predict values for the test data set of independent variables.



y_pred = logsk.predict_proba(x_test)
#Declaring a dataframe of the predicted variable data



y_pred_df = pd.DataFrame(y_pred)
#Showing only the shared content column



y_pred_1 = y_pred_df.iloc[:,[1]]
y_pred_1.head()
#Another dataframe with the dependent varible's test dataset.



y_test_df = pd.DataFrame(y_test)
#Joining two dataframes together and removing their index number column so that they appear side by side.



y_pred_1.reset_index(drop= True, inplace = True)

y_test_df.reset_index(drop= True, inplace = True)

y_pred_1 = y_pred_1.rename(columns= {1 : 'WasShared_Predicted'})

y_pred_final = pd.concat([y_test_df, y_pred_1],axis = 1)
#Showing dependent variable's actual and predicted results.



y_pred_final.head(10)
#Creating a new column with a condition to be matched with actual result.



y_pred_final['Predicted'] = y_pred_final.WasShared_Predicted.map(lambda x: 1 if x >0.5 else 0 )
y_pred_final.head(10)
from sklearn import metrics
#Creating confusion matrix.



cm = metrics.confusion_matrix(y_pred_final.wasShared, y_pred_final.Predicted)
cm
#showing the overall accuracy of the model.



metrics.accuracy_score(y_pred_final.wasShared, y_pred_final.Predicted)
#A model was built using Logistic Regression to predict if the content was shared or not.

#All the independent variables were affecting if the content will be shared or not except 'Positive_words' column as it's 

#p-value was higher.

#From the logistic regression model I can say that the number of words in the content (length_content), the number of images in the content (images), the number of videos in the content (videos), the number of keywords (keywords), if the content was read on a weekend (onWeekend), a measure of the relevancy of the content (relevancy), a measure of subjectivity in the title (title_subjectivity), a measure of sentiments in the title (title_sentiments) are the factors that can determine whether a new content will be shared or not.

#For one-unit change in these factors(the log odds ratio of wasShared (measure if the content was shared) will be increased.

#The Accuracy is 0.686(68.6%) which means the model is capable of classifying instances correctly in 68.6% cases.