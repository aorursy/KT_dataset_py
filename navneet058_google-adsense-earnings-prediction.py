#Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/Adsense.csv")

df.head()
#Get the info of the data frame

df.info()
#Calculate Some Statistics of the data set

df.describe()

df = df.dropna()

df.shape



df[df.isnull()].count()
plt.scatter(df['Clicks'],df['Estimated earnings (INR)'])

plt.title('Clicks Vs Estimated earnings (INR)')

plt.xlabel('Clicks')

plt.ylabel('Estimated earnings (INR)')

# sns.pairplot(df)
# Find more relationship between other features using pairplot sns

df.fillna(df.mean(), inplace=True)

sns.pairplot(df)
#Get the features and target values from the data-sets



X=df.drop(['Month','Active View Viewable','Estimated earnings (INR)'],axis=1)

y=df['Estimated earnings (INR)']

# Import the library

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
#Split the data into training and testing sets using train_test_split function

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1)

#Initialize the Linear Regression Model

lm=LinearRegression()



#Fit the model using training data set

lm.fit(X_train,y_train)
#predict the model using test data

y_pred=lm.predict(X_test)

# print the coefficients

print(lm.intercept_)

print(lm.coef_)
#Import Libraries for calculating model performance

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


print("Mean Absolute Error",(y_test,y_pred))

print("R2 Score",r2_score(y_test,y_pred))





print("=======================================")



output_df=pd.DataFrame({'Actual Output':y_test,'Predicted Output':y_pred})

output_df







# y_test