# import neccessary libraries

import pandas as pd

import numpy as np
# import the data and create the dataframe

df=pd.read_csv('../input/website_data.csv', skiprows=range(6))
# lets look at the overview of our data

df.head(5)
# lets see if there are any missing data

missing_data=df.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print('')
# lets look at the shape of our data

df.shape
# lets look at the data types in our variables

df.dtypes
df.info()
# create a function to convert percentage to float

def p2f(x):

    return float(x.strip('%'))/100
# update the dataframe

df=pd.read_csv('../input/website_data.csv', skiprows=range(6), converters={'Bounce Rate': p2f, 'Goal Conversion Rate': p2f})
df.head(5)
# update revenue so it is float with no $ sign

df[['Revenue']]=df[['Revenue']].replace('[$,]', '', regex=True).astype('float')
# update Avg. Session Duration to float and total seconds

df['Avg. Session Duration']=pd.to_datetime(df['Avg. Session Duration'], errors='coerce')
df['Avg. Session Duration']=pd.to_timedelta(df['Avg. Session Duration'].dt.strftime('%H:%M:%S'), errors='coerce').dt.total_seconds().astype(int)
df.head(5)
# lets look at the correlation between the variables

df.corr()
# Lets visualize some of our findings

# import neccessary libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df.hist()

plt.show()
# Lets use scatter plot to visualize the relationship

plt.figure(figsize=(15,10), dpi=80)

plt.scatter(df[['Sessions']], df[['Avg. Session Duration']], color='orange')



plt.title('Sessions vs Avg. Session Duration')

plt.xlabel('Sessions')

plt.ylabel('Avg. Session Duration')



plt.show()
plt.figure(figsize=(15,10), dpi=80)

plt.scatter(df[['Sessions']], df[['Revenue']], color='blue')



plt.title('Sessions vs Revenue')

plt.xlabel('Sessions')

plt.ylabel('Revenue')



plt.show()
plt.figure(figsize=(15,10), dpi=80)

plt.scatter(df[['Goal Conversion Rate']], df[['Revenue']], color='red')



plt.title('Goal Conversion Rate vs Revenue')

plt.xlabel('Goal Conversion Rate')

plt.ylabel('Revenue')



plt.show()
# lets split the data for 80% train and 20% set

msk=np.random.rand(len(df))<0.8

train=df[msk]

test=df[~msk]
# train the data distribution

plt.figure(figsize=(15,10), dpi=80)

plt.scatter(train['Sessions'], train['Revenue'], color='blue')

plt.title('Training the Data Distribution for Sessions and Revenue')

plt.xlabel('Sessions')

plt.ylabel('Revenue')

plt.show()
# Lets create our model (first import neccessary libraries)

from sklearn import linear_model
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['Sessions']])

train_y=np.asanyarray(train[['Revenue']])
regr.fit(train_x, train_y)
# lets see the coefficient and the intercept values

print(regr.coef_)

print(regr.intercept_)
#Lets look at the outputs based on our model

plt.figure(figsize=(15,10), dpi=80)

plt.scatter(train['Sessions'], train['Revenue'], color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x +regr.intercept_[0], '-r')

plt.xlabel('Sessions')

plt.ylabel('Revenue')
# import neccessary libraries

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['Sessions']])

test_y = np.asanyarray(test[['Revenue']])

test_y_hat = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )