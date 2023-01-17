import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#Load the csv file into a dataframe 
df = pd.read_csv("../input/BlackFriday.csv")
df.shape #(rows,columns)
df.head(7) #first seven data points 
# let's get an overall idea of the numerical columns
df.describe()
df = df.drop('User_ID',axis=1)

# Time to prepare each column before we start training.
# First lets see how many non-null values each column has - 
df.count().sort_values()
# there is around 30% data for product_category_3, lets drop it!
df = df.drop(['Product_Category_3'], axis = 1)
# Now lets fill the null values for Product_Category_2 with its mean value
df['Product_Category_2'].fillna(9.842144,inplace=True)
df.head(7)
# Now, as we have handled the missing data, we need to see if there are any outliers
# and remove them in the numerical columns.
# Let's first visualize using box plot
import seaborn as sns
sns.boxplot(x=df['Occupation'])
sns.boxplot(x=df['Product_Category_1'])
sns.boxplot(x=df['Product_Category_2'])
sns.boxplot(x=df['Purchase']) # we can see few points outside which are the outliers
# Only Purchase has outliers.So, we are going to use Inter quartile range for removing outliers.
Q1 = df['Purchase'].quantile(0.25)
Q3 = df['Purchase'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df.Purchase < (Q1 - 1.5 * IQR)) |(df.Purchase > (Q3 + 1.5 * IQR)))]
df.shape
# Now let's deal with the categorical columns
df['Stay_In_Current_City_Years'].replace('4+','4',inplace=True)
df['Stay_In_Current_City_Years'] = pd.to_numeric(df['Stay_In_Current_City_Years'])
df.head(7)
#convert the categorical columns to numeric - one column for each unique value of the categorical column
df = pd.get_dummies(df,columns=['Gender','Age','City_Category'])
df.head(7)
#drop product ID - we don't need it for purchase quantity
df.drop(columns=['Product_ID'],inplace=True)
df.head(7)
# Find out the important features using heatmap visualization
import matplotlib.pyplot as plt
plt.subplots(figsize=(16,9))
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,cmap='Reds')
plt.show()
# City_Category and Gender seem to give positive correlation.
df.corr()[['Purchase']].sort_values('Purchase')
# The correlation values are not very good. Still we will try to use all the columns which
# show a positive correlation and find a linear regression model

from sklearn.model_selection import train_test_split
train,test = train_test_split(df[['City_Category_C','Gender_M','Occupation','Age_51-55','Age_36-45',
                                 'Stay_In_Current_City_Years','Age_55+', 'Purchase']])
print(train.shape)
print(test.shape)
X_train , y_train = train.iloc[:,train.columns!='Purchase'], train[['Purchase']]
X_test , y_test = test.iloc[:,test.columns!='Purchase'], test[['Purchase']]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
r2_score = reg.score(X_test,y_test)
r2_score
#Not a great score as expected ! 
#Let's try DecisionTreeRegressor!
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train,y_train)
r2_score = reg.score(X_test,y_test)
r2_score
# Use KNN to see if we can get better results.
# Note that KNN is a non-parametric regression i.e. it does not learn from the data rather stores the whole data
# and gives the result from the nearest neighbours.
train,test = train_test_split(df)
X_train , y_train = train.iloc[:,train.columns!='Purchase'], train[['Purchase']]
X_test , y_test = test.iloc[:,test.columns!='Purchase'], test[['Purchase']]

from sklearn.neighbors import KNeighborsRegressor
clf = KNeighborsRegressor(n_neighbors=12)
clf.fit(X_train,y_train)
r2_score = clf.score(X_test,y_test)
r2_score