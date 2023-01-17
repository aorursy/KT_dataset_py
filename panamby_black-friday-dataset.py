# First I will import the basic libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

% matplotlib inline
# import the file.

bf = pd.read_csv('../input/BlackFriday.csv')

# Take a look on it!

bf.head()
# One more look...

bf.info()
# Let's see if there are missing values.

plt.figure(figsize=(24,16))

sns.heatmap(bf.isnull())

# We can see that there are missing values only in "Product_Category_2"and "Product_Category_3""
# This time I will do different and, before to solve missing value problems, 
# I will see if I can do some conclusions form datas.
# First Conclusion: Men go to Black Friday more than women.

sns.countplot(bf.Gender)
# How much Men and Women people spent in Black Friday.
# Men ---> blue bar
# Women ---> orange bar


bf[bf.Gender == 'M']['Purchase'].hist(bins=50)
bf[bf.Gender == 'F']['Purchase'].hist(bins=50)

# apparently the same proportion with the last graph.
# Second Conclusion: People who has 26-35 participate in Black Friday much more than others. 
# Followed by 36-45, 18-25, 46-50, 51-55, 55+ and 0-17. 
# In summary we have three groups with the almost the same quantity of people between each element within the group.
# Group #1 36-45/18-25; group #2 46-50/51-55 and group #3 55+/0-17.

sns.countplot(bf.Age)
# How much different Ages people spent in Black Friday.
# 26-35 ---> blue bar
# 36-45 ---> orange bar
# 18-25 ---> green bar
#--------------------
# 46-50 ---> blue bar
# 51-55 ---> orange bar
# 55+ ---> green bar
# 0-17 ---> red bar


plt.subplot(2,2,1)
bf[bf.Age == '26-35']['Purchase'].hist(bins=50, figsize=(16,10))

plt.subplot(2,2,2)
bf[bf.Age == '36-45']['Purchase'].hist(bins=50)
bf[bf.Age == '18-25']['Purchase'].hist(bins=50)

plt.subplot(2,2,1)
bf[bf.Age == '46-50']['Purchase'].hist(bins=50)
bf[bf.Age == '51-55']['Purchase'].hist(bins=50)

plt.subplot(2,2,2)
bf[bf.Age == '55+']['Purchase'].hist(bins=50)
bf[bf.Age == '0-17']['Purchase'].hist(bins=50)

plt.tight_layout()

# apparently the same proportion with the last graph.
# Third Conclusion: We can't do much conclusions here, but is possible to realize that people with
# lows and highs occupation number go out to Black Friday more than who was mediums occupation number.
# Between lows and highs, low occupation number go out more than high occupation number.

sns.countplot(bf.Occupation)
# Fourth Conclusion: People who lives in City B bought much more than others. 
# Followed by City C and A, with the almost the same number of people.

sns.countplot(bf.City_Category)
# How much people who lives in different cities spent in Black Friday.
# City B ---> blue bar
# City C ---> orange bar
# City A ---> green bar

plt.figure(figsize=(12,8))
bf[bf.City_Category == 'B']['Purchase'].hist(bins=50)
bf[bf.City_Category == 'C']['Purchase'].hist(bins=50)
bf[bf.City_Category == 'A']['Purchase'].hist(bins=50)

# Fifth Conclusion: We can see that, in less amount of money, people who lives in City A and C spent almost the same.

# Sixth Conclusion: People who are married attend the event more than who aren't. 

sns.countplot(bf.Marital_Status)
# How much Married and No Married people spent in Black Friday.
# No maried ---> blue bar
# Married ---> orange bar

plt.figure(figsize=(12,8))
bf[bf.Marital_Status == 0]['Purchase'].hist(bins=50)
bf[bf.Marital_Status == 1]['Purchase'].hist(bins=50)

# apparently the same proportion with the last graph.
# Seventh Conclusion: People who has less time in the same city go to Black Friday more than has more time.
# The exception is who has less than 1 (one) year.

sns.countplot(bf.Stay_In_Current_City_Years)
# How much people, who stay in the same city a couple of time, spent in Black Friday.
# 1 year ---> blue bar
# 2 years ---> orange bar
# 0 year ---> green bar
#--------------------
# 3 years ---> blue bar
# 4+ years ---> orange bar


plt.subplot(2,2,1)
bf[bf.Stay_In_Current_City_Years == '1']['Purchase'].hist(bins=50, figsize=(16,10))

plt.subplot(2,2,2)
bf[bf.Stay_In_Current_City_Years == '2']['Purchase'].hist(bins=50)
bf[bf.Stay_In_Current_City_Years == '0']['Purchase'].hist(bins=50)

plt.subplot(2,2,1)
bf[bf.Stay_In_Current_City_Years == '3']['Purchase'].hist(bins=50)
bf[bf.Stay_In_Current_City_Years == '4+']['Purchase'].hist(bins=50)

# apparently the same proportion with the last graph.
sns.countplot(bf.Product_Category_1)
sns.countplot(bf.Product_Category_2)
sns.countplot(bf.Product_Category_3)
# We can see the same behavior and both male and female people, when we look in Age column.
# Hence, no relevant conclusions.

sns.countplot(data=bf, x=bf.Gender, hue=bf.Age)
# We can see the same behavior and both male and female people, when we look in Marital Status column.
# Hence, no relevant conclusions.

sns.countplot(data=bf, x=bf.Marital_Status, hue=bf.Gender)
# We will see now about the distribution 
bf.Purchase.hist(bins=50, figsize=(12,8), by=bf.Age)
bf.Purchase.hist(bins=50, figsize=(12,8), by=bf.Gender)
bf.Purchase.hist(bins=50, figsize=(12,8), by=bf.Occupation)
bf.Purchase.hist(bins=50, figsize=(12,8), by=bf.City_Category)
bf.Purchase.hist(bins=50, figsize=(12,8), by=bf.Stay_In_Current_City_Years)
# We can realize, when we see the graphs above, that, regardless the parameters, the graphs has almost the same shape.
# It's a very curious observation. Everyone has almost the same behavior according to money spend.
sns.boxplot(x=bf.Purchase)
# Now let's start to fill the missing values.

bf.Product_Category_2.isnull().sum()

# This columns don't have a lot of missing value.
bf.Product_Category_3.isnull().sum()

# Despite having a lot of missing values, I choose to use this column with fill values.
# I'm trying to understand the values in this column to figure out how can I solve this.

bf.Product_Category_2.value_counts()
# This column has few numbers, so I can calculate the mean and use it to fill the missing values.
# I won't use the lowest number.
# mean = 15.5

bf.Product_Category_3.value_counts().head()
# Fill the missing values and check!

bf.Product_Category_3.fillna(value=15.5, inplace=True)

bf.Product_Category_3.head(10)
# I will see if I can use the mean in the boxplot graph to figure out what number I will fill.
# I will use Age and Gende.

sns.boxplot(x=bf.Product_Category_2, y=bf.Age, hue=bf.Gender)

# I think that this graph will help me to fill the missing values in this column.
# This function will fill the missing values according to the means in the graph above.

def impute_value(cols):
    Product = cols[0]
    Age = cols[1]
    Gender = cols[2]
    
    if pd.isnull(Product):
        if Age == '0-17':
            return 8.0
        elif Age == '18-25':
            return 8.0
        elif Age == '26-35':
            if Gender == 'M':
                return 8.0
            else:
                return 11.0
        elif Age == '36-45':
            if Gender == 'M':
                return 9.0
            else:
                return 11.0
        elif Age == '46-50':
            return 11.0
        else:
            if Gender == 'M':
                return 11.0
            else:
                return 13.0
    else:
        return Product
    
# Now I will apply the function wihtin the column.

bf['Product_Category_2'] = bf[['Product_Category_2','Age', 'Gender']].apply(impute_value,axis=1)
# Let's take a look!

bf.head()
# Let's see if there are any missing values remaining.

plt.figure(figsize=(12,8))

sns.heatmap(bf.isnull())
# Now the dataset is ready!
# Let's build the model.

# Split the dataset

X = bf.iloc[:,2:-1].values
y = bf.iloc[:, -1].values
X
y
# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
X
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Feature Scaling

from  sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# I will use some Regression models and figure out what will be the best, that is, the lowest "Mean Squared Error".
# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results

y_pred = regressor.predict(X_test)
# Mean Squared Error

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly,y_train)

# I tried other "degree" values and figure out that the number "3" is the better.
# Predicting the Test set results

y_pred = regressor.predict(poly_reg.fit_transform(X_test))
# Mean Squared Error

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 600, random_state = 0)
regressor.fit(X_train, y_train)

# I tried other "n_estimators" values and figure out that the number "600" is the better.
y_pred = regressor.predict(X_test)
# Mean Squared Error

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
# Comparing the "Mean Squared Error" between the three models, 
# we can realize that the "Random Forest Regression" is the better model for this problem.
# About the Black Friday dataset, we can see some conclusions below:

#1 - Men go to Black Friday more than women;
#2 - People who has 26-35 participate in Black Friday much more than others. 
# Followed by 36-45, 18-25, 46-50, 51-55, 55+ and 0-17. 
# In summary we have three groups with the almost the same quantity of people between each element within the group.
# Group #1 36-45/18-25; group #2 46-50/51-55 and group #3 55+/0-17;
#3 - We can't do much conclusions here, but is possible to realize that people with
# lows and highs occupation number go out to Black Friday more than who was mediums occupation number.
# Between lows and highs, low occupation number go out more than high occupation number;
#4 - People who lives in City B bought much more than others. 
# Followed by City C and A, with the almost the same number of people;
#5 - We can see that, in less amount of money, people who lives in City A and C spent almost the same;
#6 - People who are married attend the event more than who aren't; &
#7 - People who has less time in the same city go to Black Friday more than has more time.
# The exception is who has less than 1 (one) year.

# Observation: 

# We can realize that, regardless the parameters, the graphs has almost the same shape.
# It's a very curious observation. Everyone has almost the same behavior according to money spend.
# This dataset was Amazing!!!!
# Until the next one!!!