# Numerical libraries

import numpy as np   

# Import Linear Regression machine learning library

from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as smf

# to handle data in form of rows and columns 

import pandas as pd    

# importing ploting libraries

import matplotlib.pyplot as plt   

#importing seaborn for statistical plots

import seaborn as sns

# To enable plotting graphs in Jupyter notebook

%matplotlib inline 
# reading the CSV file into pandas dataframe

mpg_df = pd.read_csv("../input/car-mpg.csv")  
type(mpg_df)
mpg_df.head()
# Check top few records to get a feel of the data structure

mpg_df.shape
# drop the car name column as it is useless for the model

mpg_df = mpg_df.drop('car_name', axis=1)
mpg_df.shape
mpg_df.head()
# Replace the numbers in categorical variables with the actual country names in the origin col

mpg_df['origin'] = mpg_df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
mpg_df['origin'].value_counts()
# Convert categorical variable into dummy/indicator variables. As many columns will be created as distinct values

mpg_df = pd.get_dummies(mpg_df, columns=['origin'])
mpg_df.head()
#Lets analysze the distribution of the dependent (mpg) column

mpg_df.describe().transpose()
mpg_df['hp'][120:130]
# Note:  HP column is missing the describe output. That indicates something is not right with that column
#Check if the hp column contains anything other than digits 

# run the "isdigit() check on 'hp' column of the mpg_df dataframe. Result will be True or False for every row

# capture the result in temp dataframe and dow a frequency count using value_counts()

# There are six records with non digit values in 'hp' column

temp = pd.DataFrame(mpg_df.hp.str.isdigit())

temp[temp['hp'] == False]
# On inspecting records number 32, 126 etc, we find "?" in the columns. Replace them with "nan"

#Replace them with nan and remove the records from the data frame that have "nan"

mpg_df = mpg_df.replace('?', np.nan)
#Let us see if we can get those records with nan

mpg_df[mpg_df.isnull().any(axis=1)]
# There are various ways to handle missing values. Drop the rows, replace missing values with median values etc. 
#of the 398 rows 6 have NAN in the hp column. We will drop those 6 rows. Not a good idea under all situations

#note: HP is missing becauses of the non-numeric values in the column. 

#mpg_df = mpg_df.dropna()
#instead of dropping the rows, lets replace the missing values with median value. 

mpg_df.median()
# replace the missing values in 'hp' with median value of 'hp' :Note, we do not need to specify the column names

# every column's missing value is replaced with that column's median respectively

mpg_df = mpg_df.fillna(mpg_df.median())
mpg_df['hp'].dtype
# The "hp" column was treated as an object when data was loaded into the dataframe as it contained "?"



mpg_df['hp']=mpg_df['hp'].astype('float64')

mpg_df.describe().transpose()
#let us look at each attribute and understand it's distribution

mpg_df.describe().transpose()
# Study the distribution of the data in each column. Columns which do not have random distributions may not be good 

# for model as random processes cannot be modeled

mpg_df.hp.hist()
# Let us visually inspect the central values and the spread

sns.boxplot(mpg_df.hp, color = "yellow", orient = "h")
# Let us do a correlation analysis among the different dimensions and also each dimension with the dependent dimension

# This is done using scatter matrix function which creates a dashboard reflecting useful information about the dimensions

# The result can be stored as a .png file and opened in say, paint to get a larger view 



mpg_df_attr = mpg_df.iloc[:,0:7]



axes = pd.plotting.scatter_matrix(mpg_df_attr)

plt.tight_layout()

plt.savefig('mpg_pairpanel.png')
sns.pairplot(mpg_df,diag_kind='kde')
mpg_df.corr()
m1=smf.ols('mpg~cyl+disp+hp+wt+yr+car_type',mpg_df).fit()
m1.summary()
from sklearn.model_selection import train_test_split
X=mpg_df.drop(['mpg','acc'],axis=1)

#X=mpg_df[['disp']]

Y=mpg_df['mpg']
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=2)
xtrain.shape
xtest.shape
LR=LinearRegression()

LR.fit(xtrain,ytrain)

mpg_pred=LR.predict(xtest)
LR.score(xtest,ytest)  # Adjusted R square
from sklearn import metrics

np.sqrt(metrics.mean_squared_error(ytest,mpg_pred))  #RMSE
rmse=np.sqrt(np.mean((ytest-mpg_pred)**2))

rmse
plt.plot(xtest['disp'],ytest,'*')

plt.plot(xtest['disp'],mpg_pred,'+')
np.corrcoef(ytest,mpg_pred)
0.92544341*0.92544341  #R square
#The data distribution across various dimensions except 'Acc' do not look normal

#Close observation between 'mpg' and other attributes indicate the relationship is not really linear

#relation between 'mpg' and 'hp' show hetroscedacity... which will impact model accuracy

#How about 'mpg' vs 'yr' surprising to see a positive relation
# Copy all the predictor variables into X dataframe. Since 'mpg' is dependent variable drop it

X = mpg_df.drop(['mpg','yr','acc'], axis=1)



# Copy the 'mpg' column alone into the y dataframe. This is the dependent variable

y = mpg_df['mpg']

y.shape
X.head()
#Let us break the X and y dataframes into training set and test set. For this we will use

#Sklearn package's data splitting function which is based on random function



from sklearn.model_selection import train_test_split
# Split X and y into training and test set in 75:25 ratio



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
# invoke the LinearRegression function and find the bestfit model on training data

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)
# Let us explore the coefficients for each of the independent attributes

for idx, col_name in enumerate(X_train.columns):

    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[idx]))
X_train.columns
regression_model.coef_[2]
regression_model.intercept_
# Let us check the intercept for the model



intercept = regression_model.intercept_



print("The intercept for our model is {}".format(intercept))
# we can write our linear model as:

# Y=−21.11–0.35×X1+0.03×X2–0.02×X3–0.01×X4+0.12×X5+0.85×X6–1.90×X7+0.74×X8+1.16×X9
# Model score - R2 or coeff of determinant

# R^2=1–RSS / TSS



regression_model.score(X_test, y_test)
# So the model explains 84.4% of the variability in Y using X
# Let us check the sum of squared errors by predicting value of y for test cases and 

# subtracting from the actual y for the test cases



rmse = np.sqrt(np.mean((regression_model.predict(X_test)-y_test)**2))

rmse
# predict mileage (mpg) for a set of attributes not in the training or test set

y_pred = regression_model.predict(X_test)
# Since this is regression, plot the predicted y value vs actual y values for the test data

# A good model's prediction will be close to actual leading to high R and R2 values

plt.scatter(y_test, y_pred)
type(y_test)
len(y_test)
len(y_pred)
np.corrcoef(y_test,y_pred)
0.86056386*0.86056386
# To scale the dimensions we need scale function which is part of sckikit preprocessing libraries



from sklearn import preprocessing



# scale all the columns of the mpg_df. This will produce a numpy array

mpg_df_scaled = preprocessing.scale(mpg_df)
mpg_df_scaled
#convert the numpy array back into a dataframe 



mpg_df_scaled = pd.DataFrame(mpg_df_scaled, columns=mpg_df.columns)
#browse the contents of the dataframe. Check that all the values are now z scores



mpg_df_scaled
# Copy all the predictor variables into X dataframe. Since 'mpg' is dependent variable drop it

X = mpg_df_scaled.drop('mpg', axis=1)



# Copy the 'mpg' column alone into the y dataframe. This is the dependent variable

y = mpg_df_scaled[['mpg']]

# Split X and y into training and test set in 75:25 ratio



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
# invoke the LinearRegression function and find the bestfit model on training data



regression_model = LinearRegression()

regression_model.fit(X_train, y_train)
# Let us explore the coefficients for each of the independent attributes



for idx, col_name in enumerate(X_train.columns):

    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))
# Model score - R2 or coeff of determinant

# R^2=1–RSS / TSS



regression_model.score(X_test, y_test)
# Let us check the sum of squared errors by predicting value of y for test cases and 

# subtracting from the actual y for the test cases



mse = np.mean((regression_model.predict(X_test)-y_test)**2)
# underroot of mean_sq_error is standard deviation i.e. avg variance between predicted and actual



import math



math.sqrt(mse)
# predict mileage (mpg) for a set of attributes not in the training or test set

y_pred = regression_model.predict(X_test)
# Since this is regression, plot the predicted y value vs actual y values for the test data

# A good model's prediction will be close to actual leading to high R and R2 values

plt.scatter(y_test, y_pred)