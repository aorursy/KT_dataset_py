# To enable plotting graphs in Jupyter notebook

%matplotlib inline 
# Numerical libraries

import numpy as np   



# Import Linear Regression machine learning library

from sklearn.linear_model import LinearRegression



# to handle data in form of rows and columns 

import pandas as pd    



# importing ploting libraries

import matplotlib.pyplot as plt   



#importing seaborn for statistical plots

import seaborn as sns
import os

os.listdir('../input/carmpg')
# reading the CSV file into pandas dataframe

mpg_df = pd.read_csv("../input/carmpg/car-mpg (1).csv")  
# Check top few records to get a feel of the data structure

mpg_df.head(5)
# drop the car name column as it is useless for the model

mpg_df = mpg_df.drop('car_name', axis=1)
mpg_df
# Replace the numbers in categorical variables with the actual country names in the origin col

mpg_df['origin'] = mpg_df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})

mpg_df.head()
mpg_df.info()
# Convert categorical variable into dummy/indicator variables. As many columns will be created as distinct values

# This is also kown as one hot coding. The column names will be America, Europe and Asia... with one hot coding

# Like feature scaling

mpg_df = pd.get_dummies(mpg_df, columns=['origin'])
mpg_df
#Lets analysze the distribution of the dependent (mpg) column

mpg_df.describe()
mpg_df.info()
# Note:  HP column is missing the describe output. That indicates something is not right with that column
#Check if the hp column contains anything other than digits 

# run the "isdigit() check on 'hp' column of the mpg_df dataframe. Result will be True or False for every row

# capture the result in temp dataframe and dow a frequency count using value_counts()

# There are six records with non digit values in 'hp' column

temp = pd.DataFrame(mpg_df.hp.str.isdigit())  # if the string is made of digits store True else False  in the hp column 

# in temp dataframe



temp[temp['hp'] == False]   # from temp take only those rows where hp has false

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

# every column's missing value is replaced with that column's median respectively  (axis =0 means columnwise)

#mpg_df = mpg_df.fillna(mpg_df.median())



mpg_df = mpg_df.apply(lambda x: x.fillna(x.median()),axis=0)

mpg_df.dtypes
mpg_df['hp'] = mpg_df['hp'].astype('float64')  # converting the hp column from object / string type to float

# Let us do a correlation analysis among the different dimensions and also each dimension with the dependent dimension

# This is done using scatter matrix function which creates a dashboard reflecting useful information about the dimensions

# The result can be stored as a .png file and opened in say, paint to get a larger view 



mpg_df_attr = mpg_df.iloc[:, 0:11]



#axes = pd.plotting.scatter_matrix(mpg_df_attr)

#plt.tight_layout()

#plt.savefig('d:\greatlakes\mpg_pairpanel.png')



sns.pairplot(mpg_df_attr, diag_kind='kde')   # to plot density curve instead of histogram



#sns.pairplot(mpg_df_attr)  # to plot histogram, the default
#The data distribution across various dimensions except 'Acc' do not look normal

#Close observation between 'mpg' and other attributes indicate the relationship is not really linear

#relation between 'mpg' and 'hp' show hetroscedacity... which will impact model accuracy

#How about 'mpg' vs 'yr' surprising to see a positive relation
# Copy all the predictor variables into X dataframe. Since 'mpg' is dependent variable drop it

X = mpg_df.drop('mpg', axis=1)



# Copy the 'mpg' column alone into the y dataframe. This is the dependent variable

y = mpg_df[['mpg']]

#Let us break the X and y dataframes into training set and test set. For this we will use

#Sklearn package's data splitting function which is based on random function



from sklearn.model_selection import train_test_split
# Split X and y into training and test set in 75:25 ratio



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
# invoke the LinearRegression function and find the bestfit model on training data



regression_model = LinearRegression()

regression_model.fit(X_train, y_train)
# explaining enumerate

for idx, col_name in enumerate(X_train.columns):

    print (idx,col_name)
# Let us explore the coefficients for each of the independent attributes



for idx, col_name in enumerate(X_train.columns):

    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))
# Let us check the intercept for the model



intercept = regression_model.intercept_[0]



print("The intercept for our model is {}".format(intercept))
# we can write our linear model as:

# Y=−21.11–0.35×X1+0.03×X2–0.02×X3–0.01×X4+0.12×X5+0.85×X6–1.90×X7+0.74×X8+1.16×X9
# Model score - R2 or coeff of determinant

# R^2=1–RSS / TSS



regression_model.score(X_test, y_test)
# So the model explains 85% of the variability in Y using X
#  Iteration -2 



#Since on many dimensions, the relationship is not really linear, let us try polynomial models (quadratic)
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



poly = PolynomialFeatures(degree=2, interaction_only=True)

#interaction_only is to provide interaction between columns

X_train_ = poly.fit_transform(X_train)



X_test_ = poly.fit_transform(X_test)



poly_clf = linear_model.LinearRegression()



poly_clf.fit(X_train_, y_train)



y_pred = poly_clf.predict(X_test_)



#print(y_pred)

print(poly_clf.score(X_train_, y_train))

print(poly_clf.score(X_test_, y_test))
print(X.shape)

print(X_train_.shape)

poly
# Even with polynomial function, we are not getting better results. What Next?  
sns.jointplot('disp','mpg',kind='scatter',data=mpg_df)
sns.jointplot('hp','mpg',kind='scatter',data=mpg_df)
sns.jointplot('wt','mpg',kind='scatter',data=mpg_df)
#The above 3 columns show they are in polynomial regression with the mileage column.So use only these 3 instead of all columns
X_siva=mpg_df.iloc[:,2:5]

X_siva.head()
y_siva=mpg_df.iloc[:,0]

y_siva.head()
from sklearn.model_selection import train_test_split
X_siva_train,X_siva_test,y_siva_train,y_siva_test=train_test_split(X_siva,y_siva,test_size=0.3,random_state=1)
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



poly = PolynomialFeatures(degree=2, interaction_only=True)
X_siva_train_poly=poly.fit_transform(X_siva_train)

X_siva_test_poly=poly.fit_transform(X_siva_test)
linea=linear_model.LinearRegression()
linea.fit(X_siva_train_poly,y_siva_train)
y_pre=linea.predict(X_siva_test_poly)
linea.score(X_siva_test_poly,y_siva_test)
print(X_siva_train.shape)

print(X_siva_train_poly.shape)
from statsmodels.formula.api import ols

import statsmodels.api as sm
formula = 'mpg ~ disp + hp + wt'

model = ols(formula, mpg_df).fit() 

model.summary()