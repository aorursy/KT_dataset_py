import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
# setting file path

df=pd.read_csv('../input/carprice-dataset/CarPrice.csv')

df.head() # gives only the first 5 rows
# rows, column

df.shape
df.info()
df.describe()
# dropping the car_ID as it is not affecting the car price



df.drop('car_ID',axis=1,inplace=True)

df.head()
# Checking if the dataframe has any missing values



print(df.isnull().values.any())
# to see the data type of each column

df.dtypes
# Outlier Analysis of target variable with maximum amount of Inconsistency



outliers = ['price']

plt.rcParams['figure.figsize'] = [8,8]

sns.boxplot(data=df[outliers], orient="v", palette="Set1",whis=1.5, saturation=1, width=0.7)

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Price Range", fontweight = 'bold')

plt.xlabel("Continous Variable", fontweight = 'bold')

df.shape
# putting all subcategories into a single category

# Using only the car company names



df['CarName']=df['CarName'].str.split(' ',expand=True)

df['CarName'].head()
# checking the unique car companies



df['CarName'].unique()
# renaming the typos in car company names

#replace(wrong:correct)



df['CarName']=df['CarName'].replace({'maxda':'mazda',

                                     'nissan':'Nissan', 

                                     'toyouta':"toyota",

                                     'porcshce':'porsche',

                                     'vokswagen':'volkswagen',

                                     'vw':'volkswagen'

                                    })

# changing the datatype of 'symboling' from int64 to string as it is a categorical variable as per dictionary file.

# check the result of 'df.dtypes' above.



df['symboling']=df['symboling'].astype(str)

df['symboling'].head()
# checking for duplicate values



df.loc[df.duplicated()]



# when no rows are printed, means no duplicate values
# Segregation of columns into numerical and categorical variables

# df.select_dtypes?



cat_col = df.select_dtypes(include='object').columns

num_col = df.select_dtypes(exclude='object').columns

df_cat = df[cat_col]

df_num = df[num_col]


df_cat.head(2)
df_num.head(2)
df['CarName'].value_counts()
# Visualizing the different car names available



plt.figure(figsize=(15,8))

ax=df['CarName'].value_counts().plot(kind='bar',stacked=True, colormap = 'Set2')

plt.title(label = 'CarName')

plt.xlabel("Names of the Car",fontweight = 'bold')

plt.ylabel("Count of Cars",fontweight = 'bold')

plt.show()
# Visualizing the distribution of car prices



plt.figure(figsize=(8,8))

plt.title('Car Price Distribution Plot')

sns.distplot(df['price']) #The distplot shows the distribution of a univariate set of observations.
ax = sns.pairplot(df[num_col])
plt.figure(figsize=(20, 15))

plt.subplot(3,3,1)

sns.boxplot(x = 'doornumber', y = 'price', data = df)

plt.subplot(3,3,2)

sns.boxplot(x = 'fueltype', y = 'price', data = df)

plt.subplot(3,3,3)

sns.boxplot(x = 'aspiration', y = 'price', data = df)

plt.subplot(3,3,4)

sns.boxplot(x = 'carbody', y = 'price', data = df)

plt.subplot(3,3,5)

sns.boxplot(x = 'enginelocation', y = 'price', data = df)

plt.subplot(3,3,6)

sns.boxplot(x = 'drivewheel', y = 'price', data = df)

plt.subplot(3,3,7)

sns.boxplot(x = 'enginetype', y = 'price', data = df)

plt.subplot(3,3,8)

sns.boxplot(x = 'cylindernumber', y = 'price', data = df)

plt.subplot(3,3,9)

sns.boxplot(x = 'fuelsystem', y = 'price', data = df)

plt.show()
plt.figure(figsize=(25, 6))



plt.subplot(1,3,1)

plt1 = df['cylindernumber'].value_counts().plot(kind = 'bar')

plt.title('Number of cylinders')

plt1.set(xlabel = 'Number of cylinders', ylabel='Frequency of Number of cylinders')



plt.subplot(1,3,2)

plt1 = df['fueltype'].value_counts().plot(kind = 'bar')

plt.title('Fuel Type')

plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of Fuel type')



plt.subplot(1,3,3)

plt1 = df['carbody'].value_counts().plot(kind = 'bar')

plt.title('Car body')

plt1.set(xlabel = 'Car Body', ylabel='Frequency of Car Body')
plt.figure(figsize = (10,6))

sns.boxplot(x = 'fuelsystem', y = 'price', hue = 'fueltype', data = df)

plt.show()
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'carbody', y = 'price', hue = 'enginelocation', data = df)

plt.show()
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'cylindernumber', y = 'price', hue = 'fueltype', data = df)

plt.show()
plt.figure(figsize=(20, 6))



dfx = pd.DataFrame(df.groupby(['CarName'])['price'].mean().sort_values(ascending = False))

dfx.plot.bar()

plt.title('Car Company Name vs Average Price')

plt.show()
plt.figure(figsize=(10, 10))



dfx=df.groupby(['carbody'])['price'].mean().sort_values(ascending=False)

dfx.plot.bar()

plt.title('Car Body Name vs Average Price')

plt.show()
# doubt

# Binning the Car Companies based on avg prices of each car Company.

# Binning - putting into buckets



df['price'] = df['price'].astype('int')

dfx = df.copy()

grouped = dfx.groupby(['CarName'])['price'].mean()

print(grouped)



dfx = dfx.merge(grouped.reset_index(), how='left', on='CarName')

bins = [0,10000,20000,40000]

label =['Budget_Friendly','Medium_Range','TopNotch_Cars']

df['Cars_Category'] = pd.cut(dfx['price_y'], bins, right=False, labels=label)

df.head()
# List of significant columns

sig_col = ['Cars_Category','fueltype', 'aspiration','carbody','drivewheel','enginelocation', 'wheelbase', 'carlength', 'carwidth', 

           'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg', 'price']

len(sig_col)
# Keeping only the significant columns in the data frame



df = df[sig_col]
# Categorical variables found previously

cat_col
# List of significant categorical variables

sig_cat_col = ['Cars_Category','fueltype','aspiration','carbody','drivewheel','enginelocation','enginetype','cylindernumber']
# Get the dummy variables for the categorical feature and store it in a new variable - 'dummy1'



dummy1 = pd.get_dummies(df[sig_cat_col])

print(dummy1.shape)

dummy1
# It is a good practice to always drop the first dummy after performing One Hot encoding

# Because the dropped dummy can be explained as the linear combination of the others.

# Therefore, drop_first = True



dummy1 = pd.get_dummies(df[sig_cat_col],drop_first=True)

print(dummy1.shape)

dummy1
# concatenating the dataframe with the dummy variables

df = pd.concat([df,dummy1], axis = 1)
# dropping the significant categorial columns as we have already made and added the dummy variables for the same in the dataframe

df.drop(sig_cat_col, axis = 1, inplace = True)

df.shape
df
# We specify this so that the train and test data set always have the same rows, respectively

# We divide the df into 70/30 ratio



np.random.seed(0)



from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train
df_test
# Numerical variables found previously

num_col
# List of significant numerical variables

sig_num_col = ['wheelbase', 'carlength', 'carwidth', 'curbweight',

       'enginesize', 'boreratio', 'horsepower','citympg', 'highwaympg', 'price']
# We apply feature scaling only on the numerical variables

# since categorical variables are already converted to 0 and 1 using dummies.



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



df_train[sig_num_col] = scaler.fit_transform(df_train[sig_num_col])
df_train.head() 
plt.figure(figsize=(20,20))

sns.heatmap(df_train.corr(), cmap= 'coolwarm')

plt.show()
col = ['highwaympg','citympg','horsepower','enginesize','curbweight','carwidth']
# Scatter Plot of independent variables vs dependent variables



plt.figure(figsize=(18,15))



plt.subplot(2,3,1)

sns.scatterplot(x=col[0],y='price',data =df)



plt.subplot(2,3,2)

sns.scatterplot(x=col[1],y='price',data =df)



plt.subplot(2,3,3)

sns.scatterplot(x=col[2],y='price',data =df)



plt.subplot(2,3,4)

sns.scatterplot(x=col[3],y='price',data =df)



plt.subplot(2,3,5)

sns.scatterplot(x=col[4],y='price',data =df)



plt.subplot(2,3,6)

sns.scatterplot(x=col[5],y='price',data =df)



'''

fig,axes = plt.subplots(2,3,figsize=(18,15))

for seg,colm in enumerate(col):

    x,y = seg//3,seg%3

    an=sns.scatterplot(x=colm, y='price' ,data=df, ax=axes[x,y])

    plt.setp(an.get_xticklabels(), rotation=45)

   

plt.subplots_adjust(hspace=0.5)

'''
y_train = df_train.pop('price') # dependent variable

x_train = df_train # Taking all the independent variables into x_train
y_train
x_train
import statsmodels.api as sm



x_train_copy = x_train
# Add a constant

x_train_copy1 = sm.add_constant(x_train_copy['horsepower'])



# Create a first fitted model

#1st model

lr1=sm.OLS(y_train,x_train_copy1).fit()
lr1.params
# Let's visualise the data with a scatter plot and the fitted regression line



plt.scatter(x_train_copy1.iloc[:, 1], y_train)

plt.plot(x_train_copy1.iloc[:, 1], 0.8062*x_train_copy1.iloc[:, 1], 'r')

plt.show()
print(lr1.summary())
# Add a constant

x_train_2copy = sm.add_constant(x_train[['horsepower','curbweight']])



# Create a 2nd fitted model

lr2 = sm.OLS(y_train, x_train_2copy).fit()
lr2.params
print(lr2.summary())
# Add a constant

x_train_3copy = sm.add_constant(x_train[['horsepower','curbweight', 'enginesize']])



# Create a 2nd fitted model

lr3 = sm.OLS(y_train, x_train_3copy).fit()
lr3.params
print(lr3.summary())
from sklearn.linear_model import LinearRegression



lm = LinearRegression()

lm.fit(x_train, y_train)
# Running RFE with the output number of the variable equal to 15

from sklearn.feature_selection import RFE



rfe=RFE(lm,15)

rfe=rfe.fit(x_train,y_train)
list(zip(x_train.columns,rfe.support_,rfe.ranking_))

# List of Columns which supports the RFE

# Selecting the variables which are in support



col_sup = x_train.columns[rfe.support_]

col_sup

#Dropping 'enginetype_rotor' as per the rfe values in google colab

col_sup = col_sup.drop(col_sup[10])



# adding 'cylindernumber_two' as per the rfe values in google colab

col_sup = col_sup.insert(14,'cylindernumber_two')
col_sup
# Creating X_train dataframe with RFE selected variables



x_train_rfe = x_train[col_sup]

x_train_rfe
# Adding a constant variable and Build a first fitted model

import statsmodels.api as sm  

x_train_rfec = sm.add_constant(x_train_rfe)

lm_rfe = sm.OLS(y_train,x_train_rfec).fit()



#Summary of linear model

print(lm_rfe.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe.values, i) for i in range(x_train_rfe.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe1 = x_train_rfe.drop('cylindernumber_twelve', 1)



# Adding a constant variable and Build a second fitted model



x_train_rfe1c = sm.add_constant(x_train_rfe1)

lm_rfe1 = sm.OLS(y_train, x_train_rfe1c).fit()



#Summary of linear model

print(lm_rfe1.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe1.values, i) for i in range(x_train_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe2 = x_train_rfe1.drop('cylindernumber_six', 1)



# Adding a constant variable and Build a second fitted model



x_train_rfe2c = sm.add_constant(x_train_rfe2)

lm_rfe2 = sm.OLS(y_train, x_train_rfe2c).fit()



#Summary of linear model

print(lm_rfe2.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe2.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe2.values, i) for i in range(x_train_rfe2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe3 = x_train_rfe2.drop('carbody_hardtop', 1)



# Adding a constant variable and Build a second fitted model



x_train_rfe3c = sm.add_constant(x_train_rfe3)

lm_rfe3 = sm.OLS(y_train, x_train_rfe3c).fit()



#Summary of linear model

print(lm_rfe3.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe3.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe3.values, i) for i in range(x_train_rfe3.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe4 = x_train_rfe3.drop('enginetype_ohcv', 1)



# Adding a constant variable and Build a second fitted model



x_train_rfe4c = sm.add_constant(x_train_rfe4)

lm_rfe4 = sm.OLS(y_train, x_train_rfe4c).fit()



#Summary of linear model

print(lm_rfe4.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe4.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe4.values, i) for i in range(x_train_rfe4.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe5 = x_train_rfe4.drop('enginetype_dohcv', 1)



# Adding a constant variable and Build a second fitted model



x_train_rfe5c = sm.add_constant(x_train_rfe5)

lm_rfe5 = sm.OLS(y_train, x_train_rfe5c).fit()



#Summary of linear model

print(lm_rfe5.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe5.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe5.values, i) for i in range(x_train_rfe5.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe6 = x_train_rfe5.drop('cylindernumber_five', 1)



# Adding a constant variable and Build a second fitted model



x_train_rfe6c = sm.add_constant(x_train_rfe6)

lm_rfe6 = sm.OLS(y_train, x_train_rfe6c).fit()



#Summary of linear model

print(lm_rfe6.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe6.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe6.values, i) for i in range(x_train_rfe6.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe6.columns
# Dropping highly correlated variables and insignificant variables



x_train_rfe7 = x_train_rfe6.drop('cylindernumber_two', 1)



# Adding a constant variable and Build a second fitted model



x_train_rfe7c = sm.add_constant(x_train_rfe7)

lm_rfe7 = sm.OLS(y_train, x_train_rfe7c).fit()



#Summary of linear model

print(lm_rfe7.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe7.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe7.values, i) for i in range(x_train_rfe7.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe8 = x_train_rfe7.drop('curbweight', 1,)



# Adding a constant variable and Build a sixth fitted model

x_train_rfe8c = sm.add_constant(x_train_rfe8)

lm_rfe8 = sm.OLS(y_train, x_train_rfe8c).fit()



#Summary of linear model

print(lm_rfe8.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe8.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe8.values, i) for i in range(x_train_rfe8.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe9 = x_train_rfe8.drop('cylindernumber_four', 1,)



# Adding a constant variable and Build a sixth fitted model

x_train_rfe9c = sm.add_constant(x_train_rfe9)

lm_rfe9 = sm.OLS(y_train, x_train_rfe9c).fit()



#Summary of linear model

print(lm_rfe9.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe9.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe9.values, i) for i in range(x_train_rfe9.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe10 = x_train_rfe9.drop('carbody_sedan', 1,)



# Adding a constant variable and Build a sixth fitted model

x_train_rfe10c = sm.add_constant(x_train_rfe10)

lm_rfe10 = sm.OLS(y_train, x_train_rfe10c).fit()



#Summary of linear model

print(lm_rfe10.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe10.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe10.values, i) for i in range(x_train_rfe10.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe11 = x_train_rfe10.drop('carbody_wagon', 1,)



# Adding a constant variable and Build a sixth fitted model

x_train_rfe11c = sm.add_constant(x_train_rfe11)

lm_rfe11 = sm.OLS(y_train, x_train_rfe11c).fit()



#Summary of linear model

print(lm_rfe11.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe11.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe11.values, i) for i in range(x_train_rfe11.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



x_train_rfe12 = x_train_rfe11.drop('carbody_hatchback', 1,)



# Adding a constant variable and Build a sixth fitted model

x_train_rfe12c = sm.add_constant(x_train_rfe12)

lm_rfe12 = sm.OLS(y_train, x_train_rfe12c).fit()



#Summary of linear model

print(lm_rfe12.summary())
# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe12.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe12.values, i) for i in range(x_train_rfe12.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Predicting the price of training set.

y_train_price = lm_rfe12.predict(x_train_rfe12c)
# Plot the histogram of the error terms

# error between the actual price and the predicted price by our model



sns.distplot((y_train - y_train_price),bins=20)

plt.title('Error Term Analysis')

plt.xlabel('Errors')

plt.show()

df_test[sig_num_col] = scaler.transform(df_test[sig_num_col])

df_test.shape
y_test = df_test.pop('price')

x_test = df_test
# Adding Constant

x_test_1 = sm.add_constant(x_test)



x_test_new = x_test_1[x_train_rfe12c.columns]
# Making predictions using the final model

y_pred = lm_rfe12.predict(x_test_new)
y_pred
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

plt.title('y_test vs y_pred', fontsize=20)   

plt.xlabel('y_test ', fontsize=18)                       

plt.ylabel('y_pred', fontsize=16)  
from sklearn.metrics import r2_score



r2_score(y_test, y_pred)
# Predicting the price of training set.

y_train_price2 = lm_rfe9.predict(x_train_rfe9c)

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price2), bins = 20)

fig.suptitle('Error Terms Analysis', fontsize = 20)                   

plt.xlabel('Errors', fontsize = 18)
x_test_2 = x_test_1[x_train_rfe9c.columns]
# Making predictions using the final model

y_pred2 = lm_rfe9.predict(x_test_2)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred2)

fig.suptitle('y_test vs y_pred2', fontsize=20)   

plt.xlabel('y_test ', fontsize=18)                       

plt.ylabel('y_pred2', fontsize=16)
r2_score(y_test, y_pred2)