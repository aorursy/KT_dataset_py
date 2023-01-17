# Importaing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
dataset = pd.read_csv("../input/Admission_Predict.csv")
dataset.head()
# Removing column 'Serial No.' from dataframe

dataset.drop('Serial No.', axis = 1 , inplace = True)
dataset.info()
dataset.describe()
# This is another method which we can use to check for null values.



round(( dataset.isnull().sum() / len(dataset.index) )*100 , 2)
# Creating a derived column



avg_score = round(((dataset.loc[:,['GRE Score', 'TOEFL Score', 'SOP', 'LOR ', 'CGPA']].sum(axis = 1))/(340+120+5+5+10))*100, 2)

dataset.insert(7, 'Average_score', avg_score)

dataset.head()
corr_matrix = dataset.corr()

round( corr_matrix , 3)
plt.figure(figsize = (15,12))

sns.heatmap(corr_matrix , annot = True)

plt.show()
sns.pairplot(dataset)

plt.show()
#defining a normalisation function 

def normalize (x): 

    return ( (x-np.mean(x))/ (max(x) - min(x)))



def inv_normalize (y , x):

    return ( (y*( max(x) - min(x) )) + np.mean(x))



scaled_df = dataset.apply(normalize)



# to inverse normalizaion

#for column in dataset.columns:

#    scaled_df[column] = inv_normalize(scaled_df[column] , dataset[column])

#    scaled_df[column] = scaled_df[column].astype(type(dataset[column][0]))
scaled_df.head()
# Dividing data in X and Y

X = scaled_df.iloc[:,[0,1,2,3,4,5,6,7]]

y = scaled_df.iloc[:,8]



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# creating regression model

X_trainsm = sm.add_constant(X_train)

lm1 = sm.OLS(y_train, X_trainsm).fit()



print(lm1.summary())
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removed Average_score

X = scaled_df.iloc[:,[0,1,2,3,4,5,6]]

y = scaled_df.iloc[:,8]



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# creating regression model

X_trainsm = sm.add_constant(X_train)

lm2 = sm.OLS(y_train, X_trainsm).fit()



print(lm2.summary())
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removed CGPA

X = scaled_df.iloc[:,[0,1,2,3,4,6]]

y = scaled_df.iloc[:,8]



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# creating regression model

X_trainsm = sm.add_constant(X_train)

lm3 = sm.OLS(y_train, X_trainsm).fit()

print(lm3.summary())
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removed SOP

X = scaled_df.iloc[:,[0,1,2,4,6]]

y = scaled_df.iloc[:,8]



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# creating regression model

X_trainsm = sm.add_constant(X_train)

lm4 = sm.OLS(y_train, X_trainsm).fit()

print(lm4.summary())
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removed GRE

X = scaled_df.iloc[:,[1,2,4,6]]

y = scaled_df.iloc[:,8]



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# creating regression model

X_trainsm = sm.add_constant(X_train)

lm5 = sm.OLS(y_train, X_trainsm).fit()

print(lm5.summary())
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removed University Rating

X = scaled_df.iloc[:,[1,4,6]]

y = scaled_df.iloc[:,8]



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# creating regression model

X_trainsm = sm.add_constant(X_train)

lm6 = sm.OLS(y_train, X_trainsm).fit()

print(lm6.summary())
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# As TOEFL is highly correlated with GRE, CGPA and Average_value , tried with different combinations and chose CGPA



X = scaled_df.iloc[:,[4,5,6]]

y = scaled_df.iloc[:,8]



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# creating regression model

X_trainsm = sm.add_constant(X_train)

lm7 = sm.OLS(y_train, X_trainsm).fit()

print(lm7.summary())
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
lm = LinearRegression()

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)



# Note lm7 directly could have been used for which the below code will be required

#X_testsm = sm.add_constant(X_test)

#y_pred = lm7.predict(X_testsm)
# Actual vs Predicted



c = [i for i in range(1,81,1)]

plt.figure(figsize = (10,8))

plt.plot(c,y_test, color = 'blue', linewidth = 3 , linestyle = '-')

plt.plot(c,y_pred, color = 'red', linewidth = 3 , linestyle = '-')

plt.suptitle("Actual vs Predicted", fontsize = 20)

plt.xlabel("Index", fontsize = 18)

plt.ylabel('Probability of getting into a college', fontsize = 16)

plt.show()
# Error terms

c = [i for i in range(1,81,1)]

fig = plt.figure(figsize = (10,8))

plt.plot(c,y_test-y_pred, color="blue", linewidth=3, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('y_test-y_pred', fontsize=16)

plt.show()
# Error terms scatter plot

c = [i for i in range(1,81,1)]

plt.figure(figsize = (10,8))

plt.scatter(c, y_test-y_pred)

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('y_test-y_pred', fontsize=16) 

plt.show()
# Plotting the error terms to understand the distribution.

plt.figure(figsize = (10,8))

sns.distplot((y_test-y_pred),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label

plt.show()
print("r2 value of train data : " + str(lm.score(X_train,y_train)))

print("r2 value of test data : " + str(lm.score(X_test,y_test)))

print('RMSE :', np.sqrt(mean_squared_error(y_test, y_pred)))
lm.coef_

coeff_df = pd.DataFrame(lm.coef_, X_test.columns, columns = ['Coefficient'])

coeff_df
# Visualizeing coefficients

plt.figure(figsize = (10,8))

sns.barplot(x = 'Coefficient', y = coeff_df.index, data = coeff_df)

plt.suptitle('Coefficient strength plot', fontsize = 18)

plt.show()