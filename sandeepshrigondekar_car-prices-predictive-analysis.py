# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



# import all important libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
df_car = pd.read_csv("../input/CarPrice_Assignment.csv")

df_car.head()
df_car.shape
df_car.head()
df_car.info()
df_car.describe()
df_car['carcompany'] = df_car.CarName.str.split().str[0]

df_car['carmodel'] = df_car.CarName.str.split().str[1]

df_car.carcompany = df_car.carcompany.replace({'maxda' : 'mazda' , 'vw':'volkswagen','vokswagen':'volkswagen',

                                                       'Nissan':'nissan', 'porcshce':'porsche','toyouta':'toyota'})
# drop car id , car name and carmodel as we dont need them for further analysis OR model building

df_car.drop(['car_ID'],axis=1,inplace=True)

df_car.drop(['CarName'],axis=1,inplace=True)

df_car.drop(['carmodel'],axis=1,inplace=True)
plt.figure(figsize=(20, 20))

plt.subplot(2,1,1)

ax1 = df_car.carcompany.value_counts().plot('bar')

plt.title('Car Companies')

plt.xticks(rotation=45,fontsize=12)

ax1.set(xlabel = 'Car Companies', ylabel='Favorite Car Company')



plt.subplot(2,1,2)

plt.title('Average car price of each company')

plt.xticks(rotation=45,fontsize=12)

ax2=sns.barplot(x='carcompany', y='price', data=df_car, estimator=np.mean)

ax2.set_xlabel("Car Companies")

ax2.set_ylabel("Avg Price (Dollars)")



plt.show()
plt.figure(figsize=(16, 10))



plt.subplot(3,2,1)

ax1 = sns.countplot(df_car['symboling'])

ax1.set(xlabel = 'Risk', ylabel= 'Count of Cars')



plt.subplot(3,2,2)

ax2=sns.barplot(x='symboling', y='price', data=df_car, estimator=np.mean)

ax2.set_xlabel("Risk")

ax2.set_ylabel("Avg Price (Dollars)")



plt.subplot(3,2,3)

ax1 = sns.countplot(df_car['carbody'])

ax1.set(xlabel = 'Car Body', ylabel= 'Count of Cars')



plt.subplot(3,2,4)

ax1 = sns.countplot(df_car['doornumber'])

ax1.set(xlabel = 'No Of Doors', ylabel= 'Count of Cars')



plt.subplot(3,2,5)

ax1 = sns.countplot(df_car['drivewheel'])

ax1.set(xlabel = 'Wheel Drive Type', ylabel= 'Count of Cars')



plt.subplot(3,2,6)

ax1 = sns.countplot(df_car['enginelocation'])

ax1.set(xlabel = 'Engine Location', ylabel= 'Count of Cars')



plt.show()

# plt.tight_layout()
plt.figure(figsize=(20, 14))



plt.subplot(3,1,1)

plt.xticks(rotation=45,fontsize=12)

ax1=sns.barplot(x='cylindernumber', y='price', data=df_car, estimator=np.mean)

ax1.set(xlabel = '# Cylinders', ylabel= 'Count of Cars')



plt.subplot(3,1,2)

plt.xticks(rotation=90,fontsize=12)

ax2=sns.barplot(x='horsepower', y='price', data=df_car, estimator=np.mean)

ax2.set_xlabel("Horsepower")

ax2.set_ylabel("Avg Price (Dollars)")



plt.subplot(3,1,3)

plt.xticks(rotation=45,fontsize=12)

ax2=sns.barplot(x='enginetype', y='price', data=df_car, estimator=np.mean)

ax2.set_xlabel("Engine Type")

ax2.set_ylabel("Avg Price (Dollars)")



plt.show()
plt.figure(figsize=(20, 20))

plt.subplot(3,3,1)

sns.boxplot(x = 'fueltype', y = 'price', data = df_car)

plt.subplot(3,3,2)

sns.boxplot(x = 'aspiration', y = 'price', data = df_car)

plt.subplot(3,3,3)

sns.boxplot(x = 'carbody', y = 'price', data = df_car)

plt.subplot(3,3,4)

sns.boxplot(x = 'doornumber', y = 'price', data = df_car)

plt.subplot(3,3,5)

sns.boxplot(x = 'drivewheel', y = 'price', data = df_car)

plt.subplot(3,3,6)

sns.boxplot(x = 'enginelocation', y = 'price', data = df_car)

plt.subplot(3,3,7)

sns.boxplot(x = 'enginetype', y = 'price', data = df_car)

plt.subplot(3,3,8)

sns.boxplot(x = 'cylindernumber', y = 'price', data = df_car)

plt.subplot(3,3,9)

sns.boxplot(x = 'fuelsystem', y = 'price', data = df_car)

plt.show()
plt.figure(figsize = (20, 20))

sns.pairplot(df_car)

plt.show()
plt.figure(figsize = (16, 10))

sns.heatmap(df_car.corr(), annot = True, cmap="YlGnBu")

plt.show()
df_car['fuel-economy'] = df_car.highwaympg / df_car.citympg

df_car['car-dimension'] = df_car.carwidth / df_car.carheight

df_car['car-perf'] = df_car.horsepower / df_car.enginesize

df_car['car-load'] = df_car.curbweight / df_car.carlength



# drop original variables  as we dont need it anymore

df_car.drop(['highwaympg'],axis=1,inplace=True)

df_car.drop(['citympg'],axis=1,inplace=True)

df_car.drop(['carwidth'],axis=1,inplace=True)

df_car.drop(['carheight'],axis=1,inplace=True)

df_car.drop(['carlength'],axis=1,inplace=True)

df_car.drop(['horsepower'],axis=1,inplace=True)

df_car.drop(['curbweight'],axis=1,inplace=True)

df_car.drop(['enginesize'],axis=1,inplace=True)
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Car Price Distribution Plot')

sns.distplot(df_car.price)



plt.subplot(1,2,2)

plt.title('Car Price Spread')

sns.boxplot(y=df_car.price)



plt.show()
print(df_car.price.describe(percentiles = [0.25,0.50,0.75,1]))
categorical_cols=['fueltype','aspiration','enginelocation','carbody','drivewheel','enginetype','fuelsystem',

                  'carcompany']



# num_cols = ['wheelbase','boreratio','stroke','compressionratio','peakrpm','highwaympg','citympg','cylindernumber','doornumber',

#             'symboling','carheight','carlength','carwidth','curbweight','enginesize','horsepower','price']



num_cols = ['wheelbase','boreratio','stroke','compressionratio','peakrpm','cylindernumber','doornumber',

            'symboling','price','fuel-economy','car-dimension','car-perf','car-load']



cnt = len (categorical_cols) + len(num_cols)

print(cnt)
# Encoding : Convert binary categorical variables to 1's and 0's

df_car.doornumber = df_car.doornumber.replace({'two' : 2,'four' : 4})

df_car.cylindernumber = df_car.cylindernumber.replace({'two' : 2,'three': 3,'four' : 4,'five' : 5, 'six': 6, 

                                                       'eight' : 8, 'twelve' : 12})
df_car.head(10)
# Dummies : Let's convert categorical variables into numerical variables

df_ft = pd.get_dummies(df_car['fueltype'], drop_first = True)

df_car = pd.concat([df_car, df_ft], axis = 1)



df_asp = pd.get_dummies(df_car['aspiration'], drop_first = True)

df_car = pd.concat([df_car, df_asp], axis = 1)



df_el = pd.get_dummies(df_car['enginelocation'], drop_first = True)

df_car = pd.concat([df_car, df_el], axis = 1)



df_cb = pd.get_dummies(df_car['carbody'], drop_first = True)

df_car = pd.concat([df_car, df_cb], axis = 1)



df_dwhl = pd.get_dummies(df_car['drivewheel'], drop_first = True)

df_car = pd.concat([df_car, df_dwhl], axis = 1)



df_engtp = pd.get_dummies(df_car['enginetype'], drop_first = True)

df_car = pd.concat([df_car, df_engtp], axis = 1)



df_fs = pd.get_dummies(df_car['fuelsystem'], drop_first = True)

df_car = pd.concat([df_car, df_fs], axis = 1)



df_cc = pd.get_dummies(df_car['carcompany'], drop_first = True)

df_car = pd.concat([df_car, df_cc], axis = 1)
df_car.drop(columns=categorical_cols,axis=1,inplace=True)
df_car.shape
df_train, df_test = train_test_split(df_car, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.shape
df_test.shape
scaler = MinMaxScaler()

df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

df_train.head()
df_train.describe()
y_train = df_train.pop('price')

X_train = df_train
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 20)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col_sel = X_train.columns[rfe.support_]

col_sel
col_not_sel = X_train.columns[~rfe.support_]

col_not_sel
X_train_rfe = X_train[col_sel]

X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

print(lm.summary())
# Calculate the VIFs for the new model

vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# High VIF and High pVal

X_train_new1 = X_train_rfe.drop(["idi"], axis = 1)
X_train_lm1 = sm.add_constant(X_train_new1)

lm = sm.OLS(y_train,X_train_lm1).fit()   # Running the linear model

print(lm.summary())
# Calculate the VIFs for the new model

vif = pd.DataFrame()

X = X_train_new1

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new2 = X_train_new1.drop(["chevrolet"], axis = 1)
X_train_lm2 = sm.add_constant(X_train_new2)

lm = sm.OLS(y_train,X_train_lm2).fit()   # Running the linear model

print(lm.summary())
# Calculate the VIFs for the new model

vif = pd.DataFrame()

X = X_train_new2

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new3 = X_train_new2.drop(["compressionratio"], axis = 1)
X_train_lm3 = sm.add_constant(X_train_new3)

lm = sm.OLS(y_train,X_train_lm3).fit()   # Running the linear model

print(lm.summary())
# Calculate the VIFs for the new model

vif = pd.DataFrame()

X = X_train_new3

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new4 = X_train_new3.drop(["gas"], axis = 1)
X_train_lm4 = sm.add_constant(X_train_new4)

lm = sm.OLS(y_train,X_train_lm4).fit()   # Running the linear model

print(lm.summary())
# Calculate the VIFs for the new model

vif = pd.DataFrame()

X = X_train_new4

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new5 = X_train_new4.drop(["l"], axis = 1)
X_train_lm5 = sm.add_constant(X_train_new5)

lm = sm.OLS(y_train,X_train_lm5).fit()   # Running the linear model

print(lm.summary())
# Calculate the VIFs for the new model

vif = pd.DataFrame()

X = X_train_new5

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new5 = X_train_new5.drop(["const"], axis = 1)
y_train_price = lm.predict(X_train_lm5)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
df_test[num_cols] = scaler.transform(df_test[num_cols])
y_test = df_test.pop('price')

X_test = df_test
# # Now let's use our model to make predictions.



# # Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new5.columns]



# # Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions with the final model (model-6)

y_pred = lm.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)   
r2_score(y_test, y_pred)