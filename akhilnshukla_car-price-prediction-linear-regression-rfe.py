import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Importing statsmodel library for statistical summary

import statsmodels.api as sm 



#Importing sklearn methods

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Importing sklearn RFE and LinearRegression methods

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression



pd.set_option('display.max_columns', 600)

pd.set_option('display.max_rows', 50)
dfcar = pd.read_csv('../input/car-data/CarPrice_Assignment.csv')

dfcar.head()
dfcar.info()
dfcar.describe().T
sns.set(font_scale=2)

sns.pairplot(dfcar)
plt.figure(figsize = (20,10))  

sns.set(font_scale=1.5)

sns.heatmap(dfcar.corr(),annot = True)
#Lets drop the highwaympg for reasons mentioned above.



dfcar = dfcar.drop(['highwaympg'], axis=1)
plt.figure(figsize = (20,10))

corr1 = dfcar[['wheelbase','carlength','carwidth','curbweight','carheight','enginesize','price']].corr()

corr1.style.background_gradient(cmap='Greens')
#Lets drop the curbweight, carwidth and wheelbase column for reasons mentioned above.



dfcar = dfcar.drop(['curbweight', 'carwidth', 'wheelbase'], axis=1)
sns.set(font_scale=1)

plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)

sns.boxplot(y = 'fueltype', x = 'price', data = dfcar)

plt.subplot(3,3,2)

sns.boxplot(y = 'aspiration',x = 'price', data = dfcar)

plt.subplot(3,3,3)

sns.boxplot(y = 'doornumber', x = 'price', data = dfcar)

plt.subplot(3,3,4)

sns.boxplot(y = 'carbody', x = 'price', data = dfcar)

plt.subplot(3,3,5)

sns.boxplot(y = 'drivewheel', x = 'price', data = dfcar)

plt.subplot(3,3,6)

sns.boxplot(y = 'enginelocation', x = 'price', data = dfcar)

plt.subplot(3,3,7)

sns.boxplot(y = 'enginetype', x = 'price', data = dfcar)

plt.subplot(3,3,8)

sns.boxplot(y = 'cylindernumber', x = 'price', data = dfcar)

plt.subplot(3,3,9)

sns.boxplot(y = 'fuelsystem', x = 'price', data = dfcar)



plt.show()
dfcar['CarComp'] = dfcar['CarName'].str.replace('-',' ')

dfcar['CarComp'] = dfcar['CarComp'].str.split(' ', n=1, expand=True)[0]
#convert all car company names to lowercase

dfcar['CarComp'] = dfcar['CarComp'].str.lower()



#Note some car company name are entered with typos/ or lowercase. Lets fix that.

correct_name = {'maxda' : 'mazda', 'porcshce' : 'porsche', 'toyouta' : 'toyota', 'vokswagen' : 'volkswagen', 'vw' : 'volkswagen'}

dfcar['CarComp'] = dfcar['CarComp'].replace(correct_name, regex=True)
#Lets drop the CarName and car_ID column as we dont need it anymore

dfcar = dfcar.drop(['CarName', 'car_ID'], axis=1)
#Lets also visualize the derived feature CarComp



plt.figure(figsize=(15,8))

sns.boxplot(x = 'price', y = 'CarComp', data = dfcar)
## Ordinal Features:



# dfcar.fueltype.unique()           # 'gas', 'diesel'



dfcar.fueltype = dfcar.fueltype.map({'gas':0, 'diesel':1})



# dfcar.aspiration.unique()         # 'std', 'turbo'



dfcar.aspiration = dfcar.aspiration.map({'std':0, 'turbo':1})



# dfcar.doornumber.unique()         # 'two', 'four'



dfcar.doornumber = dfcar.doornumber.map({'two':0, 'four':1})



# dfcar.enginelocation.unique()     # 'front', 'rear'



dfcar.enginelocation = dfcar.enginelocation.map({'front':0, 'rear':1})



dfcar
## Nominal Features:



car = pd.get_dummies(dfcar)



# From now on we will use `car` dataset only.
car = car.drop(['CarComp_mercury'], axis=1)

car
# Let us split the dataset 'car' into train and test data (70:30 ratio) using sklearn train_test_split() method



df_train, df_test = train_test_split(car, train_size = 0.7, test_size = 0.3, random_state = 100)
# Fit on train data



# Create list num_vars(contains the list of numeric predictor variables):



num_vars = ['symboling','carlength','carheight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','price']



# Lets normalize both df_train and df_test here.



df_train[num_vars] = df_train[num_vars].apply(lambda x: (x- np.mean(x))/(x.max() - x.min()))

df_test[num_vars] = df_test[num_vars].apply(lambda x: (x- np.mean(x))/(x.max() - x.min()))
df_train.describe()
# Let's check the correlation coefficients to see which variables are highly correlated. 



corr = df_train[df_train.columns].corr()

corr.style.background_gradient(cmap='coolwarm')
sns.pairplot(df_train, x_vars=['carlength', 'carheight'], y_vars='price',height=4, aspect=1, kind='scatter')

plt.show()
sns.pairplot(df_train, x_vars=['enginesize', 'horsepower', 'stroke'], y_vars='price',height=4, aspect=1, kind='scatter')

plt.show()
sns.pairplot(df_train, x_vars=['boreratio', 'compressionratio','peakrpm', 'citympg'], y_vars='price',height=4, aspect=1, kind='scatter')

plt.show()
y_train = df_train.pop('price')

X_train = df_train
# Running RFE with the output number of the variable equal to 15

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col_in = X_train.columns[rfe.support_]

col_in
col_out = X_train.columns[~rfe.support_]

col_out
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col_in]
def LRM_Summ(df):

    # Display the linear model summary

    

    #Adding a constant

    X_train_lm = sm.add_constant(df)

    

    #Fit

    lm = sm.OLS(y_train,X_train_lm).fit()

    

    #Printing the statistics summary

    print(lm.summary())

    

    #Returning the X_train_lm dataset and the VIF table for all the features present in the model

    return X_train_lm, lm
def LRM_VIF_Corr(df):    

    

    # Calculate the VIFs for the current model

    vif = pd.DataFrame()

    X = df

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    print('\n')

    print(vif)

    

    

    #Printing the correlation matrix of features in the current model along with price

    col_var = list(vif.Features)

    col_var = col_var + ['price']

    corr = car[col_var].corr()

    return(corr.style.background_gradient(cmap='coolwarm'))
X_train_lmod1, lmod1 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
X_train_rfe = X_train_rfe.drop(['enginetype_rotor'], axis=1)

car = car.drop(['enginetype_rotor'], axis=1)



X_train_lmod2, lmod2 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
X_train_rfe = X_train_rfe.drop(['enginetype_dohcv'], axis=1)

car = car.drop(['enginetype_dohcv'], axis=1)



X_train_lmod3, lmod3 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
X_train_rfe = X_train_rfe.drop(['cylindernumber_eight'], axis=1)



car = car.drop(['cylindernumber_eight'], axis=1)



X_train_lmod4, lmod4 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
X_train_rfe = X_train_rfe.drop(['cylindernumber_four'], axis=1)



car = car.drop(['cylindernumber_four'], axis=1)



X_train_lmod5, lmod5 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
X_train_rfe = X_train_rfe.drop(['cylindernumber_twelve'], axis=1)



car = car.drop(['cylindernumber_twelve'], axis=1)



X_train_lmod6, lmod6 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
X_train_rfe = X_train_rfe.drop(['stroke'], axis=1)



car = car.drop(['stroke'], axis=1)



X_train_lmod7, lmod7 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
X_train_rfe = X_train_rfe.drop(['boreratio'], axis=1)



car = car.drop(['boreratio'], axis=1)



X_train_lmod8, lmod8 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
X_train_rfe = X_train_rfe.drop(['cylindernumber_three'], axis=1)



car = car.drop(['cylindernumber_three'], axis=1)



X_train_lmod9, lmod9 = LRM_Summ(X_train_rfe)

LRM_VIF_Corr(X_train_rfe)
y_train_pred = lmod9.predict(X_train_lmod9)
res = y_train - y_train_pred

sns.distplot(res, color = 'blue')
y_test = df_test.pop('price')

X_test = df_test
# Adding  constant variable to test dataframe

X_test_lmod9 = sm.add_constant(X_test)

X_test_lmod9.head()
# Creating X_test_lm9 dataframe by dropping variables from X_test_lm9

X_test_lmod9 = X_test_lmod9.drop(col_out, axis=1)

X_test_lmod9 = X_test_lmod9.drop(['enginetype_dohcv', 'enginetype_rotor', 'cylindernumber_eight',

       'cylindernumber_four', 'cylindernumber_three', 'stroke', 'boreratio', 'cylindernumber_twelve'], axis=1)

X_test_lmod9.head()
#Predict

y_test_pred = lmod9.predict(X_test_lmod9)
#Evaluate the model

r2_score(y_true = y_test, y_pred = y_test_pred)
# Actual vs Predicted

index = [i for i in range(1,63,1)]

fig = plt.figure()

plt.plot(index,y_test, color="blue", linewidth=3.5, linestyle="-")     #Plotting Actual

plt.plot(index,y_test_pred, color="purple",  linewidth=3.5, linestyle="-")  #Plotting predicted

fig.suptitle('Actual and Predicted')              # Plot heading 

plt.xlabel('Index')                               # X-label

plt.ylabel('Car Price')  
#Plotting y_test against y_pred to see the relationship.

fig = plt.figure()

plt.scatter(y_test,y_test_pred)

fig.suptitle('y_test vs y_test_pred')              # Plot heading 

plt.xlabel('y_test')                          # X-label

plt.ylabel('y_test_pred')     
# Error terms

fig = plt.figure(figsize = (16,6))



index = [i for i in range(1,63)]



#To see the randomness with dots

plt.subplot(1,2,1)

plt.scatter(index, y_test-y_test_pred, color = 'red')



fig.suptitle('Error Terms')              # Plot heading 

plt.xlabel('Index')                      # X-label

plt.ylabel('y_test - y_test_pred')                # Y-label





# To join the randomness with dots to see if it has any pattern

plt.subplot(1,2,2)

plt.plot(index, y_test-y_test_pred, color="red")



fig.suptitle('Error Terms')              # Plot heading 

plt.xlabel('Index')                      # X-label

plt.ylabel('y_test - y_test_pred')                # Y-label
# Let us plot the histogram plot of the error terms to see if they follow the Normal Distribution.



fig = plt.figure(figsize = (4,4))



sns.distplot((y_test - y_test_pred),bins=10)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)             