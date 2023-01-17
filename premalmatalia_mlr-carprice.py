# Importing all required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder #to encode categorical values

from sklearn.preprocessing import MinMaxScaler # to scale the data

from sklearn.model_selection import train_test_split # To create train test split

from sklearn.feature_selection import RFE # to calculate RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor # To calculate VIF

import statsmodels.api as sm  

import warnings
###supress warnings

warnings.filterwarnings('ignore')
#Importing dataset

carprice = pd.read_csv("../input/CarPrice_Assignment.csv")
#Let's explore the top 5 rows

carprice.head(5)

carprice.shape ## (205,26)

carprice.info()

carprice.columns
##Understand values in different categorical columns

print(carprice.symboling.value_counts()) 

print("\n")

print(carprice.fueltype.value_counts())

print("\n")

print(carprice.aspiration.value_counts())

print("\n")

print(carprice.doornumber.value_counts())

print("\n")

print(carprice.carbody.value_counts())

print("\n")

print(carprice.drivewheel.value_counts())

print("\n")

print(carprice.enginelocation.value_counts())

print("\n")

print(carprice.enginetype.value_counts())

print("\n")

print(carprice.cylindernumber.value_counts())

print("\n")

print(carprice.fuelsystem.value_counts())
carprice.describe()
carprice.drop('car_ID',axis=1,inplace=True)
carprice['CarName'] = carprice['CarName'].str.split(' ',n=1,expand=True)[0]

carprice.rename(columns ={'CarName':'carcompany'},inplace=True)
carprice.isnull().sum()

##No null values in dataset
carprice.carcompany.value_counts().sort_index()
carprice['carcompany'].replace({"maxda": "mazda", "porcshce": "porsche","toyouta": "toyota","vokswagen":"volkswagen",

                            "vw":"volkswagen","Nissan":"nissan"}, inplace=True)
carprice_1 = carprice[['carlength','carwidth', 'carheight', 'curbweight','price']]

sns.pairplot(carprice_1)

plt.show()
carprice_2 = carprice[['enginesize','boreratio', 'stroke', 'compressionratio',

                       'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']]

sns.pairplot(carprice_2)

plt.show()
plt.figure(figsize = (20,10))  

sns.heatmap(carprice.corr(),annot = True,cmap="YlGnBu")
plt.figure(figsize=(20, 20))

plt.subplot(4,3,1)

sns.boxplot(x = 'symboling', y = 'price', data = carprice)

plt.subplot(4,3,2)

sns.boxplot(x = 'fueltype', y = 'price', data = carprice)

plt.subplot(4,3,3)

sns.boxplot(x = 'aspiration', y = 'price', data = carprice)

plt.subplot(4,3,4)

sns.boxplot(x = 'doornumber', y = 'price', data = carprice)

plt.subplot(4,3,5)

sns.boxplot(x = 'carbody', y = 'price', data = carprice)

plt.subplot(4,3,6)

sns.boxplot(x = 'drivewheel', y = 'price', data = carprice)

plt.subplot(4,3,7)

sns.boxplot(x = 'enginelocation', y = 'price', data = carprice)

plt.subplot(4,3,8)

sns.boxplot(x = 'enginetype', y = 'price', data = carprice)

plt.subplot(4,3,9)

sns.boxplot(x = 'cylindernumber', y = 'price', data = carprice)

plt.subplot(4,3,10)

sns.boxplot(x = 'fuelsystem', y = 'price', data = carprice)

plt.show()
plt.figure(figsize=(15, 20))

sns.boxplot(x = 'price', y = 'carcompany', data = carprice,orient="h")
le = LabelEncoder()

carprice['fueltype'] = le.fit(carprice['fueltype']).transform(carprice['fueltype'])

carprice['aspiration'] = le.fit(carprice['aspiration']).transform(carprice['aspiration'])

carprice['doornumber'] = le.fit(carprice['doornumber']).transform(carprice['doornumber'])

carprice['enginelocation'] = le.fit(carprice['enginelocation']).transform(carprice['enginelocation'])
def create_dummies(df,col):

    '''This function is to create dummy variables for given dataframe and columns

        **Parameter Details:**

        df- dataframe name 

        col- Column name for which dummy variables need to be created

    '''

    dummy_df = pd.get_dummies(df[[col]], drop_first = True)

    # Concat original dataframe and dummy_df

    df = pd.concat([df,dummy_df],axis=1)

    # Drop column as we have created the dummies for it

    df.drop([col], axis = 1, inplace = True)

    return df
carprice = create_dummies(carprice,'carbody')

carprice = create_dummies(carprice,'drivewheel')

carprice = create_dummies(carprice,'enginetype')

carprice = create_dummies(carprice,'fuelsystem')

carprice = create_dummies(carprice,'carcompany')



##Verify increase in number of columns

carprice.head()
carprice['cylindernumber'] = carprice['cylindernumber'].map({'two':2, 'three': 3,'four': 4, 

                                                             'five': 5,'six': 6,'eight': 8,'twelve':12})
carprice['symboling'] = carprice['symboling'].map({ -2 : 'safe', -1: 'safe',0: 'safe', 

                                                    1: 'risky',2: 'risky',3:'risky'})

carprice = create_dummies(carprice,'symboling')
df_train,df_test = train_test_split(carprice,train_size=0.7,test_size=0.3,random_state=100)
print(df_train.shape)

print(df_test.shape)
#num_vars= ['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','cylindernumber',

#            'compressionratio','horsepower','peakrpm','highwaympg','citympg','price']

num_vars= ['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','cylindernumber',

           'compressionratio','horsepower','peakrpm','highwaympg','citympg']

scaler = MinMaxScaler()

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
## All the above variables should have min value of 0 and max value of 1

df_train.describe()
y_train = df_train.pop('price')

X_train = df_train
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)             # running RFE

rfe = rfe.fit(X_train, y_train)
#list(zip(X_train.columns,rfe.support_,rfe.ranking_))
## This will give us columns selected by RFE model

rfe_col = X_train.columns[rfe.support_]

print(rfe_col)
# Creating X_train dataframe with RFE selected variables

X_train_rfe = X_train[rfe_col]
# Adding a constant variable 

X_train_rfe = sm.add_constant(X_train_rfe)

# Running the simple ordinary least square linear model

lm = sm.OLS(y_train,X_train_rfe).fit()   

print(lm.summary())
# Calculate the VIFs for the new model

def VIF_calculation(X_df):

    vif = pd.DataFrame()

    X = X_df

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return vif
print(VIF_calculation(X_train_rfe))
X_train_2 = X_train_rfe.drop(['compressionratio'],axis=1,)

X_train_2 = sm.add_constant(X_train_2)

lm_2 = sm.OLS(y_train,X_train_2).fit()

print(lm_2.summary())
##Checking VIF again

print(VIF_calculation(X_train_2))
plt.figure(figsize=(10,10))

sns.heatmap(X_train_2.corr(),annot=True,cmap="YlGnBu")

plt.show()
X_train_3 = X_train_2.drop(['carlength'],axis=1)

X_train_3 = sm.add_constant(X_train_3)

lm_3 = sm.OLS(y_train,X_train_3).fit()

print(lm_3.summary())
print(VIF_calculation(X_train_3))
plt.figure(figsize=(10,10))

sns.heatmap(X_train_3.corr(),annot=True,cmap="YlGnBu")

plt.show()
X_train_4 = X_train_3.drop(['enginetype_l'],axis=1)

X_train_4 = sm.add_constant(X_train_4)

lm_4 = sm.OLS(y_train,X_train_4).fit()

print(lm_4.summary())
print(VIF_calculation(X_train_4))
plt.figure(figsize=(10,10))

sns.heatmap(X_train_4.corr(),annot=True,cmap="YlGnBu")

plt.show()
X_train_5 = X_train_4.drop(['curbweight'],axis=1)

X_train_5 = sm.add_constant(X_train_5)

lr_5 = sm.OLS(y_train,X_train_5).fit()

print(lr_5.summary())
print(VIF_calculation(X_train_5))
plt.figure(figsize=(10,10))

sns.heatmap(X_train_5.corr(),annot=True,cmap="YlGnBu")

plt.show()
X_train_6 = X_train_5.drop(['enginesize'],axis=1)

X_train_6 = sm.add_constant(X_train_6)

lr_6 = sm.OLS(y_train,X_train_6).fit()

print(lr_6.summary())
print(VIF_calculation(X_train_6))
plt.figure(figsize=(10,10))

sns.heatmap(X_train_6.corr(),annot=True,cmap="YlGnBu")

plt.show()
X_train_7 = X_train_6.drop(['carcompany_porsche'],axis=1)

X_train_7 = sm.add_constant(X_train_7)

lr_7 = sm.OLS(y_train,X_train_7).fit()

print(lr_7.summary())
print(VIF_calculation(X_train_7))
X_train_8 = X_train_7.drop(['stroke'],axis=1)

X_train_8 = sm.add_constant(X_train_8)

lr_8 = sm.OLS(y_train,X_train_8).fit()

print(lr_8.summary())
print(VIF_calculation(X_train_8))
plt.figure(figsize=(10,10))

sns.heatmap(X_train_8.corr(),annot=True,cmap="YlGnBu")

plt.show()
X_train_9 = X_train_8.drop(['boreratio'],axis=1)

X_train_9 = sm.add_constant(X_train_9)

lm_9 = sm.OLS(y_train,X_train_9).fit()

print(lm_9.summary())
print(VIF_calculation(X_train_9))
X_train_10 = X_train_9.drop(['peakrpm'],axis=1)

X_train_10 = sm.add_constant(X_train_10)

lm_10 = sm.OLS(y_train,X_train_10).fit()

print(lm_10.summary())
print(VIF_calculation(X_train_10))
X_train_11 = X_train_10.drop(['fueltype'],axis=1)

X_train_11 = sm.add_constant(X_train_11)

lm_11 = sm.OLS(y_train,X_train_11).fit()

print(lm_11.summary())
print(VIF_calculation(X_train_11))
plt.figure(figsize=(5,5))

sns.heatmap(X_train_11.corr(),annot=True,cmap="YlGnBu")

plt.show()
X_train_12 = X_train_11.drop(['carcompany_peugeot'],axis=1)

X_train_12 = sm.add_constant(X_train_12)

lm_12 = sm.OLS(y_train,X_train_12).fit()

print(lm_12.summary())
print(VIF_calculation(X_train_12))
sns.heatmap(X_train_12.corr(),annot=True,cmap="YlGnBu")

plt.show()
y_train_pred = lm_12.predict(X_train_12)

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_pred),bins=20)

plt.title('Error Terms - Distribution Plot')                   

plt.xlabel('Errors')                      

plt.show()



pp_x = sm.ProbPlot(y_train)

pp_y = sm.ProbPlot(y_train_pred)

sm.qqplot_2samples(pp_x, pp_y,line='45')

plt.title('Error Terms - QQ Plot')                   

plt.xlabel('y_train_pred')                      

plt.ylabel('y_train')

plt.show()
# Error terms

c = [i for i in range(1,144,1)]

fig = plt.figure(figsize=(12,4))

plt.subplot(131)

plt.plot(c,y_train-y_train_pred, color="blue", linewidth=2.5, linestyle="-")

plt.xlabel('Index', fontsize=12)                      

plt.ylabel('y_train - y_train_pred', fontsize=12) 

plt.title('Error Terms: Line Graph')



plt.subplot(132)

plt.scatter(c,y_train - y_train_pred)

plt.xlabel('Index', fontsize=12)                     

plt.ylabel('y_train - y_train_pred', fontsize=12) 

plt.title('Error Terms: Scatter plot')



plt.subplot(133)

plt.scatter(y_train,y_train_pred)

plt.xlabel('y_train', fontsize=12)                         

plt.ylabel('y_train_pred', fontsize=12)                        

plt.title('y_train vs y_train_pred')



plt.tight_layout()

plt.show()
from sklearn.model_selection import cross_val_score, cross_val_predict

#from sklearn import metrics

from sklearn import datasets, linear_model



lm = linear_model.LinearRegression()

model = lm.fit(X_train_12,y_train)

scores = cross_val_score(model, X_train_12, y_train, scoring='r2')

print ('Cross-validated scores:', scores)

print('Cross-validated mean score:', scores.mean())
col_12 = ['enginelocation', 'carwidth','cylindernumber', 'carcompany_bmw','price']

num_vars_12 = ['carwidth','cylindernumber']



df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test_12 = df_test[col_12]

y_test = df_test_12.pop('price')

X_test = df_test_12
X_test.head()

y_test.head()

X_test.describe()
# Making predictions

X_test = sm.add_constant(X_test)

y_pred = lm_12.predict(X_test)

y_pred

# Actual vs Predicted

c = [i for i in range(1,63,1)]

fig = plt.figure(figsize=(8,6))

plt.plot(c,y_test, color="blue", linewidth=3.5, linestyle="-")  #Plotting Actual

plt.plot(c,y_pred, color="red",  linewidth=3.5, linestyle="-")  #Plotting predicted

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Car Price', fontsize=16)

plt.show()
from sklearn.metrics import mean_squared_error, r2_score

#mse = mean_squared_error(y_test, y_pred)

r2_score = r2_score(y_test, y_pred)

#print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r2_score)