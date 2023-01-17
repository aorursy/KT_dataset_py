# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Importing CarPrice_Assignment.csv
carDetails = pd.read_csv('../input/CarPriceAssignment.csv')
# Looking at first five rows
carDetails.head()
carDetails.info()
# Let's look at some statistical information about our dataframe.
carDetails.describe()
# Let's plot a pair plot of all variables in our dataframe
sns.set(font_scale=2)
sns.pairplot(carDetails)
plt.figure(figsize = (20,10))  
sns.heatmap(carDetails.corr(),annot = True)
carDetails.drop(['carwidth','curbweight','wheelbase','highwaympg'], axis =1, inplace = True)
#we can also remove carID  as its just a serial number 
carDetails.drop(['car_ID'], axis =1, inplace = True)

carDetails.info()
# Plotting price 
c = [i for i in range(1,206,1)]
fig = plt.figure()
plt.scatter(c,carDetails['price'])
fig.suptitle('price vs index', fontsize=20)              # Plot heading 
plt.xlabel('index', fontsize=18)                          # X-label
plt.ylabel('price', fontsize=16)  

# # carDetails = carDetails.ix[carDetails['price'] <= 25000]
# # carDetails.describe()
# import numpy

# arr = carDetails['price']
# elements = numpy.array(arr)

# mean = numpy.mean(elements, axis=0)
# sd = numpy.std(elements, axis=0)

# final_list1 = [x for x in arr if (x < mean - 2 * sd)]
# final_list2 = [x for x in arr if (x > mean + 2 * sd)]
# print(len(final_list1))
# print(len(final_list2))

# print(final_list1)
# print(final_list2)


# carDetails = carDetails.ix[carDetails['price'] <= 30000]
# carDetails.describe()
carDetails["CarName"] = carDetails["CarName"].str.replace('-', ' ')
carDetails.CarName.unique()

carDetails["CarName"] = carDetails.CarName.map(lambda x: x.split(" ", 1)[0])
# As we have some redundant data in carName lets fix it 
carDetails.CarName = carDetails['CarName'].str.lower()
carDetails['CarName'] = carDetails['CarName'].str.replace('vw','volkswagen')
carDetails['CarName'] = carDetails['CarName'].str.replace('vokswagen','volkswagen')
carDetails['CarName'] = carDetails['CarName'].str.replace('toyouta','toyota')
carDetails['CarName'] = carDetails['CarName'].str.replace('porcshce','porsche')
carDetails['CarName'] = carDetails['CarName'].str.replace('maxda','mazda')
carDetails['CarName'] = carDetails['CarName'].str.replace('maxda','mazda')

carDetails.CarName.unique()
# carDetails.info()
# Converting Yes to 1 and No to 0
carDetails['fueltype'] = carDetails['fueltype'].map({'gas': 1, 'diesel': 0})
carDetails['aspiration'] = carDetails['aspiration'].map({'std': 1, 'turbo': 0})
carDetails['doornumber'] = carDetails['doornumber'].map({'two': 1, 'four': 0})
carDetails['enginelocation'] = carDetails['enginelocation'].map({'front': 1, 'rear': 0})
carDetails.info()
# carDetails.head()
df = pd.get_dummies(carDetails)
df.head()
# df.info()

#defining a normalisation function 
cols_to_norm = ['symboling', 'carlength', 'carheight', 
         'enginesize', 'boreratio', 'stroke', 'compressionratio','horsepower', 'peakrpm', 'citympg', 'price']
# Normalising only the numeric fields 
normalised_df = df[cols_to_norm].apply(lambda x: (x-np.mean(x))/ (max(x) - min(x)))
normalised_df.head()

df['symboling'] = normalised_df['symboling']
df['carlength'] = normalised_df['carlength']
df['carheight'] = normalised_df['carheight']
df['enginesize'] = normalised_df['enginesize']
df['boreratio'] = normalised_df['boreratio']
df['stroke'] = normalised_df['stroke']
df['price'] = normalised_df['price']
df['compressionratio'] = normalised_df['compressionratio']
df['horsepower'] = normalised_df['horsepower']
df['peakrpm']= normalised_df['peakrpm']
df['citympg'] = normalised_df['citympg']
df.head()

refinedcol = df.columns
refinedcol
# Putting feature variable to X
# df.info()
# df.columns
X = df[['symboling', 'fueltype', 'aspiration', 'doornumber', 'enginelocation',
       'carlength', 'carheight', 'enginesize', 'boreratio', 'stroke',
       'compressionratio', 'horsepower', 'peakrpm', 'citympg',
       'CarName_alfa', 'CarName_audi', 'CarName_bmw', 'CarName_buick',
       'CarName_chevrolet', 'CarName_dodge', 'CarName_honda', 'CarName_isuzu',
       'CarName_jaguar', 'CarName_mazda', 'CarName_mercury',
       'CarName_mitsubishi', 'CarName_nissan', 'CarName_peugeot',
       'CarName_plymouth', 'CarName_porsche', 'CarName_renault',
       'CarName_saab', 'CarName_subaru', 'CarName_toyota',
       'CarName_volkswagen', 'CarName_volvo', 'carbody_convertible',
       'carbody_hardtop', 'carbody_hatchback', 'carbody_sedan',
       'carbody_wagon', 'drivewheel_4wd', 'drivewheel_fwd', 'drivewheel_rwd',
       'enginetype_dohc', 'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc',
       'enginetype_ohcf', 'enginetype_ohcv', 'enginetype_rotor',
       'cylindernumber_eight', 'cylindernumber_five', 'cylindernumber_four',
       'cylindernumber_six', 'cylindernumber_three', 'cylindernumber_twelve',
       'cylindernumber_two', 'fuelsystem_1bbl', 'fuelsystem_2bbl',
       'fuelsystem_4bbl', 'fuelsystem_idi', 'fuelsystem_mfi',
       'fuelsystem_mpfi', 'fuelsystem_spdi', 'fuelsystem_spfi']]

# # # Putting response variable to y
y = df['price']
#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)
# help(rfe)
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
rfe = RFE(lm, 15)             # running RFE
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)           # Printing the boolean results
print(rfe.ranking_)  
X_train.columns[rfe.support_]
#variables that are to be dropped
X_train.columns
col = X_train.columns[~rfe.support_]
col
print("Before droping of columns")
X_train.columns
X_train1 = X_train.drop(col,1)
print("After Droping of columns")
X_train1.columns

df.head()
# Adding a constant variable 
import statsmodels.api as sm  
X_train1 = sm.add_constant(X_train1)
lm_1 = sm.OLS(y_train,X_train1).fit() # Running the linear model
print(lm_1.summary())
def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)
df.drop(col, axis =1, inplace = True)
# df.head()
# Calculating Vif value
# df.head()
vif_cal(input_data=df, dependent_col="price")
plt.figure(figsize = (20,10))  
sns.heatmap(df.corr(),annot = True)
# Dropping highly correlated variables and insignificant variables
X_train2 = X_train1.drop('enginetype_rotor', 1)
# Creating a second fitted model
lm_2 = sm.OLS(y_train,X_train2).fit()
#Let's see the summary of our second linear model
print(lm_2.summary())
df.drop('enginetype_rotor', axis =1, inplace = True)
# Calculating Vif value
vif_cal(input_data= df, dependent_col="price")
# Dropping highly correlated variables and insignificant variables
X_train3 = X_train2.drop('cylindernumber_eight', 1)
# Creating a third fitted model 
lm_3 = sm.OLS(y_train,X_train3).fit()
#Let's see the summary of our third linear model
print(lm_3.summary())
df.drop('cylindernumber_eight', axis =1, inplace = True)
# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")
# Dropping highly correlated variables and insignificant variables 
X_train4 = X_train3.drop('enginetype_dohcv', 1)
# Creating a fourth fitted model
lm_4 = sm.OLS(y_train,X_train4).fit()
#Let's see the summary of our fourth linear model
print(lm_4.summary())
df.drop('enginetype_dohcv', axis =1, inplace = True)
# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")
# Dropping highly correlated variables and insignificant variables
X_train5 = X_train4.drop('cylindernumber_four', 1)
# Creating a fifth fitted model
lm_5 = sm.OLS(y_train,X_train5).fit()
#Let's see the summary of our fifth linear model
print(lm_5.summary())
df.drop('cylindernumber_four', axis =1, inplace = True)
# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")
plt.figure(figsize = (20,10))  
sns.heatmap(df.corr(),annot = True)
# Dropping highly correlated variables and insignificant variables
X_train6 = X_train5.drop('cylindernumber_twelve', 1)
# Creating a sixth fitted model
lm_6 = sm.OLS(y_train,X_train6).fit()
#Let's see the summary of our sixth linear model
print(lm_6.summary())
df.drop('cylindernumber_twelve', axis =1, inplace = True)
# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")
# Dropping highly correlated variables and insignificant variables
X_train7 = X_train6.drop('stroke', 1)
# Creating a seventh fitted model
lm_7 = sm.OLS(y_train,X_train7).fit()
#Let's see the summary of our seventh linear model
print(lm_7.summary())
df.drop('stroke', axis =1, inplace = True)
# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")
# Dropping highly correlated variables and insignificant variables
X_train8 = X_train7.drop('boreratio', 1)
# Creating a eighth fitted model
lm_8 = sm.OLS(y_train,X_train8).fit()
#Let's see the summary of our eighth linear model
print(lm_8.summary())
df.drop('boreratio', axis =1, inplace = True)
# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")
# Dropping highly correlated variables and insignificant variables
X_train9 = X_train8.drop('cylindernumber_three', 1)
# Creating a ninth fitted model
lm_9 = sm.OLS(y_train,X_train9).fit()
#Let's see the summary of our ninth linear model
print(lm_9.summary())
df.drop('cylindernumber_three', axis =1, inplace = True)
# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")
plt.figure(figsize = (20,10))  
sns.heatmap(df.corr(),annot = True)
# Adding  constant variable to test dataframe
X_test_m9 = sm.add_constant(X_test)
# Creating X_test_m12 dataframe by dropping variables from X_test_m12
X_test_m9 = X_test_m9.drop(col, axis=1)
X_test_m9 = X_test_m9.drop(['cylindernumber_three','enginetype_rotor','cylindernumber_eight',
                              'enginetype_dohcv','cylindernumber_four','cylindernumber_twelve','stroke','boreratio'], axis=1)
X_test_m9.info()

# Making predictions
y_pred_m9 = lm_9.predict(X_test_m9)
y_pred_m9
# Actual vs Predicted
c = [i for i in range(1,63,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=3.5, linestyle="-")     #Plotting Actual
plt.plot(c,y_pred_m9, color="red",  linewidth=3.5, linestyle="-")  #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Car Price', fontsize=16)  
#Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred_m9)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)     
# Error terms
fig = plt.figure()
c = [i for i in range(1,63,1)]
# plt.plot(c,y_test-y_pred_m9, color="blue", linewidth=2.5, linestyle="-")
plt.scatter(c,y_test-y_pred_m9)

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
# Plotting the error terms to understand the distribution.
fig = plt.figure()
sns.distplot((y_test-y_pred_m9),bins=15)
fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)             
import numpy as np
from sklearn import metrics
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_m9)))
