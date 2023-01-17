import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Suppress Warnings

import warnings

warnings.filterwarnings('ignore')
auto = pd.read_csv("../input/CarPricePrediction.csv")

auto.head()
auto.info()
# auto.rename(columns={"CarName":"company"})



auto["CarName"] = auto.CarName.map(lambda x: x.split(" ", 1)[0])



# As we have some redundant data in carName lets fix it 

auto.CarName = auto['CarName'].str.lower()

auto['CarName'] = auto['CarName'].str.replace('vw','volkswagen')

auto['CarName'] = auto['CarName'].str.replace('vokswagen','volkswagen')

auto['CarName'] = auto['CarName'].str.replace('toyouta','toyota')

auto['CarName'] = auto['CarName'].str.replace('porcshce','porsche')

auto['CarName'] = auto['CarName'].str.replace('maxda','mazda')

auto['CarName'] = auto['CarName'].str.replace('maxda','mazda')



auto.CarName.unique()
auto.info()
sns.set(font_scale=2)

sns.pairplot(auto)

plt.show()
plt.figure(figsize = (20, 20))

sns.set(font_scale=1.25)

sns.heatmap(auto.corr(), annot = True, cmap="YlGnBu")

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(y='CarName', x='price', data = auto)
plt.figure(figsize=(20,20))



plt.subplot(3,3,1)

sns.boxplot(x='fueltype', y='price', data = auto)



plt.subplot(3,3,2)

sns.boxplot(x='aspiration', y='price', data = auto)



plt.subplot(3,3,3)

sns.boxplot(x='doornumber', y='price', data = auto)



plt.subplot(3,3,4)

sns.boxplot(x='enginelocation', y='price', data = auto)



plt.subplot(3,3,5)

sns.boxplot(x='drivewheel', y='price', data = auto)



plt.subplot(3,3,6)

sns.boxplot(x='enginetype', y='price', data = auto)



plt.subplot(3,3,7)

sns.boxplot(x='cylindernumber', y='price', data = auto)



plt.subplot(3,3,8)

sns.boxplot(x='fuelsystem', y='price', data = auto)



plt.subplot(3,3,9)

sns.boxplot(x='carbody', y='price', data = auto)

auto.drop(['car_ID'], axis =1, inplace = True)
# Converting Yes to 1 and No to 0

auto['fueltype'] = auto['fueltype'].map({'gas': 1, 'diesel': 0})

auto['aspiration'] = auto['aspiration'].map({'std': 1, 'turbo': 0})

auto['doornumber'] = auto['doornumber'].map({'two': 1, 'four': 0})

auto['enginelocation'] = auto['enginelocation'].map({'front': 1, 'rear': 0})
auto.drop(['carwidth','curbweight','wheelbase','highwaympg'], axis =1, inplace = True)

auto.info()


# # drivewheel, enginetype, cylindernumber, fuelsystem, carbody



# dummy_drivewheel = pd.get_dummies(auto['drivewheel'], drop_first = True)

# dummy_enginetype = pd.get_dummies(auto['enginetype'], drop_first = True)

# dummy_cylindernumber = pd.get_dummies(auto['cylindernumber'], drop_first = True)

# dummy_fuelsystem = pd.get_dummies(auto['fuelsystem'], drop_first = True)

# dummy_carbody = pd.get_dummies(auto['carbody'], drop_first = True)





# ## Concatenating all the Dummy Variables to the DataFrame



# auto = pd.concat([auto, dummy_drivewheel,dummy_enginetype, dummy_cylindernumber, dummy_fuelsystem, dummy_carbody], axis = 1)





# # Dropping the Columns where Dummy variables have been created

# auto.drop(['drivewheel'],axis = 1, inplace=True)

# auto.drop(['enginetype'],axis = 1, inplace=True)

# auto.drop(['cylindernumber'],axis = 1, inplace=True)

# auto.drop(['fuelsystem'],axis = 1, inplace=True)

# auto.drop(['carbody'],axis = 1, inplace=True)

# auto.head()
df = pd.get_dummies(auto)

df.columns

df.head()
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

from sklearn.model_selection import train_test_split
y = df.pop('price')

X = df
X.info()
from sklearn.model_selection import train_test_split



np.random.seed(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression



# Running RFE with the output number of the variable equal to 15

lm = LinearRegression()



# Running RFE -Recursive Feature Elimination

rfe = RFE(lm, 15)             

rfe = rfe.fit(X_train, y_train)



# Printing the boolean results

print(rfe.support_)           

print(rfe.ranking_)  
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# variables to be dropped

col = X_train.columns[~rfe.support_] 

col
X_train.columns

X_train.drop(col,1, inplace = True)

X_train.columns
plt.figure(figsize = (15, 15))

sns.set(font_scale=1.0)

sns.heatmap(X_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
X_train_1 = X_train
# Add a constant

X_train_1 = sm.add_constant(X_train_1)



# Create Model1

lr1 = sm.OLS(y_train, X_train_1).fit()

print(lr1.params)

print(lr1.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Function to create a dataframe that will contain the names of all the feature variables and their respective VIFs



def calculate_VIF(data_frame):

    vif = pd.DataFrame(columns = ['Features', 'VIF'])

    vif['Features'] = data_frame.columns

    vif['VIF'] = [variance_inflation_factor(data_frame.values, i) for i in range(data_frame.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return vif
# Model1 VIF

calculate_VIF(X_train)
plt.figure(figsize = (15, 15))

sns.set(font_scale=1)

sns.heatmap(X_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
X_train_2 = X_train_1.drop('enginetype_rotor', 1)



# Add a constant

X_train_2 = sm.add_constant(X_train_2)



# Create Model1

lr2 = sm.OLS(y_train, X_train_2).fit()

print(lr2.params)

print(lr2.summary())
# Model2 VIF

X_train.drop('enginetype_rotor', axis =1, inplace = True)

calculate_VIF(X_train)
X_train_3 = X_train_2.drop('cylindernumber_eight', 1)



# Add a constant

X_train_3 = sm.add_constant(X_train_3)



# Create Model1

lr3 = sm.OLS(y_train, X_train_3).fit()

print(lr3.params)

print(lr3.summary())
# Model3 VIF

X_train.drop('cylindernumber_eight', axis =1, inplace = True)

calculate_VIF(X_train)
X_train_4 = X_train_3.drop('enginetype_dohcv', 1)



# Add a constant

X_train_4 = sm.add_constant(X_train_4)



# Create Model1

lr4 = sm.OLS(y_train, X_train_4).fit()

print(lr4.params)

print(lr4.summary())
# Model4 VIF

X_train.drop('enginetype_dohcv', axis =1, inplace = True)

calculate_VIF(X_train)
X_train_5 = X_train_4.drop('cylindernumber_four', 1)



# Add a constant

X_train_5 = sm.add_constant(X_train_5)



# Create Model1

lr5 = sm.OLS(y_train, X_train_5).fit()

print(lr5.params)

print(lr5.summary())
# Model5 VIF

X_train.drop('cylindernumber_four', axis =1, inplace = True)

calculate_VIF(X_train)
X_train_6 = X_train_5.drop('cylindernumber_twelve', 1)



# Add a constant

X_train_6 = sm.add_constant(X_train_6)



# Create Model1

lr6 = sm.OLS(y_train, X_train_6).fit()

print(lr6.params)

print(lr6.summary())
# Model6 VIF

X_train.drop('cylindernumber_twelve', axis =1, inplace = True)

calculate_VIF(X_train)
plt.figure(figsize = (10, 10))

sns.set(font_scale=1)

sns.heatmap(X_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
X_train_7 = X_train_6.drop('stroke', 1)



# Add a constant

X_train_7 = sm.add_constant(X_train_7)



# Create Model1

lr7 = sm.OLS(y_train, X_train_7).fit()

print(lr7.params)

print(lr7.summary())
# Model7 VIF

X_train.drop('stroke', axis =1, inplace = True)

calculate_VIF(X_train)
X_train_8 = X_train_7.drop('boreratio', 1)



# Add a constant

X_train_8 = sm.add_constant(X_train_8)



# Create Model1

lr8 = sm.OLS(y_train, X_train_8).fit()

print(lr8.params)

print(lr8.summary())
# Model8 VIF

X_train.drop('boreratio', axis =1, inplace = True)

calculate_VIF(X_train)
X_train_9 = X_train_8.drop('cylindernumber_three', 1)



# Add a constant

X_train_9 = sm.add_constant(X_train_9)



# Create Model1

lr9 = sm.OLS(y_train, X_train_9).fit()

print(lr9.params)

print(lr9.summary())
# Model9 VIF

X_train.drop('cylindernumber_three', axis =1, inplace = True)

calculate_VIF(X_train)
y_train_price = lr9.predict(X_train_9)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 10)

# Plot heading 

fig.suptitle('Error Terms', fontsize = 20)                  

plt.xlabel('Errors', fontsize = 18)    
X_train.columns
X_test_m9 =  X_test[['carlength', 'enginesize', 'CarName_audi', 'CarName_bmw','CarName_buick', 'CarName_porsche', 'cylindernumber_two']]

X_test_m9.head()

# Adding  constant variable to test dataframe

X_test_m9 = sm.add_constant(X_test_m9)



y_pred_m9 = lr9.predict(X_test_m9)
# Actual vs Predicted

c = [i for i in range(1,63,1)]



fig = plt.figure()

plt.figure(figsize = (6, 6))



plt.plot(c,y_test, color="red", linewidth=2, linestyle="-")     

plt.plot(c,y_pred_m9, color="green",  linewidth=2, linestyle="-")  



fig.suptitle('Actual vs Predicted', fontsize=15)              



plt.xlabel('Index', fontsize=15)                              

plt.ylabel('Car Price', fontsize=15)  
#Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.figure(figsize = (5,5))

plt.scatter(y_test,y_pred_m9)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)     
from sklearn.metrics import r2_score

r2_score(y_test, y_pred_m9)