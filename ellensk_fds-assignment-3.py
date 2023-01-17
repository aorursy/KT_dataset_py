import pandas as pd
import numpy as np

df = pd.read_csv('../input/sales-data-2015/sales_data_2015.csv', index_col=0, low_memory=False)

df.head()
df.info()
len(df)
len(df.bldg_ctgy.unique())
unique_bldg_ctgy = df.bldg_ctgy.unique()

print(len(df.bldg_ctgy.unique()))

i = 0
bldg_ctgy_dict = dict([])
while  i < len(unique_bldg_ctgy):
    bldg_ctgy_dict[unique_bldg_ctgy[i]] = i
    i += 1
 
# print(bldg_ctgy_dict)
# print(unique_bldg_ctgy[38], " code number is:", bldg_ctgy_dict[unique_bldg_ctgy[38]])
df2 = df[['Sale_id','yr_built', 'tot_sqft', 'borough', 'block', 'lot','tot_unit', 'tax_cls_s', 'usable', 'bldg_ctgy', 'res_unit', 'com_unit', 'land_sqft', 'zip', 'price']].copy()
df2.head()
# for category in df2.bldg_ctgry:
#     df2.bldg_ctgry = bldg_ctgy_dict[category]
for key in bldg_ctgy_dict:
#     df2.bldg_ctgy = df2.bldg_ctgy.replace(['old value'],'new value')
    df2.bldg_ctgy = df2.bldg_ctgy.replace([key],bldg_ctgy_dict[key])
#     df2.bldg_ctgy[key] =  bldg_ctgy_dict[key]
    
df2.bldg_ctgy.astype('int32').dtypes
df2.info
# df2['sqft'] = np.zeros(shape = len(df2))
# df2.head()

df2['sqft'] = df['land_sqft'] + df['tot_sqft']
# Checking for NaN values
df2.isna().sum().sort_values(ascending=False)
# We also don't want any attibute to be 0, which is presumably a missing entry
# Year built
print("Entries with incorrect year: ", len(df2[df2.yr_built == 0]))
# Total square feet
print("Entries with incorrect tot_sqft: ", len(df2[df2.sqft == 0]))
# Borough
print("Entries with incorrect borough: ",len(df2[df2.borough == 0]))
# Block
print("Entries with incorrect block: ",len(df2[df2.block == 0]))
# Lot 
print("Entries with incorrect lot: ",len(df2[df2.lot == 0]))
# # Residual units  
# print("Entries with incorrect residual units: ",len(df2[df2.res_unit == 0]))
# # Commercials units
# print("Entries with incorrect commercial units: ",len(df2[df2.com_unit == 0]))
# Total units
print("Entries with incorrect total units: ",len(df2[df2.tot_unit == 0]))
# Tax class present
print("Entries with incorrect tax class on time of sale: ",len(df2[df2.tax_cls_s == 0]))
# Reusable with N label --> this mean been given away
print("Entries with N usable: ",len(df2[df2.usable == 'N']))
# Zip code
print("Entries with invalid zip code: ",len(df2[df2.zip == 0]))
# Price
print("Entries with incorrect price: ",len(df2[df2.price == 0]))
# Order of columns: yr_built, tot_sqft, borough, block, lot, res_unit, com_unit, tot_unit, price
# (df2.res_unit != 0) & (df2.com_unit != 0)
# sales_df = sales_df[sales_df['usable'].str.contains("Y")]
# df2 = df2[(df2.tot_sqft != 0) | (df2.land_sqft != 0)]
# df2 = df2[(df2.yr_built != 0) & ((df2.tot_sqft != 0) | (df2.land_sqft != 0)) & (df2.borough != 0) & (df2.block != 0) & (df2.lot != 0) & (df2.tot_unit != 0) & (df2.tax_cls_s != 0) & (df2.usable == "Y") & (df2.zip != 0) & (df2.price != 0)]
df2 = df2[(df2.yr_built != 0) & (df2.sqft != 0) & (df2.borough != 0) & (df2.block != 0) & (df2.lot != 0) & (df2.tot_unit != 0) & (df2.tax_cls_s != 0) & (df2.usable == "Y") & (df2.zip != 0) & (df2.price != 0)]
print("Original dataset length: ",len(df)) 
print("Cleaned dataset length: ", len(df2))
# df2['sqft'] = np.where(df2['tot_sqft'] == 0 , df2['land_sqft'], df2['tot_sqft'])
# df2.head()
#Check that the cleaning steps worked
print(len(df2[df2.yr_built == 0]))
print(len(df2[df2.tot_sqft == 0]))
print(len(df2[df2.price == 0]))
print(len(df2[df2.tot_unit == 0]))

# remove 
df2 = df2.drop(columns = ['usable'])

df2.info()
# import seaborn as sns
# import matplotlib.pyplot as plt

# #Set the size of the plot
# plt.figure(figsize=(15,6))

# # Plot the data and configure the settings
# sns.boxplot(x='price', data=df2)
# plt.ticklabel_format(style='plain', axis='x')
# plt.title('Boxplot of sale price in USD')
# plt.show()

# #Set the size of the plot
# plt.figure(figsize=(15,6))

# # Plot the data and configure the settings
# sns.distplot(df2['price'])
# plt.title('Histogram of SALE PRICE in USD')
# plt.ylabel('Normed Frequency')
# plt.show()

# #Set the size of the plot
# plt.figure(figsize=(15,6))

# #Get the data and format it
# x = df2[['price']].sort_values(by='price').reset_index()
# x['Property Proportion'] = 1
# x['Property Proportion'] = x['Property Proportion'].cumsum()
# x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# # Plot the data and configure the settings
# plt.plot(x['Property Proportion'],x['price'], linestyle='None', marker='o')
# plt.title('Cumulative Distribution of Properties according to Price')
# plt.xlabel('Percentage of Properties in ascending order of Price')
# plt.ylabel('Sale Price')
# plt.ticklabel_format(style='plain', axis='y')
# plt.show()
## Remove observations that fall outside those caps
print(len(df2))
print(len(df2[df2.price < 100000]))
print(len(df2[df2.price > 1200000]))
df2 = df2[(df2['price'] > 100000) & (df2['price'] < 50000000)]
print(len(df2))
import seaborn as sns
import matplotlib.pyplot as plt

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x='price', data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of sale price in USD')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['price'])
plt.title('Histogram of SALE PRICE in USD')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['price']].sort_values(by='price').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['price'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to Price')
plt.xlabel('Percentage of Properties in ascending order of Price')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x=df2.sqft, data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of square feet')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['sqft'])
plt.title('Histogram of square feet')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['sqft']].sort_values(by='sqft').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['sqft'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to Square Feet')
plt.xlabel('Percentage of Properties in ascending order of Square Feet')
plt.ylabel('Square Feet')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
# Remove observations that fall outside those caps
print(len(df2))
print(len(df2[df2.sqft < 1000]))
print(len(df2[df2.sqft > 8000]))
df2 = df2[(df2['sqft'] > 1000) & (df2['sqft'] < 8000)]
# df2 = df2[(df2['sqft'] < 8500)]
print(len(df2))
# Plot the data and configure the settings
sns.boxplot(x=df2.sqft, data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of square feet')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['sqft'])
plt.title('Histogram of square feet')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['sqft']].sort_values(by='sqft').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['sqft'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to Square Feet')
plt.xlabel('Percentage of Properties in ascending order of Square Feet')
plt.ylabel('Square Feet')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x=df2.yr_built, data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of yr_built')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['yr_built'])
plt.title('Histogram of yr_built')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['yr_built']].sort_values(by='yr_built').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['yr_built'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to yr_built')
plt.xlabel('Percentage of Properties in ascending order of yr_built')
plt.ylabel('Square Feet')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
# Remove observations that fall outside those caps
print(len(df2))
print(len(df2[df2.yr_built < 1850]))
df2 = df2[(df2['yr_built'] > 1850)]
# df2 = df2[(df2['sqft'] < 8500)]
print(len(df2))
#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x=df2.yr_built, data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of yr_built')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['yr_built'])
plt.title('Histogram of yr_built')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['yr_built']].sort_values(by='yr_built').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['yr_built'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to yr_built')
plt.xlabel('Percentage of Properties in ascending order of yr_built')
plt.ylabel('Square Feet')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x=df2.tot_unit, data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of tot_unit')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['tot_unit'])
plt.title('Histogram of tot_unit')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['tot_unit']].sort_values(by='tot_unit').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['tot_unit'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to tot_unit')
plt.xlabel('Percentage of Properties in ascending order of tot_unit')
plt.ylabel('Square Feet')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
# Remove observations that fall outside those caps
print(len(df2))
print(len(df2[df2.tot_unit > 4]))
df2 = df2[(df2['tot_unit'] < 4)]
# df2 = df2[(df2['sqft'] < 8500)]
print(len(df2))
#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x=df2.tot_unit, data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of tot_unit')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['tot_unit'])
plt.title('Histogram of tot_unit')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['tot_unit']].sort_values(by='tot_unit').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['tot_unit'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to tot_unit')
plt.xlabel('Percentage of Properties in ascending order of tot_unit')
plt.ylabel('Square Feet')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x=df2.lot, data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of lot')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['lot'])
plt.title('Histogram of lot')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['lot']].sort_values(by='lot').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['lot'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to lot')
plt.xlabel('Percentage of Properties in ascending order of lot')
plt.ylabel('lot')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
# Remove observations that fall outside those caps
print(len(df2))
print(len(df2[df2.lot > 200]))
df2 = df2[(df2['lot'] < 200)]
# df2 = df2[(df2['sqft'] < 8500)]
print(len(df2))
#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.boxplot(x=df2.lot, data=df2)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of lot')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

# Plot the data and configure the settings
sns.distplot(df2['lot'])
plt.title('Histogram of lot')
plt.ylabel('Normed Frequency')
plt.show()

#Set the size of the plot
plt.figure(figsize=(15,6))

#Get the data and format it
x = df2[['lot']].sort_values(by='lot').reset_index()
x['Property Proportion'] = 1
x['Property Proportion'] = x['Property Proportion'].cumsum()
x['Property Proportion'] = 100* x['Property Proportion'] / len(x['Property Proportion'])

# Plot the data and configure the settings
plt.plot(x['Property Proportion'],x['lot'], linestyle='None', marker='o')
plt.title('Cumulative Distribution of Properties according to lot')
plt.xlabel('Percentage of Properties in ascending order of lot')
plt.ylabel('lot')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
# importing one hot encoder from sklearn 
# There are changes in OneHotEncoder class 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
import pandas as pd
   
# Creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# Passing building category column (label encoded values of bldg_ctgry)
enc_df = pd.DataFrame(enc.fit_transform(df2[['borough']]).toarray())
end = len(df2) -1
# We reset the indexes of the different data frames and then merge them so that the one-hot encoding categories will be part of 
# our cleaned dataset as well
df2.reset_index(drop=True)
enc_df.reset_index(drop = True)

df2 = pd.concat([df2.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)
# df2.to_csv (r'./\cleaned_df.csv', index = False, header=True)
X = df2
Y = df2.price
Y = np.array(Y).reshape(-1)

import matplotlib.pyplot as plt
%matplotlib inline

f,ax = plt.subplots(1,8)
f.set_figwidth(15,10)
f.tight_layout(pad=2.0)
# (1) Prices - Year built scatterplot
ax[0].scatter(Y, X.yr_built)
ax[0].set_xlabel("Prices")
ax[0].set_ylabel("Year built")
ax[0].set_title("Prices vs Year built")
#ax[0].set_xlim(0,5e8)
#ax[0].set_ylim(1800,2050)
# (2) Prices - Total sqft scatterplot
# ax[1].scatter(Y, X.tot_sqft)
# ax[1].set_xlabel("Prices")
# ax[1].set_ylabel("Total sqft")
# ax[1].set_title("Prices vs Total sqft")
#ax[1].set_xlim(0,5e8)
#ax[1].set_ylim(0,1e6)
# (2) Prices - Borough scatterplot
ax[1].scatter(Y, X.borough)
ax[1].set_xlabel("Prices")
ax[1].set_ylabel("Borough")
ax[1].set_title("Prices vs Borough")
# (3) Prices - Block scatterplot
ax[2].scatter(Y, X.block)
ax[2].set_xlabel("Prices")
ax[2].set_ylabel("Block")
ax[2].set_title("Prices vs Block")
# (4) Prices - Lot scatterplot
ax[3].scatter(Y, X.lot)
ax[3].set_xlabel("Prices")
ax[3].set_ylabel("Lot")
ax[3].set_title("Prices vs Lot")
# (5) Prices - Total Units scatterplot
ax[4].scatter(Y, X.tot_unit)
ax[4].set_xlabel("Prices")
ax[4].set_ylabel("Total Units")
ax[4].set_title("Prices vs Total Units")
# (6) Prices - Tax class at time of sale scatterplot
ax[5].scatter(Y, X.tax_cls_s)
ax[5].set_xlabel("Prices")
ax[5].set_ylabel("Tax class at time of sale")
ax[5].set_title("Prices vs Tax class at time of sale")
# (7) Prices - Building category scatterplot
ax[6].scatter(Y, X.zip)
ax[6].set_xlabel("Prices")
ax[7].set_ylabel("Zip")
ax[6].set_title("Prices vs Zip")
# (8) Prices - Building category scatterplot
ax[7].scatter(Y, X.sqft)
ax[7].set_xlabel("Prices")
ax[7].set_ylabel("Sqft")
ax[7].set_title("Prices vs Sqft")

# f2,ax2 = plt.subplots()
# # f2.set_figwidth(15,10)
# # (8) Prices - Building category scatterplot
# ax2.scatter(Y, X.bldg_ctgy)
# ax2.set_xlabel("Prices")
# ax2.set_ylabel("Building category")
# ax2.set_title("Prices vs Building category")
print('Correlation matrix for Price and Year built: \n \n', np.corrcoef(Y, X.yr_built))
print('\n \n Correlation matrix for Price and Total sqft: \n \n', np.corrcoef(Y, X.tot_sqft))
print('\n \n Correlation matrix for Price and Borough: \n \n', np.corrcoef(Y, X.borough))
print('\n \n Correlation matrix for Price and Block : \n \n', np.corrcoef(Y, X.block))
print('\n \n Correlation matrix for Price and Lot: \n \n', np.corrcoef(Y, X.lot))
print('\n \n Correlation matrix for Price and Total Units: \n \n', np.corrcoef(Y, X.tot_unit))
print('\n \n Correlation matrix for Price and Tax class at time of sale: \n \n', np.corrcoef(Y, X.tax_cls_s))
print('\n \n Correlation matrix for Price and Zip: \n \n', np.corrcoef(Y, X.zip))
print('\n \n Correlation matrix for Price and Sqft: \n \n', np.corrcoef(Y, X.sqft))
# Remove column bldg_ctgry because because we already have the one-hot encoding for that
X = X.drop(columns = ['bldg_ctgy'])
# We will also remove the columns lot and block as they describe very similar things like borough but they have a lower correlation score
# X = X.drop(columns = ['lot','block'])
X.reset_index(drop=True)


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

# For the VIF score, we will test only the factors that aren't part of the one-hot encoding categories
# Find index of last column that's not a one-hot encoding category
sqft_index = X.columns.get_loc('sqft')
print(sqft_index)
# X2 = df2.columns[:sqft_index+1] incorrect
X2 = X.iloc[:,:sqft_index+1]

X2 = X2.drop(columns = ['Sale_id','price','res_unit','com_unit','tot_sqft', 'land_sqft','borough', 'lot', 'zip', 'yr_built'])
vif = calc_vif(X2)
print(vif)

X = X.drop(columns = ['borough','lot', 'tot_sqft','tot_unit', 'land_sqft','zip','yr_built']) 
X.to_csv(r'./cleaned_df.csv', index = False, header=True)

X = X.drop(columns = ['Sale_id','price']) 
# # Assigning X to age and size and Y to price
# X = df2.drop(columns=['price'])
# Y = df2.price
# Y = np.array(Y).reshape(-1)
# print(X.shape,Y.shape)
X
# Splitting the data with 80:20 ratio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor 

# train, validate, test = np.split(sales_df2.sample(frac=1, random_state=42), [int(.8*len(sales_df2)), int(.9*len(sales_df2))])

X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, Y, test_size = .10, random_state = 40)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size = .10, random_state = 40)

# Create the linear regression model
lr = LinearRegression()

# Fitting the model
lr.fit(X_train, y_train)

# R^2 scores
print('Linear Regression')
print('R^2 scores')
print('Train:', lr.score(X_train, y_train))
print('Validation:', lr.score(X_validation, y_validation))

# MAE
# evaluate the model
yhat = lr.predict(X_validation)
# evaluate predictions
mae = mean_absolute_error(y_validation, yhat)
print('\nMAE: %.3f' % mae)

# Print the 5-fold cross-validation scores
lr.fit(X_train_validation, y_train_validation)

cv_scores = cross_val_score(lr, X_train_validation, y_train_validation, cv=5)
print('\n')
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

lasso = Lasso(alpha=1.0, normalize=True, max_iter=1e5)
lasso.fit(X_train_validation, y_train_validation)

# R^2 scores
print('\nLasso Regression')
print('R^2 scores')
print('Train:', lasso.score(X_train, y_train))
print('Validation:', lasso.score(X_validation, y_validation))

# MAE
# evaluate the model
yhat = lasso.predict(X_validation)
# evaluate predictions
mae = mean_absolute_error(y_validation, yhat)
print('\nMAE: %.3f' % mae)

# Print the 5-fold cross-validation scores
lasso.fit(X_train_validation, y_train_validation)

cv_scores = cross_val_score(lasso, X_train_validation, y_train_validation, cv=5)
print('\n')
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
regressor.fit(X_train_validation, y_train_validation)   

# R^2 scores
print('\nRandom Forest Regression')
print('R^2 scores')
print('Train:', regressor.score(X_train, y_train))
print('Validation:', regressor.score(X_validation, y_validation))

# MAE
# evaluate the model
yhat = regressor.predict(X_validation)
# evaluate predictions
mae = mean_absolute_error(y_validation, yhat)
print('\nMAE: %.3f' % mae)

# Print the 5-fold cross-validation scores
regressor.fit(X_train_validation, y_train_validation)

cv_scores = cross_val_score(regressor, X_train_validation, y_train_validation, cv=5)
print(cv_scores)

print("\nAverage 5-Fold CV Score: {}".format(np.mean(cv_scores)))
print('Test Set Validation\n')

# R^2 scores
print('Linear Regression')
print('R^2 scores')
print('Train:', lr.score(X_test, y_test))
print('Validation:', lr.score(X_test, y_test))

# MAE
# evaluate the model
yhat = lr.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('\nMAE: %.3f' % mae)

# R^2 scores
print('\nLasso Regression')
print('R^2 scores')
print('Train:', lasso.score(X_test, y_test))
print('Validation:', lasso.score(X_test, y_test))

# MAE
# evaluate the model
yhat = lasso.predict(X_validation)
# evaluate predictions
mae = mean_absolute_error(y_validation, yhat)
print('\nMAE: %.3f' % mae)

# Print the 5-fold cross-validation scores
lasso.fit(X_train_validation, y_train_validation)


# R^2 scores
print('\nRandom Forest Regression')
print('R^2 scores')
print('Train:', regressor.score(X_test, y_test))
print('Validation:', regressor.score(X_test, y_test))

# MAE
# evaluate the model
yhat = regressor.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('\nMAE: %.3f' % mae)
# # Splitting the data with 80:20 ratio
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import cross_val_score

# # train, validate, test = np.split(sales_df2.sample(frac=1, random_state=42), [int(.8*len(sales_df2)), int(.9*len(sales_df2))])

# X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, Y, test_size = .20, random_state = 40)
# X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size = .10, random_state = 40)

# # Create the linear regression model
# lr = LinearRegression()

# # Fitting the model
# lr.fit(X_train, y_train)

# # R^2 scores
# print('Train:', lr.score(X_train, y_train))
# print('Validation:', lr.score(X_validation, y_validation))

# # MAE
# # evaluate the model
# yhat = lr.predict(X_validation)
# # evaluate predictions
# mae = mean_absolute_error(y_validation, yhat)
# print('\nMAE: %.3f' % mae)

# # Print the 5-fold cross-validation scores
# lr.fit(X_train_validation, y_train_validation)

# print("\nCross Validation:") 
# cv_scores = cross_val_score(lr, X_train_validation, y_train_validation, cv=5)
# print(cv_scores)

# print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# print('\nTest Set Scores')
# # R^2 scores
# print('Test:', lr.score(X_test, y_test))

# # MAE
# # evaluate the model
# yhat = lr.predict(X_test)
# # evaluate predictions
# mae = mean_absolute_error(y_test, yhat)
# print('\nMAE: %.3f' % mae)

# from sklearn.ensemble import RandomForestRegressor 

# # X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, Y, test_size = .10, random_state = 40)
# # train, validate, test = np.split(df_eleni_final.sample(frac=1, random_state=66),[int(.8*len(df_eleni_final)), int(.9*len(df_eleni_final))])

# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)  

# # Fitting the model
# regressor.fit(X_train, y_train)

# # R^2 scores
# print('Train:', regressor.score(X_train, y_train))
# print('Validation:', regressor.score(X_validation, y_validation))
# # MAE
# # evaluate the model
# yhat = regressor.predict(X_validation)
# # evaluate predictions
# mae = mean_absolute_error(y_validation, yhat)
# print('\nMAE: %.3f' % mae)

# regressor.fit(X_train_validation, y_train_validation)  

# cv_scores = cross_val_score(regressor, X_train_validation, y_train_validation, cv=5)
# print(cv_scores)

# print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# print('\nTest Set Scores')
# # R^2 scores
# print('Test:', regressor.score(X_test, y_test))

# # MAE
# # evaluate the model
# yhat = regressor.predict(X_test)
# # evaluate predictions
# mae = mean_absolute_error(y_test, yhat)
# print('\nMAE: %.3f' % mae)
# from sklearn.ensemble import IsolationForest

# # identify outliers in the training dataset
# iso = IsolationForest(contamination=0.3)
# yhat = iso.fit_predict(X_train)
# from sklearn.neighbors import LocalOutlierFactor

# # identify outliers in the training dataset
# lof = LocalOutlierFactor()
# yhat = lof.fit_predict(X_train)
# print(X_train.shape, y_train.shape)
# # select all rows that are not outliers
# mask = yhat != -1
# xtest = X_train.to_numpy()
# X_train, y_train = xtest[mask, :], y_train[mask]
# # summarize the shape of the updated training dataset
# print(X_train.shape, y_train.shape)