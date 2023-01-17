import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 



from scipy.stats import pointbiserialr



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
data_dekho = pd.read_csv("../input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv")

data_dekho.shape
data_dekho.tail() 
data_dekho.info()
data_dekho["name"]
data_dekho.drop(['name'], axis=1, inplace=True)

data_dekho.head() 
data_dekho['transmission'].unique()
transmission = data_dekho['transmission'] 

transmission_clean = [0 if i == "Manual"  else 1  for i in data_dekho['transmission']]



transmission_clean = np.array(transmission_clean)

transmission_clean.shape
data_dekho.drop(['transmission'], axis=1, inplace=True)

data_dekho['transmission'] = transmission_clean
data_dekho['owner'].unique()
owner_unique_names = data_dekho['owner'].unique()

owner_unique_names = pd.Series(index=owner_unique_names, data=[0,1,2,3,4])

dict(owner_unique_names)
data_dekho['owner'].replace(dict(owner_unique_names), inplace=True)

data_dekho.head() 
fuel_unique_names = data_dekho['fuel'].unique()

fuel_unique_names = pd.Series(index=fuel_unique_names, data=[0,1,2,3,4])

dict(fuel_unique_names)

data_dekho['fuel'].replace(dict(fuel_unique_names), inplace=True)

data_dekho.head() 
seller_type_unique_names = data_dekho['seller_type'].unique()

seller_type_unique_names = pd.Series(index=seller_type_unique_names, data=[0,1,2])

dict(seller_type_unique_names)

data_dekho['seller_type'].replace(dict(seller_type_unique_names), inplace=True)

data_dekho.head() 
data = data_dekho.drop('selling_price', axis=1) 

data['price'] = data_dekho['selling_price'] 
data.head() 
data.info() 
data.count() 
pd.isnull(data).any()
data.head() 
plt.figure(figsize=(10,6))

plt.hist(data['price'], bins=50,ec='black', color='#2196f3') 

plt.xlabel('prices', fontsize=14)

plt.ylabel('Nr of Prices', fontsize=14)

plt.title("The Distribution of The Target Variable", fontsize=14)

plt.show() 
data['price'].skew()
data["price"].min()
data["price"].max()
data['price'].mean() 
plt.figure(figsize=(10,7))



sns.boxplot(x=data['price'])

plt.xlabel("price",fontsize=14)

plt.show()
Q1 = data['price'].quantile(0.25)

Q3 = data['price'].quantile(0.75)

IQR = Q3 - Q1

print(IQR)
res = (data['price'] < (Q1 - 1.5 * IQR)) | (data['price'] > (Q3 + 1.5 * IQR))

print(res[res.values == True].count(), "outliers")
log_prices = np.log(data['price'])

data_log_prices = data.drop(['price'], axis=1)

data_log_prices['price'] = log_prices
plt.figure(figsize=(10,6))

plt.hist(data_log_prices['price'], bins=50,ec='black', color='#2196f3') 

plt.xlabel('prices', fontsize=14)

plt.ylabel('Nr of Prices', fontsize=14)

plt.title(f"The Distribution of The Target Variable skew:{str(round(data_log_prices['price'].skew(),3))}", fontsize=14)

plt.show() 
data.iloc[12,:] # example of an outlier
out_idx = res[res.values == True].index

data_rem_out = data.drop(index=out_idx)

data_rem_out.shape
plt.figure(figsize=(10,6))

plt.hist(data_rem_out['price'], bins=50,ec='black', color='#2196f3') 

plt.xlabel('prices', fontsize=14)

plt.ylabel('Nr of Prices', fontsize=14)

plt.title(f"The Distribution of The Target Variable skew:{str(round(data_rem_out['price'].skew(),3))}", fontsize=14)

plt.show() 
data = data_log_prices
plt.figure(figsize=(10,6))

freq = data['year'].value_counts()

plt.bar(x=freq.index, height=freq.values)

plt.xlabel('years', fontsize=14)

plt.ylabel('Nr of Years', fontsize=14)

plt.title(f"The Distribution of The Year Variable", fontsize=14)

plt.show()
data['year'].min() 
data['year'].max() 
data.loc[data['year'] < 2000, 'price'] .mean() 
data.loc[data['year'] > 2000, 'price'] .mean() 
plt.figure(figsize=(10,6))

plt.hist(data['km_driven'], bins=50,ec='black', color='#2196f3') 

plt.xlabel('prices', fontsize=14)

plt.ylabel('Nr of Prices', fontsize=14)

plt.title("The Distribution of The Target Variable", fontsize=14)

plt.show() 
data['km_driven'].mean() 
data['km_driven'].min() 
data['km_driven'].max() 
plt.bar(x=fuel_unique_names.index, height=data['fuel'].value_counts())

plt.show() 
round(data.describe())
data.corr() # Pearson Correlation Coefficients
mask = np.zeros_like(data.corr())

triangle_indices = np.triu_indices_from(mask)

mask[triangle_indices] = True

mask
plt.figure(figsize=(16,10))

sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={"size": 14})

sns.set_style('white')

plt.xticks(fontsize=11)

plt.yticks(fontsize=11)

plt.show()
cor = round(data['price'].corr(data['km_driven']),3) 

sns.lmplot(x="km_driven", y="price", data=data, height=6, 

           line_kws={'color': 'cyan'}, scatter_kws={'color': 'purple', 'alpha': 0.7})

plt.title(f'price vs km corr:{cor}', fontsize=14)

plt.show() 
plt.figure(figsize=(10,6))

sns.distplot(data['km_driven'])

plt.title(f"The Histogram of the km_driven skew:{round(data['km_driven'].skew(),3)}")

plt.show()
data['price'].corr(np.log(data['km_driven']))
km_log = np.log(data['km_driven'])

data['km_driven'] = km_log

data.head() 
cor = round(data['price'].corr(data['km_driven']),3) 

sns.lmplot(x="km_driven", y="price", data=data, height=6, 

           line_kws={'color': 'cyan'}, scatter_kws={'color': 'purple', 'alpha': 0.7})

plt.title(f'price vs km corr:{cor}', fontsize=14)

plt.show() 
plt.figure(figsize=(10, 6), dpi=300)

plt.scatter(data['year'], data['km_driven'], color='indigo', s=80, alpha=0.7)

plt.title(f"Year vs Km_Driven Corr: {round(data['year'].corr(data['km_driven']),3)}")

plt.xlabel('Year', fontsize=14)

plt.ylabel('Km_Driven', fontsize=14)          

plt.show() 
%%time



sns.pairplot(data)

plt.show()
log_target = data['price']

features = data.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, log_target, test_size=0.2)
# Model using log price and log km_driven 

regr = LinearRegression() 

model_log_price_km = regr.fit(X_train, y_train)



log_price_log_km = regr.score(X_train, y_train)





print('Intercept is', round(regr.intercept_,3))

print('R-squared for training set is', regr.score(X_train, y_train))

print('R-squared for testing set is', regr.score(X_test, y_test))



pd.DataFrame(regr.coef_, columns=['coef'], index=features.columns)
target = np.e**data['price']

features = data.drop(['price'], axis=1)

features['km_driven'] = np.e**data['km_driven']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
# Model using normal price and normal km_driven 

regr = LinearRegression() 

regr.fit(X_train, y_train)



norm_price_norm_km = regr.score(X_train, y_train)



print('Intercept is', round(regr.intercept_,3))

print('R-squared for training set is', regr.score(X_train, y_train))

print('R-squared for testing set is', regr.score(X_test, y_test))



pd.DataFrame(regr.coef_, columns=['coef'], index=features.columns)
# Model using log price and norm km_driven 



target = data['price']

features = data.drop(['price'], axis=1)

features['km_driven'] = np.e**data['km_driven']



X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)





regr = LinearRegression() 

regr.fit(X_train, y_train)



log_price_norm_km = regr.score(X_train, y_train)



print('Intercept is', round(regr.intercept_,3))

print('R-squared for training set is', regr.score(X_train, y_train))

print('R-squared for testing set is', regr.score(X_test, y_test))



pd.DataFrame(regr.coef_, columns=['coef'], index=features.columns)
arr = np.asanyarray([log_price_log_km, log_price_norm_km, norm_price_norm_km])



pd.DataFrame(arr, columns=['R-Squared'], index=['LOG PRICE AND LOG KM', 'LOG PRICE AND NORMAL KM', 'NORMAL PRICE AND NORMAL KM'])
X_incl_const = sm.add_constant(X_train)

model = sm.OLS(y_train, X_incl_const) 

results = model.fit() 

round(results.pvalues, 3)
log_km_x_train = X_train

log_km_x_train['km_driven'] = np.log(X_train['km_driven'])

X_incl_const_log_km = sm.add_constant(log_km_x_train)

model = sm.OLS(y_train, X_incl_const_log_km) 

results = model.fit() 

round(results.pvalues, 3)
variance_inflation_factor(exog=np.asanyarray(X_incl_const_log_km), exog_idx=1)
vifs = [variance_inflation_factor(exog=np.asanyarray(X_incl_const_log_km), exog_idx=i) 

        for i in range(len(X_incl_const.columns))]

pd.DataFrame(np.asanyarray(vifs).reshape(1,7),  columns=X_incl_const.columns, index=['VIF'])
# Model using log price and norm km_driven 



X_incl_const = sm.add_constant(X_train)

model = sm.OLS(y_train, X_incl_const)

results = model.fit() 



print("R-squared is", results.rsquared)

print("BIC is", results.bic)
# Model using log price without km_driven 



X_incl_const = sm.add_constant(X_train.drop(['km_driven'], axis=1))

model = sm.OLS(y_train, X_incl_const)

results = model.fit() 



print("R-squared is", results.rsquared)

print("BIC is", results.bic)
# Predicted log prices vs Actual Log prices 



regr = LinearRegression().fit(X_train, y_train) 



predicted_values = pd.Series(regr.predict(X_train))

corr = np.round(y_train.corr(predicted_values), 3)



plt.figure(figsize=(10,6))



plt.scatter(x=predicted_values, y=y_train)

plt.plot(y_train, y_train, c='red')

plt.title(f"Predicted log prices vs Actual Log prices {corr}", fontsize=14)

plt.xlabel('Predicted Price',fontsize=14)

plt.ylabel('Actual Price', fontsize=14) 





# residual vs predicted values 

plt.figure(figsize=(10,6))

y = np.asanyarray(y_train)

y_hat = np.asanyarray(predicted_values)

resi = y - y_hat



plt.scatter(x=predicted_values, y=resi, c="skyblue",alpha=0.7)

plt.xlabel('Residual', fontsize=14)

plt.ylabel('Predicted Values', fontsize=14)

plt.title("Residual vs Predicted Values", fontsize=14)



plt.figure(figsize=(10,6))

sns.distplot(resi)

plt.title(f'The Distribution of the Residuals Skew:{round(pd.Series(resi).skew(), 2)}', fontsize=14)



plt.show() 
print("R-squared is", regr.score(X_train, y_train))