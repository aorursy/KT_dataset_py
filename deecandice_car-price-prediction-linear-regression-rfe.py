import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
cars = pd.read_csv('../input/CarPrice_Assignment.csv')
cars.head(10)
cars.shape  #205 observations and 26 columns
#checking dtypes and null values of columns
cars.info()
#need to convert cylindernumber from object to numeric
cars.cylindernumber.unique()
cars['symboling'] = cars['symboling'].astype('category')
cars.describe() #statistical summary of numeric features
#checking number of columns of each data type for general EDA

cars.dtypes.value_counts()
#Splitting company name from CarName column
CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0])
cars.insert(3,"CompanyName",CompanyName)
cars.drop(['CarName'],axis=1,inplace=True)
cars.head()
cars.CompanyName.unique()
cars.CompanyName = cars.CompanyName.str.lower()

def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

cars.CompanyName.unique()
#Checking for duplicates
cars.loc[cars.duplicated()]
#plotting count of company names

plt.figure(figsize=(30, 8))
plt1=sns.countplot(x=cars.CompanyName, data=cars, order= cars.CompanyName.value_counts().index)
plt.title('Company Wise Popularity', size=14)
plt1.set_xlabel('Car company', fontsize=14)
plt1.set_ylabel('Frequency of Car Body', fontsize=14)
plt1.set_xticklabels(plt1.get_xticklabels(),rotation=360, size=14)
plt.show()
#Visualize the dependent variable

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(cars.price)

plt.subplot(1,2,2)
plt.title('Car Price Spread')
sns.boxplot(y=cars.price)

plt.show()
print(cars.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))
cars.columns
#plotting company wise average price of car

plt.figure(figsize=(30, 6))

df = pd.DataFrame(cars.groupby(['CompanyName'])['price'].mean().sort_values())
df=df.reset_index(drop=False)
plt1=sns.barplot(x="CompanyName", y="price", data=df)
plt1.set_title('Car Range vs Average Price', size=14)
plt1.set_xlabel('Car company', fontsize=14)
plt1.set_ylabel('Price', fontsize=14)
plt1.set_xticklabels(plt1.get_xticklabels(),rotation=360, size=14)
plt.show()
#Binning the Car Companies based on avg prices of each Company.
cars['price'] = cars['price'].astype('int')
temp = cars.copy()
table = temp.groupby(['CompanyName'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left',on='CompanyName')
bins = [0,10000,20000,40000]
cars_bin=['Budget','Medium','Highend']
cars['segment'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)
cars.head()
cars.drop(['CompanyName'], axis =1, inplace = True)
print(cars.select_dtypes(['object']).columns)
# Convert objects to categorical variables
obj_cats = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']

for colname in obj_cats:
    cars[colname] = cars[colname].astype('category') 
print(cars.select_dtypes(['object','category']).columns)
def plot_charts(var1, var2):
    plt.figure(figsize=(15, 10))   
    plt.subplot(2,2,1)
    plt.title('Histogram of '+ var1)
    sns.countplot(cars[var1], palette=("husl"))
    plt1.set(xlabel = '%var1', ylabel='Frequency of'+ '%s'%var1)
    
    plt.subplot(2,2,2)
    plt.title(var1+' vs Price')
    sns.boxplot(x=cars[var1], y=cars.price, palette=("husl"))
    
    plt.subplot(2,2,3)
    plt.title('Histogram of '+ var2)
    sns.countplot(cars[var2], palette=("husl"))
    plt1.set(xlabel = '%var2', ylabel='Frequency of'+ '%s'%var2)
    
    plt.subplot(2,2,4)
    plt.title(var2+' vs Price')
    sns.boxplot(x=cars[var2], y=cars.price, palette=("husl"))
    
    plt.show() 
plot_charts('segment', 'fueltype')
plot_charts('aspiration', 'doornumber')
plot_charts('drivewheel', 'carbody')
plot_charts('enginelocation', 'enginetype')
plot_charts('cylindernumber', 'fuelsystem')
plt.figure(figsize=(15, 5))    
plt.subplot(1,2,1)
plt.title('Histogram of '+ 'symboling')
sns.countplot(cars['symboling'], palette=("husl"))
plt1.set(xlabel = '%symboling', ylabel='Frequency of'+ '%s'%'symboling')
    
plt.subplot(1,2,2)
plt.title('symboling'+' vs Price')
sns.boxplot(x=cars['symboling'], y=cars.price, palette=("husl"))
plt.show()
# checking numeric columns

cars.select_dtypes(include=['float64','int64']).columns
def scatter(Col,figNum):
    plt.subplot(4,2,figNum)
    plt.scatter(cars[Col],cars['price'])
    plt.title(Col+' vs Price')
    plt.ylabel('Price')
    plt.xlabel(Col)

plt.figure(figsize=(10,20))

scatter('carlength', 1)
scatter('carwidth', 2)
scatter('carheight', 3)
scatter('curbweight', 4)


plt.tight_layout()
#dropping card_Id because it has all unique values

cars.drop(['carheight'], axis =1, inplace = True)
def pp(x,y,z):
    sns.pairplot(cars, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='reg')
    plt.show()

pp('enginesize', 'boreratio', 'stroke')
pp('compressionratio', 'horsepower', 'peakrpm')
pp('wheelbase', 'citympg', 'highwaympg')
#np.corrcoef(cars['carlength'], cars['carwidth'])[0, 1]
#Fuel economy
# Combined fuel economy is a weighted average of City and Highway MPG values that is calculated by 
#weighting the City value by 55% and the Highway value by 45%.
cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])
cars.drop(['citympg','highwaympg'], axis =1, inplace = True)
plt.figure(figsize=(8,6))

plt.title('Fuel economy vs Price')
sns.scatterplot(x=cars['fueleconomy'],y=cars['price'],hue=cars['drivewheel'])
plt.xlabel('Fuel Economy')
plt.ylabel('Price')

plt.show()
plt.tight_layout()
#dropping card_Id because it has all unique values

cars.drop(['car_ID'], axis =1, inplace = True)
#drop Compression Ratio, Stoke and Peakrpm show no obvious correlation b/w them and car price.

cars.drop(['compressionratio', 'stroke','peakrpm','doornumber'], axis =1, inplace = True)
cars.columns
#numeric variables

num_vars=cars.select_dtypes(include=['float64','int64']).columns
num_vars
plt.figure(figsize=(20,10))
sns.heatmap(cars[num_vars].corr(),cmap = 'YlGnBu',linewidth = 1,annot= True, annot_kws={"size": 9})
plt.title('Variable Correlation')
plt.show()
cars.info()
cars.head()
sns.pairplot(cars)
plt.show()
cars.select_dtypes(include=['category']).columns
# Defining the map function
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
# Applying the function to the cars dataframe
cars = dummies('symboling',cars)
cars = dummies('segment',cars)
cars = dummies('fueltype',cars)
cars = dummies('aspiration',cars)
cars = dummies('carbody',cars)
cars = dummies('drivewheel',cars)
cars = dummies('enginelocation',cars)
cars = dummies('enginetype',cars)
cars = dummies('cylindernumber',cars)
cars = dummies('fuelsystem',cars)
cars.head(10)
cars.shape #the dataframe has 205 observations and 45 features
# Importing train_test_split to train the data for model building
from sklearn.model_selection import train_test_split

np.random.seed(0)
cars_train, cars_test = train_test_split(cars, train_size = 0.7, test_size = 0.3, random_state = 100)
print(cars.select_dtypes(['int64','float64']).columns)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize',
       'boreratio', 'horsepower', 'price', 'fueleconomy']
cars_train[num_vars] = scaler.fit_transform(cars_train[num_vars])
cars_train.describe()
cars_train.head()
from sklearn.model_selection import train_test_split

#Dividing data into X and y variables
y_train = cars_train.pop('price')
X_train = cars_train
X_train.head()
X_train.describe()
#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()
def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X

#Calculating VIF for the model
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)
X_train_new = build_model(X_train_rfe,y_train)
#dropping constant to calculate VIF
X_train_new.drop('const', axis = 1, inplace=True)
#checking VIF
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["enginesize"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#dropping constant to calculate VIF
X_train_new.drop('const', axis = 1, inplace=True)
#checking VIF
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["fueleconomy"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#dropping constant to calculate VIF
X_train_new.drop('const', axis = 1, inplace=True)
#checking VIF
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["curbweight"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#dropping constant to calculate VIF
X_train_new.drop('const', axis = 1, inplace=True)
#checking VIF
checkVIF(X_train_new)
#drop 'wagon' because it has high P value
X_train_new = X_train_new.drop(["wagon"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#dropping constant to calculate VIF
X_train_new.drop('const', axis = 1, inplace=True)
#checking VIF
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["horsepower"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#checking VIF
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["hatchback"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#checking VIF
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["dohcv"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#checking VIF
checkVIF(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)   
X_train_new.columns
# Scaling the test data
cars_test[num_vars] = scaler.fit_transform(cars_test[num_vars])
#Dividing test data into X and y
y_test = cars_test.pop('price')
X_test = cars_test
#take a look at X_test and y_test
X_test.head()
y_test.head()
# Now let's use our model to make predictions.

# Now let's use our model to make predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)
#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   
print(lm.summary())