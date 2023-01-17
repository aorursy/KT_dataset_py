import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize


with open("../input/data.dat") as json_file:
    json_data = json.load(json_file)
    houses=pd.DataFrame(json_data)
#The loaded JSON data has been saved in a Python dictionary. 
#Using json_normalize, flattening the "data" dictionary into a table and saving it in a DataFrame "df".
df=json_normalize(json_data['houses'])
print(df.head(5))
df['date'].value_counts()
import matplotlib.pyplot as plt
%matplotlib inline

#plotting bar graph to audit "date"
df['date'].value_counts().plot(kind="line",figsize=(15,5))
plt.show()

# there is a different date formate for "23052014T000000" replacing it with "20140523T000000"
df['date'].replace(['23052014T000000'],['20140523T000000'],inplace=True) 

# there is a date value with 20140631T000000, where as June month does not contain 31st and hence considering it has Irregularities. changing date 20140631T000000 to 20140701T000000  
df['date'].replace(['20140631T000000'],['20140701T000000'],inplace=True)

#'address' is split according to "," and stored into 'address_list'
df['address_list'] = df['address'].str.split(', ')
#from 'address_list', col[0] is asigned to street,col[1] is asigned to city
#col[2] is asigned to statezip and col[3] is asigned to country
df['street'] = df['address_list'].apply(lambda col: col[0])
df['city']=df['address_list'].apply(lambda col: col[1])
df['statezip']=df['address_list'].apply(lambda col: col[2])
df['country']=df['address_list'].apply(lambda col: col[3])
df.head(3)
#Dropping 'address' column and also dummy list created as 'address_list'
df.drop('address',axis=1, inplace=True)
df.drop('address_list', axis=1, inplace = True)
df.head(3)
#"room" attribute values are extracted into 'bathrooms' and 'bedrooms',and "room" attribute is dropped once its values are extracted
df['bathrooms'] = df.rooms.str.extract('Number of bathrooms: (\d.\d+)', expand = True)
df['bedrooms'] = df.rooms.str.extract('Number of bedrooms: (\d+)', expand = True)
df.drop('rooms', axis=1, inplace = True)
df.head(3)

#splitting the values of 'area.sqft_living/sqft_lot' according to "="
df['area.sqft_living/sqft_lot_list'] = df['area.sqft_living/sqft_lot'].str.split('=')
# col[1] has the values of sqft_living and sqft_lot, hence storing it in 'area.sqft_living/sqft_lot_list_list1'
df['area.sqft_living/sqft_lot_list_list1'] = df['area.sqft_living/sqft_lot_list'].apply(lambda col: col[1])
df['area.sqft_living/sqft_lot_list_list2'] = df['area.sqft_living/sqft_lot_list_list1'].str.split('\ ')
df['sqft_living']=df['area.sqft_living/sqft_lot_list_list2'].apply(lambda col: col[0])
df['sqft_lot']=df['area.sqft_living/sqft_lot_list_list2'].apply(lambda col: col[1])
#dropping all dummy list used to store while splitting values of 'area.sqft_living/sqft_lot'
df.drop('area.sqft_living/sqft_lot_list_list1',axis=1, inplace=True)
df.drop('area.sqft_living/sqft_lot_list_list2', axis=1, inplace = True)
df.drop('area.sqft_living/sqft_lot_list', axis=1, inplace = True)
df.drop('area.sqft_living/sqft_lot', axis=1, inplace = True)

#renaming the columns from "area.sqft_above" to "sqft_above" and "area.sqft_basement" to "sqft_basement"
df.rename(index=str, columns={"area.sqft_above": "sqft_above", "area.sqft_basement": "sqft_basement"},inplace=True)
df.head(3)
#right stripping the value of 'sqft_living'
df['sqft_living'] = df['sqft_living'].map(lambda x: x.rstrip('\\'))
df.head(2)
#Trying find the Irregularities in sqft_living by applying, sqft_basement + sqft_above = sqft_living
df['sqft_temp'] =  df[['sqft_basement', 'sqft_above']].sum(axis=1)

#comparing the temp values with 'sqft_living'
df[df['sqft_temp'] != df['sqft_living']].index
#print(df.iloc[[4338]])

#df.ix[4338, 'sqft_living'] = df.ix[4338,['sqft_basement', 'sqft_above']].sum()

#print(df.iloc[[4338]])
#print(df.iloc[[4339]])

#df.ix[4339, 'sqft_living'] = df.ix[4339,['sqft_basement', 'sqft_above']].sum()

#print(df.iloc[[4339]])
#changing the datetime format.
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%dT%H:%m:%s')
#Changing the data types
df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].astype(float)
df[['sqft_lot','sqft_living']] = df[['sqft_lot','sqft_living']].astype(np.int64)
df['price'] = df['price'].apply(np.int64)
df.info()
#rearranging the dataframe according to the requirement
df = df[['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
         'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city', 'statezip', 
         'country']]
df.head()
 # 'O' for Objects
#df.describe(include=['O'])
df.describe(include="all")
df['city'].value_counts()
#City
# there is a Lexical error in city "sammamish" which is replaced with the average value of city named "Sammamish"
df['city'].replace(['sammamish'],['Sammamish'],inplace=True) 

# there is a Lexical error in city "Samamish" which is replaced with the average value of city named "Sammamish"
df['city'].replace(['Samamish'],['Sammamish'],inplace=True) 

# there is a Lexical error in city "Seaattle" which is replaced with the average value of city named "Seattle"
df['city'].replace(['Seaattle'],['Seattle'],inplace=True) 

# there is a Lexical error in city "Seatle" which is replaced with the average value of city named "Seattle"
df['city'].replace(['Seatle'],['Seattle'],inplace=True) 
 
# there is a Lexical error in city "seattle" which is replaced with the average value of city named "Seattle"
df['city'].replace(['seattle'],['Seattle'],inplace=True)

# there is a Lexical error in city "Issaguah" which is replaced with the average value of city named "Issaquah"
df['city'].replace(['Issaguah'],['Issaquah'],inplace=True)

# there is a Lexical error in city "Woodenville" which is replaced with the average value of city named "Woodinville"
df['city'].replace(['Woodenville'],['Woodinville'],inplace=True)
 
# there is a Lexical error in city "redmond" which is replaced with the average value of city named "Redmond"
df['city'].replace(['redmond'],['Redmond'],inplace=True)

# there is a Lexical error in city "Redmund" which is replaced with the average value of city named "Redmond"
df['city'].replace(['Redmund'],['Redmond'],inplace=True)

# there is a Lexical error in city "Redmund" which is replaced with the average value of city named "Redmond"
df['city'].replace(['Redmonde'],['Redmond'],inplace=True)

# there is a Lexical error in city "auburn" which is replaced with the average value of city named "Auburn"
df['city'].replace(['auburn'],['Auburn'],inplace=True)

# there is a Lexical error in city "Auburnt" which is replaced with the average value of city named "Auburn"
df['city'].replace(['Auburnt'],['Auburn'],inplace=True)

# there is a Lexical error in city "Sureline" which is replaced with the average value of city named "Shoreline "
df['city'].replace(['Sureline'],['Shoreline'],inplace=True)

# there is a Lexical error in city "Bellvue" which is replaced with the average value of city named "Bellevue "
df['city'].replace(['Bellvue'],['Bellevue'],inplace=True)

# there is a Lexical error in city "Belleview" which is replaced with the average value of city named "Bellevue "
df['city'].replace(['Belleview'],['Bellevue'],inplace=True)

# there is a Lexical error in city "Snogualmie" which is replaced with the average value of city named "Snoqualmie"
df['city'].replace(['Snogualmie'],['Snoqualmie'],inplace=True)

# there is a Lexical error in city "Coronation" which is replaced with the average value of city named "Carnation"
df['city'].replace(['Coronation'],['Carnation'],inplace=True)

# there is a Lexical error in city "Kirklund" which is replaced with the average value of city named "Kirkland"
df['city'].replace(['Kirklund'],['Kirkland'],inplace=True)

#The above changes can aslo be done as show in below code,
#df.city.replace({"sammamish":"Sammamish", "Samamish": "Sammamish", "Seaattle":"Seattle", "Seatle":"Seattle",
#"seattle":"Seattle", "Issaguah":"Issaquah"}, inplace=True) 
df.city.value_counts()
df['bathrooms'].value_counts()
#Bathroom
# there is a abnormal data value in "bathrooms", "1.70" which is replaced with the average value"1.75"
df['bathrooms'].replace([1.70],[1.75],inplace=True) 

# there is a lexical error in "bathrooms", "1.05" which is replaced with the value "1.50"
df['bathrooms'].replace([1.05],[1.50],inplace=True) 

# there is a abnormal data value in "bathrooms", "2.55" which is replaced with the nearest value of bathrooms named "2.50"
df['bathrooms'].replace([2.55],[2.50],inplace=True) 

# there is a abnormal data value in "bathrooms", "2.30" which is replaced with the average value of bathrooms named "2.25"
df['bathrooms'].replace([2.30],[2.25],inplace=True) 

# there is a lexical error in "bathrooms", "2.57" which is replaced with the average value of bathrooms named "2.75"
df['bathrooms'].replace([2.57],[2.75],inplace=True) 

df['bathrooms'].value_counts()
df[df.duplicated(keep=False)]
#dropping the row which are duplictaes
df.drop_duplicates(keep="first", inplace=True)
df.info()
df.isnull().sum()
# unique values of 'yr_renovated'
df['yr_renovated'].unique()
#creating dummy dataframe to use it for predicting the null values for yr_renovated by using mean
df_impute= df.copy()
df_impute.head(5)

#converting the yr_renovated column, from float64 to int64 and replacing all NaN to '0'
df_impute['yr_renovated'] = np.nan_to_num(df_impute['yr_renovated']).astype(np.int64)
#For linear regression we use sklearn (built in python library) and import linear regression from it.
from sklearn.linear_model import LinearRegression
#Initializing Linear Regression to a variable reg
reg = LinearRegression()
#we know that 'yr_renovated' are to be predicted , hence we set labels (output) as 'yr_renovated' column
labels = df_impute['yr_renovated']
#Converting dates to 1’s and 0’s so that it doesn’t influence our data much
#We use 0 for houses which are new that is built after 2014.
conv_dates = [1 if values == 2014 else 0 for values in df_impute.date]
df_impute['date'] = conv_dates
train1 = df_impute.drop(['city','street','country','statezip','price'],axis=1)
#We again import another dependency to split our data into train and test
from sklearn.cross_validation import train_test_split
#train data is set to 90% and 10% of the data to be my test data , and randomized the splitting of data by using random_state.
X_train, X_test, y_train, y_test = train_test_split(train1,labels,test_size = 0.10, random_state=2)
reg.fit(X_train,y_train)
reg.score(X_test,y_test)
df["yr_renovated"].fillna(df.groupby(["yr_built","condition"])["yr_renovated"].transform("mean"), inplace=True)
df.yr_renovated.describe()
df.isnull().sum()
df['yr_renovated'].fillna(0, inplace=True)
df_impute = df.copy()
#we know that 'yr_renovated' are to be predicted , hence we set labels (output) as 'yr_renovated' column
labels = df_impute['yr_renovated']
#Converting dates to 1’s and 0’s so that it doesn’t influence our data much
#We use 0 for houses which are new that is built after 2014.
conv_dates = [1 if values == 2014 else 0 for values in df_impute.date]
df_impute['date'] = conv_dates
train1 = df_impute.drop(['city','street','country','statezip','price'],axis=1)
reg.fit(X_train,y_train)
reg.score(X_test,y_test)
df_imptd1 = df.copy()
#we know that 'yr_renovated' are to be predicted , hence we set labels (output) as 'yr_renovated' column
labels = df_imptd1['yr_renovated']
#Converting dates to 1’s and 0’s so that it doesn’t influence our data much
#We use 0 for houses which are new that is built after 2014.
conv_dates = [1 if values == 2014 else 0 for values in df_imptd1.date]
df_imptd1['date'] = conv_dates
train1 = df_imptd1.drop(['city','street','country','statezip','price'],axis=1)
#train data is set to 90% and 10% of the data to be my test data , and randomized the splitting of data by using random_state.
X_train, X_test, y_train, y_test = train_test_split(train1,labels,test_size = 0.10, random_state=2)
reg.fit(X_train,y_train)
reg.score(X_test,y_test)
df['yr_renovated'] = df_impute['yr_renovated'].astype(np.int64)
df.price.value_counts()
#checking how many values are < 1
df_impute[df_impute['price'] < 1] 
df[["price","bedrooms","bathrooms","sqft_living","sqft_lot","sqft_above","yr_built","sqft_living","sqft_lot"]].describe()
#replacing all the 0.0 to NaN
df['price'] = df['price'].replace(0.0, np.nan)
#replacing all NaN to mean values of price 
df["price"].fillna(df_impute.groupby(["bedrooms","bathrooms","city","statezip"])["price"].transform("mean"), inplace=True)
df.price.describe()
df.isnull().sum()
#dropping all NaN values from price
df.dropna(subset=['price'],axis=0,inplace=True)
df_final = df.copy()
import matplotlib.pyplot as plt
%matplotlib inline

df.boxplot(figsize=(20,10))
bp = df.boxplot(column='price',figsize=(10,15))
# We can see a bunch of price above 0.5, then something around 2.5, the outliers are:
df[df['price'] > 2.0] 
# plotting baoxplot to check outliers price vs bedrooms
bp = df.boxplot(column='price', by = 'bedrooms',figsize=(15,10))
df[df['price'] > 2.0] 
#creating a dummy dataframe to check on price vs date of property sold.
df_dummy = df
#df_dummy['date'] = pd.to_datetime(df_dummy['date'])
# plot price vs date

#df_dummy['Year'] = df_dummy['date'].dt.year
#df_dummy.boxplot(column='price', by='Year', figsize=(10,10))

#df.set_index(['date',df.date.dt.year])['price'].unstack().boxplot()
df[(df['price'] > 0.5) & (df['price'] > 1.0) & (df['price'] > 2.5)].describe()
# sqft_lot
df.boxplot(column='sqft_lot', figsize=(10,10))
df[(df['sqft_lot'] > 400000) & (df['sqft_lot'] > 600000)] 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib import rcParams

%matplotlib inline 
%pylab inline 
sns.pairplot(data=df, x_vars=['bathrooms','bedrooms','sqft_living','sqft_lot','sqft_above','waterfront'], y_vars=["price"])
df.describe()
import statsmodels.api as sm
from statsmodels.formula.api import ols
m = ols('price ~ sqft_living',df).fit()
print (m.summary())
m = ols('price ~ sqft_living + bedrooms + view + condition',df).fit()
print (m.summary())
sns.jointplot('sqft_living','price', data=df, size=10, alpha=.5)
sns.jointplot('sqft_living','price', data=df, xlim=(500,3500), ylim=(100000,1000000), size=10, alpha=.5)
df["statezip"].nunique()

df['statezip'].value_counts()
df.groupby('statezip')['price'].mean()
zip_WA_98103 = df['statezip'] == "WA 98103"  # True if zip is 98103
sns.jointplot('sqft_living','price', data=df[zip_WA_98103], size=10, alpha=.5)
df.describe()
import seaborn as sns
import mpl_toolkits

#
df['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine

#PRIcE Vs SQFT_LIVING
plt.scatter(df.price,df.sqft_living)
plt.title("Price Vs Square feet")
plt.xlabel('Square feet')
plt.ylabel('Price')
plt.show()
#PRICE Vs BEDROOMS
plt.scatter(df.bedrooms,df.price)
plt.title("Bedroom and Price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine()
#Total sqft including basement vs price and waterfront vs price
plt.scatter((df['sqft_living']+df['sqft_basement']),df['price'])
plt.title("sqft_living and sqft_basement vs Price")
plt.xlabel("sqft_living and sqft_basement")
plt.ylabel("Price")
plt.show()
plt.scatter(df.waterfront,df.price)
plt.title("Waterfront Vs Price (0 = No Waterfront )")
plt.xlabel("waterfront")
plt.ylabel("Price")
plt.show()
plt.scatter(df.floors,df.price)
plt.title("floors Vs Price")
plt.xlabel("floors")
plt.ylabel("Price")
plt.show()
plt.scatter(df.condition,df.price)
plt.title("condition Vs Price")
plt.xlabel("condition")
plt.ylabel("Price")
plt.show()
#For linear regression we use sklearn (built in python library) and import linear regression from it.
from sklearn.linear_model import LinearRegression
#Initializing Linear Regression to a variable reg
reg = LinearRegression()
#we know that 'yr_renovated' are to be predicted , hence we set labels (output) as 'yr_renovated' column
labels = df['price']
#Converting dates to 1’s and 0’s so that it doesn’t influence our data much
#We use 0 for houses which are new that is built after 2014.
conv_dates = [1 if values == 2014 else 0 for values in df.date]
df['date'] = conv_dates
train1 = df.drop(['city','street','country','statezip','price'],axis=1)

#We again import another dependency to split our data into train and test
from sklearn.cross_validation import train_test_split
#train data is set to 90% and 10% of the data to be my test data , and randomized the splitting of data by using random_state.
X_train, X_test, y_train, y_test = train_test_split(train1,labels,test_size = 0.10, random_state=2)
map(pd.np.shape,[X_train, X_test, y_train, y_test])
reg.fit(X_train,y_train)
reg.score(X_test,y_test)

df_final.info()
df_final.head()
filename = 'output.csv'
df_final.to_csv(filename, encoding='utf-8', index=False)
