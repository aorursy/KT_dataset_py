'''
Dataset : Craigslist Used Car Dataset

Team Members :    1] PES2201800116 : Aniketh D Urs 
                  2] PES2201800656 : Manoj Mahesh Patil
                  3] PES2201800480 : Purushotham S
                  4] PES2201800646 : Mahammad Thufail

Problem Statement : To predict the price of the used car using Prediction Models that we create.

Introduction : When we want to sell used cars, one of the biggest problems is deciding reasonable selling prices for the cars. 
An effective way to solve this problem is to use a machine-learning model that can predict car prices.
'''


'''Exploratory Data Analysis and Data Cleaning'''
#First we import Pandas 
import pandas as pd
#Next we import the dataset 
df_original = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')
#Now , let's look at the contents of our dataset 
import numpy as np

df = df_original.copy()

df.iloc[np.r_[0:3, -3:0]]
'''Cleaning of Dataset'''
irrelevant_cols = ['id', 'url', 'region_url', 'vin', 'image_url', \
                   'description', 'county']

df = df.drop(columns=irrelevant_cols)
# The "Price" column is our target column , so let's move it to the last of the dataset for convinience.
col_list = ['price']

rearranged_cols = np.hstack((df.columns.difference(col_list, sort=False), col_list))

df = df.reindex(columns=rearranged_cols)
# Now we will make sure that the string values in the dataset are in lower case and there should be no spaces in between
for column in df.columns[1:]:
    if df[column].dtype == 'object':
        df[column] = df[column].str.lower().str.strip()
df.info()
# As you can see above that some of the columns have been removed and the price column has been moved to the end of the dataset
# Now let us visualize these NULL values using graphs 
import seaborn as sns
import matplotlib.pyplot as plt

heat_map = sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='Blues')
_ = heat_map.set_xticklabels(heat_map.get_xticklabels(), color='#6eafd7')
'''
Missing values can lead to errors in machine-learning models. To avoid these errors, we can use the following workarounds:
         1] Remove selected rows that contain missing values.
         2] Replace missing values with estimates by using scikit-learn imputers.

As we want our algorithms to be accurate, we must retain as much of the car data as possible.
This means that we will have to impute many of the missing values.
At the same time, we want to minimize instances of incorrect data.
So, we will delete selected rows as well.
'''
'''What are Extra Tree Regressor and Bayesian Ridge ?

Extratreesregressor divides a target dataset into smaller subsets. 
Then, it uses multiple decision trees, or extra trees, on the subsets to determine how various attributes of the dataset interrelate.
It combines the findings of the trees to generate an average value for each null field.

Unlike Extratreesregressor, BayesianRidge uses linear regression to determine relationships between variables. 
Based on these relationships, it generates regularized values for non-null fields.'''
# Now we import necessary modules for ExtraTreeRegressor and Bayesian Ridge
import warnings
warnings.filterwarnings('ignore')

from sklearn.experimental import enable_iterative_imputer

from sklearn import preprocessing
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor

imputers = [
    BayesianRidge(),
    ExtraTreesRegressor(n_estimators=10, random_state=0),
]
# Now we divide the columns into 2 groups , i.e categorical and numerical 
from collections import Counter

numerical = ['year', 'odometer', 'lat', 'long']

categorical = list((Counter(df.columns) -\
                    Counter(numerical + ['manufacturer', 'model', 'price'])).elements())
# First, we will use Extratreesregressor to fill out the null fields of the numerical columns.
sr_numerical = df[numerical]
imp_numerical = IterativeImputer(imputers[1])
imputed_vals = imp_numerical.fit_transform(sr_numerical)
df[numerical] = imputed_vals
# The numerical columns have no NULL vaues in them now 
df.isnull().sum()[numerical]
''' Now we will use Bayesian Ridge to remove NULL values in categorical column .
But the algorithm cannot understand the data as it is in string format . 
So we have to encode it .
'''
def encode(data_col):
    #A function that transforms non-null values
    vals = np.array(data_col.dropna())
    # Reshaping the non-null data of a column
    reshaped_data = vals.reshape(-1,1)
    # Encoding the reshaped data
    encoded_data = encoder.fit_transform(reshaped_data)
    # Assigning the encoded values to the corresponding column values
    data_col.loc[data_col.notnull()] = np.squeeze(encoded_data)
    return data_col
# Now let us use the encode function
sr_categorical = df[categorical]
encoder = preprocessing.LabelEncoder()

# Using a for loop to iterate through each categorical column and
# filling out its null fields
for column in categorical:
    encode(sr_categorical[column])
    imp_categorical = IterativeImputer(BayesianRidge())
    imputed_vals_cat = imp_categorical.fit_transform(sr_categorical[column].values.reshape(-1, 1))
    imputed_vals_cat = imputed_vals_cat.astype('int64')
    imputed_vals_cat = pd.DataFrame(imputed_vals_cat)
    imputed_vals_cat = encoder.inverse_transform(imputed_vals_cat.values.reshape(-1, 1))
    sr_categorical[column] = imputed_vals_cat

df[categorical]= sr_categorical
# We have successfully removed the null values in the categorical group also.
df.isnull().sum()[categorical]
# Now let us take a peek at our dataset 
df.head()
df.loc[:, ['region', 'manufacturer', 'model']]\
[df.model.str.startswith(r'$500', na=False)]
# Let us see how many unique values our dataset contains 
df.apply(pd.Series.nunique)
# The dataset is fairly clean now. Let us save it as a CSV file.
df.to_csv('vehicles_eda.csv', index=False)
'''Visualization'''
# We will be using  Seaborn Displot for plotting graphs .
# Remember our Target Variable is : "Price"
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(6,3)})

def plot_histogram(col, color_val='#005c9d',\
                   x_label='Price [x10\u2076 USD]', y_label='Frequency',\
                   title_text='Distribution of car prices'):
    sns.distplot(col, kde=False, color=color_val)
    
    ax = plt.gca()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_text)
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)

    plt.show()
# We will also import the CSV file that we had saved in the previous step.
df = pd.read_csv('vehicles_eda.csv')
# Now we plot histogram 
price_mill = df.price/10**6
plot_histogram(price_mill)
# The above graph shows that the maximum number of Price values are around zero.
# This is because we have taken the range wrong .
# So we scale the Price range between USD 0 - 60,000 range 
plot_histogram(df.price[df.price<60000])
# Apparently a large number of Prices are around 0 . 
# We calculate the mean , median 
print('Mean:', df.price.mean())
print()
print('Median: ', df.price.median())
print('Max. price: ', df.price.max())
# The min value is 0 , whereas the max value , as you can see above is huge !
# Why is this ?
# This is because of the presence of outliers.
# Let's look at some attributes that are of higher values 
cols = ['region', 'year', 'manufacturer', 'model', 'price']

df.loc[:, cols][df.price>100000].sort_values(by='price', ascending=False).head(10)
# As you can see above , the prices of the cars is in Billions , which is impossible.
# To remove these outliers we use Inter Quartile Range (IQR)
# But to apply IQR , the data should be uniform
# But our target columne i.e Price columns is not uniform 
# So we use Logarithmic Function to bring uniformity and add a modified version of the Price column.
df.insert(17, 'logprice', np.log1p(df['price']))
# To check whether the column is uniform or not , we plot the graph for the new column i.e "logprice"
plot_histogram(df.logprice) 
# Now the data is uniform.
# We can now remove the outlier using IQR .
# But before that we convert the "Price" attribute to string , so that IQR is not applied on that , later we will once again bring it 
# to int64 format.
df['price'] = df.price.astype(str)
# Now , we will apply IQR 
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
Q1
Q3
IQR
# Now we update the dataset , by removing the outliers using IQR 
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
# As said earlier , we once again bring back the data type of the "Price" attribute to int64 format
df['price'] = df.price.astype(np.int64)
# As the dataset is quite clean now , we will delete all those car models that appear less than 1000 times.
# This step will dramatically reduce the chance of unrealistic model names and manufacturer-model combinations appearing in our dataset. 
# It will also ensure that the proposed machine-learning models have enough relevant data to understand the interrelations between car attributes or characteristics and their prices.

df = df.groupby("model").filter(lambda x: len(x) >= 1000)
df.reset_index(drop=True, inplace=True)
# Next , we will fill the missing values of manufacturer column with the mode of that column i.e most occuring value.
df['manufacturer'] = df.groupby('model').manufacturer.transform(
    lambda x: x.fillna(x.mode()[0])
)
# Next, let us sort the dataset and browse through some of its rows.
df.sort_values(by=['year','manufacturer', 'price'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.iloc[np.r_[0:3, -3:0]]
# The structure and standard summary statistics of the updated dataset are as follows: 
df.info()
df.describe()
# Mean , Median , Mode of the Target Variable 
print('Mean: ', round(df.price.mean()))
print()
print('Median: ', round(df.price.median()))
print()
print('Mode: ', df.price.mode()[0])
# The mean > median > mode 
# i.e most of the data is on the lower side 
plot_histogram(df.price)
# Now , plotting graph 
plt.figure(figsize=(10, 4))
plt.xticks(rotation=90)
sns.countplot(df.manufacturer);
# Next, let us look at the counts of some of the other categorical variables.
categ_x = categorical.copy()
categ_x.remove('region')
categ_x.remove('state')

fig, ax = plt.subplots(3, 3, figsize=(20, 15))
for variable, subplot in zip(categ_x, ax.flatten()):
    sns.countplot(df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.tight_layout()
# Next, let us look at how prices are interrelated with various categorical variables.
plt.figure(figsize=(10, 4))
plt.xticks(rotation=90)
sns.barplot(x='manufacturer', y='price', data=df);
# As the graph above indicates, Ram Trucks lead on the price front.
fig, ax = plt.subplots(3, 3, figsize=(20, 15))
for var, subplot in zip(categ_x, ax.flatten()):
    sns.barplot(x=var, y='price', data=df, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.tight_layout()
# The graphs above show that , new cars, diesel cars, cars that have liens on them, 
#and cars with four-wheel drives have higher average prices than other types of cars in their respective categories.
# Let us also look at how prices vary with year of manufacture
year = df.year.astype(np.int64)
price = df.price
plt.figure(figsize=(10, 4))
plt.xticks(rotation=90)
sns.barplot(year, price);
# The above graph shows that prices increase fairly consistently with year.
# Now let us plot graphs with 3 variables 
factor_combos = [('fuel', 'condition'), ('condition', 'size'),\
                 ('fuel', 'cylinders'), ('transmission', 'size'),\
                 ('size', 'drive'), ('drive', 'size')]
fig, ax = plt.subplots(3, 2, figsize=(20, 15))
for var, subplot in zip(factor_combos, ax.flatten()):
    sns.barplot(x=var[0], y='price', hue=var[1], data=df, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.tight_layout()
# So, all the car characteristics in our dataset have some impact on the target variable, price.