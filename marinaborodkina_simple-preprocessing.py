import pandas as pd

import numpy as np



# Import and suppress warnings

import warnings

warnings.filterwarnings('ignore')



# Visualiazation

from seaborn import countplot



# Generate dataset

Cars = {'Date': ['10.10.2018', '14.10.2018', '07.11.2018', '20.11.2018', '06.12.2018', '01.01.2019', '07.01.2019', '07.02.2019'],

        'Brand': ['Honda Civic', np.NaN, 'Toyota Corolla', 'Ford Focus',  np.NaN, 'Audi A4', np.NaN, 'Honda'],

        'Price1': [22000, 25000, 27000, np.NaN, 35000, 15000, 1000, 1500],

        'Price2': [23000, 21000, 25000, np.NaN, 35000, 11000, 1200, 1100],

        'Engine': ['150.0 horsepower', '100.0 horsepower', '250.0 horsepower', '100.5 horsepower', '50.0 horsepower', '40.0 horsepower', '50.5 horsepower', '45.0 horsepower'],

        'Color': ['red', 'blue', 'green', 'red', np.NaN, 'blue', 'red', 'yellow'],

        'Year': [2000, 2010, 2015, 2011, 2019, 2005, 1999, 1995],

        'Label': [1, 0, 0, 0, 0, 0, 1, 1]

        }



df = pd.DataFrame(Cars, columns= ['Date', 'Brand', 'Price1', 'Price2', 'Engine', 'Color', 'Year', 'Label'])

df
# Columns of the DataFrame

df.columns
# Shape (number of columns, rows)

df.shape
# Types of the columns

df.dtypes
# Statistic for numerical columns

df.describe()
df.info()
round(df['Color'].value_counts(normalize=True)*100, 2)
# Check the share of the classes

print('Share of the classes:')

print(round(df['Label'].value_counts(normalize=True)*100, 2))

countplot(x='Label', data=df)
from sklearn.model_selection import train_test_split



# Create a data with all columns except Label

df_X = df.drop('Label', axis=1)



# Create a category_desc labels dataset

df_y = pd.DataFrame(df['Label'])



# Use stratified sampling to split up the dataset according to the df_y dataset

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, stratify=df_y)



# Print out the Label counts on the training y labels

print(round(y_train['Label'].value_counts(normalize=True)*100, 2))
from sklearn.linear_model import LogisticRegression



def check_score(X_, y_):

    

    # Split X and the y labels into training and test sets

    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_)



    lgr = LogisticRegression()

    # Fit lgr to the training data

    lgr.fit(X_train_, y_train_)



    # Score knn on the test data and print it out

    return(lgr.score(X_test_, y_test_)) 
# Number of missing values per column

print('Number of missing values per column:')

df.isnull().sum()
# Delete all rows with missing

print('Number of rows:', df.shape[0])

print('Number of rows after deleting all rows with missing:', df.dropna().shape[0])

df.dropna()
# Subset dataset without missing

df_no_missing = df[df.notnull()]

df_no_missing
# Delete all cols with missing

print('Number of crows:', df.shape[1])

print('Number of cows after deleting all cows with missing:', df.dropna(axis=1).shape[1])

df.dropna(axis=1)
# Columns with at least 5 not missing

df.dropna(axis=1, thresh=5)
# Fill in missing values

df['Brand'].fillna('missing', inplace=True)

df['Color'].fillna('missing', inplace=True)



df['Price1'].fillna(df['Price1'].mean(), inplace=True)

df['Price2'].fillna(df['Price2'].mean(), inplace=True)



df
# Check Data Types

df.dtypes
# Reduce memory usage

if df['Price1'].min() > np.finfo(np.float32).min and df['Price1'].max() < np.finfo(np.float32).max:

    df['Price1'] = df['Price1'].astype(np.float32)

    

if df['Price2'].min() > np.finfo(np.float32).min and df['Price2'].max() < np.finfo(np.float32).max:

    df['Price2'] = df['Price2'].astype(np.float32)



if df['Year'].min() > np.iinfo(np.int32).min and df['Year'].max() < np.iinfo(np.int32).max:

    df['Year'] = df['Year'].astype(np.int32)



# Check Data Types

df.dtypes
df.var()
df['log_Year'] = np.log(df['Year'])

df['log_Year'].var()
# Import StandardScaler from scikit-learn

from sklearn.preprocessing import StandardScaler



# Create the scaler

ss = StandardScaler()



# Take a subset of the DataFrame you want to scale 

df_subset = df[['Price1', 'Price2', 'Year']]



# Apply the scaler to the DataFrame subset

df_subset_scaled = ss.fit_transform(df_subset)



df_subset_scaled
df['Price_type'] = 'low'

df.loc[(df['Price1'] > 10000), 'Price_type'] = 'higth'

df[['Price1', 'Price_type']] 
from sklearn.preprocessing import LabelEncoder



# Set up the LabelEncoder object

enc = LabelEncoder()



# Apply the encoding to the Color column

df['Color_enc'] = enc.fit_transform(df['Color'])



# Compare the two columns

print(df[['Color', 'Color_enc']].head())
# Transform the Color column and concatinate with DataSet

df = pd.concat([df, pd.get_dummies(df['Price_type'])], axis=1)



df[['Price_type', 'higth', 'low']]
# Create a list of the columns to average

price_columns = ['Price1', 'Price2']



# Use apply to create a mean column

df['Price_average'] = df.apply(lambda row: row[price_columns].mean(), axis=1)

df['Price_average_log'] = np.log(df['Price_average'])



# Take a look at the results

print(df[['Price1', 'Price2', 'Price_average', 'Price_average_log']] )
# First, convert string column to date column

df['Date_converted'] = pd.to_datetime(df['Date'])



# Extract just the month and year from the converted column

df['Date_month'] = df.apply(lambda row: row['Date_converted'].month, axis=1)

df['Date_year'] = df.apply(lambda row: row['Date_converted'].year, axis=1)



# Take a look at the converted and new month columns

print(df[['Date_converted', 'Date_month', 'Date_year']])
import re



# Write a pattern to extract numbers and decimals

def return_hp(str):

    pattern = re.compile(r'\d+\.\d+')

    

    # Search the text for matches

    hp = re.match(pattern, str)

    

    # If a value is returned, use group(0) to return the found value

    if hp is not None:

        return float(hp.group(0))

        

# Apply the function to the Length column and take a look at both columns

df['Engine_hp'] = df['Engine'].apply(lambda row: return_hp(row))

df[['Engine', 'Engine_hp']]
# transformation into a text vector

from sklearn.feature_extraction.text import TfidfVectorizer



# Create the vectorizer method

tfidf_vec = TfidfVectorizer()



# Transform the text into tf-idf vectors

df['Brand_text_tfidf'] = tfidf_vec.fit_transform(df['Brand'])



df[['Brand', 'Brand_text_tfidf']]
# Create a list of redundant column names to drop

to_drop = ['Brand', 'Price1', 'Price2', 'Price_type', 'higth', 'Price_average', 'Engine', 'Color', 'Year', 'Date', 'Date_converted']



# Drop those columns from the dataset

df_subset_1 = df.drop(to_drop, axis=1)



# Print out the head of the new dataset

df_subset_1
# Print out the column correlations of the dataset

df_subset_1.corr()
# Take a minute to find the column where the correlation value is greater than 0.75 at least twice

to_drop = ['log_Year', 'Price_average_log']



# Drop that column from the DataFrame

df_subset_2 = df_subset_1.drop(to_drop, axis=1)



# Print out the column correlations of the current dataset

df_subset_2.corr()
df_subset_2
# Score on the test data and print it out

df_X = df_subset_2.drop(['Label', 'Brand_text_tfidf'], axis=1)

check_score(df_X, df_subset_2['Label'])
from sklearn.decomposition import PCA



# Set up PCA and the X vector for diminsionality reduction

pca = PCA()

df_X = df_subset_2.drop(['Label', 'Brand_text_tfidf'], axis=1)



# Apply PCA to the dataset X vector

transformed_X = pca.fit_transform(df_X)



# Look at the percentage of variance explained by the different components

print(pca.explained_variance_ratio_)
# Score on the test data and print it out

check_score(transformed_X, df['Label'])