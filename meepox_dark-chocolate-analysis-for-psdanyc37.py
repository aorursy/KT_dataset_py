## imports
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot, iplot, init_notebook_mode
import warnings
from subprocess import check_output
from IPython.core.display import display, HTML
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('precision', 4)
warnings.simplefilter('ignore')
init_notebook_mode()
display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
## Load data
choko = pd.read_csv('../input/flavors_of_cacao.csv')
choko.shape # How many revies we have
# Explore first 5 rows
choko.tail(1000).T
# Explore description
choko.describe(include='all').T
# Explore datatypes
choko.dtypes
## Before we continue - rename some columns, 
original_colnames = choko.columns
new_colnames = ['company', 'species', 'REF', 'review_year', 'cocoa_p',
                'company_location', 'rating', 'bean_typ', 'country']
choko = choko.rename(columns=dict(zip(original_colnames, new_colnames)))
## And modify data types
choko['cocoa_p'] = choko['cocoa_p'].str.replace('%','').astype(float)/100
choko.head(10)
# Explore description
choko.describe(include='all').T
## Look at most frequent species
choko['species'].value_counts().head(10)
## Is where any N/A values in origin country?
choko['country'].isnull().value_counts()

## Replace origin country
choko['country'] = choko['country'].fillna(choko['species'])
choko['country'].isnull().value_counts()
## Look at most frequent origin countries
choko['country'].value_counts().head(10)
## We see that a lot of countries have ' ' value - means that this is 100% blend. Let's look at this
choko[choko['country'].str.len()==1]['species'].unique()
## Is there another way to determine blends?
choko[choko['species'].str.contains(',')]['species'].nunique()
## Is there any misspelling/reduction?
choko['country'].sort_values().unique()
## Text preparation (correction) func
def txt_prep(text):
    replacements = [
        ['-', ', '], ['/ ', ', '], ['/', ', '], ['\(', ', '], [' and', ', '], [' &', ', '], ['\)', ''],
        ['Dom Rep|DR|Domin Rep|Dominican Rep,|Domincan Republic', 'Dominican Republic'],
        ['Mad,|Mad$', 'Madagascar, '],
        ['PNG', 'Papua New Guinea, '],
        ['Guat,|Guat$', 'Guatemala, '],
        ['Ven,|Ven$|Venez,|Venez$', 'Venezuela, '],
        ['Ecu,|Ecu$|Ecuad,|Ecuad$', 'Ecuador, '],
        ['Nic,|Nic$', 'Nicaragua, '],
        ['Cost Rica', 'Costa Rica'],
        ['Mex,|Mex$', 'Mexico, '],
        ['Jam,|Jam$', 'Jamaica, '],
        ['Haw,|Haw$', 'Hawaii, '],
        ['Gre,|Gre$', 'Grenada, '],
        ['Tri,|Tri$', 'Trinidad, '],
        ['C Am', 'Central America'],
        ['S America', 'South America'],
        [', $', ''], [',  ', ', '], [', ,', ', '], ['\xa0', ' '],[',\s+', ','],
        [' Bali', ',Bali']
    ]
    for i, j in replacements:
        text = re.sub(i, j, text)
    return text
choko['country'].str.replace('.', '').apply(txt_prep).unique()
## Replace country feature
choko['country'] = choko['country'].str.replace('.', '').apply(txt_prep)
## Looks better
choko['country'].value_counts().tail(30)
## How many countries may contain in Blend?
(choko['country'].str.count(',')+1).value_counts()
## Is there any misspelling/reduction in company location?
choko['company_location'].sort_values().unique()
## We need to make some replacements
choko['company_location'] = choko['company_location']\
.str.replace('Amsterdam', 'Holland')\
.str.replace('U.K.', 'England')\
.str.replace('Niacragua', 'Nicaragua')\
.str.replace('Domincan Republic', 'Dominican Republic')

choko['company_location'].sort_values().unique()
## Is there any misspelling/reduction in company name?
choko['company'].str.lower().sort_values().nunique() == choko['company'].sort_values().nunique()
## Let's define blend feature
choko['is_blend'] = np.where(
    np.logical_or(
        np.logical_or(choko['species'].str.lower().str.contains(',|(blend)|;'),
                      choko['country'].str.len() == 1),
        choko['country'].str.lower().str.contains(',')
    )
    , 1
    , 0
)
## How many blends/pure cocoa?
choko['is_blend'].value_counts()
## Look at 5 blends/pure rows
choko.groupby('is_blend').head(5)
## Look at distribution of Cocoa %
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(choko['cocoa_p'], ax=ax)   #kde=false if you just want counts
ax.set_title('Cocoa %, Distribution')
plt.show()
## Look at distribution of rating
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(choko['rating'], ax=ax)
ax.set_title('Rating, Distribution')
plt.show()

## Look at distribution of rating of 'Blend vs Not Blend'
fig, ax = plt.subplots(figsize=[16,4])
for i, c in choko.groupby('is_blend'):
    sns.distplot(c['rating'], ax=ax, label=['Not Blend', 'Blend'][i])
ax.set_title('Rating, Distribution, hue=Blend')
ax.legend()
plt.show()
## What's better? Pure or blend?
fig, ax = plt.subplots(figsize=[6, 6])
sns.boxplot(
    data=choko,
    x='is_blend',
    y='rating',
)
ax.set_title('Boxplot, Rating by Blend/Pure')
##Boxplots of relationship of ratings with cocoa percentage
fig, ax = plt.subplots(figsize=[20, 16])
sns.boxplot(
    data=choko,
    y='rating',
    x='cocoa_p'
)
ax.set_title('Boxplot, Rating for Cocoa Percentages')
##Bucketing the cocoa percentage to better identify what range of cocoa percentage is best (5% increment buckets)
bucket_array = np.linspace(.40,1,7)
choko['bucket_cocoa_p']=pd.cut(choko['cocoa_p'], bucket_array)
choko.head()
##Boxplots of relationship of ratings with binned cocoa percentage
fig, ax = plt.subplots(figsize=[20, 10])
sns.boxplot(
    data=choko,
    y='bucket_cocoa_p',
    x='rating'
)
ax.set_title('Boxplot, Rating for Cocoa Bucket  Percentages')
## Look at boxplot over the countries, even Blends
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choko,
    y='country',
    x='rating'
)
ax.set_title('Boxplot, Rating for countries (+blends)')

## But how can we see what country is biggest contributor in rating?
choko_ = pd.concat([pd.Series(row['rating'], row['country'].split(',')) for _, row in choko.iterrows()]
         ).reset_index()
choko_.columns = ['country', 'rating']
choko_['mean_rating'] = choko_.groupby(['country'])['rating'].transform('mean')

## Look at boxplot over the countries (contributors in blends)
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choko_.sort_values('mean_rating', ascending=False),
    y='country',
    x='rating'
)
ax.set_title('Boxplot, Rating for countries (contributors)')
##Get rid of the countries with low amount of ratings.
choko_ = pd.concat([pd.Series(row['rating'], row['country'].split(',')) for _, row in choko.iterrows()]
         ).reset_index()
choko_.columns = ['country', 'rating']

choko_country_counts = choko_['country'].value_counts()
print(choko_country_counts)
#Get all the countries that have >50 counts after splitting data.
##choko_sig_countries
choko_sig_countries = choko_country_counts[choko_country_counts >= 50].index.tolist()

## removing  country==' ' rows
choko_sig_countries = list(filter(lambda x: x != ' ', choko_sig_countries))

chokolat = choko_.loc[choko_['country'].isin(choko_sig_countries)]

##print(chokolat)
           
## Look at boxplot over the counties that had more than 100 ratings
fig, ax = plt.subplots(figsize=[20, 16])
sns.boxplot(
    data=chokolat,
    y='rating',
    x='country'
)
ax.set_title('Boxplot, Rating for countries (contributors)')
## Prepare full tidy choko_ dataframe, splitting countries to seperate rows for rows that have blends
def choko_tidy(choko):
    data = []
    for i in choko.itertuples():
        for c in i.country.split(','):
            data.append({
                'company': i.company,
                'species': i.species,
                'REF': i.REF,
                'review_year': i.review_year,
                'cocoa_p': i.cocoa_p,
                'company_location': i.company_location,
                'rating': i.rating,
                'bean_typ': i.bean_typ,
                'country': c,
                'is_blend': i.is_blend,
            })
    return pd.DataFrame(data)
        
choko_ = choko_tidy(choko)
print(choko_.shape, choko.shape)
## Exploring our tidied data that we split off every blends countries into their own row.
choko_.head()
##Preparing our Datatable for multiple linear regression testing by dropping columns and filtering for only significant countries.
choko_reg = choko_.drop(['REF','bean_typ','company','company_location','review_year','species'], axis=1)

##choko_sig_countries (looking at countries with >100 to test for regression effects)
choko_country_counts = choko_['country'].value_counts()
choko_sig_countries = choko_country_counts[choko_country_counts >= 100].index.tolist()

choko_reg = choko_reg.loc[choko_reg['country'].isin(choko_sig_countries)].reset_index()
choko_reg = choko_reg.drop(['index'],axis=1)
print(choko_reg)
#Breaking out Data into Independent and Dependent Variables
X = choko_reg.iloc[:,:-1].values
Y = choko_reg.iloc[:,3].values

# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1]= labelencoder_X.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

#Eliminate Dummy Error
X = X[:,1:]

# Divide dataset Train set & Test set 
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


# Fit the multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
trained = regressor.fit(X_train, Y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Building the optimal model
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((945,1)).astype(int), values = X, axis = 1)

# Backward Elimination
X_opt = X[:, [0,1,2,3,4,5,6]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())

# Backward Elimination (removed 4th column => Dominican Republic)
X_opt = X[:, [0,1,2,3,5,6]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())


# Backward Elimination (removed 2nd column => Madagascar)
X_opt = X[:, [0,1,3,5,6]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())
