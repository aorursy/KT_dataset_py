%matplotlib inline

import numpy as np 
import pandas as pd 

# import libraries for graph generations and mapping geodata
import matplotlib.pyplot as plt
import geoplot
import mapclassify
import seaborn as sns
import geopandas as gpd

# import the quintessential ML modules
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestRegressor

# we will use SequenceMatcher to quickly merge datasets by country code
from difflib import SequenceMatcher

# since our dataset isn't too large, we'll set the column/row display limits accordingly.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 100)
np.set_printoptions(suppress=True)

# let's import our datasets
## key dataset with suicide rates and basic socio-economic data
suicide = pd.read_csv('/kaggle/input/suicide-rates-from-1986-to-2016/suicide.csv')
## country-wise religions by both sum and percentage
religion = pd.read_csv('/kaggle/input/world-religions/national.csv')
## 'Countries of the World' dataset aka the World Factbook
cotw = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')


# default kaggle code that highlights all of the files in our directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

suicide.head()
# change 'Republic of Korea' to 'South Korea' to remove any confusion.
suicide['country'] = suicide['country'].replace(['Republic of Korea'], 'South Korea')
canada = suicide[suicide['country'] == 'Canada']

canada.head()
# let's analyze year-by-year suicides per 100k pop

can_pivot_sex = canada.pivot_table(columns=['sex'], index=['year'], values=['suicides/100k pop'], aggfunc=np.sum)
years = [col[1] for col in can_pivot_sex.columns]

can_pivot_sex
can_pivot_sex.plot(figsize=(5,3))
can_pivot_age = canada.pivot_table(columns=['age'], index=['year'], values=['suicides/100k pop'], aggfunc=np.sum)

can_pivot_age
can_pivot_age.plot()
# we will limit the dataset to 2014 as stated in the description above.
suicide_2014 = suicide[suicide['year'] == 2014]

# let's confirm that there are 78 countries in this dataset.
suicide_2014['country'].nunique()
# let's aggregate the dataset as per the relevant columns
# the aggregation function for HDI will need be `np.mean` whereas for the others it will be `np.sum`

pivot_dict = {'suicides_no': np.sum,
             'population': pd.Series.mode,
             'HDI for year': np.mean,
             'gdp_per': pd.Series.mode}

suicide_2014 = suicide_2014.pivot_table(['suicides_no', 'population', 'HDI for year', 'gdp_per_capita ($)'],
                                        index=['country'], aggfunc= 
                                        {'suicides_no': np.sum, 'population': np.sum, 'HDI for year': np.mean,
                                        'gdp_per_capita ($)': np.mean})

suicide_2014
# Let's calculate the 'suicide rate per 100k'on our own instead of concatenating the original column to our post-pivot table
suicide_2014['suicides_100k'] = suicide_2014['suicides_no'] / suicide_2014['population'] * 100000

# further exploratory study: should we remove the countries with `0` suicide rates - will they skew the data?

# we will create a function to find the code from the `wikipedia-iso-country-codes` dataset. 
# to do that, the function first uses a `SequenceMatcher` to match the country names in both datasets.

def code_finder(df, col):
    c_codes = pd.read_csv('/kaggle/input/iso-country-codes-global/wikipedia-iso-country-codes.csv')
    countries = []
    codes = []
    for country in col:
        highest_score = 0
        idx = 0
        for c in c_codes['English short name lower case']:
            score = SequenceMatcher(None, country, c).ratio()
            if score > highest_score:
                highest_score = score
                best_match = c
                code = c_codes.iloc[idx,2]
            idx += 1
        countries.append(country)
        codes.append(code)
    return countries, codes

countries, codes = code_finder(suicide_2014, suicide_2014.index)

# let's create a dataframe to depict the findings and merge them conveniently
codes_df = pd.DataFrame(index=countries)
codes_df['codes'] = codes

codes_df
merged_df = suicide_2014.merge(codes_df, how='left', left_index=True, right_index=True)
merged_df.head(10)
# moving forward, we will use the country code as our index, particularly as it will make 
# our job easier working with geopandas
merged_df.set_index('codes', inplace=True)
merged_df.head(10)
# let's save our target column (for the map) to a variable
variable = 'suicides_100k'

shape_file = "../input/world-shapefiles/ne_10m_admin_0_countries.shp"
map_df = gpd.read_file(shape_file)

# Join chloropleth map data with our dataset
map_df = map_df.set_index('ADM0_A3').join(merged_df)
fig, ax = plt.subplots(1, figsize=(15,12))
map_df.plot(variable, cmap='coolwarm', linewidth=0.8, ax=ax, legend=True,  edgecolor="gray")
df = merged_df.copy()
# let's merge the CIA Factbook data with our dataset. 
# note that we need to set the 'Country' column as the index as our `code_finder` function works with the index.

cotw.set_index('Country', inplace=True)
countries, codes = code_finder(cotw, cotw.index)
cotw['codes'] = codes
cotw['country'] = countries
merged_df2 = df.join(cotw.set_index('codes'), how='outer')
merged_df2 = merged_df2[~merged_df2.index.duplicated(keep='first')]

# train = merged_df2[~merged_df2['suicides_100k'].isnull()]
# train.shape
df = merged_df2.copy()
df['Region'] = df['Region'].str.strip()
df['country'] = df['country'].str.strip()
df.head()
# let's add country-wise religious data with our dataset
# we'll extract the data from 2010 as that those represent latest available figures for our for all countries on our initial dataset (`suicide`))

# set 'state' as index to enable usability with `code_finder` function
religion_2010 = religion[religion['year'] == 2010]
religion_2010.set_index('state', inplace=True)

countries, codes = code_finder(religion_2010, religion_2010.index)
religion_2010['codes'] = codes
religion_2010['state'] = religion_2010.index
religion_2010.set_index('codes', inplace=True)
# before merging, find any column that intersect between both data sets
df_cols = set(df.columns)
religion_cols = set(religion_2010.columns)
col_intersect = set.intersection(df_cols, religion_cols)

# one culprit found: 'population'
religion_2010.drop('population', axis=1, inplace=True)
religion_2010 = religion_2010[~religion_2010.index.duplicated(keep='first')]
# merge df with religion data
merged_df3 = df.join(religion_2010, how='left')
merged_df3

#merged_df3 = merged_df3[~merged_df3.index.duplicated(keep='last')]
#merged_df3[~merged_df3['suicides_100k'].isnull()]
# we can see that 'South Korea' and 'Puerto Rico' have zero HDI. 
# let's replace these with real world values from 2014.

merged_df3.loc['KOR','HDI for year'] = 0.896
merged_df3.loc['PRI','HDI for year'] = 0.845
# let's create a new dataset so we can easily revert to this state if required
df = merged_df3.copy()

df
# let's identify and clean columns that require preparation

clean_cols = [
    'Pop. Density (per sq. mi.)', 'Arable (%)', 'Coastline (coast/area ratio)', 'Net migration',
    'Infant mortality (per 1000 births)', 'Literacy (%)', 'Phones (per 1000)',
    'Phones (per 1000)', 'Crops (%)','Other (%)','Birthrate','Deathrate',
    'Agriculture','Industry','Climate', 'Service' ]

def cleaner(val):
    out = str(val).replace(',','.')
    out = float(out)
    return out

for c in clean_cols:
    df[c] = df[c].map(lambda x: cleaner(x), na_action='ignore')
df.loc['PRI','state'] = 'Puerto Rico'
df['Climate'] = df['Climate'].astype(float)
df.isnull().sum()
null_cols = set(df.isnull().sum().sort_values().index)
numeric_columns = set(df.select_dtypes(np.number).columns)
null_cols = set.intersection(null_cols, numeric_columns)
null_cols.remove('suicides_100k')
null_cols
# let's deal with numerical null values
# finding region-wise mean values for all empty columns for easy extraction 
df_pivot_region = df.pivot_table(null_cols, 'Region', aggfunc=np.mean)
df_pivot_region.head()
df_pivot_region.loc['NORTHERN AFRICA']['HDI for year'] = .775
df_pivot_region
# let's fill in the null values
def na_filler(df):
    # we iterate through each column that contains null values
    for col in null_cols:
        
        # extract all null value indexes for each column
        idx = df.Region[df.isnull()[col]].index
        
        # find out the `Region` value for each null value in each column
        region = df.Region[df.isnull()[col]]
        
        # find the mean value from our pivot table by indexing by region and column
        result = df_pivot_region.loc[region][col]
        
        #create a dictionary with index as key and region-based mean as value through dictionary comprehension
        result_dict = {idx[i]: result[i] for i in range(len(idx))}
        
        # assign result to missing values
        df[col] = df[col].fillna(result_dict)
    
na_filler(df)
        
train = df[~df['suicides_100k'].isnull()]
sns.boxplot(x='suicides_100k', y='Region', data=train, fliersize=5)
plt.xticks(rotation=45)
regions = pd.get_dummies(df['Region'])
df = df.merge(regions, left_index=True, right_index=True)
df.head()
# Let's see which columns have a high correlation with our target variable of 'suicides_100k'
corr = df.corr()['suicides_100k'].sort_values(ascending=False)
corr
# as expected from such a complex phenomenon like suicide, there is no feature that shows high correlation with our target variable
# let's try remove columns that may introduce excess noise when predicting suicide rates for missing countries

columns_significant_corr = list(corr.index[(corr > .17) | (corr < -.17)])
columns_significant_corr.append('state')

df = df.loc[:,columns_significant_corr]
df

# lets plot a headmap to see correlations visually
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr())
df.columns
# let's remove columns we will not be needing in our machine learning model

# remove `suicide_no` we dont have any suicide data for data
# remove all non-percent religious columns so as not to overlap percentage columns
# we'll remove the 'shia' and 'sunni' columns and leave in the 'islam' column, as the latter is the sum of the prior two for several countries.
del_columns = ['confucianism_all', 'suicides_no', 'noreligion_all',
               'christianity_easternorthodox','ibadhi_percent', 
               'islam_ibadhi','shi’a_percent', 'sunni_percent', 'otherislam_percent', 'ibadhi_percent',
              'Other (%)']

df.drop(del_columns, axis=1, inplace=True)
df1 = df[~df['suicides_100k'].isnull()]
df2 = df[df['suicides_100k'].isnull()]

df_ml = pd.concat([df1, df2], axis=0)
print(df.shape, df_ml.shape)

# to make our job easier for slicing, let's reset the index
df_ml.reset_index(inplace=True)
codes = df_ml['codes']
df_ml.head()
# slice our target variable for the train dataset
train_y = df_ml.loc[:77, 'suicides_100k']
df_ml.drop(['codes', 'state', 'suicides_100k'], axis=1, inplace=True)
df_ml.head()
# make a new copy of the dataset for normalization and training
df_normalized = df_ml.copy()
# normalize dataset
normalizer = Normalizer()
df_normalized = normalizer.fit_transform(df_normalized)
df_normalized.shape
# let's create train and test sets
train_X = df_normalized[:78,:]
test_X = df_normalized[78:,:]
# we'll use the RandomForestRegressor() to estimate the values for the missing countries
rf = RandomForestRegressor()
model = rf.fit(train_X, train_y)
result = model.predict(test_X)
predict_df = df_ml.copy()
predict_df['codes'] = codes
predict_df.loc[:78,'suicides_100k'] = train_y
predict_df.loc[78:,'suicides_100k'] = result
predict_df.head()
# finally, let's visualize our map and compare it to the one available from `Our World in Data` that uses the same core dataset to see how far off we are.
variable = 'suicides_100k'
map_df = gpd.read_file(shape_file)

# drop Antartica from the shapefile
map_df.drop(172, axis=0, inplace=True)

merged_df = map_df.set_index('ADM0_A3').join(predict_df.set_index('codes'))
fig, ax = plt.subplots(1, figsize=(20,15))
merged_df.plot(variable, cmap='Blues', linewidth=0.8, ax=ax, legend=True, edgecolor="gray")
