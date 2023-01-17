import numpy as np                                                 # Implemennts milti-dimensional array and matrices

import pandas as pd                                                # For data manipulation and analysis

import pandas_profiling

import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy

import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics

%matplotlib inline

sns.set()



from subprocess import check_output



#If you get UnicodeDecodeError - 'utf8' codec can't decode, invalid continuation byte error, either use engine='python'

#or encoding='latin-1' options

carsale = pd.read_csv("../input/car-sale-advertisements/car_ad.csv",engine="python")     # Importing car_sale dataset using pd.read_csv
carsale.shape    # This will print the number of rows and comlumns of the Data Frame
carsale.columns  # This will print the names of all columns.
carsale.head()   # Will give you first 5 records
carsale.tail()   # This will print the last n rows of the Data Frame
carsale.info() # This will give Index, Datatype and Memory information
# Use include='all' option to generate descriptive statistics for all columns

# You can get idea about which column has missing values using this

carsale.describe() 
carsale.isnull().sum() # Will show you null count for each column, but will not count Zeros(0) as null
profile = pandas_profiling.ProfileReport(carsale)

profile.to_file(outputfile="carsale_before_preprocessing_1.html")
carsale.replace({'engType': 'Other', 'price': 0, 'mileage': 0}, np.nan, inplace=True)
profile = pandas_profiling.ProfileReport(carsale)

profile.to_file(outputfile="carsale_before_preprocessing_2.html")
carsale.drop_duplicates(inplace=True) #inplace used to modify the dataset with applied command

carsale.shape
def get_median_price(x):

    brand = x.name[0]

    if x.count() > 0:

        return x.median() # Return median for a brand/model if the median exists.

    elif carsale.groupby(['car'])['price'].count()[brand] > 0:

        brand_median = carsale.groupby(['car'])['price'].apply(lambda x: x.median())[brand]

        return brand_median # Return median of brand if particular brand/model combo has no median,

    else:                 # but brand itself has a median for the 'price' feature. 

        return carsale['price'].median() # Otherwise return dataset's median for the 'price' feature.

    

price_median = carsale.groupby(['car','model'])['price'].apply(get_median_price).reset_index()

price_median.rename(columns={'price': 'price_med'}, inplace=True)

price_median.head()
def fill_with_median(x):

    if pd.isnull(x['price']):

        return price_median[(price_median['car'] == x['car']) & (price_median['model'] == x['model'])]['price_med'].values[0]

    else:

        return x['price']

    

carsale['price'] = carsale.apply(fill_with_median, axis=1)

carsale.head()
def get_median_engV(x):

    brand = x.name[0]

    if x.count() > 0:

        return x.median() # Return median for a brand/model if the median exists.

    elif carsale.groupby(['car'])['engV'].count()[brand] > 0:

        brand_median = carsale.groupby(['car'])['engV'].apply(lambda x: x.median())[brand]

        return brand_median # Return median of brand if particular brand/model combo has no median,

    else:                 # but brand itself has a median for the 'engV' feature. 

        return carsale['engV'].median() # Otherwise return dataset's median for the 'engV' feature.

    

engV_median = carsale.groupby(['car','model'])['engV'].apply(get_median_engV).reset_index()

engV_median.rename(columns={'engV': 'engV_med'}, inplace=True)

engV_median.head()
def fill_with_median(x):

    if pd.isnull(x['engV']):

        return engV_median[(engV_median['car'] == x['car']) & (engV_median['model'] == x['model'])]['engV_med'].values[0]

    else:

        return x['engV']

    

carsale['engV'] = carsale.apply(fill_with_median, axis=1)

carsale.head()
def get_median_mileage(x):

    brand = x.name[0]

    if x.count() > 0:

        return x.median() # Return median for a brand/model if the median exists.

    elif carsale.groupby(['car'])['mileage'].count()[brand] > 0:

        brand_median = carsale.groupby(['car'])['mileage'].apply(lambda x: x.median())[brand]

        return brand_median # Return median of brand if particular brand/model combo has no median,

    else:                 # but brand itself has a median for the 'mileage' feature. 

        return carsale['mileage'].median() # Otherwise return dataset's median for the 'mileage' feature.

    

mileage_median = carsale.groupby(['car','model'])['mileage'].apply(get_median_mileage).reset_index()

mileage_median.rename(columns={'mileage': 'mileage_med'}, inplace=True)

mileage_median.head()
def fill_with_median(x):

    if pd.isnull(x['mileage']):

        return mileage_median[(mileage_median['car'] == x['car']) & (mileage_median['model'] == x['model'])]['mileage_med'].values[0]

    else:

        return x['mileage']

    

carsale['mileage'] = carsale.apply(fill_with_median, axis=1)

carsale.head()
def get_drive_mode(x):

    brand = x.name[0]

    if x.count() > 0:

        return x.mode() # Return mode for a brand/model if the mode exists.

    elif carsale.groupby(['car'])['drive'].count()[brand] > 0:

        brand_mode = carsale.groupby(['car'])['drive'].apply(lambda x: x.mode())[brand]

        return brand_mode # Return mode of brand if particular brand/model combo has no mode,

    else:                 # but brand itself has a mode for the 'drive' feature. 

        return carsale['drive'].mode() # Otherwise return dataset's mode for the 'drive' feature.

    

drive_modes = carsale.groupby(['car','model'])['drive'].apply(get_drive_mode).reset_index().drop('level_2', axis=1)

drive_modes.rename(columns={'drive': 'drive_mode'}, inplace=True)

drive_modes.head()
def fill_with_mode(x):

    if pd.isnull(x['drive']):

        return drive_modes[(drive_modes['car'] == x['car']) & (drive_modes['model'] == x['model'])]['drive_mode'].values[0]

    else:

        return x['drive']

    

carsale['drive'] = carsale.apply(fill_with_mode, axis=1)

carsale.head()
def get_engType_mode(x):

    brand = x.name[0]

    if x.count() > 0:

        return x.mode() # Return mode for a brand/model if the mode exists.

    elif carsale.groupby(['car'])['engType'].count()[brand] > 0:

        brand_mode = carsale.groupby(['car'])['engType'].apply(lambda x: x.mode())[brand]

        return brand_mode # Return mode of brand if particular brand/model combo has no mode,

    else:                 # but brand itself has a mode for the 'engType' feature. 

        return carsale['engType'].mode() # Otherwise return dataset's mode for the 'engType' feature.

    

engType_modes = carsale.groupby(['car','model'])['engType'].apply(get_engType_mode).reset_index().drop('level_2', axis=1)

engType_modes.rename(columns={'engType': 'engType_mode'}, inplace=True)

engType_modes.head()
def fill_with_mode(x):

    if pd.isnull(x['engType']):

        return engType_modes[(engType_modes['car'] == x['car']) & (engType_modes['model'] == x['model'])]['engType_mode'].values[0]

    else:

        return x['engType']

    

carsale['engType'] = carsale.apply(fill_with_mode, axis=1)

carsale.head()
carsale.isnull().sum()
import pandas_profiling

profile = pandas_profiling.ProfileReport(carsale)

profile.to_file(outputfile="carsale_after_preprocessing.html")
carsale.car.value_counts().head(10).plot.bar()

plt.title("Top 10 car brands on sale")
carsale[carsale.price.isin(carsale.price.nlargest())].sort_values(['car','model','body','mileage','price'])
carsale[carsale.price.isin(carsale.price.nsmallest())].sort_values(['car','model','body','mileage','price'])
sns.countplot(y='body', data=carsale, orient='h', hue='registration')

plt.title("Most preferred body type used in 1953-2016")
sns.countplot(x='engType', data=carsale, orient='h')

plt.title("Most preferred engType used over the years")
carsale.sort_values(['car','model','body','mileage','year'])



df = carsale.groupby('year')['registration'].value_counts().sort_values(ascending=False)

df = pd.DataFrame(df)

df.rename(columns={'registration': 'RegCounts'}, inplace=True)

df.reset_index(inplace=True)

display(df.head())

sns.lineplot(data=df, x='year', y='RegCounts', hue='registration')

#sns.scatterplot(data=df, x='year', y='RegCounts', hue='registration')

plt.title("Years group having max sale/registration")

sns.lineplot(data=carsale, y='price', x='year', hue='drive')

plt.title("year - price lineplot (1950 - 2010)")
sns.lineplot(data=carsale[carsale.year >= 2010], y='price', x='year', hue='drive')

plt.title("year - price lineplot (2010 - 2016)")
sns.lineplot(x='mileage',y='price',data=carsale, hue='engType')

plt.title("mileage - price line Plot")
sns.heatmap(carsale.corr(),annot=True, linewidths=.5)

plt.title("Heatmap for Highest correlated features for Carsale datset")
sns.lmplot('year','price', carsale, fit_reg=False, hue='engType')

plt.title("Price distribution over the year w.r.t to engType")
sns.pairplot(carsale, hue='engType', palette="viridis", height=3)