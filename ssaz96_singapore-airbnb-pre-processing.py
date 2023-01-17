import pandas as pd
import numpy as np
from scipy import stats
def describe_attribute(df,x):
    print(df[x].value_counts().sort_values())
    print('NA values count: ', df[x].isna().sum())
    print('Unique values count: ', df[x].nunique())
    print('Data Type: ', df[x].dtype)
def detect_outlier(data_1):
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
df = pd.read_csv('singapore_airbnb_dirty_data.csv')
df.head(10)
df.columns.tolist()
#check missing values in data
df.isnull().sum().sort_values(ascending=True)
# df.shape
# Count of unique values in each attribute
df.nunique().sort_values()
# Missing values in name column: 44
# Strange values in name column:  Fix case, remove extra spaces and characters. Number and Letters.
df['name'] = df['name'].apply(lambda x: str(x).capitalize().strip().replace('@', ' at ').replace('  ', ' '))

# Fill with concatenated neighbourhood and room_type
df['name'].fillna(str(df['neighbourhood'] +" "+  df['room_type']).capitalize(), inplace = True)
# Remove string values in id column
# df['id'] = pd.to_numeric(df['id'],  downcast='integer' ,errors='coerce')
# Missing values in id column: 12
# df['id1'].isna().sum()

# Drop Column
df.drop('id', axis = 1, inplace=True)
# df['host_id'].nunique() #2707 values

# Remove date values in host_id column
df['host_id'] = pd.to_numeric(df['host_id'],  downcast='integer' ,errors='coerce')
df['host_id'].isna().sum()
df['host_id'].fillna('-1', inplace=True)
# Strange values in host_name column : Encoding issue
# Incomplete name values in host_name column
df['host_name'] = df['host_name'].str.capitalize()
# Remove Numbers from Host Name
df['host_name'] = df['host_name'].str.replace(r"\d", "")
# Missing values in room_type column
df['room_type'].fillna(df['room_type'].value_counts().idxmax(), inplace=True)

# Numbers in room_type column
vc = df['room_type'].value_counts() 
# Get value_counts with occurrence < 2
vc = vc[vc < 2]
# Replace these values with mode.
df['room_type'].replace(vc.index.tolist(), df['room_type'].value_counts().idxmax(), inplace=True)
# Negative values in price column : Take absolute
df['price']= df['price'].apply(lambda x: np.abs(x))

# Potential outliers in price column
import seaborn as sns
%matplotlib inline

sns.boxplot(x = df['price'])
df['price'].describe()

# sns.boxplot(x = df[df['price'] < 500]['price'])
# df[df['price'] < 500]['price'].describe()
# Predict outliers in price
df['price'] = np.abs(df['price'])
price_df = (df['price'])
outliers=[]

# these are valid prices how ever they are special cases so we drop these values.
outlier_datapoints = detect_outlier(price_df)
print(np.sort(outlier_datapoints))

# replace outliers with nan
# df['price'].replace(outlier_datapoints, np.nan, inplace = True)
# String values in minimum_nights column
df['minimum_nights'] = pd.to_numeric(df['minimum_nights'], errors='coerce', downcast='integer')

# Missing values in minimum_nights column
df['minimum_nights'].fillna(1, inplace=True)
# Convert to datetime
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

# Missing values in last_review column : 1990-01-01
df['last_review'].fillna('1990-01-01', inplace=True)
# Missing values in reviews_per_month column : 0
df['reviews_per_month'].fillna(0.0, inplace=True)
# Negative values in calculated_host_listings_count column
df_copy = df.copy()

# Calculate count of each host_id 
count_df = df_copy.groupby('host_id').size().to_frame().reset_index()

# Merge the output and rename column
df_copy = pd.merge(df_copy, count_df, left_on='host_id', right_on='host_id')
df_copy = df_copy.drop('calculated_host_listings_count', axis = 1)
df_copy.rename(columns={0: 'calculated_host_listings_count'}, inplace=True)
df = df_copy
# Missing values in availability_365 column
df['availability_365'] = pd.to_numeric('availability_365', errors='coerce', downcast='integer')
df['availability_365'].fillna('0', inplace=True)
df['availability_365'] = df['availability_365'].astype(int)
# Check for potential dirty values in the remaining columns as well (latitude, longitude, number_of_reviews)
describe_attribute(df, 'number_of_reviews')
# Missing values in neighbourhood_group column
# describe_attribute(df, 'neighbourhood_group')

# Part 1: Eport unique rows where neighbourhood_group is na
df_temp = df[['neighbourhood_group', 'neighbourhood']].drop_duplicates()
df_temp = df_temp[df_temp['neighbourhood_group'].isna()]
# df_temp.to_csv('neighbourhood_group_mappings.csv', index=False)
# Part 2: Read as dictionary
df_temp = df.copy()
df_neighbourhood_group = pd.read_csv('neighbourhood_group_mappings.csv');
dict_neighbourhood_group = df_neighbourhood_group.set_index('neighbourhood').to_dict()['neighbourhood_group']
dict_neighbourhood_group.get('Outram')

# Copy the neighbourhood where neighbourhood_group is na.
df_temp['neighbourhood_group'].fillna(df_temp['neighbourhood'], inplace=True)
# Replace with dictionary
df_temp['neighbourhood_group'].replace(dict_neighbourhood_group,regex=False,inplace=True)
describe_attribute(df_temp, 'neighbourhood_group')
df = df_temp
# Missing values in neighbourhood column
# describe_attribute(df, 'neighbourhood')

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="singapore_airbnb_cleaning",timeout=2)
location = geolocator.reverse('1.31267,103.87457')
print(location.raw.get('address').get('suburb'))

import warnings
warnings.filterwarnings(action='once')

reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def geodecode(la, ln):
    return reverse((str(la)+','+str(ln)), language='en')

df_copy = df.copy()
df_copy2 = df_copy[df_copy['neighbourhood'].isna()]

# Fetch addresses where neighbourhood is na.
for index, row in df_copy2.iterrows():
    df_copy2.at[index,'neighbourhood'] = geodecode(row['latitude'],row['longitude']).raw.get('address').get('suburb')

df_copy.loc[df_copy['neighbourhood'].isna() , 'neighbourhood'] = df_copy2['neighbourhood'].apply(lambda x: x)
# df_copy.loc[df_copy['neighbourhood'].str.contains(',')]['neighbourhood'].tolist()
# df_copy['neighbourhood'].isna().sum()

# Check Where none is
df_copy.loc[df_copy['neighbourhood'].isna()] 
# Fill single value to Boon Keng (Google coordinates)
df_copy['neighbourhood'] = df_copy['neighbourhood'].fillna('Boon Keng')
# df_copy['neighbourhood'].unique().tolist()
# df_copy['neighbourhood'].value_counts()
# Final Check
df_copy.isna().sum()
# Export 
df_copy.to_csv('singapore_airbnb_clean_data.csv', index_label='row_id')