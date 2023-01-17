!pip install descartes pandas geopandas numpy
import pandas as pd
import geopandas as gpd
import os



input_dir = '../input/tamil-nadu-school-data-ssa/'

print(input_dir)
os.listdir('../input')
# 2011_Dist.Json is a geoJSON file obtained from https://github.com/datameet/maps

# Unless otherwise states, the map dataset is shared under Creative Commons Attribution-ShareAlike 2.5 India license.
enroll_school_df = pd.read_csv(f'{input_dir}/enrollement_schoolmanagement_2.csv')
enroll_school_df.info()
enroll_school_df.head()
# Convert the District Names to title case
enroll_school_df['District'] = enroll_school_df['District'].apply(lambda x: x.title())
enroll_school_df.head()
# Show all District Names
enroll_school_df['District'].unique()
len(enroll_school_df['District'].unique())
# What's the district State Total?
enroll_school_df[enroll_school_df['District'] == 'State Total']
# Is it sum of other districts?



state_total_df = enroll_school_df[enroll_school_df['District'] == 'State Total']

district_df = enroll_school_df[enroll_school_df['District'] != 'State Total']
enroll_school_df.columns
cols = enroll_school_df.columns[2:]

for col in cols:

    assert state_total_df[col].sum() == district_df[col].sum()
# Yes, drop the 'State Total' Row
enroll_school_df.drop(enroll_school_df[enroll_school_df['District'] == 'State Total'].index, inplace=True)
enroll_school_df.tail()
# Grand Total column name has a space at the end. Rename the Column.

enroll_school_df.rename(columns={'Grand Total ': 'Grand Total'}, inplace=True)
# Check the renamed column name

enroll_school_df.columns
%matplotlib inline

def draw_bar(df, x_column, y_column, vertical=True, to_sort=True, figsize=(5, 5), title='Bar Graph', 

             stacked=False):

    if to_sort:

        df = df.sort_values(y_column)

    df.plot.bar(x=x_column, y=y_column, figsize=figsize, title=title, stacked=stacked)

        

    
# Show district Enrollment
draw_bar(enroll_school_df, y_column='Grand Total', x_column='District', figsize=(15, 7), 

         title='Enrollment Grand Total')
### Top 3 Enrollment in the districts: Kancheepuram, Chennai, Thiruvallur

### Least 3 Enrollment in the districts: Perambulur, The Nilgris, Ariyalur
enroll_school_df[enroll_school_df['District'].isin(['Kancheepuram', 'Chennai', 'Thiruvallur'])][['District', 'Grand Total']]
enroll_school_df[enroll_school_df['District'].isin(['Perambalur', 'The Nilgiris', 'Ariyalur'])][['District', 'Grand Total']]
enroll_school_df['Grand Total'].describe()
# On an average every district has 3,85,360 enrollments.
enroll_school_df['Grand Total'].sum()
# Total Enrollments in schools are 1,23,31,525.
# Now let's map the districts and these values in the Tamil Nadu Map
districts_gdf = gpd.read_file(f'../input/2011-dist/2011_Dist.json')
districts_gdf.head()
districts_gdf['ST_NM'].unique()
# filter TN Districts

tn_districts_gdf = districts_gdf[districts_gdf['ST_NM'] == 'Tamil Nadu']
tn_districts_gdf.head()
len(tn_districts_gdf)
len(enroll_school_df)
# Rename the DISTRICT column to 'District'
tn_districts_gdf.rename(columns={'DISTRICT': 'District'}, inplace=True)
tn_districts_gdf.columns
tn_districts_gdf['District'].values
enroll_school_df['District'].values
# There is a typo in Krishnagiri in enroll_gender_df

enroll_school_df.replace(['Krishanagiri'], 'Krishnagiri', inplace=True)
# There is difference in spelling for Nagappatinam

enroll_school_df.replace(['Nagapattinam'], 'Nagappattinam', inplace=True)
tn_districts_gdf.replace(['Virudunagar'], 'Virudhunagar', inplace=True)
# Merge the tn_districts geometry information into enrollement df.

enroll_school_df = enroll_school_df.merge(tn_districts_gdf[['District', 'geometry']], on='District')
len(enroll_school_df)
enroll_school_df.columns
import numpy as np



def draw_map(df, column, annotation_column='District', figsize=(20, 20)):

    ax = df.plot(column=column, legend=True, figsize=figsize)

    if df[column].dtype == np.float:

        _ = df.apply(lambda x: ax.annotate(s=f"{x[annotation_column]}: {x[column]:.3f}", 

                                           xy=x.geometry.centroid.coords[0], ha='center'),axis=1)

    else:

        _ = df.apply(lambda x: ax.annotate(s=f"{x[annotation_column]}: {x[column]}", 

                                           xy=x.geometry.centroid.coords[0], ha='center'),axis=1)
from geopandas import GeoDataFrame

# We need geodataframe to plot, so maintain a copy of the dataframe

enroll_school_gdf = GeoDataFrame(enroll_school_df)

draw_map(enroll_school_gdf, column='Grand Total')
# Compute Grand Total Girls Vs Grand Total Boys Ratio
grand_gender_ratio = enroll_school_df['Grand Total Girls'] / enroll_school_df['Grand Total Boys'].astype(float)

enroll_school_df['Grand Gender Ratio'] = grand_gender_ratio

enroll_school_gdf['Grand Gender Ratio'] = grand_gender_ratio
# How many Girls are enrolled?



girls = enroll_school_df['Grand Total Girls'].sum()

print(girls)
# How many boys are enrolled?

boys = enroll_school_df['Grand Total Boys'].sum()

print(boys)
total = enroll_school_df['Grand Total'].sum()

print(f'Total % of girls enrolled is {(girls/total) * 100}')

print(f'Total % of boys enrolled is {(boys/total) * 100}')
draw_bar(enroll_school_df, x_column='District', y_column='Grand Gender Ratio', 

         title='District Wise Enrollment Gender Ratio', figsize=(16, 7))
# All the districts grand gender ratio is > 0.8
def print_district_info(df, districts, cols_to_print):

    print(df[df['District'].isin(districts)][cols_to_print])
# Top 3 Districts: Thiruvarur, Thoothukkudi, Chennai

# Least 3 Districts: Nammakkal, Perambalur, Dharmapuri
print_district_info(enroll_school_df, districts=['Thiruvarur', 'Thoothukkudi', 'Chennai'], 

                    cols_to_print=['District', 'Grand Gender Ratio'])
print_district_info(enroll_school_df, districts=['Namakkal', 'Perambalur', 'Dharmapuri'], 

                    cols_to_print=['District', 'Grand Gender Ratio'])
# Let's do Histogram of the ratio
def hist(df, column, title="Histogram", figsize=(5, 5)):

    df[column].plot.hist(figsize=figsize, title=title)
hist(enroll_school_df, 'Grand Gender Ratio', figsize=(7, 7), title='Enrollment Grand Gender Ratio')
enroll_school_df['Grand Gender Ratio'].describe()
# Average Gender ratio among the districts is 0.96

# Lowest Gender ration among the districts is 0.90
# What are the districts with greater than or equal to 1 as gender ratio
enroll_school_df[enroll_school_df['Grand Gender Ratio'] >= 1]['District']
# What are the districts with greater than or equal to 0.95 as gender ratio

enroll_school_df[enroll_school_df['Grand Gender Ratio'] >= 0.95]['District']
# Plot the grand gender ratio in Map
draw_map(enroll_school_gdf, column='Grand Gender Ratio')
enroll_school_df.columns
# Govt Girls vs Govt Boys Ratio

govt_gender_ratio = enroll_school_df['Govt Girls'] / enroll_school_df['Govt Boys'].astype(float)

enroll_school_df['Govt Gender Ratio'] = govt_gender_ratio

enroll_school_gdf['Govt Gender Ratio'] = govt_gender_ratio
# Describe the govt gender ratio

enroll_school_df['Govt Gender Ratio'].describe()
draw_bar(enroll_school_df, x_column='District', y_column='Govt Gender Ratio', 

         title='District Wise Enrollment Govt School Gender Ratio', figsize=(16, 7))
# All the districts govt gender ratio is > 0.8

enroll_school_df[enroll_school_df['Govt Gender Ratio'] > 1]['District']
# Highest three districts: Tirunelveli, Thoothukudi, Chennai

# Lowest three districts: The Nilgris, Kanniyakumari, Ariyalur
print_district_info(enroll_school_df, districts=['Tirunelveli', 'Thoothukkudi', 'Chennai'], 

                    cols_to_print=['District', 'Govt Gender Ratio'])
print_district_info(enroll_school_df, districts=['The Nilgiris', 'Kanniyakumari', 'Ariyalur'], 

                    cols_to_print=['District', 'Grand Gender Ratio'])
draw_map(enroll_school_gdf, column='Govt Gender Ratio')
# Private Aided Boys vs Private Aided Girls Ratio

gender_ratio = enroll_school_df['Private Aided Girls'] / enroll_school_df['Private Aided Boys'].astype(float)

enroll_school_df['Private Aided Gender Ratio'] = gender_ratio

enroll_school_gdf['Private Aided Gender Ratio'] = gender_ratio
# Describe the private aided gender ratio

enroll_school_df['Private Aided Gender Ratio'].describe()
draw_bar(enroll_school_df, x_column='District', y_column='Private Aided Gender Ratio', 

         title='District Wise Enrollment Private Aided School Gender Ratio', figsize=(16, 7))
# Highest 3 districts: The Nilgris, Ariyalur, Karur

# Lowest 3 districts: Dharmapuri, Tiruvannamalai, Pudukkotai
print_district_info(enroll_school_df, districts=['The Nilgiris', 'Ariyalur', 'Karur'], 

                    cols_to_print=['District', 'Private Aided Gender Ratio'])
print_district_info(enroll_school_df, districts=['Dharmapuri', 'Tiruvannamalai', 'Pudukkottai'], 

                    cols_to_print=['District', 'Private Aided Gender Ratio'])
draw_map(enroll_school_gdf, column='Private Aided Gender Ratio')
# Private UnAided Boys vs Private UnAided Girls Ratio

gender_ratio = enroll_school_df['Private Unaided Girls'] / enroll_school_df['Private Unaided Boys'].astype(float)

enroll_school_df['Private Unaided Gender Ratio'] = gender_ratio

enroll_school_gdf['Private Unaided Gender Ratio'] = gender_ratio
# Describe the private aided gender ratio

enroll_school_df['Private Unaided Gender Ratio'].describe()
draw_bar(enroll_school_df, x_column='District', y_column='Private Unaided Gender Ratio', 

         title='District Wise Enrollment Private Unaided School Gender Ratio', figsize=(16, 7))
# Highest Ratio: Chennai, Coimbatore, Kanniyakumari

# Lowest Ratio: Ariyalur, Dharmapuri, Perambalur
print_district_info(enroll_school_df, districts=['Chennai', 'Coimbatore', 'Kanniyakumari'], 

                    cols_to_print=['District', 'Private Unaided Gender Ratio'])
print_district_info(enroll_school_df, districts=['Ariyalur', 'Dharmapuri', 'Perambalur'], 

                    cols_to_print=['District', 'Private Unaided Gender Ratio'])
# No district has ratio > 1.
enroll_school_df[enroll_school_df['Private Unaided Gender Ratio'] > 1]['District']
draw_map(enroll_school_gdf, column='Private Unaided Gender Ratio')
### Describe on all the gender ratio

gender_cols = ['Govt Gender Ratio', 'Private Unaided Gender Ratio', 'Private Aided Gender Ratio']

enroll_school_df[gender_cols].describe()
# Govt Gender Ratio, Private Aided Gender Ratio has mean ratio greater than 1.
# Let's draw district wise line chart for ratio
def draw_line_chart(df, x, y, title='Line Chart', figsize=(10, 10), subplots=False):

    df.plot.line(x=x, y=y, title=title, figsize=figsize, subplots=subplots)
draw_line_chart(enroll_school_df, x='District', y=gender_cols, title='District wise gender ratio',

               figsize=(15, 7))
def draw_scatter_chart(df, x, y, title='Line Chart', figsize=(10, 10), subplots=False):

    df.plot.scatter(x=x, y=y, title=title, figsize=figsize, subplots=subplots)
# Let's draw bar graph with all four gender cols in 5 different bar charts

draw_bar(enroll_school_df[:6], x_column='District', y_column=gender_cols, title='District wise gender ratio',

               figsize=(15, 7), stacked=False)
draw_bar(enroll_school_df[6:12], x_column='District', y_column=gender_cols, title='District wise gender ratio',

               figsize=(15, 7), stacked=False)
draw_bar(enroll_school_df[12:18], x_column='District', y_column=gender_cols, title='District wise gender ratio',

               figsize=(15, 7), stacked=False)
draw_bar(enroll_school_df[18:24], x_column='District', y_column=gender_cols, title='District wise gender ratio',

               figsize=(15, 7), stacked=False)
draw_bar(enroll_school_df[24:], x_column='District', y_column=gender_cols, title='District wise gender ratio',

               figsize=(15, 7), stacked=False)
# Let's calculate the variance for each district on gender columns

gender_variance = {}

for idx, row in enroll_school_df.iterrows():

    dist = row['District']

    gender_variance[dist] = row[gender_cols].values.var()
gender_variance
# Lower the variance, uniform the ratio
var_df = pd.DataFrame({'District': list(gender_variance.keys()), 'Variance': list(gender_variance.values())})
var_df.head()
draw_bar(var_df, x_column='District', y_column='Variance', figsize=(16, 7))
# Add variance to each district

enroll_school_gdf['Variance'] = [np.nan] * len(enroll_school_gdf)

for idx, row in var_df.iterrows():

    enroll_school_gdf.loc[enroll_school_gdf['District'] == row['District'], 'Variance'] = row['Variance']
enroll_school_gdf.columns
enroll_school_gdf['Variance'] = enroll_school_gdf['Variance'].astype(float)

draw_map(enroll_school_gdf, column='Variance')
enroll_school_gdf['Variance'].describe()
# The Nilgris has highest gender ratio difference in different school types.
enroll_school_df[enroll_school_df['District'] == 'The Nilgiris'][gender_cols]
### How much is the grand enrollment percentage numbers?

govt_total = enroll_school_df['Govt Total'].sum()

private_aided_total = enroll_school_df['Private Aided Total'].sum()

private_unaided_total = enroll_school_df['Private Unaided Total'].sum()



labels = ['Govt Total', 'Private Aided Total', 'Private Unaided Total']

x = pd.DataFrame({'labels': labels,

                  'data': [govt_total, private_aided_total, private_unaided_total]}, index=labels)
x.head()
x.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', title='Enrollment ratio across different of schools')
# 1. 43.09% enrollments in Private Unaided Total

# 2. 19.8% enrollments in Private Aided Total

# 3. 62.89% enrollments in Private School

# 4. 37.80% enrollment in Govt School
### How much is the girls enrollment percentage numbers?

govt_girls_total = enroll_school_df['Govt Girls'].sum()

private_aided_girls_total = enroll_school_df['Private Aided Girls'].sum()

private_unaided_girls_total = enroll_school_df['Private Unaided Girls'].sum()
labels = ['Govt Girls', 'Private Aided Girls', 'Private Unaided Girls']

y_girls = pd.DataFrame({'labels': labels,

                  'data': [govt_girls_total, private_aided_girls_total, private_unaided_girls_total]},

                 index=labels)

y_girls.head()
y_girls.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 

               title='Enrollment girls ratio across different of type of schools')
# 1. 21.09% girls enrolled in Private Aided School

# 2. 39.70% girls enrolled in Private Unaided School

# 3. 60.79% girls enrolled in Private School

# 4. 39.11% girls enrolled in Govt School
### How much is the boys enrollment percentage numbers?

govt_boys_total = enroll_school_df['Govt Boys'].sum()

private_aided_boys_total = enroll_school_df['Private Aided Boys'].sum()

private_unaided_boys_total = enroll_school_df['Private Unaided Boys'].sum()
labels = ['Govt Boys', 'Private Aided Boys', 'Private Unaided Boys']

y_boys = pd.DataFrame({'labels': labels,

                  'data': [govt_boys_total, private_aided_boys_total, private_unaided_boys_total]},

                 index=labels)

y_boys.head()
y_boys.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 

           title='Enrollment boys ratio across different of type of schools')
# 1. 18.54% boys enrolled in Private Aided School

# 2. 46.38% boys enrolled in Private Unaided School

# 3. 64.92% boys enrolled in Private School

# 4. 35.08% boys enrolled in Govt School
girls = y_girls['data'].sum()

boys = y_boys['data'].sum()

labels = ['Girls', 'Boys']

data = [girls, boys]

y = pd.DataFrame({'labels': labels, 'data': data}, index=labels)

y.head()
y.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 

           title='Enrollment girls vs boys')
# Overall Girls and boys enrollment are same
y_girls.columns
girls = enroll_school_df['Private Aided Girls'].sum()

boys = enroll_school_df['Private Aided Boys'].sum()

labels = ['Girls', 'Boys']

data = [girls, boys]

y = pd.DataFrame({'labels': labels, 'data': data}, index=labels)

y.head()
y.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 

           title='Private Aided Enrollment girls vs boys')
# In private Aided girls enrollement is higher than boys
girls = enroll_school_df['Private Unaided Girls'].sum()

boys = enroll_school_df['Private Unaided Boys'].sum()

labels = ['Girls', 'Boys']

data = [girls, boys]

y = pd.DataFrame({'labels': labels, 'data': data}, index=labels)

y.head()

y.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 

           title='Private Unaided Enrollment girls vs boys')
# In private unaided, there is significant difference in girls vs boys enrollment. Difference is ~10%
# Let's see the enrollment genderwise data see if any other data is available



enroll_gender_df = pd.read_csv(f'{input_dir}/enrollment_genderwise_0.csv')
enroll_gender_df.info()
# this dataset has primary school, middle school, high school, higher secondary school data district wise, 

# but missing private/govt split.
enroll_gender_df['District'].values
# remove district name with `nan` or select the district from previous dataframe
len(enroll_gender_df)
enroll_gender_df.replace(['Krishanagiri'], 'Krishnagiri', inplace=True)

enroll_gender_df.replace(['Nagapattinam'], 'Nagappattinam', inplace=True)
len(enroll_gender_df)
districts = enroll_school_df['District'].values

enroll_gender_df = enroll_gender_df[enroll_gender_df['District'].isin(districts)]
len(enroll_school_df), len(enroll_gender_df)
enroll_gender_df.head()
# We had already seen District wise Grand Total students enrollment. Let's skip
enroll_gender_df.columns
# See District Primary School Total students enrollment 

draw_bar(enroll_gender_df, x_column='District', y_column='Primary School Total', 

         title='District wise Primary School Enrollment', figsize=(16, 7))
# Highest Primary School Enrollment in Vellore, Vilippuram, Tirunnelveli

# Lowest Primary School Enrollment in The Nilgris, Perambalur, Karur
print_district_info(enroll_gender_df, districts=['Vellore', 'Viluppuram', 'Tirunelveli'], 

                    cols_to_print=['District', 'Primary School Total'])
print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Karur'], 

                    cols_to_print=['District', 'Primary School Total'])
enroll_gender_df = enroll_gender_df.merge(tn_districts_gdf[['District', 'geometry']], on='District')

enroll_gender_gdf =  GeoDataFrame(enroll_gender_df)

len(enroll_gender_df), len(enroll_gender_gdf)
draw_map(enroll_gender_gdf, 'Primary School Total')
# Girls Enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='Primary School Girls', 

         title='District wise Primary School Girls Enrollment', figsize=(16, 7))
# Highest Girls Enrollment: Vellore, Tirunnelveli, Viluppuram

# Lowest Girls Enrollment: The Nilgris, Perambalur, Ariyalur 
print_district_info(enroll_gender_df, districts=['Vellore', 'Viluppuram', 'Tirunelveli'], 

                    cols_to_print=['District', 'Primary School Girls'])
print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Ariyalur'], 

                    cols_to_print=['District', 'Primary School Total'])
draw_map(enroll_gender_gdf, 'Primary School Girls')
# Boys Enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='Primary School Boys', 

         title='District wise Primary School Boys Enrollment', figsize=(16, 7))
# Highest boys enrollment: Vellore, Viluppuram, Tirunelveli

# Lowest boys enrollment: The Nilgris, Perambalur, Karur
print_district_info(enroll_gender_df, districts=['Vellore', 'Viluppuram', 'Tirunelveli'], 

                    cols_to_print=['District', 'Primary School Boys'])
print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Karur'], 

                    cols_to_print=['District', 'Primary School Boys'])
draw_map(enroll_gender_gdf, 'Primary School Boys')
# Pie chart: See Primary School Girls vs Boys enrollment 
enroll_gender_df[['Primary School Boys', 'Primary School Girls']].sum().plot.pie(autopct="%.2f")
# See Grand Total Girls vs Grand Total Boys enrollment Ratio

gender_ratio = enroll_gender_df['Primary School Girls'] / enroll_gender_df['Primary School Boys'].astype(float)

enroll_gender_df['Primary School Gender Ratio'] = gender_ratio

enroll_gender_gdf['Primary School Gender Ratio'] = gender_ratio
enroll_gender_df['Primary School Gender Ratio'].describe()
# Districts with gender >= 1

enroll_gender_df[enroll_gender_df['Primary School Gender Ratio'] >= 1]['District']
# 9 districts ratio is greater than or equal  to 1
draw_bar(enroll_gender_df, 'District', 'Primary School Gender Ratio', 

         title='District wise Primary School Gender Ratio', figsize=(16, 7))
# highest gender ratio: Kanniyakumari, the nilgris, Kancheepuram

# lowest gender ratio: Ariyalur, Cuddalore, Perambalur
print_district_info(enroll_gender_df, districts=['Kanniyakumari', 'The Nilgiris', 'Kancheepuram'], 

                    cols_to_print=['District', 'Primary School Gender Ratio'])
print_district_info(enroll_gender_df, districts=['Ariyalur', 'Cuddalore', 'Perambalur'], 

                    cols_to_print=['District', 'Primary School Gender Ratio'])
draw_map(enroll_gender_gdf, 'Primary School Gender Ratio')
# See District Middle School Total students enrollment 
draw_bar(enroll_gender_df, x_column='District', y_column='Middle School Total', 

         title='District wise Middle School Enrollment', figsize=(16, 7))
# Highest Primary School Enrollment in Vilippuram, Kancheepuram, Tirunnelveli

# Lowest Primary School Enrollment in The Nilgris, Perambalur, Ariyalur
print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Viluppuram', 'Tirunelveli'], 

                    cols_to_print=['District', 'Middle School Total'])
print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Ariyalur'], 

                    cols_to_print=['District', 'Middle School Total'])
draw_map(enroll_gender_gdf, 'Middle School Total')
# Girls Enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='Middle School Girls', 

         title='District wise Middle School Girls Enrollment', figsize=(16, 7))
# Highest girls enrollment: Viluppuram, Kancheepuram, Vellore

# Lowest girls enrollment: The Nilgris, Perambualur, Ariyalur
print_district_info(enroll_gender_df, districts=['Vellore', 'Viluppuram', 'Kancheepuram'], 

                    cols_to_print=['District', 'Middle School Girls'])
print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Ariyalur'], 

                    cols_to_print=['District', 'Middle School Girls'])
draw_map(enroll_gender_gdf, 'Middle School Girls')
# boys Enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='Middle School Boys', 

         title='District wise Middle School Boys Enrollment', figsize=(16, 7))
# Highest boys enrollment: Viluppuram, Kancheepuram, Tirunelveli

# Lowest boys enrollment: The Nilgris, Perambualur, Ariyalur
print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Viluppuram', 'Tirunelveli'], 

                    cols_to_print=['District', 'Middle School Boys'])
print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Ariyalur'], 

                    cols_to_print=['District', 'Middle School Boys'])
draw_map(enroll_gender_gdf, 'Middle School Boys')
# Pie chart: See Middle School Girls vs Boys enrollment 
enroll_gender_df[['Middle School Girls', 'Middle School Boys']].sum().plot.pie(autopct="%.2f")
# See Grand Total Girls vs Grand Total Boys enrollment Ratio

gender_ratio = enroll_gender_df['Middle School Girls'] / enroll_gender_df['Middle School Boys'].astype(float)

enroll_gender_df['Middle School Gender Ratio'] = gender_ratio

enroll_gender_gdf['Middle School Gender Ratio'] = gender_ratio
enroll_gender_df['Middle School Gender Ratio'].describe()
# Districts with gender >= 1

enroll_gender_df[enroll_gender_df['Middle School Gender Ratio'] >= 1]['District']
# 12 districts ratio is greater than or equal  to 1
draw_bar(enroll_gender_df, 'District', 'Middle School Gender Ratio', 

         title='District wise Middle School Gender Ratio', figsize=(16, 7))
# highest gender ratio: Viluppuram, Dindigul, Thiruvarur

# lowest gender ratio: Chennai, Kanniyakumari, Tiruppur
print_district_info(enroll_gender_df, districts=['Dindigul', 'Viluppuram', 'Tirunelveli'], 

                    cols_to_print=['District', 'Middle School Gender Ratio'])
print_district_info(enroll_gender_df, districts=['Chennai', 'Kanniyakumari', 'Tiruppur'], 

                    cols_to_print=['District', 'Middle School Gender Ratio'])
draw_map(enroll_gender_gdf, 'Middle School Gender Ratio')
# See High School enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='High School Total', 

         title='District wise High School Enrollment', figsize=(16, 7))
# Highest Primary School Enrollment in Kancheepuram, Thiruvallur, Vellore

# Lowest Primary School Enrollment in Perambalur, Theni, Ariyalur
print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Thiruvallur', 'Vellore'], 

                    cols_to_print=['District', 'High School Total'])
print_district_info(enroll_gender_df, districts=['Perambalur', 'Theni', 'Ariyalur'], 

                    cols_to_print=['District', 'High School Total'])
draw_map(enroll_gender_gdf, 'High School Total')
# Girls Enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='High School Girls', 

         title='District wise High School Girls Enrollment', figsize=(16, 7))
# Highest Primary School Enrollment in Kancheepuram, Thiruvallur, Vellore

# Lowest Primary School Enrollment in Perambalur, Theni, Ariyalur
print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Thiruvallur', 'Vellore'], 

                    cols_to_print=['District', 'High School Girls'])
print_district_info(enroll_gender_df, districts=['Perambalur', 'Theni', 'Ariyalur'], 

                    cols_to_print=['District', 'High School Girls'])
draw_map(enroll_gender_gdf, 'High School Girls')
# Boys Enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='High School Boys', 

         title='District wise High School Boys Enrollment', figsize=(16, 7))
# Highest Primary School Enrollment in Kancheepuram, Thiruvallur, Vellore

# Lowest Primary School Enrollment in Perambalur, theni, Ariyalur
print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Vellore', 'Thiruvallur'], 

                    cols_to_print=['District', 'High School Boys'])
print_district_info(enroll_gender_df, districts=['Perambalur', 'Theni', 'Ariyalur'], 

                    cols_to_print=['District', 'High School Boys'])
draw_map(enroll_gender_gdf, 'High School Boys')
# Pie chart: See High School Girls vs Boys enrollment 
enroll_gender_df[['High School Girls', 'High School Boys']].sum().plot.pie(autopct="%.2f")
# See Grand Total Girls vs Grand Total Boys enrollment Ratio

gender_ratio = enroll_gender_df['High School Girls'] / enroll_gender_df['High School Boys'].astype(float)

enroll_gender_df['High School Gender Ratio'] = gender_ratio

enroll_gender_gdf['High School Gender Ratio'] = gender_ratio
enroll_gender_df['High School Gender Ratio'].describe()
# Districts with gender >= 1

enroll_gender_df[enroll_gender_df['High School Gender Ratio'] >= 1]['District']
# There is no district in high school enrollment women outnumber men
draw_bar(enroll_gender_df, 'District', 'High School Gender Ratio', 

         title='District wise High School Gender Ratio', figsize=(16, 7))
# highest gender ratio: Nagapattinam, Erode, Pudukkotai

# lowest gender ratio: Madurai, Theni, Thiruvarur
print_district_info(enroll_gender_df, districts=['Nagappattinam', 'Erode', 'Pudukkottai'], 

                    cols_to_print=['District', 'High School Gender Ratio'])
print_district_info(enroll_gender_df, districts=['Madurai', 'Theni', 'Thiruvarur'], 

                    cols_to_print=['District', 'High School Gender Ratio'])
draw_map(enroll_gender_gdf, 'High School Gender Ratio')
enroll_gender_df.columns
# See Hr.Secondary School enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='Hr.Secondary School Total', 

         title='District wise Hr.Secondary School Enrollment', figsize=(16, 7))
# Highest Primary School Enrollment in Chennai, Kancheepuram, Thiruvallur 

# Lowest Primary School Enrollment in Ariyalur, Perambalur, The Nilgris,
print_district_info(enroll_gender_df, districts=['Chennai', 'Kancheepuram', 'Thiruvallur'], 

                    cols_to_print=['District', 'Hr.Secondary School Total'])
print_district_info(enroll_gender_df, districts=['Ariyalur', 'Perambalur', 'The Nilgiris'], 

                    cols_to_print=['District', 'Hr.Secondary School Total'])
draw_map(enroll_gender_gdf, 'Hr.Secondary School Total')
# Girls Enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='Hr.Secondary School Girls', 

         title='District wise Hr.Secondary School Girls Enrollment', figsize=(16, 7))
# Highest Primary School Enrollment in Chennai, Kancheepuram, Thiruvallur

# Lowest Primary School Enrollment in Perambalur, Ariyalur, The Nilgris
print_district_info(enroll_gender_df, districts=['Chennai', 'Kancheepuram', 'Thiruvallur'], 

                    cols_to_print=['District', 'Hr.Secondary School Girls'])
print_district_info(enroll_gender_df, districts=['Perambalur', 'Ariyalur', 'The Nilgiris'], 

                    cols_to_print=['District', 'Hr.Secondary School Girls'])
draw_map(enroll_gender_gdf, 'Hr.Secondary School Girls')
# Boys Enrollment
draw_bar(enroll_gender_df, x_column='District', y_column='Hr.Secondary School Boys', 

         title='District wise Hr.Secondary School Boys Enrollment', figsize=(16, 7))
# Highest Primary School Enrollment in Chennai, Kancheepuram, Thiruvallur

# Lowest Primary School Enrollment in Ariyalur, The Nilgris, Perambalur
print_district_info(enroll_gender_df, districts=['Chennai', 'Kancheepuram', 'Thiruvallur'], 

                    cols_to_print=['District', 'Hr.Secondary School Boys'])
print_district_info(enroll_gender_df, districts=['Ariyalur', 'The Nilgiris', 'Perambalur'], 

                    cols_to_print=['District', 'Hr.Secondary School Boys'])
draw_map(enroll_gender_gdf, 'Hr.Secondary School Boys')
# Pie chart: See Hr.Secondary School Girls vs Boys enrollment 
enroll_gender_df[['Hr.Secondary School Girls', 'Hr.Secondary School Boys']].sum().plot.pie(autopct="%.2f")
# See Grand Total Girls vs Grand Total Boys enrollment Ratio

gender_ratio = enroll_gender_df['Hr.Secondary School Girls'] / enroll_gender_df['Hr.Secondary School Boys'].astype(float)

enroll_gender_df['Hr.Secondary School Gender Ratio'] = gender_ratio

enroll_gender_gdf['Hr.Secondary School Gender Ratio'] = gender_ratio
enroll_gender_df['Hr.Secondary School Gender Ratio'].describe()
# Districts with gender >= 1

enroll_gender_df[enroll_gender_df['Hr.Secondary School Gender Ratio'] >= 1]['District']
# 13 districts ratio is greater than or equal  to 1
draw_bar(enroll_gender_df, 'District', 'Hr.Secondary School Gender Ratio', 

         title='District wise Hr.Secondary School Gender Ratio', figsize=(16, 7))
# highest gender ratio: Ariyalur, Thothukkudi, Thiruvarur

# lowest gender ratio: Dharmapuri, Permabalur, Nammakkal
print_district_info(enroll_gender_df, districts=['Ariyalur', 'Thoothukkudi', 'Thiruvarur'], 

                    cols_to_print=['District', 'Hr.Secondary School Gender Ratio'])
print_district_info(enroll_gender_df, districts=['Dharmapuri', 'Perambalur', 'Namakkal'], 

                    cols_to_print=['District', 'Hr.Secondary School Gender Ratio'])
draw_map(enroll_gender_gdf, 'Hr.Secondary School Gender Ratio')
## Describe on all gender ratio columns

enroll_gender_df[['Primary School Gender Ratio', 'Middle School Gender Ratio',

                  'High School Gender Ratio', 'Hr.Secondary School Gender Ratio']].describe()
# 1. Mean is lowest in High School Gender ratio, the difference is significant. 

# But picks up in Higher Secondary school. 

# So lot of people drop in high school, those survive continue in higher secondary. 

# Is the data for high school corruput?



# 2. Std deviation for High School ratio is low, this pattern is common in all districts in TN.



# 3. there is not even a single district in TN, where high school enrollment ratio is greater than or equal to 1.



# Hr.Secondary Ratio

# highest gender ratio: Ariyalur, Thothukkudi, Thiruvarur

# lowest gender ratio: Dharmapuri, Permabalur, Nammakkal



# High School Ratio

# highest gender ratio: Nagapattinam, Erode, Pudukkotai

# lowest gender ratio: Madurai, Theni, Thiruvarur



# Middle School Ratio

# highest gender ratio: Viluppuram, Dindigul, Thiruvarur

# lowest gender ratio: Chennai, Kanniyakumari, Tiruppur



# Primary School Ratio

# highest gender ratio: Kanniyakumari, the nilgris, Kancheepuram

# lowest gender ratio: Ariyalur, Cuddalore, Perambalur



# Ariyalur which has low primary school gender ratio, has highest hr secondary school gender ratio

# Chennai has lowest middle school gender ratio

# Madurai, Theni shows up  lowest gender ratio in middle school

# thiruvarur which is one of the highest gender ratio in Hr.Secondary, Middle School has lowest gender ratio 

# in High School
### Teacher information
teacher_df = pd.read_csv(f'{input_dir}/no.ofteachers_0.csv')
teacher_df.info()
teacher_df['District'].values
teacher_df['District'].values[:32]
teacher_df.replace(['Krishanagiri'], 'Krishnagiri', inplace=True)
teacher_df.replace(['Nagapattinam'], 'Nagappattinam', inplace=True)
len(teacher_df)
teacher_df = teacher_df[teacher_df['District'].isin(enroll_school_df['District'])]
len(teacher_df)
teacher_df.head()
# Rename the column, and merge it with enroll_school_df, and enroll_school_gdf

teacher_df.rename(columns={'Govt': 'Govt Teachers', 'Pvt Aided': 'Private Aided Teachers',

                           'Pvt Unaided': 'Private Unaided Teachers'}, inplace=True)
teacher_df.columns
teacher_df.head()
len(enroll_school_df), len(enroll_school_df.columns)
enroll_school_df = enroll_school_df.merge(teacher_df[['District', 'Govt Teachers', 'Private Aided Teachers',

                                   'Private Unaided Teachers', 'Total Teachers']], on='District')
len(enroll_school_df), len(enroll_school_df.columns)
enroll_school_df.head()
enroll_school_gdf = enroll_school_gdf.merge(teacher_df[['District', 'Govt Teachers', 'Private Aided Teachers',

                                                        'Private Unaided Teachers', 'Total Teachers']],

                                            on='District')
enroll_school_df.columns
# Calculate the Students vs Govt Teachers Ratio

ratio = enroll_school_df['Govt Total'] / enroll_school_df['Govt Teachers']

enroll_school_df['Govt Student Teacher Ratio'] = ratio

enroll_school_gdf['Govt Student Teacher Ratio'] = ratio



ratio = enroll_school_df['Private Aided Total'] / enroll_school_df['Private Aided Teachers']

enroll_school_df['Private Aided Student Teacher Ratio'] = ratio

enroll_school_gdf['Private Aided Student Teacher Ratio'] = ratio



ratio = enroll_school_df['Private Unaided Total'] / enroll_school_df['Private Unaided Teachers']

enroll_school_df['Private Unaided Student Teacher Ratio'] = ratio

enroll_school_gdf['Private Unaided Student Teacher Ratio'] = ratio



ratio = enroll_school_df['Grand Total'] / enroll_school_df['Total Teachers']

enroll_school_df['Grand Student Teacher Ratio'] = ratio

enroll_school_gdf['Grand Student Teacher Ratio'] = ratio



# Govt Student Teacher Ratio



enroll_school_df['Govt Student Teacher Ratio'].describe()
draw_bar(enroll_school_df, 'District', y_column='Govt Student Teacher Ratio',

         title='Govt Student Teacher Ratio', figsize=(16, 7))
# Lower the ratio better it is

# Better ratio: The Nilgiris, Sivaganga, Ramnathapuram

# Higher ratio: Villuppuram, Krishnagiri, Tiruvannamalai
print_district_info(enroll_school_df, districts=['The Nilgiris', 'Sivaganga', 'Ramanathapuram'], 

                    cols_to_print=['District', 'Govt Student Teacher Ratio'])
print_district_info(enroll_school_df, districts=['Viluppuram', 'Krishnagiri', 'Tiruvannamalai'], 

                    cols_to_print=['District', 'Govt Student Teacher Ratio'])
draw_map(enroll_school_gdf, 'Govt Student Teacher Ratio')
# Districts with ratio less than mean.



enroll_school_df[enroll_school_df['Govt Student Teacher Ratio'] <= 19.3]['District']
# 16 districts
# see 'Private Aided Student Teacher Ratio'

col = 'Private Aided Student Teacher Ratio'
enroll_school_df[col].describe()
# Mean/Std is worse.
draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))
# Lower the better

# Better: KanniyaKumari, The Nilgris, Namakkal

# Highest: Thiruvallur, Vellore, Krishnagiri
print_district_info(enroll_school_df, districts=['Kanniyakumari', 'The Nilgiris', 'Namakkal'], 

                    cols_to_print=['District', 'Private Aided Student Teacher Ratio'])
print_district_info(enroll_school_df, districts=['Thiruvallur', 'Vellore', 'Krishnagiri'], 

                    cols_to_print=['District', 'Private Aided Student Teacher Ratio'])
draw_map(enroll_school_gdf, col)
enroll_school_df[enroll_school_df[col] <= 30.31]['District']
#18 districts less than or equal to mean
# see 'Private Unaided Student Teacher Ratio'

col = 'Private Unaided Student Teacher Ratio'
enroll_school_df[col].describe()
enroll_school_df[enroll_school_df[col] <= 19.5]['District']
#20 Districts
draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))
# Lower the better

# Better: Theni, Karur, Erode

# Highest: Kancheepuram, thiruvallur, Viluppuram
print_district_info(enroll_school_df, districts=['Theni', 'Karur', 'Erode'], 

                    cols_to_print=['District', 'Private Unaided Student Teacher Ratio'])
print_district_info(enroll_school_df, districts=['Kancheepuram', 'Thiruvallur', 'Viluppuram'], 

                    cols_to_print=['District', 'Private Unaided Student Teacher Ratio'])
draw_map(enroll_school_gdf, col)
# See 'Grand Student Teacher Ratio'

col = 'Grand Student Teacher Ratio'
enroll_school_df[col].describe()
enroll_school_df[enroll_school_df[col] <= 20.99]['District']
#14 districts
draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))
# Lower is better

# Better: The nilgris, Karur, KanniyaKumari

# Highest: Kancheepuram, Viluppuram, Chennai
print_district_info(enroll_school_df, districts=['The Nilgiris', 'Karur', 'Kanniyakumari'], 

                    cols_to_print=['District', 'Grand Student Teacher Ratio'])
print_district_info(enroll_school_df, districts=['Kancheepuram', 'Viluppuram', 'Chennai'], 

                    cols_to_print=['District', 'Grand Student Teacher Ratio'])
cols = ['Govt Student Teacher Ratio', 'Private Aided Student Teacher Ratio',

       'Private Unaided Student Teacher Ratio', 'Grand Student Teacher Ratio']

enroll_school_df[cols].describe()
# Govt school student teacher ratio is lowest 19.3

# Private Aided Student teacher ratio is highest 30.3

# Private Unaided student teacher ratio is close to govt school

# Lowest std deviation is Private Unaided school.

# TN Student Teacher Ratio is 21.



# Govt School



# Better ratio: The Nilgiris, Sivaganga, Ramnathapuram

# Higher ratio: Villuppuram, Krishnagiri, Tiruvannamalai



# Private Aided



# Better: KanniyaKumari, The Nilgris, Namakkal

# Highest: Thiruvallur, Vellore, Krishnagiri



# Private Unaided 



# Better: Theni, Karur, Erode

# Highest: Kancheepuram, thiruvallur, Viluppuram



# Overall

# Better: The nilgris, Karur, KanniyaKumari

# Highest: Kancheepuram, Viluppuram, Chennai
management_df = pd.read_csv(f'{input_dir}/managementwise_schools_0.csv')
management_df.columns
management_df.head()
management_df['District'] = management_df['District'].apply(lambda x: x.title())
management_df['District'].values
# Rename values

management_df.replace(['Krishanagiri'], 'Krishnagiri', inplace=True)

management_df.replace(['Nagapattinam'], 'Nagappattinam', inplace=True)
len(management_df)
# Filter the valid districts
management_df = management_df[management_df['District'].isin(enroll_school_df['District'])]
len(management_df)
management_df.columns
# Rename columns

management_df.rename(columns={'Govt': 'Govt Schools',

                              'Pvt Aided': 'Private Aided Schools',

                              'Pvt Unaided': 'Private Unaided Schools',

                              'Grand Total': 'Grand Total Schools'}, inplace=True)
management_df.columns
len(enroll_school_df.columns)
# Merge with scholl enrollment df and gdf

enroll_school_df = enroll_school_df.merge(management_df[['District', 'Govt Schools',

                                                         'Private Aided Schools',

                                                         'Private Unaided Schools',

                                                         'Grand Total Schools']], on='District')
len(enroll_school_df.columns)
len(enroll_school_df)
# Merge with scholl enrollment df and gdf

enroll_school_gdf = enroll_school_gdf.merge(management_df[['District', 'Govt Schools',

                                                         'Private Aided Schools',

                                                         'Private Unaided Schools',

                                                         'Grand Total Schools']], on='District')
# See districts with more schools
draw_bar(enroll_school_df, 'District', 'Grand Total Schools', 'Districtwise Total Schools', figsize=(16, 7))
# Highest Schools in: Vellore, Villupuram, Tirunelveli

# Lowest Schools in: Perambalur, The Nilgris, Ariyalur
print_district_info(enroll_school_df, districts=['Vellore', 'Viluppuram', 'Tirunelveli'], 

                    cols_to_print=['District', 'Grand Total Schools'])
print_district_info(enroll_school_df, districts=['Perambalur', 'The Nilgiris', 'Ariyalur'], 

                    cols_to_print=['District', 'Grand Total Schools'])
# this information is not that useful because, let's take student school ratio, teacher school ratio
enroll_school_df.columns
# Calculate the Students vs School Ratio

ratio = enroll_school_df['Govt Total'] / enroll_school_df['Govt Schools']

enroll_school_df['Govt Student School Ratio'] = ratio

enroll_school_gdf['Govt Student School Ratio'] = ratio



ratio = enroll_school_df['Private Aided Total'] / enroll_school_df['Private Aided Schools']

enroll_school_df['Private Aided Student School Ratio'] = ratio

enroll_school_gdf['Private Aided Student School Ratio'] = ratio



ratio = enroll_school_df['Private Unaided Total'] / enroll_school_df['Private Unaided Schools']

enroll_school_df['Private Unaided Student School Ratio'] = ratio

enroll_school_gdf['Private Unaided Student School Ratio'] = ratio



ratio = enroll_school_df['Grand Total'] / enroll_school_df['Grand Total Schools']

enroll_school_df['Grand Student School Ratio'] = ratio

enroll_school_gdf['Grand Student School Ratio'] = ratio

# See 'Govt Student School Ratio'

col = 'Govt Student School Ratio'

enroll_school_df[col].describe()
draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))
# Chennai, Kancheepuram, Viluppuram school size is large

# Ramanathapuram, Sivaganga, The nulgris school size is small
print_district_info(enroll_school_df, districts=['Chennai', 'Kancheepuram', 'Viluppuram'], 

                    cols_to_print=['District', 'Govt Student School Ratio'])
print_district_info(enroll_school_df, districts=['Ramanathapuram', 'Sivaganga', 'The Nilgiris'], 

                    cols_to_print=['District', 'Govt Student School Ratio'])
draw_map(enroll_school_gdf, col)
# See 'Private Aided Student School Ratio'

col = 'Private Aided Student School Ratio'

enroll_school_df[col].describe()
draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))
# Thoothukkudi, The Nilgris, Tirunelvi school size is smaller

# Krishnagirir, Salem, Chennai school size is higher
draw_map(enroll_school_gdf, col)
# See 'Private Unaided Student School Ratio'

col = 'Private Unaided Student School Ratio'

enroll_school_df[col].describe()
draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))
# Pudukkotai, Thiruvarur, Ariyalur has least school size

# Chennai, Thiruvallur, Coimbatore has highest school size
col = 'Private Unaided Student School Ratio'

print_district_info(enroll_school_df, districts=['Chennai', 'Thiruvallur', 'Coimbatore'], 

                    cols_to_print=['District', col])
print_district_info(enroll_school_df, districts=['Pudukkottai', 'Thiruvarur', 'Ariyalur'], 

                    cols_to_print=['District', col])
draw_map(enroll_school_gdf, col)
# See 'Grand Student School Ratio'

col = 'Grand Student School Ratio'

enroll_school_df[col].describe()
draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))
# Sivaganga, Ramnathapuram, Pudukkotai has small school size in general

# Chennai, Kancheepuram, Thiruvallur has largest school size in general
col = 'Grand Student School Ratio'

print_district_info(enroll_school_df, districts=['Ramanathapuram', 'Pudukkottai', 'Sivaganga'], 

                    cols_to_print=['District', col])
print_district_info(enroll_school_df, districts=['Chennai', 'Kancheepuram', 'Thiruvallur'], 

                    cols_to_print=['District', col])
draw_map(enroll_school_gdf, col)
cols = ['Private Unaided Student School Ratio', 'Private Aided Student School Ratio', 'Govt Student School Ratio']

enroll_school_df[cols].describe()
# Private unaided schools have largest school student ratio

# govt school have largest school student ratio

# private aided schools have largest standard deviation
enroll_school_df.columns
draw_line_chart(x='District', df=enroll_school_df, y=cols, figsize=(16, 7))