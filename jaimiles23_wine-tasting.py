import pandas as pd 

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline



import numpy as np 

import seaborn as sns





#CHECK VERSION

!pip install pywrangle --upgrade  #! runs cmd lines

import pywrangle as pw





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print("\nSetup Complete :)")
## File Path

filepath_winereviews = "../input/wine-reviews/winemag-data_first150k.csv"



## Load Data

df_winereviews = pd.read_csv(

    filepath_or_buffer = filepath_winereviews,

    sep = ",",

    header = 0,

    index_col = 0    # index unnamed ID column

)



df_winereviews.head()
## Inspect dataframe

df_winereviews.info()

print(df_winereviews.shape)

print(df_winereviews.columns)
## Use Pywrangle to show nulls per column

pw.show_col_nulls(

    df_winereviews,

    show_null_heatmap = True,

    show_null_corr_matrix = True,

)
##########

# Columns to drop

##########

df_info = pw.record_df_info(df_winereviews)



needed_columns = (

    "region_1",

    "price",

    "country",

    "province",

)





df_winereviews.dropna(

    axis = 0,

    how = "any",

    subset = needed_columns,

    inplace = True

)





##########

# Dataframe changes

##########



pw.print_df_changes(df_winereviews, df_info)



##########

# Remaining null values

##########



pw.show_col_nulls(df_winereviews)

## Check dataframe

df_winereviews.info()

print(df_winereviews.shape)

print(df_winereviews.columns)
##########

# Clean all string data

##########



## Tuple of string columns and key for sentence case.

str_col_name_case = (

    ("country", 2),

    ("description", 0),

    ("designation", 0),

    ("province", 1),

    ("region_1", 1),

    ("region_2", 1),

    ("variety", 1),

    ("winery", 1),

)



## Use PW to clean string columns

df_winereviews = pw.clean_str_columns(df_winereviews, str_col_name_case)

df_winereviews.head(10)

##########

# Combine regions

##########

old_df = pw.record_df_info(df_winereviews)



def join_regions(row: object, join_str = " - ") -> str:

    """Joins the regions into new string for data_frame."""

    region_list = [row.region_1]

    

    if pd.notna(row.region_2):

        region_list += [join_str, row.region_2]

        

    return ''.join(region_list)





df_winereviews['master_region'] = df_winereviews.apply(join_regions, axis = 1)



pw.print_df_changes(df_winereviews, old_df)
## Drop region columns

old_df = pw.record_df_info(df_winereviews)



regions = ['region_1', 'region_2']

df_winereviews.drop(

    regions,

    axis = 1,

    inplace = True

)



pw.print_df_changes(df_winereviews, old_df)



df_winereviews.head()
## Check duplicate values

df_duplicates = (

    df_winereviews[df_winereviews.duplicated(keep = False) == True]

    .sort_values(by = ["country", "description"])

)



print(df_duplicates.shape)

df_duplicates.head(5)



## NOTE: may like to create an inspect method that does all 3 of these ^^
old_df = pw.record_df_info(df_winereviews)



df_winereviews.drop_duplicates( inplace = True)

df_winereviews.reset_index(drop = True, inplace = True)



pw.print_df_changes(df_winereviews, old_df)

df_winereviews.head()
df_recorded = pw.record_df_info(df_winereviews)



df_ca = df_winereviews[df_winereviews['province'] == "California"]

df_ca.drop(['country', 'province'], axis = 1, inplace = True)

df_ca.reset_index(drop = True, inplace = True)



pw.print_df_changes(df_ca, df_recorded)

df_ca.head(5)
## General description

df_ca.describe()
sns.distplot(

    a = df_ca['price'],

    kde = False

)

plt.title("Histogram of CA wine prices")
sns.scatterplot(

    x = df_ca['points'],

    y = df_ca['price']

)

plt.title("CA wine Points vs Price")
df_ca['points'].agg(['min', 'max'])
old_df = pw.record_df_info(df_ca)

df_ca['stars'] = df_ca.apply(lambda row: (row.points - 80) // 5 + 1, axis = 1)



pw.print_df_changes(df_ca, old_df)

df_ca.head()
star_labels = (

    (1, "1 star"),

    (2, "2 stars"),

    (3, "3 stars"), 

    (4, "4 stars"),

    (5, "5 stars")

)



for star, star_label in star_labels:

    sns.distplot(

        a = df_ca[ df_ca['stars'] == star]['price'],

        label = star_label,

        kde = True

    )



plt.title("Histogram of price, by stars")

plt.legend()

old_df = pw.record_df_info(df_ca)



outliers = df_ca[df_ca['price'] > 500]

outliers.reset_index(drop = True, inplace = True)

outliers.shape



pw.print_df_changes(outliers, old_df)

outliers.head()
for index, row in outliers.iterrows():

    print(

        index, row['description'],

        sep = "\t", end = "\n\n"

    )

old_df = pw.record_df_info(df_ca)



df_ca = df_ca[df_ca['price'] <= 500]



pw.print_df_changes(df_ca, old_df)  ## NOTE that these results match those from the outliers df.

df_ca.head()
plt.figure(figsize = (10,6))



plt.title("Average price, by number of stars")



sns.barplot(

    x = df_ca['stars'],

    y = df_ca['price'],

)

sns.lmplot(

    x = "points",

    y = "price",

    data = df_ca

)

plt.title("CA wine Points vs Price")



corr = round( 

    df_ca['points'].corr(df_ca['price']), 4)



print(f"Correlation of points to price: {corr}")
## Group description



pd.set_option('display.max_rows', None)



(

    df_ca

    .groupby(['stars', 'master_region'])['price']

    .describe()

    .sort_values(['stars', 'mean', 'count'], ascending = [False, True, False])

    .reset_index()

)