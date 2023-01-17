import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv("../input/beers-raw.csv")

df.head(5)
df[["brewery_location", "brewery_name"]]
breweries = df[["brewery_location", "brewery_name"]]

#resets the index of the rows to 0 and autoincrement

breweries = breweries.drop_duplicates().reset_index() #adds an index column to remember the old index value

breweries.head(5)
breweries = df[["brewery_location", "brewery_name"]]

breweries = breweries.drop_duplicates().reset_index(drop=True) #drops the index column after reset

breweries.head(5)
breweries["id"] = breweries.index #adds an id column and stores a row's index

breweries.head(5)
#merge the original dataframe (df) and breweries to map the brewery ID

beers = pd.merge(df,

                 breweries,

                 left_on=["brewery_name", "brewery_location"],

                 right_on=["brewery_name", "brewery_location"],

                 sort=True,

                 suffixes=('_beer', '_brewery'))

beers.head(5)
#drop brewery_location and brewery_name columns

beers = beers[["abv", "ibu", "id_beer",

               "name", "size", "style", "id_brewery"]]

beers.head(5)
beers_columns_rename = {

    "id_beer": "id",

    "id_brewery": "brewery_id"

}

beers.rename(inplace=True, columns=beers_columns_rename)

beers.head(5)
#this is how we usually define a function

def aFunc(x):

    return x**2

print(aFunc(6))
#this is how we define a lambda function

aLambdaFunc = lambda x: x**2

print(aLambdaFunc(6))
breweries["city"] = breweries["brewery_location"].apply(

    lambda location: location.split(",")[0])

breweries.head(5)
breweries["state"] = breweries["brewery_location"].apply(

    lambda location: location.split(",")[1])

breweries.head(5)
breweries = breweries[["brewery_name", "city", "state", "id"]]

breweries.rename(inplace=True, columns={"brewery_name": "name"})

breweries.head(5)
def string_pct_to_float(value):

    stripped = str(value).strip('%')

    try:

        return float(stripped)/100

    except ValueError:    

        return None



beers["abv"] = beers["abv"].apply(string_pct_to_float)

beers.head(5)
def string_to_int(value):

    try:

        return int(value)

    except ValueError:  

        return None

    

beers["ibu"] = beers["ibu"].apply(string_to_int)
beers.head(5)
for possible_value in set(beers["size"].tolist()): #a set maintains only unique values

    print(possible_value)
import re



def extract_ounces(value):

    stripped = value.strip("oz")

    match = re.match("(\d{1,2}\.*\d*)", value)

    if match:

        return float(match.group(0))

    else:

        return None

 

beers["ounces"] = beers["size"].apply(extract_ounces)    

del beers["size"]

beers.head(5)
for possible_value in set(beers["ounces"].tolist()): #a set maintains only unique values

    print(possible_value)
import pandas_profiling

import seaborn as sns

sns.set(color_codes=True)

sns.set_palette(sns.color_palette("muted"))



beers.head(5)
beers_and_breweries = pd.merge(beers, 

                               breweries, 

                               how='inner', 

                               left_on="brewery_id", 

                               right_on="id", 

                               sort=True,

                               suffixes=('_beer', '_brewery'))

beers_and_breweries.head(5)
beers.dtypes
def get_var_category(series):

    unique_count = series.nunique(dropna=False)

    total_count = len(series)

    if pd.api.types.is_numeric_dtype(series):

        return 'Numerical'

    elif pd.api.types.is_datetime64_dtype(series):

        return 'Date'

    elif unique_count==total_count:

        return 'Text (Unique)'

    else:

        return 'Categorical'



def print_categories(df):

    for column_name in df.columns:

        print(column_name, ": ", get_var_category(df[column_name]))

        

print_categories(beers)
print_categories(breweries)
length = len(beers["ibu"]) #includes missing and null values

length
count = beers["ibu"].count() #non-null

print(count)
number_of_missing_values = length - count

pct_of_missing_values = float(number_of_missing_values / length)

pct_of_missing_values = "{0:.1f}%".format(pct_of_missing_values*100)

print(pct_of_missing_values)
print("Minimum value: ", beers["ibu"].min())

print("Maximum value: ", beers["ibu"].max())
print(beers["ibu"].mode())
beers["ibu"].mean()
beers["ibu"].median()
beers["ibu"].quantile([.25, .5, .75])
sns.distplot(beers["ibu"].dropna());
beers[["abv", "ibu", "ounces"]].corr()
pandas_profiling.ProfileReport(beers_and_breweries)