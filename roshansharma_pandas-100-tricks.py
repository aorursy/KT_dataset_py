# basic libraries

import os

import numpy as np

import pandas as pd



# this will allow us to print all the files as we generate more in the kernel

def print_files():

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



# check Trick 91 for an example

def generate_sample_data(): # creates a fake df for testing

    number_or_rows = 20

    num_cols = 7

    cols = list("ABCDEFG")

    df = pd.DataFrame(np.random.randint(1, 20, size = (number_or_rows, num_cols)), columns=cols)

    df.index = pd.util.testing.makeIntIndex(number_or_rows)

    return df



# check Trick 91 for an example

def generate_sample_data_datetime(): # creates a fake df for testing

    number_or_rows = 365*24

    num_cols = 2

    cols = ["sales", "customers"]

    df = pd.DataFrame(np.random.randint(1, 20, size = (number_or_rows, num_cols)), columns=cols)

    df.index = pd.util.testing.makeDateIndex(number_or_rows, freq="H")

    return df



# show several prints in one cell. This will allow us to condence every trick in one cell.

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
df = pd.read_csv("../input/us-accidents/US_Accidents_May19.csv")

print("The shape of the df is {}".format(df.shape))



del df



df = pd.read_csv("../input/us-accidents/US_Accidents_May19.csv", skiprows = lambda x: x>0 and np.random.rand() > 0.01)

print("The shape of the df is {}. It has been reduced 10 times!".format(df.shape))





'''

How it works:

skiprows accepts a function that is evaluated against the integer index.

x > 0 makes sure that the headers is not skipped

np.random.rand() > 0.01 returns True 99% of the tie, thus skipping 99% of the time.

Note that we are using skiprows

'''
d = {\

"zip_code": [12345, 56789, 101112, 131415],

"factory": [100, 400, 500, 600],

"warehouse": [200, 300, 400, 500],

"retail": [1, 2, 3, 4]

}



df = pd.DataFrame(d)

df



# save to csv

df.to_csv("trick99data.csv")



df = pd.read_csv("trick99data.csv")

df

# To avoid Unnamed: 0



df = pd.read_csv("trick99data.csv", index_col=0)

# or when saving df = pd.read_csv("trick99data.csv", index = False)

df
d = {\

"zip_code": [12345, 56789, 101112, 131415],

"factory": [100, 400, 500, 600],

"warehouse": [200, 300, 400, 500],

"retail": [1, 2, 3, 4]

}



df = pd.DataFrame(d)

df



# we have to reassing



# location_type is generated automatically from the columns left after specifying id_vars (you can pass a list also)

df = df.melt(id_vars = "zip_code", var_name = "location_type", value_name = "distance")

df
# Trick 97

# Convert

d = {\

"year": [2019, 2019, 2020],

"day_of_year": [350, 365, 1]

}



df = pd.DataFrame(d)

df



# Step 1: create a combined column

df["combined"] = df["year"]*1000 + df["day_of_year"]

df



# Step 2: convert to datetime

df["date"] = pd.to_datetime(df["combined"], format = "%Y%j")

df
print(pd.__version__)

# Pandas version 0.25 or higher requiered and you need hvplot



import pandas as pd

df = pd.read_csv("../input/drinks-by-country/drinksbycountry.csv")

df



# this one is not interactve

df.plot(kind = "scatter", x = "spirit_servings", y = "wine_servings")



# run !pip install hvplot

#pd.options.plotting.backend = "hvplot"

#df.plot(kind = "scatter", x = "spirit_servings", y = "wine_servings", c = "continent")
d = {\

"col1": [2019, 2019, 2020],

"col2": [350, 365, 1],

"col3": [np.nan, 365, None]

}



df = pd.DataFrame(d)

df



# Solution 1

df.isnull().sum().sum()



# Solution 2

df.isna().sum()



# Solution 3

df.isna().any()



# Solution 4:

df.isna().any(axis = None)
df = pd.read_csv("../input/titanic/train.csv", usecols = ["Pclass", "Sex", "Parch", "Cabin"])

df



# let's see how much our df occupies in memory

df.memory_usage(deep = True)



# convert to smaller datatypes

df = df.astype({"Pclass":"int8",

                "Sex":"category", 

                "Parch": "Sparse[int]", # most values are 0

                "Cabin":"Sparse[str]"}) # most values are NaN



df.memory_usage(deep = True)
d = {"genre": ["A", "A", "A", "A", "A", "B", "B", "C", "D", "E", "F"]}

df = pd.DataFrame(d)

df



# Step 1: count the frequencies

frequencies = df["genre"].value_counts(normalize = True)

frequencies



# Step 2: establish your threshold and filter the smaller categories

threshold = 0.1

small_categories = frequencies[frequencies < threshold].index

small_categories



# Step 3: replace the values

df["genre"] = df["genre"].replace(small_categories, "Other")

df["genre"].value_counts(normalize = True)
d = {"customer": ["A", "B", "C", "D"], "sales":[1100, 950.75, "$400", "$1250.35"]}

df = pd.DataFrame(d)

df



# Step 1: check the data types

df["sales"].apply(type)



# Step 2: use regex

df["sales"] = df["sales"].replace("[$,]", "", regex = True).astype("float")

df

df["sales"].apply(type)
# Solution 1

number_or_rows = 365*24 # hours in a year

pd.util.testing.makeTimeDataFrame(number_or_rows, freq="H")



# Solution 2

num_cols = 2

cols = ["sales", "customers"]

df = pd.DataFrame(np.random.randint(1, 20, size = (number_or_rows, num_cols)), columns=cols)

df.index = pd.util.testing.makeDateIndex(number_or_rows, freq="H")

df
d = {"A":[15, 20], "B":[20, 25], "C":[30 ,40], "D":[50, 60]}

df = pd.DataFrame(d)

df



# Using insert

df.insert(3, "C2", df["C"]*2)

df



# Other solution

df["C3"] = df["C"]*3 # create a new columns, it will be at the end

columns = df.columns.to_list() # create a list with all columns

location = 4 # specify the location where you want your new column

columns = columns[:location] + ["C3"] + columns[location:-1] # reaarange the list

df = df[columns] # create te dataframe in with the order of columns you like

df

df = pd.Series(["Geordi La Forge", "Deanna Troi", "Data"]).to_frame()

df.rename({0:"names"}, inplace = True, axis = 1)

df

#                              split on first space  

df["first_name"] = df["names"].str.split(n = 1).str[0]

df["last_name"] = df["names"].str.split(n = 1).str[1]

df
df = generate_sample_data()

df.head()



# Solution 1

df[["A", "C", "D", "F", "E", "G", "B"]].head() # doesn't modify in place



# Solution 2

cols_to_move = ["A", "G", "B"]



new_order = cols_to_move + [c for c in df.columns if c not in cols_to_move] # generate your new order

df[new_order].head()



# Solutin 3: using index

cols = df.columns[[0, 5 , 3, 4, 2, 1, 6]] # df.columns returns a series with index, we use the list to iorder the index as we like --> this way we order the columns

df[cols].head()

df = generate_sample_data_datetime()

df.shape

df.head()



# Step 1: resample by D. Basically agregate by day and use to_frame() to convert it to frame

daily_sales = df.resample("D")["sales"].sum().to_frame()

daily_sales



# Step 2: filter weekends

weekends_sales = daily_sales[daily_sales.index.dayofweek.isin([5, 6])]

weekends_sales



'''

dayofweek day

0         Monday

1         Tuesday

2         Wednesday

3         Thursday

4         Friday

5         Saturday

6         Sunday

'''
print_files()



df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.head()



# Problem 1

print("The Problem relies on that we don't know the column name")

df.groupby("Pclass")["Age"].agg(["mean", "max"])



# Problem 2

print("The Problem relies on that we have multiindex")

df.groupby("Pclass").agg({"Age":["mean", "max"]})



# Solution new in pandas 0.25 and higher

print("Now we have solved the previous problems by specifyig the column final name we want.")

print("BUT IT ONLY WORKS WITH A COLUMN. TO THIS KIND OF OPERATIONS ON MULTIPLE COLUMNS CHECK THE NEXT CELL")

df.groupby("Pclass")["Age"].agg(age_mean = "mean", age_max = "max")

def my_agg(x):

    names = {

        'age_mean': x['Age'].mean(),

        'age_max':  x['Age'].max(),

        'fare_mean': x['Fare'].mean(),

        'fare_max': x['Fare'].max()

    } # define you custom colum names and operations



    return pd.Series(names, index=[ key for key in names.keys()]) # all the columns you create in the previous dictionary will be in this list comprehension



df.groupby('Pclass').apply(my_agg)



# reference

# https://stackoverflow.com/questions/44635626/rename-result-columns-from-pandas-aggregation-futurewarning-using-a-dict-with

# Do some fast feature eng on the DF

d = {"gender":["male", "female", "male"], "color":["red", "green", "blue"], "age":[25, 30, 15]}

df = pd.DataFrame(d)

df



# Solution

map_dict = {"male":"M", "female":"F"}

df["gender_mapped"] = df["gender"].map(map_dict) # using dictionaries to map values

df["color_factorized"] = df["color"].factorize()[0] # using factorize: returns a tuple of arrays (array([0, 1, 2]), Index(['red', 'green', 'blue'], dtype='object')) that's why we select [0]

df["age_compared_boolean"] = df["age"] < 18 # return a True False boolean value



df
print("This df occupies way too much space")

df = generate_sample_data()

df



print("using set_option to save some screen space")

pd.set_option("display.max_rows", 6)

df



print("use reset_option all to reset to default")

pd.reset_option("all")

df
print_files()

df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv")



# Step 1: Let's the datetype of the columns

col_types = df.dtypes.to_frame()

col_types.rename({0:"type"}, inplace = True, axis = 1)

col_types

col_types.to_csv("trick83data.csv")



# Step 2: Let's import the previous data and convert it to a dictionary

col_dict = pd.read_csv("trick83data.csv", index_col = 0)["type"].to_dict()



# Step 3: Edit the dictionary with the correct data types

print("Original dictionary")

col_dict

col_dict["country"] = "category"

col_dict["continent"] = "category"

print("Modified dictionary")

col_dict



# Step 4: Use the dictionary to import the data

df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv", dtype=col_dict)

df.dtypes



# Note: please note that you can use the dict from step1 and paste in like this

df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv", \

dtype=

{'country': 'category',

 'beer_servings': 'int64',

 'spirit_servings': 'int64',

 'wine_servings': 'int64',

 'total_litres_of_pure_alcohol': 'float64',

 'continent': 'category'})

# However, if you have many colums, this can be confusing

df.dtypes
df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv", index_col="country")

df.iloc[15:20, :].loc[:, "beer_servings":"wine_servings"]

# iloc is used to filter the rows and loc the columns
d = {"customer":["A", "B", "C", "D", "E"], "sales":[100, "100", 50, 550.20, "375.25"]}

df = pd.DataFrame(d)

# everything seems  but this operation crashes df["sales"].sum(). We have mixed data types

df.dtypes

df["sales"].apply(type) # Wow we can see that we have int, str, floats inn one column

df["sales"].apply(type).value_counts() # See the number of each value



df["sales"] = df["sales"].astype(float) # convert the data to float

df["sales"].sum()

df["sales"].apply(type).value_counts()
df = generate_sample_data().T

cols_str = list(map(str, list(df.columns))) # so that we can do df["0"] as string for the example

df.columns = cols_str



# Using pandas concatenation

# if you are ever confused about axis = 1 or axis = 0, just put axis = "columns" or axis = "rows"

pd.concat([df.loc[:, "0":"2"], df.loc[:, "6":"10"], df.loc[:, "16":"19"]], axis = "columns") # ------------------> here we are selecting columns converted to strings



# Using lists

# please ntoe that df.columns is a series with index, so we are using index to filter # -------------------------> here we are selecting the index of columns

df[list(df.columns[0:3]) + list(df.columns[6:11]) + list(df.columns[16:20])]



# Using numpy

df.iloc[:, np.r_[0:3, 6:11, 16:20]] # probably the most beautiful solution
df = generate_sample_data()

df.head()

df.shape



# absolute values

(df["A"] < 5).sum()

print("In the columns A we have {} of rows that are below 5".format((df["A"] < 5).sum()))



# percentage

(df["A"] < 5).mean()

print("In the columns A the values that are below 5 represent {}%".format((df["A"] < 5).mean()))
# let's generate some fake data

df1 = generate_sample_data()

df2 = generate_sample_data()

df3 = generate_sample_data()

# df1.head()

# df2.head()

# df3.head()

df1.to_csv("trick78data1.csv")

df2.to_csv("trick78data2.csv")

df3.to_csv("trick78data3.csv")



# Step 1 generate list with the file name

lf = []

for _,_, files in os.walk("/kaggle/working/"):

    for f in files:

        if "trick78" in f:

            lf.append(f)

            

lf



# You can use this on your local machine

#from glob import glob

#files = glob("trick78.csv")



# Step 2: assing create a new column named filename and the value is file

# Other than this we are just concatinating the different dataframes

df = pd.concat((pd.read_csv(file).assign(filename = file) for file in lf), ignore_index = True)

df.sample(10)
d = {"genre": ["A", "A", "A", "A", "A", "B", "B", "C", "D", "E", "F"]}

df = pd.DataFrame(d)

df["genre"].value_counts()



# Step 1: count the frequencies

top_four = df["genre"].value_counts().nlargest(4).index

top_four



# Step 2: update the df

df_updated = df.where(df["genre"].isin(top_four), other = "Other")

df_updated["genre"].value_counts()
df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")

df.columns = map(str.lower, list(df.columns)) # convert headers to lower type

df.shape

# select top 3 genre

top_genre = df["genre"].value_counts().to_frame()[0:3].index



# now let's filter the df with the top genre

df_top = df[df["genre"].isin(top_genre)]

df_top

df_top.shape

df_top["genre"].unique()
df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv", usecols=["Title"])

df["Words"] = df["Title"].str.count(" ") + 1

df
# Run this on you local machine

# url = "https://es.wikipedia.org/wiki/Twitter"

# tables = pd.read_html(url)

# len(tables)



# matching_tables = pd.read_html(url, match = "Followers")

# matching_tables[0]
df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")

df.head()



meta = df.pop("Metascore").to_frame()

df.head()

meta.head()
df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")

df.head()



# Using cut you can specify the bin edges

pd.cut(df["Metascore"], bins = [0, 25, 50, 75, 99]).head()



# Using qcut you can specify the number of bins and it fill generate of aproximate equal size

pd.qcut(df["Metascore"], q = 3).head()



# cut and qcut accept label bin size

pd.qcut(df["Metascore"], q = 4, labels = ["awful", "bad", "average", "good"]).head()
# you will have to run on your local machine

#from tabula import read_pdf

# df = read_pdf("test.pdf", pages = "all")
print(pd.__version__)

print(pd.show_versions())
d = {"A":[1, 2, 3, 4,], "B":[1.0, 2.0, 3.0, 4.0], "C":[1.00000, 2.00000, 3.00000, 4.000003], "D":[1.0, 2.0, 3.0, 4.0], "E":[4.0, 2.0, 3.0, 1.0]}

df = pd.DataFrame(d)

df



df["A"].equals(df["B"]) # they requiere identical datatypes

df["B"].equals(df["C"])

df["B"].equals(df["D"])

df["B"].equals(df["E"]) # and the same order



print(pd.testing.assert_series_equal(df["A"], df["B"], check_names=False, check_dtype=False)) # assertion passes
# You will have to run this on you local machine

#apple_stocks = pd.read_html("https://finance.yahoo.com/quote/AAPL/history?p=AAPL")

#pd.concat([apple_stocks[0], apple_stocks[1]])
print_files()



df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv", usecols=["continent", "beer_servings"])

df.head()



(df.assign(continent = df["continent"].str.title(),

           beer_ounces = df["beer_servings"]*12,#                                     this will allow yo set a title

           beer_galons = lambda df: df["beer_ounces"]/128).query("beer_galons > 30").style.set_caption("Average beer consumption"))
d = {"state":["ny", "CA", "Tx", "FI"], "country":["USA", "usa", "UsA", "uSa"], "pop":[1000000, 2000000, 30000, 40000]}

df = pd.DataFrame(d)

df



int_types = ["int64"]

# creating new columns

for col in df.columns:

    ctype = str(df[col].dtype)

    if ctype in int_types:

        df[f'{col}_millions'] = df[col]/1000000

    elif ctype == "object":

        df[f'{col}_new'] = df[col].str.upper()

        # you can also drop the columns

        df.drop(col, inplace = True, axis = "columns")

        

df
df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv")

df



drink = "wine"



# allows us to iterate fast over columns

df[f'{drink}_servings'].to_frame()
df = pd.DataFrame({"gender":["Male", "Female", "Female", "Male"]})

df



# Getting this nasty warning

males = df[df["gender"] == "Male"]

males["abbreviation"] = "M"



# Fixing the error

print("Fixing the warning with print")

males = df[df["gender"] == "Male"].copy()

males["abbreviation"] = "M"

males
d = {"salesperson":["Nico", "Carlos", "Juan", "Nico", "Nico", "Juan", "Maria", "Carlos"], "item":["Car", "Truck", "Car", "Truck", "cAr", "Car", "Truck", "Moto"]}

df = pd.DataFrame(d)

df



# Fixing columns

df["salesperson"] = df["salesperson"].str.title()

df["item"] = df["item"].str.title()



df["count_by_person"] = df.groupby("salesperson").cumcount() + 1

df["count_by_item"] = df.groupby("item").cumcount() + 1

df["count_by_both"] = df.groupby(["salesperson","item"]).cumcount() + 1

df
df = pd.DataFrame({"gender":["Male", "Female", "Female", "Male"]})

df



# Getting this nasty warning

df[df["gender"] == "Male"]["gender"] = 1

df[df["gender"] == "Female"]["gender"] = 0





print("Fix using loc")

df.loc[df["gender"] == "Male", "gender"] = 1

df.loc[df["gender"] == "Female", "gender"] = 0

df
url = "https://github.com/justmarkham?tab=repositories"



# run it on your local machine

# df = pd.read_json(url)

# df = df[df["fork"] == False]

# df.shape

# df.head()



# lc = ["name", "stargazers_count", "forks_count"]

# df[lc].sort_values("stargazers_count", asending = False).head(10)
d = {"salesperson":["Nico", "Carlos", "Juan", "Nico", "Nico", "Juan", "Maria", "Carlos"], "item":[10, 120, 130, 200, 300, 550, 12.3, 200]}

df = pd.DataFrame(d)

df



df["running_total"] = df["item"].cumsum()

df["running_total_by_person"] = df.groupby("salesperson")["item"].cumsum()

df



# other useful functions are cummax(), cummin(), cumprod()
d = {"orderid":[1, 1, 1, 2, 2, 3, 4, 5], "item":[10, 120, 130, 200, 300, 550, 12.3, 200]}

df = pd.DataFrame(d)

df



print("This is the output we want to aggregate to the original df")

df.groupby("orderid")["item"].sum().to_frame()



df["total_items_sold"] = df.groupby("orderid")["item"].transform(sum)

df
# we have empty rows and bad data

df = pd.read_csv("/kaggle/input/trick58data/trick58data.csv")

df



# importing correct data

df = pd.read_csv("/kaggle/input/trick58data/trick58data.csv", header = 2, skiprows = [3,4])

df
print_files()



df = pd.read_csv("/kaggle/input/imdb-data/IMDB-Movie-Data.csv")

df



gbdf = df.groupby("Genre")

gbdf.get_group("Horror")
df = pd.DataFrame({"A":["Male", "Female", "Female", "Male"], "B":["x", "y", "z", "A"], "C":["male", "female", "male", "female"], "D":[1, 2, 3, 4]})

df



# first let's use applymap to convert to standarize the text

df = df.applymap(lambda x: x.lower() if type(x) == str else x)



mapping = {"male":0, "female":1}



print("PROBLEM: Applies to the whole df but retruns None")

df.applymap(mapping.get)



print("Get the correct result but you have to specify the colums. If you don't want to do this, check the next result")

df[["A", "C"]].applymap(mapping.get)



print("Condtional apply map: if can map --> map else return the same value")

df = df.applymap(lambda x: mapping[x] if x in mapping.keys() else x)

df
df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv")

df



print("Classical filter hard to read and mantain.")

df[(df["continent"] == "Europe") & (df["beer_servings"] > 150) & (df["wine_servings"] > 50) & (df["spirit_servings"] < 60)]



print("You can split it across multiple lines to make it more readable. But it's still hard to read.")

df[

    (df["continent"] == "Europe") & 

    (df["beer_servings"] > 150) & 

    (df["wine_servings"] > 50) & 

    (df["spirit_servings"] < 60)

]



print("Solution saving criteria as objects")



cr1 = df["continent"] == "Europe"

cr2 = df["beer_servings"] > 150

cr3 = df["wine_servings"] > 50

cr4 = df["spirit_servings"] < 60



df[cr1 & cr2 & cr3 & cr4]



print("Solution using reduce")

from functools import reduce



# creates our criteria usings lambda

# lambda takes 2 parameters, x and y

# reduce combines them & for every cr in the (cr1, cr2, cr3, cr4)

criteria = reduce(lambda x, y: x & y, (cr1, cr2, cr3, cr4))

df[criteria]

df = generate_sample_data()

df["A_diff"] = df["A"].diff() # calculate the difference between 2 rows

df["A_diff_pct"] = df["A"].pct_change()*100 # calculates the porcentual variation between 2 rows



# add some style

df.style.format({"A_diff_pct":'{:.2f}%'})

df = generate_sample_data()



df.sample(frac = 0.5, random_state = 2)

df.sample(frac = 0.5, random_state = 2).reset_index(drop = True) # reset index after shuffeling

df = generate_sample_data()



df.plot(kind = "line")

df.plot(kind = "bar")

df.plot(kind = "barh")

df.plot(kind = "hist")

df.plot(kind = "box")

df.plot(kind = "kde")

df.plot(kind = "area")



# the following plots requiere x and y

df.plot(x = "A", y = "B", kind = "scatter")

df.plot(x = "A", y = "B", kind = "hexbin")

df.plot(x = "A", y = "B", kind = "pie") # here you can pass only x but you need to add subplots = True



# other plots are available through pd.plotting

# more about plotting https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
print_files()



df = pd.read_csv("/kaggle/input/titanic/train.csv")



# Solution 1: using str.cat 

df["Name"].str.cat(df["Sex"], sep = ", ").head()



# using + sign

df["Name"] + ", " + df["Sex"].head()
df = pd.read_csv("/kaggle/input/titanic/train.csv")



# Typical groupby

print("Problem: MultiIndex")

df.groupby("Pclass").agg({"Age":["mean", "max"], "Survived": "mean"})



# Please note that this has been covered in 86 and 86 bis.

# This is just one more way to do it.

print("Named aggregation")

df.groupby("Pclass").agg(avg_age = ("Age", "mean"),

                        max_age = ("Age", "max"), 

                        survival_rate = ("Survived", "mean"))
d = {"A": [100, 200, 300, 400, 100], "W":[10, 5, 0, 3, 8]}

df = pd.DataFrame(d)

df



# with replacement

df.sample(n = 5, replace = True, random_state = 2)



# adding weights

df.sample(n = 5, replace = True, random_state = 2, weights = "W")



df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv")

df.head()

df.dtypes



# Let's import the country and beer_servings columns, convert them to string and float64 respectevly

# Import only the first 5 rows and thread 0 as nans

df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv",

                    usecols=["country", "beer_servings"],

                    dtype={"country":"category", "beer_servings":"float64"},

                    nrows = 5,

                    na_values = 0.0)

df.head()

df.dtypes



# more about read_csv on https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
d = {"Team":["FC Barcelona", "FC Real Madrid"], 

    "Players":[["Ter Stegen", "Semedo", "Piqué", "Lenglet", "Alba", "Rakitic", "De Jong", "Sergi Roberto", "Messi", "Suárez", "Griezmann"], \

               ["Courtois", "Carvajal", "Varane", "Sergio Ramos", "Mendy", "Kroos", "Valverde", "Casemiro", "Isco", "Benzema", "Bale"]]}



print("Notice that we have a list of players for each team. Let's generate a row for each player.")

df = pd.DataFrame(d)

df



print("Using explode to generate new rows for each player.")

df1 = df.explode("Players")

df1



print("Reverse this operation with groupby and agg")

df["Imploded"] = df1.groupby(df1.index)["Players"].agg(list)

df
print("Default series")

ser1 = pd.Series([10, 20])

ser1



print("Let's add a NaN to an int64 series")

ser1 = pd.Series([10, 20, np.nan])

ser1 # Notice it has been converted to float64



print("But if we use Int64 than everything will work")

ser1 = pd.Series([10, 20, np.nan], dtype = "Int64")

ser1
d = {"Team":["FC Barcelona", "FC Real Madrid"], 

    "Players":["Ter Stegen, Semedo, Piqué, Lenglet, Alba, Rakitic, De Jong, Sergi Roberto, Messi, Suárez, Griezmann",

               "Courtois, Carvajal, Varane, Sergio Ramos, Mendy, Kroos, Valverde, Casemiro, Isco, Benzema, Bale"]}



print("Notice that we have a list of players for each team separated by commas. Let's generate a row for each player.")

df = pd.DataFrame(d)

df



print("Notice that we have converted to something similar seen in example 47.")

df.assign(Players = df["Players"].str.split(","))



print("Now add explode and done.")

df.assign(Players = df["Players"].str.split(",")).explode("Players")
df = generate_sample_data()

df



# create a local variable mean

mean = df["A"].mean()



# now let's use in inside a query of pandas using @

df.query("A > @mean")

# It seems that this trick is duplicated, skip to the next one

# I decided to keep in, so in the future there will be no confusion if you consult the original material

# and this kernel

d = {"Team":["FC Barcelona", "FC Real Madrid"], 

    "Players":[["Ter Stegen", "Semedo", "Piqué", "Lenglet", "Alba", "Rakitic", "De Jong", "Sergi Roberto", "Messi", "Suárez", "Griezmann"], \

               ["Courtois", "Carvajal", "Varane", "Sergio Ramos", "Mendy", "Kroos", "Valverde", "Casemiro", "Isco", "Benzema", "Bale"]]}



print("Notice that we have a list of players for each team. Let's generate a row for each player.")

df = pd.DataFrame(d)

df



print("Using explode to generate new rows for each player.")

df1 = df.explode("Players")

df1



print("Reverse this operation with groupby and agg")

df["Imploded"] = df1.groupby(df1.index)["Players"].agg(list)

df
d = {"patient":[1, 2, 3, 1, 1, 2], "visit":[2015, 2016, 2014, 2016, 2017, 2020]}

df = pd.DataFrame(d)

df.sort_values("visit")



print("Let's get the last visit for each patient")

df.groupby("patient")["visit"].last().to_frame()
import pandas as pd

from pandas.api.types import CategoricalDtype

d = {"ID":[100, 101, 102, 103], "quality":["bad", "very good", "good", "excellent"]}

df = pd.DataFrame(d)

df



print("Let's create our own categorical order.")

cat_type = CategoricalDtype(["bad", "good", "very good", "excellent"], ordered = True)

df["quality"] = df["quality"].astype(cat_type)

df



print("Now we can use logical sorting.")

df = df.sort_values("quality", ascending = True)

df



print("We can also filter this as if they are numbers. AMAZING.")

df[df["quality"] > "bad"]
df = generate_sample_data()

print("Original df")

df



df.style.hide_index().set_caption("Styled df with no index and a caption")
df = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = [2, 4, 5, 11], nrows = 10)

df



pd.get_dummies(df) # Notice that we can eliminate one column of each since this information is contained in the others



pd.get_dummies(df, drop_first=True)
df = generate_sample_data_datetime().reset_index()

df = df.sample(500)

df["Year"] = df["index"].dt.year

df["Month"] = df["index"].dt.month

df["Day"] = df["index"].dt.day

df["Hour"] = df["index"].dt.hour

df["Minute"] = df["index"].dt.minute

df["Second"] = df["index"].dt.second

df["Nanosecond"] = df["index"].dt.nanosecond

df["Date"] = df["index"].dt.date

df["Time"] = df["index"].dt.time

df["Time_Time_Zone"] = df["index"].dt.timetz

df["Day_Of_Year"] = df["index"].dt.dayofyear

df["Week_Of_Year"] = df["index"].dt.weekofyear

df["Week"] = df["index"].dt.week

df["Day_Of_week"] = df["index"].dt.dayofweek

df["Week_Day"] = df["index"].dt.weekday

df["Week_Day_Name"] = df["index"].dt.weekday_name

df["Quarter"] = df["index"].dt.quarter

df["Days_In_Month"] = df["index"].dt.days_in_month

df["Is_Month_Start"] = df["index"].dt.is_month_start

df["Is_Month_End"] = df["index"].dt.is_month_end

df["Is_Quarter_Start"] = df["index"].dt.is_quarter_start

df["Is_Quarter_End"] = df["index"].dt.is_quarter_end

df["Is_Leap_Year"] = df["index"].dt.is_leap_year

df
df = generate_sample_data()

df



# using loc --> labels

df.loc[0, "A"]



# using iloc --> position

df.iloc[0, 0]



# mixing labels and position with loc

df.loc[0, df.columns[0]]



# mixing labels and position with loc

df.loc[df.index[0], "A"]



# mixing labels and position with iloc

df.iloc[0, df.columns.get_loc("A")]



# mixing labels and position with iloc

df.iloc[df.index.get_loc(0), 0]
s = pd.Series(range(1552194000, 1552212001, 3600))

s = pd.to_datetime(s, unit = "s")

s



# set timezome to current time zone (UTC)

s = s.dt.tz_localize("UTC")

s



# set timezome to another time zone (Chicago)

s = s.dt.tz_convert("America/Chicago")

s
d = {"colum_without_space":np.array([1, 2, 3, 4, 5, 6]), "column with space":np.array([1, 2, 3, 4, 5, 6])*2}

df = pd.DataFrame(d)

df



print("Query a column without space")

df.query("colum_without_space > 4")

print("Query a column with space using backticks ``")

print("This is a backtick ``")

df.query("`column with space` > 8")
import pandas_profiling



df = generate_sample_data()



df



print("Generating report with pandas profiling")

df.profile_report()

# use pd.describe_option() to see all

# max_rows

# max_columns

# max_colwidth

# precision

# date_dayfirst

# date_yearfirst



df = generate_sample_data_datetime()[:10].reset_index()

df["sales"] = df["sales"].astype("float")

df



pd.set_option("display.max_rows",5)

pd.set_option("display.max_columns",3)

pd.set_option('display.width', 1000)

pd.set_option('display.date_dayfirst', True)

pd.describe_option()



pd.reset_option('^display.', silent=True) # restore to default

#pd.reset_option('display.width') # restore one by one
df = generate_sample_data()[:10]

df["A"] = pd.Series(["APP", "GOO", "APP", "GOO", "MIC", "MIC", "APP", "GOO", "MIC", "APP"])

df.rename(columns = {"A":"stock"}, inplace = True)

print("Original df")

df



print("Filter data using intermediate variables")

temp = df.groupby("stock").mean()

temp 



fv = temp["B"].sort_values(ascending = False)[1] # filter by the second greates. This way every time we generate sample data we will have a result

temp[temp["B"] < fv]



print("Filter using query")

df.groupby("stock").mean().query("B < {}".format(fv))

df.groupby("stock").mean().query("B < @fv")

df.groupby("stock").mean().query("B < 10")
pd.reset_option('^display.', silent=True) # restore to default



df = generate_sample_data()

df1 = df.copy(deep = True)

df = df.append(df1)



print("Imagine we have a big df where we can see all the columns ...")

df.T.head() # we are trasposing JUST TO CREATE A GIANT DF



# Solution 1

print("Solution 1 using pd.set_option display.max_columns")

pd.set_option("display.max_columns", None)

df.T.head()

pd.reset_option('^display.', silent=True) # restore to default



# Solution 2

print("Another clever solution using Traspose")

df.T.head().T
df = generate_sample_data()

df1 = df.copy(deep = True)

df1 = df1.drop([0, 1, 2], axis = "rows") # drop some index just to see the example workings

df.head()

df1.head()



pd.merge(df, df1, how = "left", indicator = True)

# Pandas is built upon numpy, so we can acess all numpy functionality from pandas

pd.np.random.rand(2, 3)

pd.np.nan
df = pd.read_csv("/kaggle/input/drinks-by-country/drinksbycountry.csv")

print("Original df")

df



print("Groupby continent beer_servings")

df.groupby("continent")["beer_servings"].mean()



print("Using agg to pass multiple functions")

df.groupby("continent")["beer_servings"].agg(["mean", "count"])



print("Using describe over a groupby object")

df.groupby("continent")["beer_servings"].describe()
df = generate_sample_data_datetime()



print("Original df")

df

print("Let's resample/groupby by month")

df.resample("M")["sales"].sum()



print("Let's resample/groupby by day")

df.resample("D")["sales"].sum()
df = generate_sample_data_datetime().reset_index()[:10]

df.rename(columns = {"index":"time"}, inplace = True)

df["sales_100"] = df["sales"]*100

print("Original df")

df.head()



# declare a formatting dict: individual for each column

fd = {"time":"{:%d/%m/%y}", "sales":"${:.2f}", "customers":"{:,}"}

df.style.format(fd)

df



# add some more formattin

(df.style.format(fd)

 .hide_index()

 .highlight_min("sales", color ="red")

 .highlight_max("sales", color ="green")

 .background_gradient(subset = "sales_100", cmap ="Blues")

 .bar("customers", color = "lightblue", align = "zero")

 .set_caption("A df with different stylings")

)
df = generate_sample_data()

df.head(2)



# Solution 1

df.rename({"A":"col_1", "B":"col_2"}, axis = "columns", inplace = True)

df.head(2)



# Solution 2

df.columns = ["col1", "col2", "col3", "col4","col5", "col6", "col7"] # list must be equal to the columns number

df.head(2)



# Solution 3

df.columns = df.columns.str.title() # apply any string method to the columns names

df.head(2)
# You will have to check this on your local machine

# Useful for fast importing

# Step 1: copy a table from excel sheet using ctrl + c (to the clipboard)

# Step 2: run this command

# df = pd.read_clipboard()
d = {"col1":[100, 120 ,140, np.nan, 160], "col2":[9, 10, np.nan, 7.5, 6.5]}

df = pd.DataFrame(d)

df.index = pd.util.testing.makeDateIndex()[0:5]

print("Original df")

df

print("DataFrame after interpolate")

df.interpolate()
print("Contains random values")

df1 = pd.util.testing.makeDataFrame() # contains random values

df1

print("Contains missing values")

df2 = pd.util.testing.makeMissingDataframe() # contains missing values

df2

print("Contains datetime values")

df3 = pd.util.testing.makeTimeDataFrame() # contains datetime values

df3

print("Contains mixed values")

df4 = pd.util.testing.makeMixedDataFrame() # contains mixed values

df4
d = {"name":["John Artur Doe", "Jane Ann Smith", "Nico P"], "location":["Los Angeles, CA", "Washington, DC", "Barcelona, Spain"]}

df = pd.DataFrame(d)

df



df[["first", "middle", "last"]] = df["name"].str.split(" ", expand = True)

df["city"] = df["location"].str.split(",", expand = True)[0]

df
d = {"day":[1, 2, 10 ,25, 12], "month":[1, 2, 4, 5, 6], "year":[2000, 2001, 2010, 2015, 2020]}

df = pd.DataFrame(d)

df["date"] = pd.to_datetime(df[["day", "month", "year"]])

df

df.dtypes
df = generate_sample_data_datetime().reset_index()

df.columns = ["date", "sales", "customers"]

df



print("Show the global usage of memory of the df")

df.info(memory_usage = "deep")

print()

print("Show the usage of memory of every column")

df.memory_usage(deep = True)
df = generate_sample_data()

df.head()



print("Writing data to a csv.zip file")

df.to_csv("trick18data.csv.zip")



print("Deleting df")

del df



print("Importing data from a csv.zip file")

df = pd.read_csv("/kaggle/working/trick18data.csv.zip", index_col=0)

df.head()



# other compression files supported .gz, .bz2, .xz
df = generate_sample_data()

print("Original df")

df



print("Using a slice (inclusive)")

df.loc[0:4, "A":"E"]



print("Using a list")

df.loc[[0,4], ["A","E"]]



print("Using a condition")

df.loc[df["A"] > 10, ["A","E"]]
df = generate_sample_data()

df["A"] = df["A"] + 5

df.rename(columns = {"A":"age"}, inplace = True)

df.sample(5)



df["age_groups"] = pd.cut(df["age"], bins = [0, 18, 65, 99], labels = ["kids", "adult", "elderly"])

df
df = pd.read_csv("/kaggle/input/titanic/train.csv")

print("Original df")

df.head()



print("Groupby and create a MultiIndex df")

print("Notice we have a df with MultiIndex (Sex and Pclass)")

df.groupby(["Sex", "Pclass"])["Survived"].mean().to_frame()



print("Reshaping using unstack")

print("Now we can interact with it like with a normal df")

df.groupby(["Sex", "Pclass"])["Survived"].mean().unstack()
# Method 1: from a dict

pd.DataFrame({"A":[10 ,20], "B":[30, 40]})



# Method 2: using numpy

pd.DataFrame(np.random.rand(2, 3), columns = list("ABC"))



# Method 3: using pandas builtin functionalities

pd.util.testing.makeMixedDataFrame()
d = {"A":[1, 2, 3], "B":[[10, 20], [40, 50], [60, 70]]}

df = pd.DataFrame(d)

print("Notice that the column B has as values lists")

df

print("Convert it to normal series")

df_ = df["B"].apply(pd.Series)

df_



print("Join the 2 df")

pd.merge(df, df_, left_index = True, right_index = True)

df = generate_sample_data()[:10]

df1 = df.copy(deep = True)

df = df.drop([0, 1, 2])

df1 = df1.drop([8, 9])

df

df1



df_one_to_one = pd.merge(df, df1, validate = "one_to_one")

df_one_to_one



df_one_to_many = pd.merge(df, df1, validate = "one_to_many")

df_one_to_many



df_many_to_one = pd.merge(df, df1, validate = "many_to_one")

df_many_to_one

print_files()

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.columns = ["Passenger ID", "Survived", "Pclass", "Name         ", "Sex", "Age", "Sib SP", "Parch", "Ticket", "Fare", "Cabin", "Embarked"] # creating column names for the example

df

df1 = df.copy(deep = True)



print("Replace all spaces with undescore and convert to lower")

print("Notice the Passenger and Sib SP column now has underscore")

df.columns = df.columns.str.replace(" ", "_").str.lower()

df.head()



print("Remove trailing (at the end) whitesapce and convert to lower")

print("Notice the Passenger and Sib SP column now has underscore")

df1.columns = df1.columns.str.lower().str.rstrip()

df1.head()
df = generate_sample_data()[["A", "B"]][:5]

df["A"] = pd.Series([15, 15, 18, np.nan, 12])

df["B"] = pd.Series([15, 15, 18, np.nan, 12])

df



print("Don't use ==, it does not handle NaN properly")

print("Notice that element 4 of each list is np.nan but == still returns False")

df["A"] == df["B"]



print("Using equals. Now we get True, so the 2 series are equal")

df["A"].equals(df["B"])



print("Equals also works for df")

df1 = df.copy(deep = True)

df.equals(df1)



print("== of df has the same issue as for series")

df == df1
print_files()



df = pd.read_csv("/kaggle/input/imdb-data/IMDB-Movie-Data.csv", \

                 usecols = ["Title", "Genre", "Year", "Metascore", "Revenue (Millions)"])

df.dtypes

df.memory_usage(deep = True)



print("Importing only a few columns and converting to proper dtype")

df = pd.read_csv("/kaggle/input/imdb-data/IMDB-Movie-Data.csv", \

                 usecols = ["Title", "Genre", "Year", "Metascore", "Revenue (Millions)"], \

                dtype = {"Genre":"category", "Metascore":"Int64", "Year":"int8"})

df.dtypes

df.memory_usage(deep = True) # notice how Genre and Year are consuming now less memory
# let's generate some fake data

df1 = generate_sample_data()

df2 = generate_sample_data()

df3 = generate_sample_data()

# df1.head()

# df2.head()

# df3.head()

df1.to_csv("trick8data1.csv", index = False)

df2.to_csv("trick8data2.csv", index = False)

df3.to_csv("trick8data3.csv", index = False)



# Step 1 generate list with the file name

lf = []

for _,_, files in os.walk("/kaggle/working/"):

    for f in files:

        if "trick8data" in f:

            lf.append(f)

            

lf



# You can use this on your local machine

#from glob import glob

#files = glob("trick8.csv")



# Step 2: we do the same as in trick 78 except we don't create a new column of the rows origin (file they came from)

df = pd.concat((pd.read_csv(file) for file in lf), ignore_index = True)

df
df = pd.util.testing.makeMissingDataframe().reset_index() # contains missing values

df.rename(columns = {"index":"A"})

df1 = df.copy(deep = True)

df



print("Calculate the % of missing values in each row")

df.isna().mean() # calculate the % of missing values in each row

print("Droping any columns that have missing values. Only column A wil remain")

df.dropna(axis = "columns") # drop any column that has missing values

print("Droping any rows that have missing values.")

df1.dropna(axis = "rows") # drop any row that has missing values

print("Droping column where missing values are above a threshold")

df.dropna(thresh = len(df)*0.95, axis = "columns") # drop any row that has missing values
df = generate_sample_data()

df_1 = df.sample(frac = 0.7)

df_2 = df.drop(df_1.index) # only works if the df index is unique



df.shape

df_1.shape

df_2.shape
d = {"col1":["1", "2", "3", "stuff"], "col2":["1", "2", "3", "4"]}

df = pd.DataFrame(d)

df.astype({"col2":"int"}) # this will fail for col1 --> ValueError: invalid literal for int() with base 10: 'stuff'



print("Notice that now stuff got converted to NaN")

df.apply(pd.to_numeric, errors = "coerce")
df = generate_sample_data_datetime()[:10].reset_index()

df["string_col"] = list("ABCDEABCDE")

df["sales"] = df["sales"].astype("float")

print("Original df")

df



print("Select numerical columns")

df.select_dtypes(include = "number")



print("Select string columns")

df.select_dtypes(include = "object")



print("Select datetime columns")

df.select_dtypes(include = ["datetime", "timedelta"])



print("Select miscelaneous")

df.select_dtypes(include = ["number", "object", "datetime", "timedelta"])



print("Select by passing the dtypes you need")

df.select_dtypes(include = ["int8", "int16", "int32", "int64", "float"])
df = generate_sample_data()[:5]

df["A"] = [1, 2, 3, 4, 5]



print("Filter using multiple |")

df[(df["A"] == 1) | (df["A"] == 3)]



print("Filter using isin")

df[df["A"].isin([1, 3])]



print("Invert using ~ (ctrl + alt + 4)")

df[~df["A"].isin([1, 3])]
df = generate_sample_data()[:5]

df



print("Reverse column order")

df.loc[:, ::-1]



print("Reverse row order")

df.loc[::-1]



print("Reverse row order and reset index")

df.loc[::-1].reset_index(drop = True)
df = generate_sample_data()[:5]

print("Original df")

df



print("Add prefix")

df.add_prefix("1_")



print("Add suffix")

df.add_suffix("_Z")










































































