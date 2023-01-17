# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# We'll import matplotlib stuff for later...
import matplotlib.pyplot as plt
import bq_helper # should let us access the data
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
# query is structured like standard SQL, just getting everything
QUERY1 = "SELECT * FROM `bigquery-public-data.usa_names.usa_1910_current`"
print("Current database will use", str(usa_names.estimate_query_size(QUERY1)), "GB")
QUERY2a = "SELECT name, gender, year, SUM(number) AS total FROM `bigquery-public-data.usa_names.usa_1910_current` "
QUERY2b = "GROUP BY year, gender, name ORDER BY year ASC"
df = usa_names.query_to_pandas(QUERY2a + QUERY2b)
# Let's look at how many entries our table has, as well as what our columns are.
print("First, see how many rows we have (dataframe index)")
print(str(len(df)))
print ("Then, we look at the columns (labels)")
print(str(df.columns.values))
# Start by separating by gender...
# To understand this, we must understand what dataframe.loc means.
# dataframe.loc[something] gives us the somethingth row of the dataframe.  Kind of like selecting an element from a python list.
# dataframe['column_name'] will give us one particular column.
# Using that with a boolean test gives us a boolean array showing which rows evaluate to true.
# Using that array of the rows passing the test as the index for loc, we get only the rows passing the test.
df_male = df.loc[df['gender'] == "M"] # table with only male names
df_female = df.loc[df['gender'] == "F"] # table with only female names
dfuniques_male = df_male.groupby(['year']).nunique() # table counting unique names, genders, and totals, for each year
dfuniques_female = df_female.groupby(['year']).nunique() # but we really only care about unique names
dfuniques_merged = pd.concat([dfuniques_male.name.rename("male names"), dfuniques_female.name.rename("female names")], axis = 1)
# Worth noting that, in this new merged dataframe, the year is the index!  Let's just pick out the '60s!
dfuniques_merged.loc[1960:1970]
dfuniques_merged.plot(title="unique names in USA by year", figsize=(10,6))
 # Just like last time, let's check this with SQL!
# We'll just check male names, for now.\n",
QUERYa1 = "SELECT year, COUNT(DISTINCT name) AS uniques FROM `bigquery-public-data.usa_names.usa_1910_current` "
# filter the dates, states, and gender
QUERYb = "WHERE gender = 'M' AND year BETWEEN 1960 AND 1979 "
QUERYc = "GROUP BY year "
QUERYd = "ORDER BY year ASC "
QUERY = QUERYa1 + QUERYb + QUERYc + QUERYd
uniquetable1 = usa_names.query_to_pandas(QUERY)
QUERYa2 = "SELECT year, COUNT(DISTINCT name) AS uniques FROM `bigquery-public-data.usa_names.usa_1910_2013` "
QUERY = QUERYa2 + QUERYb + QUERYc + QUERYd 
uniquetable2 = usa_names.query_to_pandas(QUERY)
check_uniques_merged = pd.concat([uniquetable1.uniques.rename("current"), uniquetable2.uniques.rename("2013")], axis = 1)
check_uniques_merged.head(10)
# Let's get lists of unique boys names by year in the 1960s for each of the datasets.
QUERYa1 = "SELECT DISTINCT name, year FROM `bigquery-public-data.usa_names.usa_1910_current` "
# filter the dates, states, and gender
QUERYb = "WHERE gender = 'M' AND year BETWEEN 1960 AND 1969 "
QUERYc = "ORDER BY year, name "
unique_60s_boys_curr = usa_names.query_to_pandas(QUERYa1 + QUERYb + QUERYc)
QUERYa2 = "SELECT DISTINCT name, year FROM `bigquery-public-data.usa_names.usa_1910_2013` "
unique_60s_boys_2013 = usa_names.query_to_pandas(QUERYa2 + QUERYb + QUERYc)

# Now we look for differences.
# First, let's make a table which contains all name/year pairs and a column for whether they are in one, the other, or both
# Merging "outer" means it keeps everything, whether the row is common to both frames or not.
# The indicator gives us this column which tells us whether the row is in left, right, or both.
unique_60s_boys_common = unique_60s_boys_curr.merge(unique_60s_boys_2013.drop_duplicates(), on=['name', 'year'], how='outer', indicator=True)
unique_60s_boys_common.head()
# Now, we simply select only entries present in "left_only" or "right_only"
# Left will be names only present in the current list
unique_60s_boys_common.loc[unique_60s_boys_common['_merge'] == 'left_only']
# And now, those only present in the 2013 list
unique_60s_boys_common.loc[unique_60s_boys_common['_merge'] == 'right_only']
# First, let's figure out Jean Luc, how is it in the database?
jeansearch = 'Jean*'
jeans = df.loc[df.name.str.match('Jean')]
jeans.name.unique()
# Let's look at names of characters from Star Trek: The Next Generation
tngnamelist = ["Jeanluc", "William", "Geordi", "Deanna", "Beverly", "Natasha", "Worf", "Wesley"]
tngnamesdf = df.loc[df['name'].isin(tngnamelist)]
tngnamesdf.head(10)
groupedtngnames = tngnamesdf.groupby(['year', 'name']).agg('sum')
groupedtngnames.head()
# Now we want to make each name a separate column
# There may be a simpler way to do this, but I'm not immediately aware of it.
# I do the error catching because, for some reason, Geordi is not a name that has caught on.
# The key error that is thrown should be converted to a column of zeros.
columndict = {}
for i, iname in enumerate(tngnamelist):
    try:
        columndict[iname] = groupedtngnames.xs(iname, level='name')
        columndict[iname].columns = [iname] # renaming the column for this dataframe
    except KeyError as inst:
        print ("Failed to find any instances of ", inst)
        ourindex = groupedtngnames.index.get_level_values('year').unique()
        columndict[iname] = pd.DataFrame([0]*len(ourindex), 
                                  index = ourindex)
        columndict[iname].columns = [iname]
# Now we have a dictionary of dataframes, but that's awkward and non-standard.
# Let's try to turn it into a dictionary of series.
for ikey in columndict.keys():
    columndict[ikey] = columndict[ikey].loc[:, ikey]
# Now we can easily convert the dictionary of series into a dataframe
plottable_tng = pd.DataFrame(columndict)
plottable_tng.head()
# We're almost there.  Now we need to turn those NaNs into zeros!
plottable_tng = plottable_tng.fillna(0)
# and then plot!
# First we're going to make a plot which we can annotate, then do the actual plotting.
# We get the axes (basically, the plotting area) as a return from the subplots function,
# and we specify in the dataframe.plot that we want to plot onto those axes.
# We'll also indicate the date range where the show was airing with vertical lines.
fig, ax = plt.subplots(figsize=(12, 8))
ax.axvline(1987, ls='dotted')
ax.axvline(1994, ls='dotted')
plt_tng = plottable_tng.plot(title = "Star Trek: The Next Generation Character Names", logy = True, ax=ax)
# First, let's get the total number of names for each year
df_annual_names_total = df.groupby(['year']).agg('sum')
# Let's check the total!  We'll first select the "total" column, and then sum over it.
print("the total population in the dataframe is: ", df_annual_names_total.total.sum())
df_annual_names_total.tail()
normalized_plottable_tng = plottable_tng.divide(df_annual_names_total.total, axis='index')*100000
fig, ax = plt.subplots(figsize=(12, 8))
ax.axvline(1987, ls='dotted')
ax.axvline(1994, ls='dotted')
plt_tng = normalized_plottable_tng.plot(title = "Star Trek: The Next Generation Character Names per 100,000 births", logy = True, ax=ax)
# Name popularity plot generator function
def PlotNamePopularity(df, name_list, media_title, date_list, normalize = True):
    # df is the dataframe of names, genders, states, and totals, which we extracted from the database above
    # name_list is a normal python list of first names we're interested in
    #     Group nested list objects as tuples, so it should look (for example)...
    #     [("Tom", "Thomas"), ("Dick", "Richard"), "Harry"]
    # date_list is a normal python list of integer (or float, if you really want) years to highlight
    # normalize is a boolean determining whether we want to divide out the population
    # First check validity
    if (len(name_list) < 1 or len(name_list) > 10):
        print("Please list between 1 and 10  names.")
        return -1
    # Make a version of the list which is flat, for when we need it.
    # While doing that, we'll also make a mapping dictionary
    name_map_dict = {}
    flat_list = []
    for iname in name_list:
        if (isinstance(iname, tuple) or isinstance(iname, list)):
            multiname = ""
            for jname in iname:
                flat_list.append(jname)
                multiname += jname
                multiname += "/"
            multiname = multiname[:-1] # remove trailing "/"
            for jname in iname:
                name_map_dict[jname] = multiname
        elif isinstance(iname, str):
            flat_list.append(iname)
            name_map_dict[iname] = iname
    
    # extract just the names we care about, sum over genders, group by year and name
    groupeddf = df.loc[df['name'].isin(flat_list)].groupby(['year', 'name']).agg('sum')
    # do this awkward loop to extract each name separately, allowing 
    # for the case where the name is not present
    columndict = {}
    for i, iname in enumerate(flat_list):
        try:
            columndict[iname] = groupeddf.xs(iname, level='name')
            columndict[iname].columns = [iname] # renaming the column for this dataframe
        except KeyError as inst:
            print ("Failed to find any instances of ", inst)
            ourindex = groupeddf.index.get_level_values('year').unique()
            columndict[iname] = pd.DataFrame([0]*len(ourindex), 
                                          index = ourindex)
            columndict[iname].columns = [iname]
    # Now we have a dictionary of dataframes, but that's awkward and non-standard.
    # Let's turn it into a dictionary of series.
    for ikey in columndict.keys():
        columndict[ikey] = columndict[ikey].loc[:, ikey]
    # Now turn the dictionary of series into a dataframe
    plottable_df = pd.DataFrame(columndict)
     # turn missing entries into zeroes
    plottable_df = plottable_df.fillna(0)
    
    # So far, we're using a flat list, now we group the names together
    # using the mapping dictionary we made above
    plottable_df = plottable_df.groupby(by=name_map_dict, axis=1).sum()
    
    plot_title = str(media_title) + " character names"
    # Let's take care of normalization, if we want to
    if normalize:
        df_annual_names_total = df.groupby(['year']).agg('sum')
        plottable_df = plottable_df.divide(df_annual_names_total.total, axis='index')*100000
        plot_title += " per 100,000 births"
    # Let's sort the columns of the list in order of popularity, so the top line on the 
    # legend corresponds to the top line
    new_order = plottable_df.sum().sort_values(ascending=False).index # our new order
    plottable_df = plottable_df[new_order]
    # And now we plot!
    fig, ax = plt.subplots(figsize=(12, 8))
    for linepos in date_list:
        ax.axvline(linepos, ls='dotted')
    plt_final = plottable_df.plot(title = plot_title, logy = True, ax=ax)
    return plt_final, plottable_df
# J.K. Rowling's series of books was published between 1997 and 2007
hp_name_list = ["Harry", ("Ron", "Ronald"), "Hermione", "Ginny", "Draco", "Cedric", "Albus", "Neville", "Luna"]
hp_date_list = [1997, 2007]
hp_title = "Harry Potter Book Series"
test1, test2 = PlotNamePopularity(df, hp_name_list, hp_title, hp_date_list, True)
# The Andy Griffith Show was a popular thing in its time...
ag_name_list = [("Andy","Andrew"), "Barney", "Bee", "Opie"]
ag_date_list = [1960, 1968]
ag_title = "The Andy Griffith Show"
test1, test2 = PlotNamePopularity(df, ag_name_list, ag_title, ag_date_list, True)
# Let's do the Star Trek: The Next Generation one again, but with the multi-name capability
tng_name_list = ["Jeanluc", ("William", "Will"), "Geordi", "Deanna", ("Beverly", "Bev"), ("Tasha","Natasha"), "Worf", "Wesley"]
tng_date_list = [1987, 1994]
tng_title = "Star Trek: The Next Generation"
test1, test2 = PlotNamePopularity(df, tng_name_list, tng_title, tng_date_list, True)
# How about Friends, the NBC sitcom?
friends_name_list = [("Joey", "Joseph"), "Monica", "Chandler", "Rachel", "Ross", "Phoebe"]
friends_date_list = [1994, 2004]
friends_title = "Friends"
test1, test2 = PlotNamePopularity(df, friends_name_list, friends_title, friends_date_list, True)