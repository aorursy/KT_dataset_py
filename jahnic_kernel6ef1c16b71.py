# Packages
import requests # Scraping
from bs4 import BeautifulSoup # HTML parsing
import pandas as pd
import numpy as np
import datetime
import re
df_raw = pd.read_csv('../input/data-on-sales-posted-on-redflagdeals/rfd_main.csv').iloc[:,1:]
df_raw.head()
df_raw.info()
df_raw.describe(include='all')
df_comments = pd.read_csv("../input/data-on-sales-posted-on-redflagdeals/rfd_comments.csv").loc[:,"title":]
df_comments.head()
df_comments.info()
# Delete rows with empty comments
df_comments.dropna(axis=0, inplace=True)
df_comments.info()
# Print first 20 comments
print([x for x in df_comments['comments']][0:20])
# ↑ symbols
arrow_removed = [re.sub("↑+","", str(string)) for string in df_comments['comments']]
# \n characters
newline_removed = [re.sub("\\n+"," ",string) for string in arrow_removed]
# urls
urls_removed = [re.sub(r"\bhttp.+"," ",string) for string in newline_removed]
# Assign cleaned comments back
df_comments['comments'] = pd.Series(urls_removed)

# first 100 comments of cleaned table
print([x for x in df_comments['comments']][0:20])
# Save cleaned comment table as file
df_comments.to_csv('rfd_comments.csv')

df_comments.head()
# Copy of raw data set
df = df_raw.copy()

# List of tuples: (column name, column dtype)
col_dtypes = [(col, type(x)) for x,col in zip(df.iloc[0], df.columns)]

# Print tuple for columns containing dates
for col in col_dtypes:
    if col[0] in ['creation_date', 'last_reply', 'expiry']:
        print(col[0], ': ', col[1])
def to_datetime(column_name: str) -> pd.Series:
    """
    Converts a column of either format "%b %d, %Y %I:%M %p"
    or format "%B %d, %Y" from string to date-time
    
    Args:
    date_column - name of column with dates encoded as strings
    
    Returns:
    Column elements converted to datetime in a pandas.Series object
    """    
    # Superfluous characters removed
    column_clean = df[column_name].str.replace("st","").str.replace("nd","")\
                        .str.replace("rd","").str.replace("th","").str.strip()
    
    # Check for correct length of cleaned column
    column_len = len(column_clean)
    print("Cleaned and original column are of equal lenght: ", 
          column_len == len(df[column_name]), "\n")
    
    # Convert each entry from format "%b %d, %Y %I:%M %p" to datetime
    date_column = []
    try:
        date_column = column_clean.apply(lambda x :\
                        datetime.datetime.strptime(str(x), "%b %d, %Y %I:%M %p"))
    except: 
        print("\"%b %d, %Y %I:%M %p\" is incorrect format")
        pass
    
    # Convert from format "%B %d, %Y" to datetime
    for date in df[column_name]:
        if date is not np.nan:
            try:
                date_column.append(datetime.datetime.strptime(date, "%B %d, %Y"))
            except: 
                print("\"%B %d, %Y\" is incorrect format for", date)
                break
        else: 
            date_column.append(None)
    
    if len(date_column) != column_len:
        print("\n", "Incorrect column length!\n")
    else:
        print("\n", "Column has expected length!\n")
    
    return pd.Series(date_column)
# creation_date column converted to datetime
creation_date = to_datetime('creation_date')

# Compare random slice of original and converted column
print(creation_date.iloc[99:105], "\n")
print(df.loc[99:104, 'creation_date'])
# last_reply column converted to datetime
last_reply = to_datetime('last_reply')

# Print original and new column for comparison
print(last_reply.iloc[208:215], "\n")
print(df.loc[208:214, 'last_reply'])
expiry = to_datetime('expiry')
print(expiry.iloc[150:157], "\n")
print(df.loc[150:156, 'expiry'])
# Assign datetime columns to DataFrame
df.expiry = expiry
df.last_reply = last_reply
df.creation_date = creation_date

# Verify dates
df.head()
df.loc[:, ['source', 'title']].head()
# Set of entries in 'source' column
retailer_set = set(df['source'].dropna())
print("Number of unique sources: ", len(retailer_set))
print(df.source.isnull().sum(), "missing values in source column")
replace_dict = {} # key: index; value: retailer name to replace missing source value at index

# Iterate through set of unique values from source source column
for retailer in retailer_set:
    """Fill replace dictioray with indecies and source names. Entries are made
    when a source name is found in the title column while the corresponding source entry
    is empty."""
    
    # Iterate through 'source' and 'title' columns row-by-row
    # Generate boolean array: True if unique source name (retailer) found in "title" and "source" is np.nan
    source_missing_and_in_title = np.array([retailer in title 
                                     if source is np.nan else False
                                     for title,source in zip(df.title, df.source)])
    
    # Indecies for which source_missing_and_in_title is True
    replacement_indicies = np.where(source_missing_and_in_title == True)[0]
    # Fill "replace" dictionary
    for index in replacement_indicies:
        if index not in replace_dict.keys():
            replace_dict[index] = retailer

print("Replacements found in 'title':", len(replace_dict.values()))
source_list = list(df.source) # copy of source column 
missing_start = sum([x is np.nan for x in source_list]) # missing values before cleaning
print("Missing source values before replacement:", missing_start)

for replace in replace_dict.items():
    index = replace[0]
    source_replacement = replace[1]
    source_list[index] = source_replacement

missing_end = sum([x is np.nan for x in source_list]) # missing values after cleaning
print("Missing source values after replacement:", missing_end)
replaced_count = missing_start-missing_end # number of replaced values
print(replaced_count, "missing source records have been replaced!")
df.source = source_list
print("Number of missing values as expected:", (df.source.isnull().sum() == missing_end))
# 'url' entries of rows with missing source values
url_replacement = df[df.source.isnull()].url
print(url_replacement.notnull().sum(), "missing source values have corresponding urls")
url_replacement.head()
clean_urls = {} # key: index in df, value: cleaned url
indicies = url_replacement.index

for url in zip(indicies, url_replacement):
    index = url[0]
    replacement_url = url[1]
    
    # Clean if url value not missing
    if replacement_url is not np.nan:
        url_root = replacement_url.split("//")[1].split("/")[0].split("?")[0].replace("www.", "")
        removed_domain = url_root.split(".")
        clean_urls[index] = removed_domain
    else:
        clean_urls[index] = np.nan
        
print(clean_urls)
clean_url_final = clean_urls.copy()

for item in clean_url_final.items():
    index = item[0]
    url_split = item[1]
    try:
        if len(url_split) == 2:
             # name at index 0
            clean_url_final[index] = url_split[0].title()
        
        elif ((len(url_split) == 3) 
                        and ((url_split[-1] == "com") 
                                 or (url_split[-1] == "ca") 
                                 or (url_split[-1] == "ca"))):
            # name at index 1
            clean_url_final[index] = url_split[1].title()
        
        elif ((len(url_split) == 3) 
                        and (url_split[-1] == "io")):
             # name at index 0
            clean_url_final[index] = url_split[0].title()
        else: 
              clean_url_final[index] = np.nan
    except: value = np.nan
# Add url-derived company names to DataFrame
df.loc[list(clean_url_final.keys()),'source'] = list(clean_url_final.values())
print("Missing source values remaining: ", df.source.isnull().sum())
missing_prices_df = df[df.price.isnull()]
price_in_title = ["$" in title for title in missing_prices_df.title]
print(df.price.isnull().sum(), "missing values in 'price' column")
print(sum(price_in_title), "missing prices have '$' signs in the title") 
# Display first 10 title to evaluate if the missing price could be substituted
replacement_titles = missing_prices_df[price_in_title].title
[title for title in replacement_titles][0:10]
regex = "[$]+[.,]*\d+[.,]*\d+"\
        "|[.,]*\d+[.,]*\d+[$]+"\
        "|[a-zA-Z]+[$]+[.,]*\d+[.,]*\d+"
price_replacements = replacement_titles.str.findall(regex)
print("Number of possible replacements:", len(price_replacements))
price_replacements
replacement_dict = {} # key: index; value: price to replace missing value at index

# Iterate through price lists found in price_replacements and corresonding indecies in DataFrame
for replacement in zip(price_replacements, list(price_replacements.index)):
    price_list = replacement[0]
    index = replacement[1]
    if price_list != []:
        price = price_list[0]
        price_clean = (re.search(r"\d+[.,]*\d+", price)).group().replace(",","")
        replacement_dict[index] = price_clean
        
print(len(replacement_dict), "replacements found.")
# Replace missing values
df.loc[list(replacement_dict.keys()), 'price'] = list(replacement_dict.values())
print("Remaining missing values:", df.price.isnull().sum())
df.info()
regex = "\d+\.*\d*"
matches = [re.search(regex, str(x)) for x in df.price]

# Append matches to new Series object
new_price = pd.Series()
for match in matches:
    if match != None:
        new = pd.Series(float(match.group()))
    else:
        new = pd.Series(np.nan)
    new_price = new_price.append(new, ignore_index=True)
    
# Replace old price with new price column
df.price = new_price

df.info()
# Titles for which the saving entry is missing
missing_savings_df = df[df.saving.isnull()]
print([title for title in missing_savings_df.title.head(20)])
# Titles containing the % symbol may contain information on savings
# "saving_in_title" indicates the indicies for which there is no data
# in the "saving" column and a "%" is found in the title.
saving_in_title = ["%" in title for title in missing_savings_df.title]
print(df.saving.isnull().sum(), "missing values in 'saving' column")
print(sum(saving_in_title), "rows with missing 'saving' data have a '%' symbol in their title") 
# Titles containing the % symbol in rows with missing 'saving' entries  
replacement_titles = missing_savings_df[saving_in_title].title

# Extract savings data
regex = "[.,]*\d+[.,]*\d+[%]+"
saving_replacements = replacement_titles.str.findall(regex)
print("Number of possible replacements:", len(replacement_titles))
saving_replacements
replacements = {}
index_saving_tuples = zip(saving_replacements.index, saving_replacements)
for index, saving in index_saving_tuples:
    try:
        replacements[index] = saving[0]
    except:
        print("Empty list found in 'saving_replacements'")

print("="*50)
print("Replacements found for missing savings:", len(replacements))
# Replace missing values
df.loc[list(replacement_dict.keys()), 'price'] = list(replacement_dict.values())
print("Remaining missing values:", df.saving.isnull().sum())
# df slice with missing "saving" data and no "%" symbol in "titel"
no_title_replacement = missing_savings_df[[(not replaceable) for replaceable in saving_in_title]]

# titles will be used as ids for corresponding comments
comment_ids = set(no_title_replacement.title)
# Convert ids into indecies for comments_df
comment_indecies = [(x in comment_ids) for x in df_comments.title]

# Indecies for which titles appear for the first time
index_initial_posts = [x for x in df_comments[comment_indecies]\
                       .groupby('title').apply(pd.DataFrame.first_valid_index)]

replacement_comments = df_comments.iloc[index_initial_posts]
# Search for % symbol in comments
saving_found = ["%" in str(comment) for comment in replacement_comments.comments]
print(sum(saving_found), "row(s) with missing 'saving' data have a '%' symbol in their title") 
replacement_comments[saving_found].comments.iloc[0]
df.saving.value_counts()
# Values in "saving" without % symbols
no_missing_savings = df.iloc[df.saving.dropna().index]
dollar_savings = no_missing_savings[["%" not in str(saving) for saving in no_missing_savings.saving]]
print("Entries in 'saving' without % symbol:", dollar_savings.shape[0])
existing_price = dollar_savings.price.notnull()
print("Corresponding entries with non-missing values in 'price':", existing_price.sum())
# "$" savings with missing 'price' data
no_price_index = existing_price[existing_price == False].index

# Verify data
df.iloc[no_price_index]
# Delete "$" savings without price data
df.loc[no_price_index, "saving"] = np.nan

# Verify changes
df.iloc[no_price_index]
# Regular expressions for "%" and "$" savings
regex_percent = "(\d+\.*\d*\s*%)|(%\s*\d+\.*\d*)"
regex_dollar = "(\d+\.*\d*\s*\$)|(\$\s*\d+\.*\d*)"

print(re.search(regex_percent, "20%"))
# Convert savings to proportions from 0-1
new_savings = []
for index in range(df.shape[0]):
    saving = str(df.iloc[index].saving)
    # match objects
    percent = re.search(regex_percent, saving)
    dollar = re.search(regex_dollar, saving)
    
    if percent != None:
        saving = percent.group()
        saving_clean = float(saving.replace("%","").replace(",","").strip())
        new = float(saving_clean/100)
    elif dollar != None:
        saving = dollar.group()
        saving_clean = float(saving.replace("$","").replace(",","").strip())
        price = df.iloc[index].price
        if price > 0:
            new = float(saving_clean/(price + saving_clean))
        else:
            new = 1.0
    elif saving != "nan":
        saving = re.search("\d+\.*\d*", saving)
        if saving != None:
            saving = float(saving.group())
            price = df.iloc[index].price
            new = float(saving/(price + saving))
        else:
            new = np.nan
    else:
        new = np.nan
    new_savings.append(new)
    
df["new_saving"] = new_savings
df[df.new_saving.notnull()].loc[:,['price','saving','new_saving']]
df.info()
df[df.saving.notnull() &  df.new_saving.isnull()].loc[:,['price','saving','new_saving']]
df.saving = df.new_saving.astype('float')
df.drop(["new_saving"], axis=1, inplace=True)
df.info()
df.describe()
# Less than 0 replies
df[df.replies < 0]
df.drop([328], axis=0, inplace=True)
df[df.replies < 0]
df.describe()
