# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load files

training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")

testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")

attribute_data = pd.read_csv('../input/attributes.csv')

descriptions = pd.read_csv('../input/product_descriptions.csv')
# Look at data column names

print("Test data columns")

print(testing_data.columns)

print("Attribute data columns")

print(attribute_data.columns)

print("Description data columns")

print(descriptions.columns)
# Merge descriptions

training_data = pd.merge(training_data, descriptions, 

                         on="product_uid", how="left")

print(training_data.columns)
# See what the data looks like

print(training_data)
# Merge product counts

product_counts = pd.DataFrame(pd.Series(training_data.groupby(

['product_uid']).size(), name='product_count'))



training_data = pd.merge(training_data, product_counts, 

                        left_on="product_uid", right_index=True,

                        how="left")

print(training_data[:50])
print(attribute_data)
# merge brand names

brand_names = attribute_data[attribute_data.name == "MFG Brand Name"][['product_uid', 'value']].rename(columns={"value": "brand_name"})

training_data = pd.merge(training_data, brand_names, on="product_uid", how="left")
print(brand_names)
# Let's check to see if there are any missing values in our data

print(str(training_data.info()))
# We note that brand_name has null values and replace them with "Unknown"

# Fill all products with no brand name

training_data.brand_name.fillna("Unknown", inplace=True)



# Print out the names of the products without a brand

print(training_data.product_title[training_data.brand_name == "Unknown"])
# Description of training_data

print(str(training_data.describe()))



# We note that the mean relevance score is a 2.38 which shows the data 

# tends towards higher relevance scores.  On average there are about 2 

# of each product with a max of one product being returned for 21 different

# search queries.
# Let's check the distribution and spread of the relevance column

training_data.relevance.hist()

training_data.relevance.value_counts()
# Now let's go through how many indoor and outdoor products we have

attribute_data.value[attribute_data.name == "Indoor/Outdoor"].value_counts()
# What is the distribution of the product descriptions?

(descriptions.product_description.str.len()/5).hist(bins=30)
# What does the distribution of the string lengths for product title?

(training_data.product_title.str.len() / 5).hist(bins=30)
# Now let's get an idea of how many words are in each search_term

# Are people writing really long specific queries or general 

# one word queries?



(training_data.search_term.str.count("\\s+") + 1).hist(bins=30)



(training_data.search_term.str.count("\\s+") + 1).describe()



# We note that people search with 3 words on average with a 

# standard deviation of 1 word. This means that 66% of the data 

# lies within 2-4 word search queries
# How many rows are there for each product_uid?

testing_data.product_uid.value_counts()
# What words show up the most in the training_data? How does this

# compare to the most frequent words in the test_data?

# It should be a good match if the word is in the query and the 

# product name



import collections



words_search = collections.Counter()

for title in training_data.search_term:

    words_search.update(title.lower().split())

    

total_search = sum(words_search.values())



words_query = collections.Counter()

for title in training_data.product_title:

    words_query.update(title.lower().split())

    

total_query = sum(words_query.values())

print(words_search)

print(words_query)
# How many search terms contain numbers?

print("Contains numbers", training_data.search_term.str.contains("\\d", case=False).value_counts())
# Now we know that we want to use the attribute data.  Let's look

# at which variables show up most often in the attribute data.



attribute_counts = attribute_data.name.value_counts()

print(attribute_counts)

# This shows us that ~86,000 product descriptions include the brand

# name, ~41,000 include the color family, etc.  What are the values

# of these variables that come up the most often?



def summarize_values(name, values):

    values.fillna("", inplace=True)

    counts = collections.Counter()

    for value in values:

        counts[value.lower()] += 1

    

    total = sum(counts.values())

    print("{} counts ({:,} values)".format(name, total))

    for word, count in counts.most_common(20):

        print("{}: {:.1f}% ({:,})".format(word, 100. * count / total, count))



for attribute_name in ["Color Family", "Color/Finish", "Material", "MFG Brand Name", "Indoor/Outdoor", "Commercial / Residential"]:

    summarize_values("\n" + attribute_name, attribute_data[attribute_data.name == attribute_name].value)
# Now we know that we want to use the attribute data.  Let's look

# at which variables show up most often in the attribute data.



attribute_counts = attribute_data.name.value_counts()

print(attribute_counts)

# This shows us that ~86,000 product descriptions include the brand

# name, ~41,000 include the color family, etc.  What are the values

# of these variables that come up the most often?



def summarize_values(name, values):

    values.fillna("", inplace=True)

    counts = collections.Counter()

    for value in values:

        counts[value.lower()] += 1

    

    total = sum(counts.values())

    print("{} counts ({:,} values)".format(name, total))

    for word, count in counts.most_common(20):

        print("{}: {:.1f}% ({:,})".format(word, 100. * count / total, count))



for attribute_name in ["Color Family", "Color/Finish", "Material", "MFG Brand Name", "Indoor/Outdoor", "Commercial / Residential"]:

    summarize_values("\n" + attribute_name, attribute_data[attribute_data.name == attribute_name].value)