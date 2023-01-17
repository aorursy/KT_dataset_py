%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# load files
training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")
testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")
attribute_data = pd.read_csv("../input/attributes.csv")
descriptions = pd.read_csv("../input/product_descriptions.csv")

# merge descriptions
training_data = pd.merge(training_data, descriptions, on="product_uid", how="left")

# merge product counts
product_counts = pd.DataFrame(pd.Series(training_data.groupby(["product_uid"]).size(), name="product_count"))
training_data = pd.merge(training_data, product_counts, left_on="product_uid", right_index=True, how="left")

# merge brand names
brand_names = attribute_data[attribute_data.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand_name"})
training_data = pd.merge(training_data, brand_names, on="product_uid", how="left")
training_data.brand_name.fillna("Unknown", inplace=True)
print(str(training_data.info()))
print(str(training_data.describe()))
training_data[:50]
print(attribute_data.name.value_counts())
print(attribute_data.value[attribute_data.name == "Indoor/Outdoor"].value_counts())
training_data["id_bins"] = pd.cut(training_data.id, 20, labels=False)
print(training_data.corr(method="spearman"))
training_data.describe()
training_data.relevance.hist()
training_data.relevance.value_counts()
(descriptions.product_description.str.len() / 5).hist(bins=30)
(training_data.product_title.str.len() / 5).hist(bins=30)
(training_data.search_term.str.len() / 5.).hist(bins=30)
(training_data.search_term.str.count("\\s+") + 1).hist(bins=30)
testing_data.product_uid.value_counts()
training_products = training_data.product_uid.value_counts()
testing_products = testing_data.product_uid.value_counts()
training_norm = np.sqrt((training_products ** 2).sum())
testing_norm = np.sqrt((testing_products ** 2).sum())
product_uid_cos = (training_products * testing_products).sum() / (training_norm * testing_norm)
print("Product distribution cosine:", product_uid_cos)
import collections

chars = collections.Counter()
for title in training_data.product_title:
    chars.update(title.lower())
total = sum(chars.values())

print("Title char counts")
for c, count in chars.most_common(30):
    print("0x{:02x} {}: {:.1f}%".format(ord(c),  c, 100. * count / total))
    
words = collections.Counter()
for title in training_data.search_term:
    words.update(title.lower().split())

total = sum(words.values())
print("Search word counts")
for word, count in words.most_common(200):
    print("{}: {:.1f}% ({:,})".format(word, 100. * count / total, count))
print("Indoor/outdoor", training_data.search_term.str.contains("indoor|outdoor|interior|exterior", case=False).value_counts())
print("Contains numbers", training_data.search_term.str.contains("\\d", case=False).value_counts())
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

