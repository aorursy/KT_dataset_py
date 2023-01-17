import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re

%matplotlib inline
df = pd.read_csv("../input/flipkart_com-ecommerce_sample.csv", na_values=["No rating available"])
df.info()

df.head()
# create new variable: discount_percent



df["discount_percent"] = ((df.retail_price - df.discounted_price)*100)/df.retail_price

df.discount_percent.head()
def get_nth_category(dataframe, level=1):

    """extract the level-n product category from the product category tree"""

    

    if level == 1:

        category = dataframe.product_category_tree.apply(lambda x: re.split(" >> ", x)[0]).str[2:]

    else:

        category = dataframe.product_category_tree.apply(lambda x: re.split(" >> ", x)[level:(level+1)])

    

    category = category.replace("[]", "[EMPTY_LEVEL]")  # this line does not work! Suggestions welcome :)



    return category
# print level 4 categories just to see if the function works

print(get_nth_category(df, level=4))
# get primary and secondary level product categories



df["primary_category"] = get_nth_category(df, level=1)

df["secondary_category"] = get_nth_category(df, level=2)
print(df.primary_category.head(5), "\n\n")

print(df.secondary_category.head(5))
# check missing values in the product's ratings column



print("Missing value percentage", "\n\nProduct rating: ", round(df.product_rating.isnull().sum()*100/df.shape[0], 2), "%",

      "\nOverall rating: ", round(df.overall_rating.isnull().sum()*100/df.shape[0], 2), "%")
# groupby using primary_category



groupby_df = pd.DataFrame(df.groupby("primary_category").agg({

    "discount_percent": [np.mean],

    "primary_category": ["count"]

}))



groupby_df.columns = ["_".join(col) for col in groupby_df.columns]

groupby_df = groupby_df.sort_values(by = ["primary_category_count"], ascending=False)

groupby_df = groupby_df[groupby_df.primary_category_count > 80]
groupby_df
# reset index to flatten column names as output by the groupby object



groupby_df.reset_index(inplace=True)
print(groupby_df.head())

print(groupby_df.info())

print(groupby_df.describe())
# product category vs product count



sns.barplot(data=groupby_df.sort_values(["primary_category_count"], ascending=False),

            y="primary_category", x = "primary_category_count")

plt.xlabel("Number of products")

plt.ylabel("Product Category")

# product category vs category discounts



sns.barplot(data=groupby_df.sort_values(by = ["discount_percent_mean"], ascending=False),

            y="primary_category", x = "discount_percent_mean")

plt.xlabel("Mean Discount Percentage")

plt.ylabel("Product Category")

def is_top_category(x):

    """return 1 if x is one of the top categories"""

    if x in list(groupby_df.primary_category):

        return 1

    else:

        return 0

    

df["is_top_category"] = df.primary_category.apply(is_top_category)
# subset df such that it only contains top 20 occurring categories

top_categories = df[df.is_top_category == 1]



plt.figure(figsize = [15,7])

sns.violinplot(data=top_categories, x = "primary_category", y = "discount_percent")

plt.ylabel("Discount Percentage")

plt.xlabel("Primary Product Categories")

plt.xticks(rotation=45)