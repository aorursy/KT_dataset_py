import pandas as pd



# Import all of my data

sales_train = pd.read_csv("../input/sales_train.csv")

test = pd.read_csv("../input/test.csv")

items = pd.read_csv("../input/items.csv")

item_categories = pd.read_csv("../input/item_categories.csv")

shops = pd.read_csv("../input/shops.csv")





# This step just helps me loop over these variables in the future

# I'm lazy and always try to write less code

from collections import namedtuple

Dataset = namedtuple("Dataset", "name df")

datasets = [Dataset(name="Sales train", df=sales_train),

            Dataset(name="Test", df=test),

            Dataset(name="Items", df=items),

            Dataset(name="Item categories", df=item_categories),

            Dataset(name="Shops", df=shops)]
# Get the size info

for d in datasets:

    print(d.name + ": " + str(d.df.shape))
sales_train.sample(random_state=4)
test.sample(random_state=4)
items.sample(random_state=4)
item_categories.sample(random_state=4)
shops.sample(random_state=4)
# Get the names of columns

print("Column names for each dataset")

for d in datasets:

    print('{:<16}'.format(d.name + ":"), end=" ")

    print(*d.df.columns.tolist(), sep=", ")
# Get some samples of values

for d in datasets:

    cols = d.df.columns

    print(d.name + " column examples")

    for c in cols:

        print('    ' + c + ": ", end = " ")

        print(*d.df[c].sample(5, random_state=4), sep="; ")
# Check the types

for d in datasets:

    print("• " + d.name + " types")

    print(d.df.dtypes, end="\n\n")
# Convert date to a datetime

sales_train["date"] = pd.to_datetime(sales_train["date"], format='%d.%m.%Y')
# Check if all of the item count days are ints

all(sales_train["item_cnt_day"] == sales_train["item_cnt_day"].astype(int))
# Looks for NAs

for d in datasets:

    print("• " + d.name + " number of missing values")

    print(d.df.isna().sum(), end="\n\n")
# Looks for duplicates

for d in datasets:

    print(d.name + " any duplicates?", end=" ")

    print(d.df.duplicated().any())
sales_train[sales_train.duplicated(keep=False)]
sales_train = sales_train.drop_duplicates(keep='first')
def merge_data(df):

    """Merges data from the inputed dataframe to return shop and item details.

    

    Paramters

    ---------

    df : DataFrame

        Train or test dataframe to merge lookup values to

    

    Returns

    -------

    DataFrame

        Inputted dataframe with columns shop_name, item_name,

            item_category_id, and item_category_name appended to it

    

    Raises

    ------

    Exception

        If there is a value in the inputted df that is not present in the lookup dataframes

    """

    # validate ensures that the keys are unique in the left dataset

    shop_merge = pd.merge(df, shops,

                          on="shop_id",

                          how="left",

                          validate="m:1",

                          indicator="_shop_merge")

    # This exception catches if the lookup could not be found in the right datatset

    if any(shop_merge["_shop_merge"]=="left_only"):

        raise Exception("Could not lookup value for shop")



    item_merge = pd.merge(shop_merge, items, 

                          on="item_id", 

                          how="left", 

                          validate="m:1", 

                          indicator="_item_merge")

    if any(item_merge["_item_merge"]=="left_only"):

        raise Exception("Could not lookup value for item")

    

    item_cat_merge = pd.merge(item_merge, item_categories, 

                              on="item_category_id", 

                              how="left", 

                              validate="m:1", 

                              indicator="_cat_merge")

    if any(item_cat_merge["_cat_merge"]=="left_only"):

        raise Exception("Could not lookup value for item category")



    # Drop merge columns

    item_cat_merge = item_cat_merge.drop(columns=["_shop_merge", "_item_merge", "_cat_merge"])

    return item_cat_merge 



sales_train_merge = merge_data(sales_train)

test_merge = merge_data(test)
sales_train_merge.sample(random_state=4)
from matplotlib import pyplot

sales_train_merge.plot.scatter(x="date_block_num", y="item_price")

pyplot.show()
sales_train_merge.loc[sales_train_merge["item_price"]>300000]
sales_train_merge.loc[sales_train_merge["item_id"]==6066]
sales_train_merge.plot.scatter(x="date_block_num", y="item_cnt_day")

pyplot.show()
sales_train_merge.loc[sales_train_merge["item_cnt_day"] > 500].sort_values("item_cnt_day", ascending=False)
print("Min date ", min(sales_train_merge["date"]))

print("Max date ", max(sales_train_merge["date"]))
(sales_train_merge["item_cnt_day"] >= 0 ).all()
sales_train_merge.loc[sales_train_merge["item_cnt_day"] < 0]
sales_train_merge.loc[sales_train_merge["item_cnt_day"]==0]
sales_train_merge.loc[sales_train_merge["item_price"] < 0 ]
sales_train_merge.loc[(sales_train_merge["item_id"] == 2973 ) & (sales_train_merge["shop_id"] == 32 )]
sales_train_merge.loc[sales_train_merge["item_price"] < 0, "item_price"] = None

sales_train_merge["item_price"] = sales_train_merge["item_price"].fillna(sales_train_merge["item_price"].median())
months = sales_train_merge["date"].dt.month - 1

years = (sales_train_merge["date"].dt.year - 2013)*12



my_dateblock = months + years

all(my_dateblock == sales_train_merge["date_block_num"])
train_set = set([k for k in sales_train_merge[["shop_id", "item_id"]].itertuples(index=False)])

test_set = set([k for k in test_merge[["shop_id", "item_id"]].itertuples(index=False)])

new = test_set - train_set

print("New or never sold item + shop combinations in test", len(new), "out of", len(test_set))
test_merge.columns
sales_train_merge[["item_price", "item_cnt_day"]].describe()
sales_train_agg = sales_train_merge.groupby(["item_id", "date_block_num"])["item_cnt_day"].sum()

sales_train_agg.describe()
pyplot.figure(1, figsize=(12,6))

pyplot.suptitle("Distribution of units sold")

pyplot.subplot(121)

xlims = (-20, 100)  # set x-limits

sales_train_agg.hist(range=xlims, bins=12)

pyplot.subplot(122)

sales_train_agg.to_frame().boxplot()

pyplot.show()
sales_train_agg.to_frame().boxplot()

ax = pyplot.gca()

ax.set_ylim(xlims)

pyplot.show()
print("Number of unique shops: {0:d}".format(sales_train_merge["shop_id"].nunique()))

print("Number of unique item names: {0:d}".format(sales_train_merge["item_name"].nunique()))

print("Number of unique item ids: {0:d}".format(sales_train_merge["item_id"].nunique()))

print("Number of unique categories: {0:d}".format(sales_train_merge["item_category_id"].nunique()))
pyplot.figure(figsize=(16,8))

sales_train_merge.groupby("date")["item_cnt_day"].sum().plot()

pyplot.show()
pyplot.figure(figsize=(16,8))

sales_train_merge.groupby("date_block_num")["item_cnt_day"].sum().plot()

pyplot.show()
pyplot.figure(figsize=(16,8))

sales_train_merge.groupby(pd.Grouper(key="date", freq="A"))["item_cnt_day"].sum().plot.bar()

pyplot.show()
# There's too many to plot, so let's just look at top-selling categories

topselling = sales_train_merge.groupby("item_category_id")["item_cnt_day"].sum().nlargest(10).index

sales_topselling = sales_train_merge.loc[sales_train_merge["item_category_id"].isin(topselling)]

sales_topselling.groupby(["date_block_num", "item_category_id"])["item_cnt_day"].sum().unstack().plot(figsize=(16,8))

pyplot.show()
sales_topselling["item_category_name"].unique()