# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import HTML, display

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv('../input/train.tsv', sep='\t')

def show_table(data):

    # table header

    table_data = [[""]]

    table_data[0] += list(data.columns)



    # add column type

    row = ["type"]

    for c in data:

        row.append(data[c].dtype)

    table_data.append(row)



    # add nan count

    row = ["nan"]

    for c in data:

        total = len(data[c])

        nans = total - data[c].count()

        row.append(str(nans) + " (" + str(round(100 * nans/total, 2)) + "%)")

    table_data.append(row)







    # add example

    for i in range(3):

        row = ["example " + str(i)]

        for c in data:

            row.append(data[c].iloc[i])

        table_data.append(row)







    display(HTML(

    '<table><tr>{}</tr></table>'.format(

        '</tr><tr>'.join(

            '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in table_data)

        )

    ))



show_table(data)

print("Item no description yet", data.item_description.where(data.item_description == 'No description yet').count(),data.item_description.where(data.item_description == 'No description yet').count()/len(data.name), "%")

data.describe(include = 'all')
def sum_no_brand_percentage(x):

    return x.isnull().sum() * 100 / len(x)

def sum_no_brand(x):

    return x.isnull().sum()

def abbrev(x):

    return data.loc[x.index[0]].category_name

no_brands = data.groupby('category_name')['brand_name'].agg([sum_no_brand, sum_no_brand_percentage, 'size', abbrev]).sort_values('sum_no_brand', ascending=False)

no_brands = no_brands.head(40)

f, ax = plt.subplots(figsize=(6, 17))





# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(x="size", y="abbrev", data=no_brands,

            label="Total", color="b")



# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x="sum_no_brand", y="abbrev", data=no_brands,

            label="No brand given", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set( ylabel="",

       xlabel="Items per category")

sns.despine(left=True, bottom=True)
data[data.category_name=="Other/Other/Other"].name.head(40)
# let's fill the nan values



# replace category nans with mode

data.category_name.fillna(data.category_name.mode()[0], inplace = True)



# replace brand nans with mode

data.brand_name.fillna(data.brand_name.mode()[0], inplace = True)



# raplace description nans with "No description yet"

data.item_description.fillna("No description yet", inplace = True)

data.item_description.replace(["No description yet"],[""], inplace = True)



# remove rows with zero price

data = data[data.price > 0]



# remove rows that still have nans

data.dropna(inplace=True)



# result

show_table(data)
lijstje = data.brand_name.tolist()

for el in lijstje:

    el = el.lower()

    if el[:3] == "the":

        print(el)
data.item_condition_id.hist(bins=5)



data.head(10000).boxplot(column='price' ,by='item_condition_id').set_ylim(0, 700)
def count_slash(row):

    return row.count("/")



def root(row):

    return row[:row.find('/')]



print("Depth of category")

print("Men/Shoes/Sandals -> depth = 2\n")

print(data.category_name.apply(count_slash).value_counts())

tmp = data.category_name.apply(count_slash)

print("Maar pas op, blijkbaar (zoals deze hieronder), hebben de items met meer categorie levels een categorie met slashes")

print(data.iloc[239])

print("-"*20)

print("\nRoot category distributions\n")

print(data.category_name.apply(root).value_counts())



data.price.hist(bins=10)

data.price.mean()
data.price[data.price < 200].hist(bins=10)

data.price[data.price < 200].mean()
data.price[data.price < 20].hist(bins=10)

n = len(data[data.price < 1])

print(data.price[data.price < 20].mean())

print("Zero priced products:", n, end="\n\n")

for i in range(n):

    print(data[data.price < 1].iloc[i]["name"])

    print(data[data.price < 1].iloc[i].item_description)

    print("-" * 10)
data.shipping.value_counts()
c = data.category_name.value_counts().index.tolist()[0]

cat_data = data[data.category_name == c]

prices = data[data.category_name == c].price

print("price range:", min(prices), "-", max(prices))

print("mean:", prices.mean())

print("std:", prices.std())



fig, saxis = plt.subplots(2, 2,figsize=(16,8))

sns.barplot(x = 'item_condition_id', y = 'price', data=cat_data, ax = saxis[0,0])

sns.pointplot(x = 'item_condition_id', y = 'price', data=cat_data, ax = saxis[0,1])

sns.barplot(x = 'item_condition_id', y = 'price', data=data, ax = saxis[1,0])

sns.pointplot(x = 'item_condition_id', y = 'price', data=data, ax = saxis[1,1])





print("-"*10)

print("corr price/condition", cat_data.corr()['price']['item_condition_id'])

print("-"*10)

print(cat_data.groupby(["item_condition_id"])["price"].mean())

print("-"*10)

print(cat_data.groupby(["item_condition_id"])["price"].std())

print("-"*10)

print("cat_n \t corr \t corr h/l_1 \t corr h/l_2")

table_data = [["cat_n", "corr", "corr h/l_1", "corr h/l_2"]]

for i in range(10):

    c = data.category_name.value_counts().index.tolist()[i]

    cat_data = data[data.category_name == c]

    cat_data["condition_hl_1"] = cat_data["item_condition_id"].replace([1, 2, 3, 4, 5],[0, 1, 1,1, 1])

    cat_data["condition_hl_2"] = cat_data.item_condition_id.replace([1, 2, 3, 4, 5],[0, 1, 2, 2, 2])

    table_data.append([i, round(cat_data.corr()['price']['item_condition_id'], 2), round(cat_data.corr()['price']['condition_hl_1'], 2), round(cat_data.corr()['price']['condition_hl_2'], 2)])



display(HTML(

    '<table><tr>{}</tr></table>'.format(

        '</tr><tr>'.join(

            '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in table_data)

        )

    ))

print("We splitten de categorie in 3'en \n")

print("Miss nog andere opties van de categorien overzetten, zijn de hoofdcats nodig? bla...")

data['C1'], data['C2'], data['C3'] = data['category_name'].str.split('/', 2).str



for col in ['C1', 'C2', 'C3']:

    encoder = LabelBinarizer()

    hot = encoder.fit_transform(data[col])

    print("Number for {} level categories: {}".format(col, len(hot[0])))

    data[col] = hot.tolist()



data['C2'].head(3).describe
print("Brand_name onehot encoden \n")



encoder = LabelBinarizer()

hot = encoder.fit_transform(data["brand_name"])

print("Number of brands:", len(hot[0]))

data["brand_encoded"] = hot.tolist()



data['brand_encoded'].head(3).describe
c = data.category_name.value_counts().index.tolist()[0]

prices = data[data.category_name == c].price

print("price range:", min(prices), "-", max(prices))

print("mean:", prices.mean())

print("std:", prices.std())



fig, saxis = plt.subplots(3, 2,figsize=(16,12))

for i in range(3):

    c = data.category_name.value_counts().index.tolist()[i]

    cat_data = data[data.category_name == c]

    sns.barplot(x = 'shipping', y = 'price', data=cat_data, ax = saxis[i,0])

    sns.pointplot(x = 'shipping', y = 'price', data=cat_data, ax = saxis[i,1])







print("-"*10)

print("corr price/condition", cat_data.corr()['price']['item_condition_id'])

print("-"*10)

print(cat_data.groupby(["shipping"])["price"].mean())

print("-"*10)

print(cat_data.groupby(["shipping"])["price"].std())

print("-"*10)

print("cat_n \t corr \t corr h/l_1 \t corr h/l_2")

table_data = [["cat_n", "corr"]]

for i in range(10):

    c = data.category_name.value_counts().index.tolist()[i]

    cat_data = data[data.category_name == c]

    table_data.append([i, round(cat_data.corr()['price']['shipping'], 2)])



display(HTML(

    '<table><tr>{}</tr></table>'.format(

        '</tr><tr>'.join(

            '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in table_data)

        )

    ))
