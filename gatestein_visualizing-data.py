import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn
food = pd.read_table("../input/en.openfoodfacts.org.products.tsv")
food.head()
food.isnull().mean(axis=0)
plt.figure(figsize=(5, 20))

food.isnull().mean(axis=0).plot.barh()

plt.title("Proportion of NaNs in each column")
def no_null_objects(data, columns=None):

    """

    selects rows with no NaNs

    """

    if columns is None:

        columns = data.columns

    return data[np.logical_not(np.any(data[columns].isnull().values, axis=1))]
def splitDataFrameList(df, target_column, separator):

    ''' df = dataframe to split,

    target_column = the column containing the values to split

    separator = the symbol used to perform the split



    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 

    The values in the other columns are duplicated across the newly divided rows.

    '''

    def splitListToRows(row, row_accumulator, target_column, separator):

        split_row = row[target_column].split(separator)

        for s in split_row:

            new_row = row.to_dict()

            new_row[target_column] = s

            row_accumulator.append(new_row)

    new_rows = []

    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))

    new_df = pd.DataFrame(new_rows)

    return new_df
food_countries = splitDataFrameList(no_null_objects(food, ["countries_en"]), "countries_en", ",")

countries = food_countries["countries_en"].value_counts()
countries[:20][::-1].plot.barh()
print(countries[20:].index)

print("Max count:", countries[20:].max())
countries_matrix = pd.DataFrame(np.zeros((20, 20)), countries[:20].index, countries[:20].index)

idxs = ~food.origins.isnull() & ~food.countries_en.isnull()

for from_, to_ in zip(food["origins"][idxs], food["countries_en"][idxs]):

    from_list = filter(lambda x: x in countries[:20].index, from_.split(","))

    to_list = filter(lambda x: x in countries[:20].index, to_.split(","))

    for from_c in from_list:

        for to_c in to_list:

            countries_matrix[from_c][to_c] += 1
# Replace non-ascii country name

countries_matrix.columns = countries_matrix.columns[:-2].values.tolist() + ["Reunion"] + [countries_matrix.columns[-1]]

countries_matrix.index = countries_matrix.columns[:-2].values.tolist() + ["Reunion"] + [countries_matrix.columns[-1]]
seaborn.heatmap(countries_matrix)
seaborn.heatmap(countries_matrix.drop(["France"], axis=0)

                .drop(["France"], axis=1))
df = no_null_objects(food_countries[["countries_en", "ingredients_from_palm_oil_n"]])

df[df["countries_en"].isin(countries[:20].index)].groupby("countries_en").mean().plot.barh()
df = food_countries[["countries_en", "ingredients_from_palm_oil_n"]]

df["nan_palm_oil"] = ~ df["ingredients_from_palm_oil_n"].isnull()

df[df["countries_en"].isin(countries[:20].index)].groupby("countries_en").mean().plot.barh()
import string
def imshow_letters_dist_by_component(df, component_column):

    products_with_comp = no_null_objects(df[[component_column, "generic_name"]])   

    numbers = np.zeros((26, 2))

    for obj in products_with_comp.values:

        for let in obj[1].lower():

            if let in string.ascii_letters:

                numbers[string.ascii_letters.find(let), int(obj[0]>0)] += 1

    numbers /= numbers.sum(axis=0)[np.newaxis, :]

    seaborn.heatmap(pd.DataFrame(numbers, list(string.ascii_letters[:26]), 

                                 ["No "+component_column.replace("_100g", ""), 

                                  "With "+component_column.replace("_100g", "")]).T, square=True, cbar=False)
imshow_letters_dist_by_component(food, "alcohol_100g")

# alcohol name often contains "i" and "r"
imshow_letters_dist_by_component(food, "vitamin_c_100g")
imshow_letters_dist_by_component(food, "calcium_100g")
imshow_letters_dist_by_component(food, "vitamin_e_100g")

# products with "p", "o", "r" and "t" almost never have vitamin E
imshow_letters_dist_by_component(food, "ingredients_that_may_be_from_palm_oil_n")

# no difference
food["datetime"] = food["created_datetime"].apply(str).apply(lambda x: x[:x.find("T")])
from datetime import datetime
min_date = datetime.strptime(food["datetime"].min(), "%Y-%m-%d")
products_num_by_day = np.zeros(2000)

num_er = 0.0

for obj, country in zip(food["datetime"], food["countries_en"]):

    try:

        day = (datetime.strptime(obj, "%Y-%m-%d") - min_date).days

        products_num_by_day[day] += 1

    except:

        num_er += 1

print(num_er / food.shape[0])
plt.plot(np.cumsum(products_num_by_day))

plt.xlabel("Number of days from start date")

plt.ylabel("Total num products each day")
def apply_func(x):

    try:

        return (datetime.strptime(x, "%Y-%m-%d") - min_date).days

    except:

        return None

food["exists_days"] = food["datetime"].apply(apply_func)
plt.scatter(food["exists_days"], food["additives_n"])

plt.xlabel("A number of days from start date")

plt.ylabel("number of additives")
from pandas.tools.plotting import scatter_matrix
food_nutrients = no_null_objects(food[["carbohydrates_100g", "fat_100g", "proteins_100g", "labels_en"]])
food_nutrients["labels_en"] = food_nutrients["labels_en"].str.contains("Vegan")
plt.figure(figsize=(20, 20))

seaborn.pairplot(food_nutrients, hue="labels_en", diag_kind="kde")
food_with_labels = no_null_objects(food, ["labels_en"])
key = "energy_100g"

seaborn.kdeplot(food[key], label="All")

seaborn.kdeplot(food_with_labels[food_with_labels["labels_en"].str.contains("Vegan")][key], label="Vegan")

plt.title("KDE of energy in 100g")