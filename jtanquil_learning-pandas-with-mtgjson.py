# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# read sets into a DataFrame; index is chosen as the orient so the set names are the indices
all_sets = pd.read_json("../input/AllSets-x.json", orient = "index")
all_sets.head()
# the set names are the indices, and the columns are the attributes of each set (border, cards, etc)
all_sets.shape
all_sets.columns
all_sets.describe()
# the cards column contains the cards of each set in json format, so each set of cards can be
# converted from a json object into a DataFrame
all_sets.cards = all_sets.cards.apply(lambda x: pd.read_json(json.dumps(x), orient = "records"))
all_sets.cards["RAV"].head()
# the shape of this DataFrame gives the number of cards in the set
all_sets.cards["RAV"].shape
setSizeCol = all_sets.apply(lambda x: x.cards.shape[0], axis = 1)
all_sets = all_sets.assign(setSize = setSizeCol)
all_sets.sample(10)
# before analyzing the dataset we remove some pathological cards and sets.
# first we remove sets not intended for tournament play.
# these include Un-sets, certain promotional cards, online-only Vanguard avatars, etc.
invalid_sets = ["UGL", "UNH", "pCEL", "pHHO", "VAN"]

def test_invalid_setcode(s, invalid_sets):
    
    for setname in invalid_sets:
        if s == setname:
            return True
        
    return False

all_sets = all_sets.loc[~all_sets.code.map(lambda x: test_invalid_setcode(x, invalid_sets))]
# we also remove cards that don't have the "typical" format of a Magic card
# these include cards specific to the Planechase and Archenemy formats (planes, schemes, etc),
# cards with the Conspiracy card type, and token cards
card_layouts = ["double-faced", "flip", "leveler", "meld", "normal", "split"]

all_sets.cards = all_sets.cards.apply(lambda x: x.loc[x.layout.map(lambda y: y in card_layouts)])
all_sets.cards = all_sets.cards.apply(lambda x: x.loc[x.types.map(lambda y: y != ["Conspiracy"])])
# next we modify creature cards with variable power/toughness - for the sake of numerical analysis, it
# is simpler to remove these values so the power and toughness columns can be cast as numeric columns.
def testfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def new_pt(s):
    if testfloat(s):
        return float(s)
    else:
        return np.nan
    
def fix_pts(c):
    col_list = list(c.columns)
    
    if "power" in col_list and "toughness" in col_list:
        c.loc[:, "power"] = pd.to_numeric(c.loc[:, "power"], errors = "coerce")
        c.loc[:, "toughness"] = pd.to_numeric(c.loc[:, "toughness"], errors = "coerce")
    
    return c
    
all_sets.cards = all_sets.cards.apply(lambda x: fix_pts(x))
# we remove columns that won't be useful in our analysis.
cols_to_remove = ["multiverseid", "imageName", "border", "mciNumber", "foreignNames",
                  "originalText", "originalType", "source"]

all_sets.cards = all_sets.cards.apply(lambda x: x.loc[:, list(set(x.columns) - set(cols_to_remove))])
# we standardize the columns of each cards DataFrame by taking the set-theoretic union of the columns
# and appending the remaining columns to each DataFrame.
union_set = set()
set_cols = all_sets.cards.map(lambda x: set(x.columns))

for setname in set_cols.index:
    union_set = union_set | set_cols[setname]
    
union_set
def addcols(cards, union_set):
    unused_cols = union_set - set(cards.columns)
    new_cols = pd.DataFrame(data = None, index = cards.index, columns = list(unused_cols))
    return cards.join(new_cols)
    
# after appending the columns we sort them in alphabetical order    
all_sets.cards = all_sets.cards.apply(lambda x: addcols(x, union_set))
all_sets.cards = all_sets.cards.apply(lambda x: x.reindex_axis(sorted(list(x.columns)), axis = 1))
# now we can start preparing the all_cards DataFrame, which will be a list of every tournament-legal
# Magic card
# first we select the columns from the cards DataFrames that will be useful
all_cards_columns = ['names', 'layout', 'manaCost', 'cmc', 'colors', 'colorIdentity',
                    'supertypes', 'types', 'subtypes', 'text', 'power', 'toughness',
                    'loyalty', 'rulings', 'foreignNames', 'printings', 'legalities']
# set the index of all_cards to be the name column, so we can search cards by name
all_cards = pd.DataFrame(data = None, columns = all_cards_columns)
all_cards.rename_axis("name", inplace = True)
all_cards.head()
# we want to preserve the printing/rarity information in all_cards; we represent this information
# as a dictionary where the key/value pairs are printings and rarities
def convert_printings(x, set_name):
    x["printings"] = dict.fromkeys(x["printings"])
    x["printings"].update({set_name : x["rarity"]})
    
    return x

def convert_row(row):
    row["cards"] = row["cards"].apply(lambda x: convert_printings(x, row["code"]), 
                                      axis = 1).set_index("name")
    
    return row

def filter_columns(row, all_cards_cols):
    set_cols = list(row.columns)
    intersection = list(set(set_cols) & set(all_cards_cols))
    
    return row.filter(intersection)
only_cards = all_sets.apply(lambda x: convert_row(x), axis = 1)["cards"]
only_cards = only_cards.apply(lambda x: filter_columns(x, all_cards_columns))
test = only_cards["RAV"]
test.head()
all_cards = pd.concat(list(only_cards))
all_cards.sample(10)
# there are a non-tournament legal cards remaining in this list, reprinted as promos, so we remove
# those cards from the list
all_cards = all_cards.loc[~(all_cards.printings.map(lambda x: bool(set(invalid_sets) & set(x)))
              & all_cards.supertypes.map(lambda x: x != ["Basic"]))]
all_cards.loc["Lightning Bolt"]
# merges a list of dictionaries where for each key, only one dictionary from the list will have a
# non-null value corresponding to the key. The keys of the merged dictionary will be the union of 
# the keys of the dictionaries in the list, and the corresponding value will be that non-null value
# corresponding to the key.
def merge_dicts(dicts):
    merged_dicts = {}
    
    for d in dicts:
        for k, v in d.items():
            if bool(v):
                merged_dicts.update({k : v})
    
    return merged_dicts
# loop that iterates through unique cardnames - for each cardname, check whether the card has reprints,
# and if so, update the first entry in the list of reprints with the merged printing/rarity dictionary
for cardname in all_cards.index.unique():
    reprints = all_cards.loc[cardname]
    
    # this checks that the DataFrame above actually has more than 1 card - if it had only one, then
    # reprints would instead be a column where the 16 attributes of the card are the rows
    if reprints.shape != (16,):
        merged_dicts = merge_dicts(list(reprints.printings))
        reprints.iat[0, list(reprints.columns).index("printings")].update(merged_dicts)
# for each reprinted card, the first reprint has the completed printing/rarity dictionary, so we can get
# rid of every other duplicate
all_cards = all_cards[~all_cards.index.duplicated(keep = "first")]
all_cards.describe()
all_cards.sample(10)
colorless = all_cards.loc[all_cards.colors.isnull() &
              ~all_cards.types.apply(lambda x: "Land" in x)]
all_cards.loc[colorless.index, "colors"] = colorless.colors.apply(lambda x: [])
all_cards.loc["Umezawa's Jitte"]
colors = ["White", "Blue", "Black", "Red", "Green"]
def subsets(lst):
    powerset = []
    
    for i in range(len(lst)):
        powerset += map(lambda x: list(x), list(it.combinations(lst, i)))
        
    powerset.append(lst)
    return powerset
color_combos = subsets(colors)
subsets_by_color = {}

for color_combo in color_combos:
    cards = all_cards.loc[all_cards.colors.apply(lambda x: x == color_combo)]
    subsets_by_color.update({tuple(color_combo) : cards})
cmcs = all_cards.loc[:, "cmc"].dropna()
def plot_int_hist(df_, title, x_axis, y_axis, fig_x, fig_y):
    df = df_.dropna()
    num_bins = len(np.unique(df.values))
    
    fig, ax = plt.subplots(figsize = (fig_x, fig_y))
    
    n, bins, patches = ax.hist(df, num_bins, normed = True)
    
    df_mean = df.mean()
    df_std = df.std()
    y = mlab.normpdf(bins, df_mean, df_std)
    
    ax.plot(bins, y, '--')
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    ax.set_title(title)
    plt.text(10, 0.20, "mean = " + str(round(df_mean, 5)))
    plt.text(10, 0.18, "stdev = " + str(round(df_std, 5)))
    
    fig.tight_layout()
    plt.show()
plot_int_hist(cmcs, title = "Distribution of Converted Mana Cost - All Nonland Cards",
              x_axis = "CMC", y_axis = "Percentage", fig_x = 12, fig_y = 8)
lm_pt_cmc = all_cards.loc[:, ["power", "toughness", "cmc"]]
lm_pt_cmc = lm_pt_cmc.loc[lm_pt_cmc.power.notnull() | lm_pt_cmc.toughness.notnull()]
fig, hm = plt.subplots(figsize = (15, 10))

hm.hist2d(lm_pt_cmc.power, lm_pt_cmc.toughness, bins = np.arange(-1.5, 16.5), range = ((-1, 16), (-1, 16)), 
          cmap = "summer", norm = matplotlib.colors.LogNorm())
#hm.hexbin(lm_pt_cmc.power, lm_pt_cmc.toughness, gridsize = 17, bins = "log", cmap = "summer")
hm.set_xlabel("Power")
hm.set_ylabel("Toughness")
hm.set_xticks(np.arange(-1, 16))
hm.set_yticks(np.arange(-1, 16))
hm.set_title("Power/Toughness Heatmap")
avg_cmc_pivot = pd.pivot_table(data = lm_pt_cmc, values = "cmc", index = ["power", "toughness"])
avg_cmc_pivot.index
len(avg_cmc_pivot)
avg_cmc_pivot.loc['power' == 13]
avg_cmc_pivot.loc[0.0]
unstacked = avg_cmc_pivot.unstack()
unstacked
unstacked.index
unstacked.columns
df1 = avg_cmc_pivot.index.to_frame()
df1["cmc"] = avg_cmc_pivot["cmc"]
df1
df1 = df1.loc[df1.cmc.notnull()]
df1
new_index = list(np.arange(0, 99))
df2 = pd.DataFrame(data = None, index = new_index)
df2['power'] = df1['power'].tolist()
df2['toughness'] = df1['toughness'].tolist()
df2['cmc'] = df1['cmc'].tolist()
df2
fig, hm = plt.subplots(figsize = (15, 15))

hm.scatter(df2.power, df2.toughness, s = 2250, c = df2.cmc, marker = "s", cmap = "summer")
# put the p/t heatmap and avg cmc scatter plot side by side
fig2 = plt.figure()
plt.show()

