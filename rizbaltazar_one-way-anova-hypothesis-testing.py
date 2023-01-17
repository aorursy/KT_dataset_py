# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from scipy.stats import levene, f_oneway, kruskal

from statsmodels.stats.multicomp import pairwise_tukeyhsd#, MultipleComparison



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re



house_prices_path = "/kaggle/input/house-prices-advanced-regression-techniques/"

house_prices_trainset = os.path.join(house_prices_path, "train.csv")

house_prices_description = os.path.join(house_prices_path, "data_description.txt")

description_file = open(house_prices_description, 'r')

description = description_file.read()



def what_is(term):

    if term in description:

        res = re.search(f"{term}:.*?(\\n(?=\w*:)|\Z)", description, re.DOTALL)

        if res is not None:

            print(res[0])

            return

    print(f"Term {term} not found")
full_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")

full_data.info()
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col="Id")

test_data.info()
import missingno



def get_missing(df):

    count = int(np.ceil(len(df.columns)/28))

    for i in range(0, count):

        start = i*28

        end = start + 27

        missingno.matrix(df.iloc[:,start:end])



def find_null_info(df):

    null_list = []

    for c in df.columns:

        no_null = df.isnull().sum()

        if no_null[c] > 0:

            null_list.append(c)

            print(c, "\t", no_null[c])

#             what_is(c)

    return null_list



def get_mean(df, feat, verbose=False):

    types = df[feat].unique()

    if verbose:

        print(types)

    means = {}

    for t in types:

        mean = df[df[feat]==t].SalePrice.mean()

        means[t] = mean

        if verbose:

            print("%s:\t%f" %(t, mean))

        

    return means



def sort_dict(df):

    scale_vars = []

    all_args = []

    features = df.columns

    for cf in features:

        scale_vars.append(cf)

        if cf != 'SalePrice':

            means = get_mean(df, cf)

            t = {k:v for k, v in sorted(means.items(), key=lambda item: item[1])}

            print("%s: %s" %(cf, tuple([*t])))

            all_args.append(tuple([*t]))

    return scale_vars, all_args



def cat_to_scale(df, col_name, *args):

    '''Need to be fed arguments in increasing order'''

    s = pd.Series(pd.Categorical(df[col_name]))

    if len(args) != len(s.cat.categories):

        s = s.cat.add_categories(set(args) ^ set(s.cat.categories))

    print(f"Renaming {col_name}: \t{[*args]} to: {list(i for i in range(len(args)))}")

    s = s.cat.rename_categories({arg : i for i, arg in enumerate(args)})

    return s
log_price = np.log(full_data.SalePrice)

full_data['SalePrice'] = log_price

sns.distplot(full_data['SalePrice'])
combined = pd.concat([full_data, test_data])

combined.info()
get_missing(combined)
clean_data = combined.copy()

cat_feat = clean_data.select_dtypes(include=['object'])

cat_feat['SalePrice'] = clean_data['SalePrice']

cat_feat.info()
train_clean = full_data.copy()

means = get_mean(train_clean, 'LotShape')

means
# scale_vars, all_args = sort_dict(cat_feat)
# for (var, args) in zip(scale_vars, all_args):

#     renamed_col = cat_to_scale(clean_data, var, *args)

#     clean_data[var] = renamed_col.values

    

# clean_data
clean_data.info()
# for c in clean_data.columns:

#     if str(clean_data[c].dtypes) == 'category':

#         clean_data[c] = clean_data[c].astype(int)



# clean_data.info()
train_clean = clean_data[clean_data['SalePrice'].notnull()]

train_clean.info()
house_style = "HouseStyle"

house_styles = train_clean[house_style].unique()

house_styles
what_is(house_style)
sns.boxplot(x=house_style, y="SalePrice", palette="GnBu_d", data=train_clean)
def unequal_var(df, feature, features):

    num_cols = 3

    num_feats = int(np.ceil(len(features)/num_cols))

    fig, ax = plt.subplots(num_feats, num_cols, figsize=(num_cols*10,num_feats*7))



    i = 0

    j = 0

    feat_arr = []

    for f in features:

        if j==num_cols:

            j = 0

            i += 1

        price = df[df[feature]==f].SalePrice

        feat_arr.append(np.array(price))

        sns.distplot(price, ax=ax[i][j])

        plt.setp(ax[i][j], ylabel=f)

        j += 1

    

    feat_tup = tuple(feat_arr)

    stat, p = levene(*feat_tup)

    print("Stat: %f; p-value: %.2e; Unequal variance: %s" %(stat, p, ('Yes' if p < 0.05 else 'No')))

    plt.legend()

    plt.show()

    

unequal_var(train_clean, house_style, house_styles)
price = train_clean[train_clean[house_style]==5].SalePrice

sns.distplot(price)
copy_data = train_clean.copy()

# log_price = np.log(copy_data.SalePrice)

# copy_data['SalePrice'] = log_price

sns.distplot(copy_data['SalePrice'])
house_style = "HouseStyle"

house_styles = train_clean[house_style].unique()

unequal_var(copy_data, house_style, house_styles)
eq_var = []



def uv(df, feature, features):

    eq_var = []

    feat_arr = []

    for f in features:

        price = df[df[feature]==f].SalePrice

        feat_arr.append(np.array(price))

    

    feat_tup = tuple(feat_arr)

    stat, p = levene(*feat_tup)

    print("%s p-value: %.2e; Unequal variance: %s" %(feature, p, ('Yes' if p < 0.05 else 'No')))

    

    if p > 0.05:

        eq_var.append(feature)

        return (True, feature)

    else:

        return (False, feature)



eq_vars = []

uneq_vars = []

for cf in copy_data.columns:

    feats = copy_data[cf].unique()

    v, f = uv(copy_data, cf, feats)

    if v:

        eq_vars.append(f)

    else:

        uneq_vars.append(f)

    

print(eq_vars)
feat = "MSZoning"

feats = copy_data[feat].unique()

unequal_var(copy_data, feat, feats)
feat = "SaleCondition"

feats = copy_data[feat].unique()

unequal_var(copy_data, feat, feats)
what_is("MSSubClass")
feat = "MSSubClass"

feats = copy_data[feat].unique()

unequal_var(copy_data, feat, feats)
what_is("YrSold")
feat = "YrSold"

feats = copy_data[feat].unique()

unequal_var(copy_data, feat, feats)
def mult_comp(df, feature, features):

    feat_arr = []

    for f in features:

        price = df[df[feature]==f].SalePrice

        feat_arr.append(np.array(price))

    

    feat_tup = tuple(feat_arr)

    stat, p = f_oneway(*feat_tup)

    print("%s p-value: %.2e; Unequal mean: %s" %(feature, p, ('Yes' if p < 0.05 else 'No')))

    

    if p < 0.05:

        return feature

    else:

        return None



good_feats = []

for nv in eq_vars:

    feats = copy_data[nv].unique()

    v = mult_comp(copy_data, nv, feats)

    if v != None:

        good_feats.append(v)

    

print(good_feats)
sns.boxplot(data=copy_data, x='Street', y='SalePrice')
sns.boxplot(data=copy_data, x='YrSold', y='SalePrice')
sns.boxplot(data=copy_data, x='Heating', y='SalePrice')
what_is("SaleCondition")
good_feats
pairwise = []

for gf in good_feats:

    if len(copy_data[gf].unique()) < 50:

        pairwise.append(gf)

#         print("%s: %d" %(gf, len(copy_data[gf].unique())))

        

pairwise
sns.scatterplot(data=copy_data, x="LotArea", y="SalePrice")
for cv in pairwise:

    print("%s: " %(cv))

    print(pairwise_tukeyhsd(copy_data.SalePrice, copy_data[cv]))
sns.boxplot(x="Condition2", y="SalePrice", palette="GnBu_d", data=full_data)
get_mean(copy_data, "SaleCondition")
feat = "SaleCondition"

feats = copy_data[feat].unique()

unequal_var(copy_data, feat, feats)
print(good_feats)
def kw_test(df, feature, features):

    feat_arr = []

    for f in features:

        price = df[df[feature]==f].SalePrice

        feat_arr.append(np.array(price))

    

    feat_tup = tuple(feat_arr)

    stat, p = kruskal(*feat_tup)

    print("%s p-value: %.2e; Unequal mean: %s" %(feature, p, ('Yes' if p < 0.05 else 'No')))

    

    if p < 0.05:

        return feature

    else:

        return None
good_feats1 = []

for nv in uneq_vars:

    feats = copy_data[nv].unique()

    v = kw_test(copy_data, nv, feats)

    if v != None:

        good_feats1.append(v)

    

print(good_feats1)
feat = "FireplaceQu"

feats = copy_data[feat].unique()

unequal_var(copy_data, feat, feats)
test_clean = clean_data[clean_data['SalePrice']==0]

test_clean.info()