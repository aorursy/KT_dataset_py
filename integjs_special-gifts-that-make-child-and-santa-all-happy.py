import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline

sns.set(style="whitegrid")



child_wishlist = pd.read_csv('../input/child_wishlist_v2.csv', header=None)

gift_goodkids = pd.read_csv('../input/gift_goodkids_v2.csv', header=None)
# Split child_wishlist into three

df_triplets = child_wishlist.loc[:5000, 1:]

df_twins = child_wishlist.loc[5001:45000, 1:]

df_single = child_wishlist.loc[45001:, 1:]

# remove the id in gift_goodkids for next use

df_gift_goodkids = gift_goodkids.loc[:, 1:]
appropriate_gifts =[]  # a list to save the result dataframes of every child



def find_gift(row):

    # all avaliable gifts in gift_goodkids

    app_gifts = df_gift_goodkids[df_gift_goodkids == row.name].dropna(axis=[0, 1], how='all')

    # turn app_gifts to a new dataframe with child_id, gift_id, and gift_rank

    app_gifts = app_gifts + app_gifts.columns.values - row.name

    app_gifts = app_gifts.fillna(0)

    app_gifts = pd.DataFrame(app_gifts.sum(axis=1), columns=['gift_rank'])

    app_gifts = app_gifts.reset_index().rename(columns={"index": "gift_id"})

    app_gifts.insert(0, 'child_id', row.name)

    # if the child like these gift, add child_rank to app_gift

    app_gifts.loc[app_gifts['gift_id'].isin(row), 'child_rank'] = row[row.isin(app_gifts['gift_id'])].index

    # append the final result

    appropriate_gifts.append(app_gifts)
%%time

hide_output = df_triplets.apply(find_gift, axis=1)



# here we only run on triplets for a short running time

# df_twins.apply(find_gift, axis=1)

# df_single.apply(find_gift, axis=1)
# concat all the result to one dataframe

df_appropriate_gift = pd.concat(appropriate_gifts, ignore_index=True)

df_appropriate_gift.to_csv('appropriate_gift.csv', index=None)

df_appropriate_gift.head(10)
# draw a countplot to show how much gift liked by both the child and the Santa

g = sns.countplot(x='child_rank', data=pd.DataFrame(df_appropriate_gift['child_rank'] > 0))

g.set(xlabel="Is appropriate gift liked by the child?")
# draw a countplot of top 5 children who has most appropriate gifts

child_top5 = df_appropriate_gift[df_appropriate_gift['child_id'].isin(df_appropriate_gift['child_id'].value_counts().index[0:5].values)]

g = sns.countplot(x='child_id', data=child_top5)

g.set(xlabel="Top 5 Santa liked child")
# pick one of the top 5

df_appropriate_gift.loc[df_appropriate_gift['child_id'] == 3994]