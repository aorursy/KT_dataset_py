# Import Libraries

import numpy as np 

import pandas as pd 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
d=pd.read_csv("../input/gifts.csv")

d['type'] = d['GiftId'].apply(lambda x: x.split('_')[0])

d['id'] = d['GiftId'].apply(lambda x: x.split('_')[1])
def Weight(mType):

    if mType == "horse":

        return max(0, np.random.normal(5,2,1)[0])

    if mType == "ball":

        return max(0, 1 + np.random.normal(1,0.3,1)[0])

    if mType == "bike":

        return max(0, np.random.normal(20,10,1)[0])

    if mType == "train":

        return max(0, np.random.normal(10,5,1)[0])

    if mType == "coal":

        return 47 * np.random.beta(0.5,0.5,1)[0]

    if mType == "book":

        return np.random.chisquare(2,1)[0]

    if mType == "doll":

        return np.random.gamma(5,1,1)[0]

    if mType == "blocks":

        return np.random.triangular(5,10,20,1)[0]

    if mType == "gloves":

        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
d['weight'] = d['type'].apply(lambda x: Weight(x))
sorted_weights = d.sort_values(by=['weight'])
data = pd.DataFrame(pd.np.empty((1000, 1)) * pd.np.nan, columns = ['bag_weights']) 

data['bag_weights'] = 50.0000

data['gifts'] = np.empty((1000, 0)).tolist()
for idx, row in sorted_weights.iterrows():

	data = data.sort_values(by=['bag_weights'], ascending = 0)

	for bag_idx, bag in data.iterrows():

		if row.weight <= bag.bag_weights:

			bag.gifts.append(row.GiftId)

			data.set_value(bag_idx,'bag_weights',float(bag.bag_weights) - float(row.weight))

			break
packed_bags = [[]]

for bag_idx, bag in data.iterrows():

	if len(bag.gifts)>=3:

		packed_bags.append(" ".join(bag.gifts))