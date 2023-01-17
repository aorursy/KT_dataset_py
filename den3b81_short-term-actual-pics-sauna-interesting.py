# load the modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
# get the training data 

train_df = pd.read_json('../input/train.json')
dummies = train_df['features'].str.join(sep=',').str.lower().str.get_dummies(sep=',')

dummies.head(10)
dummies.shape
dummies.sum().sort_values(ascending= False).head(40).plot(kind = 'bar', figsize = (10,5))
frequent_features = dummies.sum().sort_values(ascending= False).head(100).index;

ff_interest_df = pd.DataFrame(index = frequent_features, columns = ['low','medium','high','count'])



for feature in frequent_features:

    # select index where feature is present

    ixes = dummies[feature].astype(bool)

    temp = pd.concat([dummies[ixes][feature],train_df['interest_level']], axis = 1, join = 'inner')

    ff_interest_df.loc[feature] = temp.interest_level.value_counts()/len(temp)

    ff_interest_df.loc[feature,'count'] = len(temp)

    

print(ff_interest_df.head(5))
ff_interest_df['high'].sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))

plt.gca().set_ylabel('% of high interest')
ff_interest_df[['low','medium','high']].plot(kind = 'bar', stacked = True, figsize = (15,5))
# let's put the overall averages and count in the picture

avg_interest_levels = train_df.interest_level.value_counts()/len(train_df)



ff_interest_df.loc['AVERAGES'] = avg_interest_levels

ff_interest_df.loc['AVERAGES','count'] = len(train_df)

ff_interest_df.tail()
# let's plot again and add a second plot with the count

ff_interest_df.sort_values(by = 'count', ascending = False)[['low','medium','high']].plot(kind = 'bar', stacked = True, figsize = (15,5))

plt.figure()

ff_interest_df['count'].sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))
ff_interest_df.loc['fireplace']
ff_interest_df.sort_values(by = 'high', ascending = False).head(3)