import pandas as pd

import numpy as np
train_df = pd.read_csv("./data/train.csv")

test_df = pd.read_csv("./data/test.csv")
from collections import Counter
Counter(train_df.TARGET.values).most_common()
train_df.info()
feats_list = ['bathrooms', 'bedrooms', 'listing_id', 'price']
train_df.isna().sum()
print("before drop: ", len(train_df))

train_df.dropna(inplace=True, subset=feats_list)

print("after drop: ", len(train_df))
X_train = train_df.loc[:, feats_list]

X_test = test_df.loc[:, feats_list]



y_train = train_df.loc[:, 'TARGET'].values
X_test.shape
X_test.isna().sum()
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(multi_class='ovr', solver='lbfgs',

                        class_weight={'low':0.33, 'high':2.9, 'medium':1.})
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
from collections import Counter

Counter(y_pred).most_common()
submit = pd.DataFrame.from_dict({'Id':test_df.Id.values, 'TARGET': y_pred})

submit.to_csv("data/sumbit.csv", index=False)