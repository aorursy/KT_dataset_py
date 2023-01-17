import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.tsv', delimiter='\t', low_memory=True)
test = pd.read_csv('../input/test.tsv', delimiter='\t', low_memory=True)
train.head()
train.dtypes
# train data
train.category_name = train.category_name.astype('category')
train.item_description = train.item_description.astype('category')
train.name = train.name.astype('category')
train.brand_name = train.brand_name.astype('category')

# test data
test.category_name = train.category_name.astype('category')
test.item_description = train.item_description.astype('category')
test.name = train.name.astype('category')
test.brand_name = train.brand_name.astype('category')
train = train.drop(['item_description'], axis=1)
test = test.drop(['item_description'], axis=1)
train.name = train.name.cat.codes
train.category_name = train.category_name.cat.codes
train.brand_name = train.brand_name.cat.codes

test.name = test.name.cat.codes
test.category_name = test.category_name.cat.codes
test.brand_name = test.brand_name.cat.codes
train.head()
train_x, train_y = train.drop(['price'], axis=1), train.price
m = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3, n_estimators=100, max_depth=100)
m.fit(train_x, train_y)
m.score(train_x, train_y)
data_idx = np.random.randint(0,train_x.shape[0], 500)
pred = m.predict(train_x.iloc[data_idx])
plt.figure(figsize=(15,5))

plt.plot(pred)
plt.plot(np.array(train_y.iloc[data_idx]))

plt.legend(['Ground Truth', 'Predicted Price'], fontsize=15)

plt.title('Comparison of randomly choosed samples', fontsize=15)
plt.xticks([])
plt.grid(True)
plt.show()
preds = pd.Series(m.predict(test))
submit = pd.concat([test.test_id, preds],axis=1)
submit.columns = ['test_id','price']
submit.to_csv("./submission.csv", index=False)