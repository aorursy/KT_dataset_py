import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))

train_data = pd.read_csv("../input/train.csv")
train_data.head()
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
train_data["winPlacePerc"].describe()
corrmat = train_data.corr()
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="Blues")
#sns.set()
#cols=['winPlacePerc', 'walkDistance', 'weaponsAcquired', 'boosts', 'killPlace']
#sns.pairplot(train_data[cols], size=3)
#plt.show()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
y = train_data.winPlacePerc
fcols = ['walkDistance', 'weaponsAcquired', 'boosts', 'killPlace']
X = train_data[fcols]
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)

model = RandomForestRegressor(random_state=1, max_leaf_nodes=200)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print("MEA for Random Forest : %f" %(val_mae))
model_on_full_data = RandomForestRegressor(random_state=1, max_leaf_nodes=200)
model_on_full_data.fit(X,y)
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)

test_X = test_data[fcols]
test_predics = model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id':test_data.Id,
                       'winPlacePerc':test_predics})
output.to_csv('submission.csv', index=False)
