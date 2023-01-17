import pandas as pd

with open("../input/train.json") as train_json:

    raw_train = pd.read_json(train_json.read()).reset_index()

    

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=300)

price_df = pd.concat([raw_train['bedrooms'],raw_train['bathrooms'],raw_train['latitude'],raw_train['longitude'],raw_train['price']], axis=1)

model.fit(price_df.drop(['price'], axis=1), price_df['price'])
print(model.kneighbors(price_df.drop(['price'], axis=1).loc[2].reshape(1,-1), n_neighbors=300))
print(price_df.drop(['price'], axis=1).loc[2])

print(price_df.drop(['price'], axis=1).loc[311])
pred_price = model.predict(price_df.drop(['price'], axis=1))



price_df['predicted_price'] = pd.DataFrame(pred_price, columns=['predicted_price'])





price_df['pred_price_ratio'] = price_df['price'] / price_df['predicted_price']



price_df['interest_level'] = raw_train['interest_level']
import matplotlib.pyplot as plt

import seaborn as sns

new_price_df = price_df[price_df['pred_price_ratio'] < 4]

%matplotlib inline

plt.figure(figsize=(10,20))

sns.boxplot(x='interest_level', y='pred_price_ratio', data=new_price_df)

plt.title("Interest Level and Price / Predicted Price Ratio", fontsize=32)

plt.show()