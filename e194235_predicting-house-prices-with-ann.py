import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

df.info()
df.describe().transpose()
plt.figure(figsize=(12,8))

sns.distplot(df["price"])
sns.countplot(df["bedrooms"])
sns.barplot(x="bedrooms", y="price", data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x="sqft_living",y="price",data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x="sqft_basement", y="price", data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x="long", y="price", data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x="lat", y="price", data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x="long", y="lat",

                data=df, hue="price",

                palette="RdYlBu", edgecolor=None)
sns.boxplot(x="waterfront", y="price", data=df)
sns.boxplot(x="floors", y="price", data=df)
sns.boxplot(x="condition", y="price", data=df)
sns.boxplot(x="grade", y="price", data=df)
sns.boxplot(x="view", y="price", data=df)
df.sort_values("price", ascending=False).head(50)
df = df.sort_values("price", ascending=False).iloc[45:]
plt.figure(figsize=(12,8))

sns.scatterplot(x="long", y="lat",

                data=df, hue="price",

                palette="RdYlBu", edgecolor=None)
df.head()
df = df.drop("id", axis=1)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].apply(lambda date:date.year)
df["month"] = df["date"].apply(lambda date:date.month)
df.groupby("year").mean()["price"]
sns.barplot(x="month", y="price", data=df)
df = df.drop("date", axis=1)
df.head()
df["zipcode"].value_counts()
df = df.drop("zipcode", axis=1)
df["yr_renovated"].value_counts()
df.head()
features = ["price", "bedrooms", "sqft_living", "sqft_lot", "floors", "waterfront",

            "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",

            "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15", "year", "month"]

mask = np.zeros_like(df[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)



sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn",

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});

X = df.drop("price", axis=1)

y = df["price"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
X_train.shape
X_val.shape
X_test.shape
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam
model = Sequential()



model.add(Dense(19, activation="relu"))

model.add(Dense(16, activation="relu"))

model.add(Dense(16, activation="relu"))

model.add(Dense(8, activation="relu"))



model.add(Dense(1))



model.compile(optimizer="adam", loss="mse")
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#early_stop = EarlyStopping(monitor="val_loss", patience=1000)
check_point = ModelCheckpoint("best_model.h5", monitor="val_loss", verbose=0, save_best_only=True)
model.fit(x=X_train, y=y_train.values,

          validation_data=(X_val, y_val.values),

          batch_size=32, epochs=10000, 

          callbacks=[check_point], verbose=0)
losses = pd.DataFrame(model.history.history)
losses.plot()
from keras.models import load_model
saved_model = load_model('best_model.h5')
predictions = saved_model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test,predictions))
r2 = r2_score(y_test, predictions)
avg = np.mean(y_test)
print("Average house price in test set: {}".format(avg))

print("RMSE: {}".format(rmse))

print("R-squared score: {}".format(r2))