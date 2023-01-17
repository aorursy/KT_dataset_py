import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler
train = pd.read_csv('../input/train.tsv', sep='\t')

test = pd.read_csv('../input/test.tsv', sep='\t')
train.head()
df = train.copy()
df["brand_name"].describe()
for c in df.columns:

    print(c, sum(df[c].isnull()) / len(df))
df["brand_name"] = df["brand_name"].fillna(0)

df["brand_name"] = df["brand_name"].apply(lambda x: 1 if x else 0)

df.head()
len(df[df["item_description"] == "No description yet"]) / len(df)
df.category_name.describe()
# first we should remove the "no item description" tags

df["item_description"] = df["item_description"].apply(lambda x: "" if x == "No description yet" else x)

df["text"] = df["name"] + " " + df["category_name"] + " " + df["item_description"]

df.drop(["name", "item_description", "category_name"], axis=1, inplace=True)
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

print("unique values: {}".format(len(df["item_condition_id"].unique())))

new_cols = ["condition_{}".format(i) for i in range(1, 6)]

dummies = pd.get_dummies(df["item_condition_id"])

dummy_df = pd.DataFrame(dummies.values)

dummy_df.columns = new_cols

df.drop("item_condition_id", axis=1, inplace=True)

df = df.join(dummy_df)

df
df.index = df["train_id"]

df.drop("train_id", axis=1, inplace=True)

df["text"] = df["text"].str.lower()

df.head()
from sklearn.model_selection import KFold, train_test_split

from sklearn.svm import SVR

from sklearn import metrics
X = df.drop("name price text".split(), axis=1).values

y = df["price"].values

Xtr, Xte, ytr, yte = train_test_split(X, y)
print(Xtr.shape, ytr.shape)
ytr_scaled = scaler.fit_transform(ytr.reshape(-1, 1))

yte_scaled = scaler.transform(yte.reshape(-1, 1))
## now fit the model and see how it does
import xgboost as xgb

from xgboost import XGBRegressor as XGBR

from sklearn.neighbors import KNeighborsRegressor
# try without scaling first

xgtrain = xgb.DMatrix(Xtr, label=ytr_scaled)

xgtest = xgb.DMatrix(Xte, label=yte_scaled)

params = {#"objective": "",

         "eta": 0.1,

         "max_depth": 6,

         "nthread": 7}

watchlist = [(xgtrain, "train"),

            (xgtest, "test")]

num_round = 50
bst = xgb.train(params, xgtrain, num_round, watchlist)
predictions = bst.predict(xgtest).reshape(-1, 1)

yte = yte.reshape(-1, 1)
for i in range(0, len(predictions), 10000):

    print("predicted: {}, actual: {}".format(scaler.inverse_transform(predictions[i]),

                                             yte[i]))
knn = KNeighborsRegressor()

print(metrics.mean_squared_error(yte, knn.fit(Xtr, ytr).predict(Xte)))
df.head()
df["text"] = df["text"].apply(lambda x: str(x))
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences
from random import sample

train = sample(range(len(df)), int(0.75 * len(df)))

test = list(set(range(len(df))).difference(set(train)))

dftr = df.copy().iloc[train]

dfte = df.copy().iloc[test]
dftr
tokenizer = Tokenizer() # removes symbols by default

tokenizer.fit_on_texts(dftr["text"])
dftr["tokens"] = tokenizer.texts_to_sequences(dftr["text"])
dfte["tokens"] = tokenizer.texts_to_sequences(dfte["text"])
# first find out the typical lengths

lengths = dftr["tokens"].apply(lambda x: len(x))
print(lengths.mean())

print(lengths.median())

print(sum([1 for x in lengths if x > 100]) / len(lengths))

# only 5% over 100 in length
# split off the tokens for padding

maxlen = 100

Xtr_tokens = pad_sequences(sequences=dftr["tokens"].values, maxlen=maxlen)

Xte_tokens = pad_sequences(sequences=dfte["tokens"].values, maxlen=maxlen)
dftr.columns
# now extract the other data and concatenate the arrays

Xtr_other = dftr.drop("price text tokens".split(), axis=1).values

Xte_other = dfte.drop("price text tokens".split(), axis=1).values
Xtr = np.concatenate([Xtr_other, Xtr_tokens], axis=1)

Xte = np.concatenate([Xte_other, Xte_tokens], axis=1)
ytr = dftr["price"].values.reshape(-1, 1)

yte = dfte["price"].values.reshape(-1, 1)

scaler = StandardScaler()

ytr_s = scaler.fit_transform(ytr)

yte_s = scaler.transform(yte)
[x.shape for x in (Xtr, ytr_s, Xte, yte_s)]
input_dim = np.max(Xtr) + 1

embedding_dims = 20

print("input dim:", input_dim)
from keras.models import Sequential

from keras.layers import Dense, GlobalAveragePooling1D, Embedding

from keras.callbacks import EarlyStopping
model = Sequential([

    Embedding(input_dim=input_dim, output_dim=embedding_dims),

    GlobalAveragePooling1D(),

    Dense(1)

])



model.compile(loss="mean_squared_error",

             optimizer="adam",

             metrics=["mse", "mae"])
train_data = model.fit(Xtr[:100000], ytr_s[:100000], batch_size=100, validation_data=(Xte[:10000], yte_s[:10000]),

                      epochs=10, callbacks=[EarlyStopping(patience=2, monitor="val_loss")],

                      verbose=1)
import matplotlib.pyplot as plt

% matplotlib inline
plt.rcParams["figure.figsize"] = (12, 9)

for k in train_data.history.keys():

    plt.plot(range(len(train_data.history["val_loss"])), train_data.history[k])

plt.show()
preds = scaler.inverse_transform(model.predict(Xte))
errors = np.log(abs((preds - yte) / (yte + 0.0001)))
np.max(errors)
hist, bins = np.histogram(errors, bins=25)

width = 0.7 * (bins[1] - bins[0])

center = (bins[:-1] + bins[1:]) / 2

plt.bar(center, hist, align='center', width=width)

plt.show()
errors