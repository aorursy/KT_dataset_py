import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow_addons as tfa

from sklearn.preprocessing import LabelEncoder



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
initial = pd.read_csv("../input/prepaired-data-of-customer-revenue-prediction/test_filtered.csv", low_memory=False)

initial.head()
initial.device_deviceCategory.value_counts()
desktop = initial[initial["device_deviceCategory"] == "desktop"].sample(n=34000, random_state=42)

mobile = initial[initial["device_deviceCategory"] == "mobile"].sample(n=34000, random_state=42)

tablet = initial[initial["device_deviceCategory"] == "tablet"].sample(n=34000, random_state=42)

sample = pd.concat([desktop, mobile, tablet])

sample.head()
sample.info()
how_many_ats = {}

for col in sample.columns:

    if "@" in sample[col].value_counts().index:

        how_many_ats[col] = sample[col].value_counts()["@"]

    else:

        how_many_ats[col] = 0



pd.DataFrame(how_many_ats.items()).sort_values(1, ascending=False)
to_be_dropped = [key for key in how_many_ats if how_many_ats[key] > 1000]

clean = sample.drop(columns=to_be_dropped)

clean.info()

to_be_dropped
ats_left = {}

for col in clean.columns:

    if "@" in clean[col].value_counts().index:

        ats_left[col] = clean[col].value_counts()["@"]

    else:

        ats_left[col] = 0

        

pd.DataFrame(ats_left.items()).sort_values(1, ascending=False)
cols_to_work_on = ["device_operatingSystem", "geoNetwork_subContinent", "geoNetwork_country", "geoNetwork_continent"]

filler = {col: clean[col].value_counts().index[0] for col in cols_to_work_on}



for col in cols_to_work_on:

    clean[col].replace("@", filler[col], inplace=True)

    print(clean[col].value_counts())
clean.info()
categoricals = ["channelGrouping", "device_browser", "device_operatingSystem", "geoNetwork_continent", "geoNetwork_country", "geoNetwork_subContinent", "trafficSource_source"]

for cat in categoricals:

    print(cat + ": " + str(clean[cat].value_counts().shape[0]))
final = clean.copy()

final = pd.get_dummies(final, columns=categoricals)



print(final.columns[:20])
unused = final[["fullVisitorId", "device_isMobile", "date", "visitStartTime"]].copy()

final.drop(columns=["fullVisitorId", "device_isMobile", "date", "visitStartTime"], inplace=True)

final.head()
final.info()
final.rename({"device_deviceCategory": "Y"}, axis=1, inplace=True)

final.Y.value_counts()
label_encoder = LabelEncoder()

final["Y"] = label_encoder.fit_transform(final["Y"])

final["Y"].value_counts()
Y = final.pop("Y")

Y = pd.get_dummies(Y)

Y.head()
X_t_v, X_test, Y_t_v, Y_test = train_test_split(final, Y, test_size=0.2, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_t_v, Y_t_v, test_size=0.2, random_state=42)

X_train.shape
input_size = X_train.shape[1]

output_size = Y.shape[1]



model = tf.keras.Sequential([

                            tf.keras.Input(shape=(input_size,)),

                            tf.keras.layers.Dense(42, activation="relu"),

                            tf.keras.layers.Dense(12, activation="relu"),

                            tf.keras.layers.Dense(output_size, activation="softmax")

                            ])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="AUC")
NUM_EPOCHS = 7

BATCH_SIZE = 100



model.fit(x=X_train, y=Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, Y_val), verbose=2)
test_loss, test_accuracy = model.evaluate(X_test, Y_test)



print("\nTest loss: " + str(test_loss) + ". Test accuracy: " + str(test_accuracy*100.) + "%.")
Y_pred = pd.DataFrame(model.predict(X_test))

Y_pred.head()