#!/usr/bin/env python

# coding: utf-8



import matplotlib.pyplot as plt 

import seaborn as sns

import numpy as np 

import pandas as pd

import warnings 



get_ipython().run_line_magic('matplotlib', 'inline')



warnings.filterwarnings('ignore')



def display_all(df):

    with pd.option_context("display.max_rows", 100, "display.max_columns", 100):

        display_all(df)
cacao = pd.read_csv("../input/chocolate-bar-ratings/flavors_of_cacao.csv")
cacao.head()
cacao.columns
# Hmmm Let's fix the column names...

cacao.columns = cacao.columns.str.replace("\n", ' ', regex=True)

cacao.columns
#One more column to fix 

cacao = cacao.rename(columns={"Company\xa0 (Maker-if known)": "Company"})
cacao.info()
# Check for nulls 

cacao.isnull().sum()
cacao["Bean Type"].value_counts()
# Let's just fill na with the mode...

cacao["Bean Type"] = cacao["Bean Type"].fillna("Trinitario")

cacao["Bean Type"].isna().sum()
cacao["Broad Bean Origin"].value_counts()
#Let's fill na with the mode again...

cacao["Broad Bean Origin"] = cacao["Broad Bean Origin"].fillna("Venezuela")
# Ok cool looks like no na values 

cacao.isna().sum()
for column in cacao.columns: 

    print(cacao[column].value_counts())
# Explore the dataset -> check distribution of Chocolate Rating (Approximately Normal)

plt.figure(figsize=(12,5))



sns.distplot(cacao["Rating"], bins=15)



plt.xlabel("Rating", fontsize=17)

plt.title("Chocolate Rating Distribution", fontsize=20)

plt.show()
cacao["Cocoa Percent"] = cacao["Cocoa Percent"].apply(lambda x: int(str(x)[:2]))
cacao["Cocoa Percent"].value_counts()
plt.figure(figsize=(12,5))



sns.distplot(cacao["Cocoa Percent"], bins=20)



plt.xlabel("Cocoa Percent", fontsize=17)

plt.title("Cocoa Percent Distribution", fontsize=20)

plt.show()
plt.figure(figsize=(12,5))



sns.distplot(cacao["Review Date"], bins=12)



plt.xlabel("Review Date", fontsize=17)

plt.title("Review Date Distribution", fontsize=20)

plt.show()
# Check distribution of Target (Ratings)



plt.figure(figsize=(12,5))



sns.countplot(x="Rating", data=cacao, palette='hls')



plt.ylabel("Count")

plt.xlabel("Rating")

plt.title("Distribution of Chocolate Rating")

plt.show()
# Target is to place them into five categories 

def fix_ratings(rating):

    if rating < 0.5: 

        return 0.0 

    elif rating < 1.5: 

        return 1.0 

    elif rating < 2.5: 

        return 2.0

    elif rating < 3.5: 

        return 3.0

    elif rating < 4.5: 

        return 4.0 

    elif rating < 5.5: 

        return 5.0
cacao["Rating"] = cacao["Rating"].apply(lambda x: fix_ratings(x))
plt.figure(figsize=(12,5))



sns.countplot(x="Rating", data=cacao, palette='hls')



plt.xlabel("Rating", fontsize=17)

plt.ylabel("Count", fontsize=17)

plt.title("Rating by Count", fontsize=20)

plt.show()
X = cacao.drop("Rating", axis=1)

y = cacao["Rating"].values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cacao.info()
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown="ignore")

ohe.fit(X_train)

X_train_enc = ohe.transform(X_train)

X_test_enc = ohe.transform(X_test)
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

le.fit(y_train)

y_train_enc = le.transform(y_train)

y_test_enc = le.transform(y_test)
from keras.utils import np_utils

y_train_cat = np_utils.to_categorical(y_train_enc)

y_test_cat = np_utils.to_categorical(y_test_enc)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_mean=False)

sc.fit(X_train_enc)

X_train_enc_sc = sc.transform(X_train_enc)

X_test_enc_sc = sc.transform(X_test_enc)
X_train_enc_sc.shape
y_train_cat.shape
# Creating holdout set to test on never seen before data (about 25%)

holdout = int(.25 * X_train_enc_sc.shape[0])

x_val = X_train_enc_sc[:holdout]

partial_x_train = X_train_enc_sc[holdout:]

y_val = y_train_cat[:holdout]

partial_y_train = y_train_cat[holdout:]
from keras import layers

from keras import models

from keras import regularizers

from keras import optimizers



model = models.Sequential()



model.add(layers.Dense(8, activation='relu', input_dim=X_train_enc.shape[1], 

                       kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(5, activation='softmax'))



# sgd = optimizers.SGD(lr = 0.01, momentum = 0.9)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train, validation_data=(x_val, y_val),

                    batch_size=64, epochs=25, verbose=2)
history_dict = history.history
history_dict.keys()
training_loss = history_dict["loss"]

val_loss = history_dict["val_loss"]



epochs = range(1, len(training_loss) + 1)



plt.plot(epochs, training_loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.legend()

plt.show()
training_acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']



epochs = range(1, len(val_acc) + 1)



plt.plot(epochs, training_acc, 'bo', label="Training Accuracy")

plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')



plt.legend()

plt.show()
#Let's evaluate...

scores = model.evaluate(X_test_enc_sc, y_test_cat, batch_size=128)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))