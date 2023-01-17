import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings(action='once')
import json
import pandas as pd
import numpy as np

# Load training dataset
df_train = pd.read_json("../input/train.json")
df_test = pd.read_json("../input/test.json")
from typing import List
import re
from nltk.stem import WordNetLemmatizer

non_alphabetical_or_whitespace = re.compile(r"[^a-zA-Z\s]")
multi_whitespace = re.compile(r"\s+")
lemmatizer = WordNetLemmatizer()

def clean_ingredients(ingredients: List[str]) -> List[str]:
    result = []
    
    for ingredient in ingredients:
        temp = ingredient.lower()
        temp = non_alphabetical_or_whitespace.sub("", temp)
        temp = multi_whitespace.sub(" ", temp)
        temp = ' '.join([lemmatizer.lemmatize(word) for word in multi_whitespace.split(temp)])
        result.append(temp)
        
    return ",".join(result)

df_train["ingredients_cleaned"] = df_train["ingredients"].apply(lambda ingredients: clean_ingredients(ingredients))
df_test["ingredients_cleaned"] = df_test["ingredients"].apply(lambda ingredients: clean_ingredients(ingredients))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = CountVectorizer()
label_encoder = LabelEncoder()
vectorizer = TfidfVectorizer(binary=True)

#X = vectorizer.fit_transform(df_train["ingredients_cleaned"]).todense()
X = vectorizer.fit_transform(df_train["ingredients_cleaned"])
y = label_encoder.fit_transform(df_train["cuisine"])
from imblearn.over_sampling import SMOTE, ADASYN
import keras

# X, y = SMOTE().fit_sample(X, y)
# X, y = ADASYN().fit_sample(X, y)

y = keras.utils.to_categorical(y)
from sklearn.model_selection import train_test_split
from keras import utils

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation

def create_model1(input_dim: int):
    model = Sequential()

    model.add(Dense(1024, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(20, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def create_logistic_model(input_dim: int): 
    model = Sequential() 
    model.add(Dense(20, input_dim=input_dim, activation='softmax')) 
    batch_size = 128 
    nb_epoch = 20
    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model
# model.fit(X, y, epochs=1, batch_size=512)
features_dimension = X_train.shape[1]

model1 = create_logistic_model(features_dimension)
model1.fit(X, y, epochs=200, batch_size=32)

model2 = create_model1(features_dimension)
model2.fit(X, y, epochs=50, batch_size=64)
for model in [model1, model2]:
    print(model.evaluate(X_test, y_test, batch_size=128))
X_validation = df_test["ingredients_cleaned"].apply(lambda ingredients: vectorizer.transform([ingredients]).todense())
classification_results = []

for ingredients in X_validation:
    classification_results.append(
        label_encoder.inverse_transform(
            np.argmax(
                (model1.predict(ingredients) + model2.predict(ingredients))/2
            )
        )
    )

df_test["cuisine"] = classification_results
df_test.head()
df_test[["id", "cuisine"]].to_csv("results_multi_with_feature_engineering.csv", index=False)