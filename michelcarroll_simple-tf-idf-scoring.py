# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
training_data = pd.read_json("../input/train.json")
split_point = 0.9
training_data = training_data[:int(len(training_data)*split_point)].copy()
validation_data = training_data[int(len(training_data)*split_point):].copy()

test_data = pd.read_json("../input/test.json")

print("Num Training Data Rows: ", len(training_data))
print("Num Validation Data Rows: ", len(validation_data))
training_data.head()
training_data.dtypes
from functools import reduce
def flatten_into_ingredients(df):
    return reduce(lambda x,y: x+y, df.ingredients.values)
all_ingredients = flatten_into_ingredients(training_data)
unique_ingredients = np.unique(all_ingredients)
cuisines = training_data.cuisine.unique()
ingredients_by_cuisine = {}
for cuisine in cuisines:
    ingredients_by_cuisine[cuisine] = flatten_into_ingredients(training_data[training_data.cuisine == cuisine])
print("Total Ingredients: ", len(all_ingredients))
print("Unique Ingredients: ", len(unique_ingredients))
print("% Duplicate: ", (1 - (float(len(unique_ingredients)) / float(len(all_ingredients)))) * 100)
import math

total_number_documents = len(training_data)

def tfidf(ingredient):
    num_docs_with_ingredient_in = sum(training_data.ingredients.map(lambda x: (ingredient.name in x)))
    idf = math.log(total_number_documents / num_docs_with_ingredient_in)
    for cuisine in ingredient.index.values:
        num_ingredients_in_cuisine = len(ingredients_by_cuisine[cuisine])
        num_this_ingredient_in_cuisine = ingredients_by_cuisine[cuisine].count(ingredient.name)
        tf = num_this_ingredient_in_cuisine / num_ingredients_in_cuisine
        ingredient[cuisine] = tf * idf
    return ingredient

ingredient_frequencies = pd.DataFrame(index=unique_ingredients, columns=training_data.cuisine.unique())
ingredient_frequencies = ingredient_frequencies.apply(tfidf, axis=1)
ingredient_frequencies.head()
print(ingredient_frequencies.greek.sort_values(ascending=False).head())
print(ingredient_frequencies.italian.sort_values(ascending=False).head())

import operator

def dict_argmax(dictionary):
    return max(dictionary.items(), key=operator.itemgetter(1))[0]

def predict(row):
    scores = {}
    for cuisine in cuisines:
        total = 0
        for ingredient in row.ingredients:
            if ingredient in ingredient_frequencies.index:
                total = total + ingredient_frequencies.loc[ingredient, cuisine]
        scores[cuisine] = total
    # print(row.ingredients, row.cuisine, dict_argmax(scores))
    return dict_argmax(scores)

validation_data["prediction"] = validation_data.apply(predict, axis=1)
accuracy = sum(validation_data.cuisine == validation_data.prediction) / len(validation_data)
print("Accuracy: %", (accuracy * 100))
false_predictions = validation_data[validation_data.cuisine != validation_data.prediction].groupby(by="cuisine").id.count()
true_predictions = validation_data[validation_data.cuisine == validation_data.prediction].groupby(by="cuisine").id.count()

accuracy_breakdown = pd.DataFrame({ "right": true_predictions, "wrong": false_predictions, "accuracy": (true_predictions / (true_predictions + false_predictions)) * 100 })
accuracy_breakdown.sort_values('accuracy', inplace=True, ascending=False)

axes = accuracy_breakdown.accuracy.plot.bar()
_ = axes.set_ylabel("Accuracy")
wrongly_predicted_rows = validation_data[validation_data.cuisine != validation_data.prediction].copy()
wrongly_predicted_rows["tuple"] = (wrongly_predicted_rows["cuisine"].str.cat(wrongly_predicted_rows["prediction"], sep="-"))
wrongly_predicted_rows.groupby('tuple').id.count().sort_values(ascending=False).head()
validation_data[validation_data.cuisine == "italian"][validation_data.cuisine != validation_data.prediction].groupby('prediction').id.count().sort_values(ascending=False)
# validation_data[(validation_data.cuisine == "italian") & (validation_data.prediction == "greek")].ingredients.values
test_data["cuisine"] = test_data.apply(predict, axis=1)
test_data.drop(columns=['ingredients'], inplace=True)
test_data.head()
test_data.to_csv("logistic_sub.csv", index=False, header=True)
