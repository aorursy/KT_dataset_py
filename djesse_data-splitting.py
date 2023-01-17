import pandas as pd

data = pd.read_json("../input/train.json", orient='columns')



# Extract X (ingredients) and y (cuisines)

X = data['ingredients']

y = data['cuisine']
import pandas as pd

import numpy as np



def convert_to_binary(data, ingredient_set):



    # Extract X (ingredients)

    X = data['ingredients']

            

    # Convert set to list to create indexes

    ingredients_list = list(ingredient_set)



    # Add a column for every ingredient to the dataframe

    i = 0

    for ingredient in ingredients_list:

        i = i + 1

        

        if i % 1000 ==0:

            print(i)

        scores = []

        

        # Check whether recipe contains ingredient or not

        for rec_ing in data['ingredients']:

            if ingredient in rec_ing:

                scores.append(1)

            else:

                scores.append(0)

        

        data[ingredient] = scores



    # Drop ingredients column

    print('Dropping ingredients column')

    data = data.drop(['ingredients'],axis=1)



    # Write file to csv

    print('Writing data to csv')

    data.to_csv(r'test_binary.csv', index=False)



    print(len(data.columns))
X = data['ingredients']



# Create a unique set of ingredients

ingredient_set = set()

for ingredients in X:

    for ingredient in ingredients:

        ingredient_set.add(ingredient)
from sklearn.model_selection import train_test_split



# Divide the data into a training and a test data set. Ratio = 0.75:0.25

X_train, X_test, y_train, y_test = train_test_split(data[['id','ingredients']], y, stratify=y, train_size=0.7, test_size=0.3)
import matplotlib.pyplot as plt



# Count occurence of each cuisine for training set

cuisine_training_occurences = y_train.value_counts()



# Count occurence of each cuisine for test set

cuisine_test_occurences = y_test.value_counts()



# Plot training occurences

cuisine_training_occurences.plot(kind='bar', x='recipe', y='occurences', legend=True)

plt.title('Distribution training data set')

plt.xlabel('Cuisines')

plt.ylabel('Number of recipes')

plt.show()



# Plot test occurences

cuisine_test_occurences.plot(kind='bar', x='recipe', y='occurences', legend=True)

plt.title('Distribution test data set')

plt.xlabel('Cuisines')

plt.ylabel('Number of recipes')

plt.show()
# Concat to new dataframe

train_df = pd.concat([y_train, X_train], axis=1)

test_df = pd.concat([y_test, X_test], axis=1)



convert_to_binary(train_df, ingredient_set)
convert_to_binary(test_df, ingredient_set)