# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd



# Reading the data

train_df = pd.read_json('../input/train.json')



print(train_df[train_df.cuisine=='indian'])
train_df.head()
train_df.describe()
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

from sklearn.ensemble import RandomForestClassifier 



vect = CountVectorizer(max_features = 1000) #considering only 1000 features

ingredients = train_df['ingredients']

words_list = [' '.join(x) for x in ingredients] #Extract the ingredients and convert them to a *single list* of recipes called words_list

len(words_list)
#create a bag of words and convert to a array and then print the shape



bag_of_words = vect.fit(words_list)

bag_of_words = vect.transform(words_list).toarray()

print(bag_of_words.shape)
forest = RandomForestClassifier(n_estimators = 500) #Initilize a random forest classifier with 500 trees 

#n_estimators is the no. of trees in the forest



forest = forest.fit( bag_of_words, train_df["cuisine"] ) #fit it with the bag of words we created

#Random forests or Random decision forests - 

# learning method for classification, regression and other tasks, 

# operate by constructing a multitude of decision trees at training time and 

# outputting the class that is the *mode* of the classes (classification) or mean prediction (regression) of the individual trees. 

# Random decision forests correct for decision trees' habit of overfitting to their training set.
test_data = pd.read_json('../input/train.json') #reads the *test* file 

test_data.head()
#same thing done like we did with the training set and create an array



test_ingredients = test_data['ingredients']

test_ingredients_words = [' '.join(x) for x in test_ingredients]

test_ingredients_array = vect.transform(test_ingredients_words).toarray()
# Use the random forest to make cusine predictions

result = forest.predict(test_ingredients_array)

result
output = pd.DataFrame( data={"id":test_data["id"], "cuisine":result} ) #Copy the results (in from of arrays) to a pandas dataframe 

                                                                       #with an "id" column and a "cusine" column

output.to_csv( "Bow.csv", index=False, quoting=3 ) #If you have set a float_format then floats are converted to strings
import collections

bow = [ collections.Counter(recipe) for recipe in train_df.ingredients ]

sumbags = sum(bow, collections.Counter())
import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use(u'ggplot')

fig = pd.DataFrame(sumbags, index=[0]).transpose()[0].sort_values(ascending=False, inplace=False)[:10].plot(kind='barh')

fig.invert_yaxis()

fig = fig.get_figure()

fig.tight_layout()

#sort function is deprecated for DataFrame to sort_values and sort_index