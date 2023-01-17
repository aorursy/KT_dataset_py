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
# We are focusing on the products first so let's read in the product file



products = pd.read_csv('../input/producto_tabla.csv')
# What does this file look like?

print(products.head())
# How many unique products are there?

products.NombreProducto.nunique()


products['short_name'] = products.NombreProducto.str.extract('^(\D*)', expand = False)

products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', 

                                                       expand=False)

w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)

products['weight'] = w[0].astype('float')*w[1].map({'Kg':1000, 'g':1})

products['pieces'] = products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')

products.head()
# Has this changed the number of unique words we have?

products.short_name.nunique()
# Now that we have only the product names in one column.

# How do we extract the key parts of each word?



# First let's remove words with little meaning.

from nltk.corpus import stopwords

stopwords = set(stopwords.words("spanish"))

# Now let's get rid of all the words in the product names that are in this list of 

# stopwords

# For each row in the dataset we need to pass in all the words and only include those 

# that are not in stop words

# The row by row is handled by just passing in the entire column but I do need to 

# separate out the words I believe



products['short_name_processed'] = [(' ').join(word) for word in products['short_name']

                                              if word not in stopwords]

print(products['short_name_processed'])

# Hmm word in products['short_name'] left me with considering each character by itself



# Let me try printing each word in one row

for word in products['short_name_processed'][0]:

    print (word)
# Once again this is by letter. Thus this isn't done by word yet! Instead I shall go

# through and tokenize!

words = products['short_name_processed'].split()

meaningful_words = [w for w in words if not w in stopwords]

return (" ". join(meaningful_words))
# Facing the problem where I can't put in the entire column at once

# What is the solution doing differently?



products['short_name_processed'] = (products['short_name'].map(lambda x: " ".join([

    i for i in x.lower().split() if i not in stopwords])))



# Series.map() will take each element of the series and perform the provided function

# on it

# Thus we have the first row 'No Identificado' passed in as x to the lambda function



# Let's look at the words we have

products['short_name_processed']
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("spanish")



# Here's the example from above.  It reduces a word to its stem essentially.

print (stemmer.stem("Tortillas"))
# How do we replace all the processed words with their stems instead?

# We will do the same process as with removing stopwords except now we will pass 

# through the stemmer function.



products['short_name_processed'] = products['short_name_processed'].map(lambda x: stemmer.stem(a).join(' ') for a in x.lower().split())

    

print(products['short_name_processed'])
# How do you define x in a lambda function?

# Doesn't seem like that is the problem

# First of all we are joining elements of a list



products['short_name_processed'] = products['short_name_processed'].map(lambda x: ' '.join(

    [stemmer.stem(a) for a in x.lower().split()]))



print(products['short_name_processed'])

from sklearn.feature_extraction.text import CountVectorizer



# Here we initialize the CountVectorizer

vectorizer = CountVectorizer(analyzer='word',

                            tokenizer=None,

                            preprocessor=None,

                            stop_words=None,

                            max_features=1000)



# Now we need to use it to fit it and train it with out sample

product_bag_words = vectorizer.fit_transform(products.short_name_processed).toarray()

product_bag_words.shape

# the shape should have the same number of rows as our product data

# the number of columns is the max number of features we allowed for

# 
# What are the words in our dictionary?

print(vectorizer.get_feature_names())



# How can we get the number of times each word appears?

# Since each column in the above array represents a single word, the sum of the column

# is the number of times that word has appeared.

# Let's print the most common word



print('\n\n')

print(vectorizer.get_feature_names()[np.argmax(sum(product_bag_words))], np.argmax(sum(product_bag_words)))

print('\n\n')



# It would be cool to see this in a dictionary format

for word, count in zip(vectorizer.get_feature_names(), sum(product_bag_words)):

    print('%s: %d' %(word, count))
train = pd.read_csv('../input/train.csv', usecols=['Producto_ID', 

                                                   'Demanda_uni_equil'])

train_product_agg = train.groupby('Producto_ID', as_index=False).agg('mean')

print(train_product_agg)



# We want to have the mean of demand for each of the product IDs

# Therefore we groupby product ID and aggregate over the demand

# Thus we can now go ahead and merge the demand data with the bag of words data



product_bag_words = pd.concat([products.Producto_ID, 

                               pd.DataFrame(product_bag_words, columns=vectorizer.get_feature_names(), index=products.index)], axis=1)

product_bag_words.head()
test = pd.read_csv('../input/test.csv')
# Create the model



from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=100)

# This means that each tree will use 100 of the estimators



# Now we have to train the model on the training data

# How do we create the model? Use fit with the data and the result

# Where do we get the actual demand from?

# Our data in product_bag_words is currently sorted alphabetically, we need to sort it

# according to how it is sorted in the train data file



set_train = set(train_product_agg['Producto_ID'])

set_test = set(test['Producto_ID'])



print(len(set_train))

print(len(set_test))

print(len(set_train & set_test ))



forest = forest.fit_transform(product_bag_words, train[''])
# Can't actually use bag of words to do this since test doesn't have product names...