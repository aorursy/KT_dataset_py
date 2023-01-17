print("Let's run the first kernel....")
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
my_line1 = "Hello People, we are hte best data science team in the world."
word_token1 = word_tokenize(my_line1)
for x in word_token1:
    print("word is ",x)
print("Execution finished ......")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups


# CountVectorizer must be a class
count_vector1 = CountVectorizer(stop_words="english", max_features=500)

my_group1 = fetch_20newsgroups()
type(my_group1)


# Working of fit_transform is to fit and transform the input data.
# The fit part will see at what features it will base future transformations
transformed1 = count_vector1.fit_transform(my_group1.data)

print(transformed1)

# it will give a matrix
print(type(transformed1))

# Now, count _vector1 object have some features according to the fitted data
print(count_vector1.get_feature_names())



print(transformed1.toarray().sum(axis=0))


# Taken the log transformation to scala down the numbers ....
print(np.log(transformed1.toarray().sum(axis=0)))

sns.distplot(np.log(transformed1.toarray().sum(axis=0)))

plt.xlabel("Log Count")
plt.ylabel("Frequency")
plt.title("Distribution of 500 words count done by us")
plt.show()





