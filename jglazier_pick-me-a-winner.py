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
data = pd.read_csv('../input/winemag-data_first150k.csv') #Load the 150k reviews
data.head(5) #Observe the format
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



num_bins = 20

n, bins, patches = plt.hist(data['points'], num_bins, normed=1, facecolor='blue', alpha=0.5)

plt.title('Distribution of Wine Scores')

plt.xlabel('Score out of 100')

plt.ylabel('Frequency')



mu = 88 # mean of distribution

sigma = 3 # standard deviation of distribution



y = mlab.normpdf(bins, mu, sigma) # create the y line



plt.plot(bins, y, 'r--')



plt.scatter(data['points'], data['price'])
df = data.dropna(subset=['description'])  # drop all NaNs



df_sorted = df.sort_values(by='points', ascending=True)  # sort by points



num_of_wines = df_sorted.shape[0]  # number of wines

worst = df_sorted.head(int(0.25*num_of_wines))  # 25 % of worst wines listed

best = df_sorted.tail(int(0.25*num_of_wines))  # 25 % of best wines listed
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(stop_words='english',analyzer='word')
X1 = vectorizer.fit_transform(best['description'])

idf = vectorizer.idf_

goodlist = vectorizer.vocabulary_
goodlist
X = vectorizer.fit_transform(worst['description'])

idf = vectorizer.idf_

not_so_good_list = vectorizer.vocabulary_
not_so_good_list
import operator



sorted_good = sorted(goodlist.items(), key=operator.itemgetter(0))

sorted_bad= sorted(not_so_good_list.items(), key=operator.itemgetter(1), reverse=True)
sorted_bad
sorted_good
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(best['variety'])
from sklearn.model_selection import train_test_split #get in all our sklean modules

from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=10) #split data
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression() # Logistic regression based on the type of the wine we have 

clf.fit(x_train, y_train)

pred = clf.predict(x_test)
accuracy_score(y_test, pred)
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
reg = linear_model.Ridge(alpha = 0.5, solver = 'sag')
y = data['points']

x = vectorizer.fit_transform(data['description'])



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=32)

reg.fit(x_train, y_train)
pred = reg.predict(x_test)
r2_score(y_test, pred)
best_sorted = best.sort_values(by='price', ascending=True)  # sort by points



num_best = best.shape[0]  # number of wines

cheapestngood = best_sorted.head(int(0.25*num_of_wines))  

cheapngoodest = cheapestngood.sort_values(by = 'points', ascending = False)
cheapestngood.head(10)
cheapngoodest.head(10)
cheapestngood['region_1'].value_counts()
topareas = cheapestngood['region_1'].value_counts().head(10)
topareas
import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nltk

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import cross_val_score

from sklearn.metrics.pairwise import euclidean_distances

pd.set_option('display.max_colwidth', 1500)



vectorizer = TfidfVectorizer(stop_words='english',

                     binary=False,

                     max_df=0.95, 

                     min_df=0.15,

                     ngram_range = (1,2),use_idf = False, norm = None)

doc_vectors = vectorizer.fit_transform(data['description'])

print(doc_vectors.shape)

print(vectorizer.get_feature_names())


def comp_description(query, results_number=20):

        results=[]

        q_vector = vectorizer.transform([query])

        print("Comparable Description: ", query)

        results.append(cosine_similarity(q_vector, doc_vectors.toarray()))

        f=0

        elem_list=[]

        for i in results[:10]:

            for elem in i[0]:

                    #print("Review",f, "Similarity: ", elem)

                    elem_list.append(elem)

                    f+=1

            print("The Review Most similar to the Comparable Description is Description #" ,elem_list.index(max(elem_list)))

            print("Similarity: ", max(elem_list))

            if sum(elem_list) / len(elem_list)==0.0:

                print("No similar descriptions")

            else:

                print(data['description'].loc[elem_list.index(max(elem_list)):elem_list.index(max(elem_list))])

                
comp_description("Bright, fresh fruit aromas of cherry, raspberry, and blueberry.Youthfully with lots of sweet fruit on the palate with hints of spice and vanilla.")

comp_description("Delicate pink hue with strawberry flavors; easy to drink and very refreshing. Perfect with lighter foods. Serve chilled.")
comp_description("This wine highlights how the power of Lake County’s Red Hills seamlessly compliments the elegance and aromatic freshness of the High Valley. Aromas of plum, allspice and clove develop into flavors of fresh dark cherry and cedar on the palate. The Red Hills’ fine tannins provide a smoothly textured palate sensation from start to finish. Fresh acidity from the High Valley culminates in a bright finish of cherry with a gentle note of French oak.")
comp_description("On the nose are those awful love-heart candies, but the palate is nothing but Nesquik strawberry powder. This alcoholic Powerade is what gives box wine a bad name. Pair with BBQ chicken")
comp_description("This wine is very bad, do not drink.")
comp_description("This is the best wine I have ever drank")