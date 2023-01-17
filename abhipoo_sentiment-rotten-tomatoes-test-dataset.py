import pandas as pd

main_file_path = '../input/train.tsv'

data = pd.read_csv(main_file_path,header=0,delimiter="\t",quoting=3)



data.groupby('Sentiment').size()
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from bs4 import BeautifulSoup

import re

from sklearn.ensemble import RandomForestClassifier

import nltk

from nltk.corpus import stopwords # Import the stop word list

import numpy as np



main_file_path = '../input/train.tsv'

data = pd.read_csv(main_file_path,header=0,delimiter="\t",quoting=3)

#print(data.shape)

#print(data["Phrase"][0])



def review_to_words( raw_review ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review).get_text() 

    #

    # 2. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    #

     # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))                               

    

    # and return the result.

    #return(word)





# Get the number of reviews based on the dataframe column size

num_reviews = data["Phrase"].size



# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in range(0, num_reviews):

    # Call our function for each one, and add the result to the list of

    # clean reviews

    if( (i+1)%10000 == 0 ):

        print("Review %d of %d\n" % ( i+1, num_reviews )) 

    clean_train_reviews.append( review_to_words( data["Phrase"][i] ) )



    

# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",

                             tokenizer = None,

                             preprocessor = None,

                             stop_words = None,

                             max_features = 5000) 







train_data_features = vectorizer.fit_transform(clean_train_reviews)



train_data_features = train_data_features.toarray()



print(train_data_features.shape)



# Take a look at the words in the vocabulary

#vocab = vectorizer.get_feature_names()

#print(len(vocab))



# Sum up the counts of each vocabulary word

#dist = np.sum(train_data_features, axis=0)



# For each, print the vocabulary word and the number of times it 

# appears in the training set

#for tag, count in zip(vocab, dist):

#    print(count, tag)



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100,max_depth=50) 



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_data_features, data["Sentiment"] )



print("Model training finished")



#Testing the model



test_file_path = '../input/test.tsv'

test_data = pd.read_csv(test_file_path,header=0,delimiter="\t",quoting=3)



# Get the number of reviews based on the dataframe column size

num_reviews = test_data["Phrase"].size



# Initialize an empty list to hold the clean reviews

clean_test_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in range(0, num_reviews):

    # Call our function for each one, and add the result to the list of

    # clean reviews

    clean_test_reviews.append( review_to_words( test_data["Phrase"][i] ) )





# Get a bag of words for the test set, and convert to a numpy array    

test_data_features = vectorizer.transform(clean_test_reviews)



test_data_features = test_data_features.toarray()



print(test_data_features.shape)

    

    

# Use the random forest to make sentiment label predictions

result = forest.predict(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"PhraseId":test_data["PhraseId"], "Sentiment":result} )



# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )



print("Prediction complete")
