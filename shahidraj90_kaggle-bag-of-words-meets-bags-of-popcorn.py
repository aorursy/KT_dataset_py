# Import the pandas package, then use the "read_csv" function to read the labeled training data

import pandas as pd

from bs4 import BeautifulSoup             

import re          #import Regular expression expression

from nltk.corpus import stopwords # Import the stop word list

from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt

from tqdm import tqdm



train = pd.read_csv("../input/labeledTrainData.tsv", header=0, \

                    delimiter="\t")

train = train.drop(['id'], axis=1)    # Drop the 'id' column

train.head()  
df2 = pd.read_csv('../input/imdb_master.csv',encoding="latin-1") #Read IMDB reviews dataset for training
df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)  #Drop all other columns 

df2.columns = ["review","sentiment"]

df2.head()
df2 = df2[df2.sentiment != 'unsup']

df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0}) # Convert 'pos' to 1 and 'neg' to 0 

df2.head()
training = pd.concat([train, df2]).reset_index(drop=True) #Merge both training Dataset

training.shape
def review_to_words( raw_review ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review,).get_text() 

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
# Get the number of reviews based on the dataframe column size

num_reviews = training["review"].size



# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

print ("Cleaning and parsing the training set movie reviews...\n")

clean_train_reviews = []

for i in tqdm(range( 0, num_reviews )):

    # If the index is evenly divisible by 10000, print a message

    if( (i+1)%10000 == 0 ):

        tqdm.write ("Review %d of %d\n" % ( i+1, num_reviews ))                                                                 

    clean_train_reviews.append( review_to_words( training["review"][i] ))

print("Done")
print ("Creating the bag of words...\n")

from sklearn.feature_extraction.text import CountVectorizer



# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 6000) 



# The input to fit_transform should be a list of strings.

train_data_features = vectorizer.fit_transform(clean_train_reviews)



# Numpy arrays are easy to work with, so convert the result to an 

# array

train_data_features = train_data_features.toarray()

print ("Done")
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train_data_features,training["sentiment"],test_size=0.2,random_state=0)

print("Splitting Done")
print ("Training the random forest model...")

from sklearn.ensemble import RandomForestClassifier



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100) 



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit(x_train, y_train)

print ("Done")
result = forest.predict(x_test)

print("Predictions Done")
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

#checking accuracy score

accuracy=accuracy_score(y_test,result)

accuracy
#checking confusion matrix

cm=confusion_matrix(y_test,result)

cm
#checking f1 score

f1=f1_score(y_test,result)

f1
from nltk.corpus import stopwords # Import the stop word list



# Read the test data

test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", \

                   quoting=3 )



# Verify that there are 25,000 rows and 2 columns

print (test.shape)



# Create an empty list and append the clean reviews one by one

num_reviews = len(test["review"])

clean_test_reviews = [] 



print ("Cleaning and parsing the test set movie reviews...\n")

for i in tqdm(range(0,num_reviews)):

    if( (i+1) % 5000 == 0 ):

        tqdm.write("Review %d of %d\n" % (i+1, num_reviews))

    clean_review = review_to_words( test["review"][i] )

    clean_test_reviews.append( clean_review )



# Get a bag of words for the test set, and convert to a numpy array

test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()



# Use the random forest to make sentiment label predictions

result = forest.predict(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )



# Use pandas to write the comma-separated output file

output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3 )

print("Done")