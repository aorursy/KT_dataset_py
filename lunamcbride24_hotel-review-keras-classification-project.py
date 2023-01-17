# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf #Import tensorflow in order to use Keras
from tensorflow.keras.preprocessing.text import Tokenizer #Add the keras tokenizer for tweet tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences #Add padding to help the Keras Sequencing
import tensorflow.keras.layers as L #Import the layers as L for quicker typing
from tensorflow.keras.optimizers import Adam #Pull the adam optimizer for usage

from tensorflow.keras.losses import SparseCategoricalCrossentropy #Loss function being used
from sklearn.model_selection import train_test_split #Train Test Split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
hotel = pd.read_csv("../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv") #Load the review data into pandas
hotel.head() #Take a peek at the dataset
print(hotel.isnull().any()) #Check if there are any null values
print(hotel["Rating"].value_counts()) #Checks the rating values in case there is a weird value
print(hotel.loc[hotel["Review"] == ""]) #Checks for empty review strings
print(hotel["Review"][26]) #Print a random review to show off the structure.
punctuations = """!()-![]{};:,+'"\,<>./?@#$%^&*_~Â""" #List of punctuation to remove

#ReviewParse: Takes the stubborn punctuation off the words for a single review
#Input: the review to parse
#Output: the parsed review
def reviewParse(review):
    splitReview = review.split() #Split the review into words
    parsedReview = "".join([word.translate(str.maketrans('', '', punctuations)) + " " for word in splitReview]) #Takes the stubborn punctuation out
    return parsedReview #Returns the parsed review

hotel["CleanReview"] = hotel["Review"].apply(reviewParse) #Parse all the reviews for their punctuation and add it into a new column
hotel.head() #Take a peek at the dataset
review = hotel["CleanReview"].copy() #Use a copy of the clean reviews

#Print an example sentence to make sure everything is working
print("Example Sentence: ") 
print(review[26])

token = Tokenizer() #Initialize the tokenizer
token.fit_on_texts(review) #Fit the tokenizer to the reviews
texts = token.texts_to_sequences(review) #Convert the reviews into sequences for keras to use

#Print an example sequence to make sure everything is working
print("Into a Sequence: ")
print(texts[26])

texts = pad_sequences(texts, padding='post') #Pad the sequences to make them similar lengths

#Print an example padded sequence to make sure everything is working
print("After Padding: ")
print(texts[26])
#EncodeLabel: encode the labels into 0, 1, and 2, back to the issue of explaining positive and extremely positive to a machine
#Input: the star rating
#Output: 0, 1, or 2 indicating rating positivity/negativity
def encodeLabel(label):
    if label == 5 or label == 4: #If the rating is generally positive
        return 2 #Give the rating a 2 for positive
    if label == 3: #If the rating is generally neutral
        return 1 #Give the rating a 1 for neutral
    return 0 #Give the rating a 0 for negative

labels = ["Negative", "Neutral", "Positive"] #Give our labels a name
hotel["EncodedRating"] = hotel["Rating"].apply(encodeLabel) #Encode the ratings to positivity labels
hotel.head() #Take a peek at the dataset with the new labels
#Split the data for training and testing
textTrain, textTest, ratingTrain, ratingTest = train_test_split(texts, hotel["EncodedRating"], test_size = 0.33, random_state = 24)
size = len(token.word_index) + 1 #Set the number of words for the size
ratings = hotel["EncodedRating"].copy() #Get the encoded ratings from the dataframe

tf.keras.backend.clear_session() #Clear any previous model building

epoch = 2 #Number of runs through the data
batchSize = 32 #The number of items in each batch
outputDimensions = 16 #The size of the output
units = 256 #Dimensions of the output space

model = tf.keras.Sequential([ #Start the sequential model, doing one layer after another in a sequence
    L.Embedding(size, outputDimensions, input_length = texts.shape[1]), #Embed the model with the number of words and size
    L.Bidirectional(L.LSTM(units, return_sequences = True)), #Make it so the model looks both forward and backward at the data
    L.GlobalMaxPool1D(), #Take the max values over time
    L.Dropout(0.3), #Make the dropout 0.3, making about a third 0 to prevent overfitting
    L.Dense(64, activation="relu"), #Create a large dense layer
    L.Dropout(0.3), #Make the dropout 0.3, making about a third 0 to prevent overfitting
    L.Dense(3) #Create a small dense layer
])


model.compile(loss = SparseCategoricalCrossentropy(from_logits = True), #Compile the model with a SparseCategorical loss function
              optimizer = 'adam', metrics = ['accuracy'] #Add an adam optimizer and collect the accuracy along the way
             )

history = model.fit(textTrain, ratingTrain, epochs = epoch, validation_split = 0.2, batch_size = batchSize) #Fit the model to the data
predict = model.predict_classes(textTest) #Predict ratings based on the model
loss, accuracy = model.evaluate(textTest, ratingTest) #Get the loss and Accuracy based on the tests

#Print the loss and accuracy
print("Test Loss: ", loss)
print("Test Accuracy: ", accuracy)
from sklearn.metrics import classification_report #Import a classification report
print(classification_report(ratingTest, predict, target_names = labels)) #Print a classification report