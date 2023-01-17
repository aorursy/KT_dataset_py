import pandas as pd



dataset_fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

dataset_true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')



dataset_fake["Is_True"]=0

dataset_true["Is_True"]=1



dataset = pd.concat([dataset_fake,dataset_true]) #Merging the 2 datasets



dataset["Full_Content"] = dataset['title']+ " " + dataset['text']



dataset.sample(2)

# Cleaning the texts

import re

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords





def simplify_text(string):

    review = re.sub('[^a-zA-Z]', ' ', string)

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    return(string)

    

    

dataset['Full_Content_New']=dataset['Full_Content'].apply(simplify_text)

print("Concat Done")
features=3000

# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = features)

X = cv.fit_transform(dataset['Full_Content_New'])

y = dataset.iloc[:,4].values



# Splitting the dataset into the Training set, Validation set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size = 0.50, random_state = 0)



print("Split done")
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



# Initialising the ANN

model = Sequential()



# Adding the input layer and the first hidden layer

model.add(Dense(features, kernel_initializer='uniform',activation = 'relu', input_dim = features))

   

# Adding the output layer

model.add(Dense(1,kernel_initializer='uniform',activation = 'sigmoid'))



# Compiling the ANN

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



history = model.fit(X_train.toarray(), y_train, batch_size = 500, epochs = 12, validation_data=(X_test1.toarray(), y_test1))



# Get training and validation loss histories

train_loss = history.history['loss']

validation_loss = history.history['val_loss']



# Create count of the number of epochs

epoch_count = range(1, len(train_loss) + 1)



import matplotlib.pyplot as plt

# Visualize loss history

plt.plot(epoch_count, train_loss, 'r--')

#plt.plot(epoch_count, accuracy_val, 'b-')

plt.plot(epoch_count, validation_loss, 'g--')

plt.legend(['Training Loss', 'Validation Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show();





# Predicting the Test set results

y_pred = model.predict(X_test2.toarray())

y_pred = (y_pred > 0.5)



print("--------------------------------------------")

print("Printing classification_report for Test Set")    

from sklearn.metrics import classification_report

print (classification_report(y_test2, y_pred))

print("--------------------------------------------")


