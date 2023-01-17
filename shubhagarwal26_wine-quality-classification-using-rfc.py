import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
raw_data=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
raw_data.head()
len(raw_data)
parsed_data = raw_data[raw_data.duplicated('description', keep=False)] #Deleting the duplicate rows 

parsed_data.dropna(subset=['description']) #Deleting the rows with missing values

len(parsed_data) # Getting the length of the parsed data after deleting the above rows
len(raw_data[raw_data['points'].isna()==True])# Checking the number of missing values in our dataset
final_data=parsed_data[['description','price','points']]

final_data.head()
#Since predicting the points for each wine can be a hectic task as there is a very slight difference between points of each wine.

#So we will club the points of each wine into ratings on a scale of 1 to 5.

def simplify(points):

    if points < 84:

        return 1

    elif points >= 84 and points < 88:

        return 2 

    elif points >= 88 and points < 92:

        return 3 

    elif points >= 92 and points < 96:

        return 4 

    else:

        return 5
#adding the simplified points column to the dataset.

final_data = final_data.assign(quality = final_data['points'].apply(simplify))

final_data.head()
X = final_data['description'] #Assigning description of wine as a feature to our model.

y = final_data['quality'] # Assigning points as class label to be predicted by the model.



vectorizer = CountVectorizer()# Initializing the vectorizer using CountVectorizer Method.

vectorizer.fit(X)# Fitting the data in the vectorizer.
X = vectorizer.transform(X) #Transforming the text data using Vectorization to be used as a feature by the model.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)# Splittiing the training data and the test data

rfc = RandomForestClassifier() # Initialising the Random Forest Classifier which is the algorithm we are using in this model.

rfc.fit(X_train, y_train) # Training the model by fitting the training data into the classifier.
predictions = rfc.predict(X_test) # Predicting the test data values using the trained data model.

print(classification_report(y_test, predictions)) # Checking the results of our test data predictions
accuracy_score(y_test,predictions)*100 # Checking the accuracy of our Model on test data
plt.hist(final_data['quality'])

plt.show()