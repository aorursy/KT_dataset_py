# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
corpus_title = []
empty_set = []
y_title = []
for i in range(0, 500):
    if (str(dataset['Title'][i]) == 'nan'):
        empty_set.append(i)
        continue
    review = re.sub('[^a-zA-Z]', ' ', str(dataset['Title'][i]))
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_title.append(review)
    y_title.append(dataset.iloc[i,6])
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 250)
X_title = cv.fit_transform(corpus_title).toarray()
y_title = np.array(y_title)

corpus_descrip = []
empty_set = []
y_descrip = []
for i in range(0, 500):
    if (str(dataset['Review Text'][i]) == 'nan'):
        empty_set.append(i)
        continue
    review = re.sub('[^a-zA-Z]', ' ', str(dataset['Review Text'][i]))
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_descrip.append(review)
    y_descrip.append(dataset.iloc[i,5])

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer(max_features = 1500)
X_descrip = cv1.fit_transform(corpus_descrip).toarray()
y_descrip = np.array(y_descrip)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_title, y_title, test_size = 0.2, random_state = 0)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_descrip, y_descrip, test_size = 0.2, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_title = GaussianNB()
classifier_title.fit(X_train_t, y_train_t)

# Predicting the Test set results
y_pred_t = classifier_title.predict(X_test_t)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_t, y_pred_t)



# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_descrip = GaussianNB()
classifier_descrip.fit(X_train_d, y_train_d)

# Predicting the Test set results
y_pred_d = classifier_descrip.predict(X_test_d)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_d = confusion_matrix(y_test_d, y_pred_d)
print('**************************************************************')
print("Do you want to give us your valuable feedback ?  1 for yes , 0 to skip")
choice = input()
i = 0
feedback_data = []
while (choice == '1'):
    print('Title')
    in_title = []
    Title = str(input())
    title = Title
    Title = re.sub('[^a-zA-Z]', ' ', Title)
    Title = Title.lower()
    Title = Title.split()
    Title = [lemmatizer.lemmatize(word) for word in Title if not word in set(stopwords.words('english'))]
    Title = ' '.join(Title)
    in_title.append(Title)
    Title = cv.transform(in_title).toarray()
    Title_res =  classifier_title.predict(Title)[0]

    if (Title_res == 1):
        print ('\nThanks !!! please describe your wonderful experience with us \n')
    else:
        print('\nSorry for your trouble! Please do describe your concern, so that we can make your experience better from next time \n' )
    
    print('Description :')
    in_title = []
    Title = str(input())
    descrip = Title
    Title = re.sub('[^a-zA-Z]', ' ', Title)
    Title = Title.lower()
    Title = Title.split()
    Title = [lemmatizer.lemmatize(word) for word in Title if not word in set(stopwords.words('english'))]
    Title = ' '.join(Title)
    in_title.append(Title)
    Title = cv1.transform(in_title).toarray()
    descrip_res =  classifier_descrip.predict(Title)[0]    
    print(descrip_res)
    print('\n we have recorded your issues \n' )
    i= i+1
    feedback_data.append([title , descrip , Title_res , descrip_res])
    print("Do you want to give us your valuable feedback ?  1 for yes , 0 to skip")
    choice = input()
feedback_data = pd.DataFrame(np.array(feedback_data) , columns = ['Title' ,'description' , 'like it ?' , 'Rating'])    

feedback_data