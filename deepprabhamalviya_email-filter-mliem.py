# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#ok, so check this first, the thing they were talking about in the workshop group
df = pd.read_csv("..//input/spam_or_not_spam.csv",encoding = 'latin-1')
df
#ok, so we have now loaded the dataset. I will let you know what exactly i showed to maam, and the

#points that were given in the assignment.

#ok.

df.shape #Dataframe rowxcolumn
#Now, sir had asked to drop dupliciates, so we will invoke that function
df.drop_duplicates(inplace=True)

df.shape #New shape updated
#Remember, in the previous day, we had attributed positive and negative sentiments of the movie review

# with 1 and 0 ? I dont remember if 1 was good or 0 was good,Yeah,. so. .same  here
positive_sentiment_count = len(df[df.label == 1])

negative_sentiment_count = len(df[df.label == 0])  #This was for the moview review. 

#So we can use the same here for filtering.

#Since here also, there is a column named label and its row values are 0 and 1. Yeah. So lets plot.

print(positive_sentiment_count,negative_sentiment_count,sep="\n")
# So these many 1s and these many 0s are here. Is it ok ?
#ok.ok. Now, just making a plot with this data, this is not a point, I was following the previvous days work so i did.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)





#Imported the matplot library for plotting the chart
plt.figure(figsize = (6,5))

ax = sns.countplot(x='label',data=df)
#Ok, so these are graphs for the filtered labels with 0,1 attributes wrt to the sentences
#Now, an error will come, which is, in the dataset, there are some values that are missing.

#Remember that NaN thing ? we need to filter them out before we proceed further.naa



#Mane , in the dataset, if missing datas are there. We remove that ok. Yes, otherwise, next is countvectorizer,

#if there is missing data, it will generate error.
dp = float("NaN")

df.replace("", dp, inplace = True)

df.drop_duplicates(inplace = True)

df.dropna(inplace = True)

df.dropna(subset = ["email"],inplace = True)

df.shape

print(df.shape)

print(df.isnull().sum())

print(df)
#So any missing value removed, this is the updated dataset now.
# thus was step? not sure, we have still to remove punctuations and stopwordshi
from nltk.corpus import stopwords

stop_words = stopwords.words('english') 

#This part is a reference from maam's code

# Part for removing stopwords here
import string

def process_text(email):

    nopunc = [char for char in email if char in string.punctuation] 

    #Part for removing punctuations here

    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean_words
#Generate the freshwords in a seperate place

# We will scan & add it to the email column's values
    

    # This will check the database : It scans for words that were not present in the stopwords list.

    # Lexical words like a, an, and, the will be filtered.

    # Now, if stop words is kill, knife, it will be filtered

    # Then I am a good boy and with - will be the clean words, each I think is seperated by a comma

    # It just filters out potential trigger words to generate a clean report.ok

    
print(df['email'].head().apply(process_text))

df.shape
df
#now we will do the main portion, that is countvectorizer and traininf test.

 

    # This will check the database for lexical stop words.

    # I copy pasted from net - this example list : {‘ourselves’, ‘hers’, ‘between’, ‘yourself’, ‘but’, ‘again’, ‘there’, ‘about’, ‘once’, ‘during’, ‘out’, ‘very’, ‘having’, ‘with’, ‘they’, ‘own’, ‘an’, ‘be’, ‘some’, ‘for’, ‘do’, ‘its’, ‘yours’, ‘such’, ‘into’, ‘of’, ‘most’, ‘itself’, ‘other’, ‘off’, ‘is’, ‘s’, ‘am’, ‘or’, ‘who’, ‘as’, ‘from’, ‘him’, ‘each’, ‘the’, ‘themselves’, ‘until’, ‘below’, ‘are’, ‘we’, ‘these’, ‘your’, ‘his’, ‘through’, ‘don’, ‘nor’, ‘me’, ‘were’, ‘her’, ‘more’, ‘himself’, ‘this’, ‘down’, ‘should’, ‘our’, ‘their’, ‘while’, ‘above’, ‘both’, ‘up’, ‘to’, ‘ours’, ‘had’, ‘she’, ‘all’, ‘no’, ‘when’, ‘at’, ‘any’, ‘before’, ‘them’, ‘same’, ‘and’, ‘been’, ‘have’, ‘in’, ‘will’, ‘on’, ‘does’, ‘yourselves’, ‘then’, ‘that’, ‘because’, ‘what’, ‘over’, ‘why’, ‘so’, ‘can’, ‘did’, ‘not’, ‘now’, ‘under’, ‘he’, ‘you’, ‘herself’, ‘has’, ‘just’, ‘where’, ‘too’, ‘only’, ‘myself’, ‘which’, ‘those’, ‘i’, ‘after’, ‘few’, ‘whom’, ‘t’, ‘being’, ‘if’, ‘theirs’, ‘my’, ‘against’, ‘a’, ‘by’, ‘doing’, ‘it’, ‘how’, ‘further’, ‘was’, ‘here’, ‘than’}

    # So many types are here.

    # It just filters out potential line stop trigger words to generate a clean report.ok

    

print(df['email'].head().apply(process_text))

df.shape
from sklearn.feature_extraction.text import CountVectorizer

testdata = CountVectorizer(analyzer=process_text).fit_transform(df['email'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(testdata, df['label'],test_size = 0.30,random_state=0)

print(testdata)

print(testdata.shape)
# This is everything upto the Naiive Bayes algorithm implementation.
#All previous points have been completed.

#This latest data set is the training and test data that was generated from the filtered words (stopwords and cleanwords)

#Ok, Multinomial and something else. 
# I forgot the name, but since the dataset and train test was 0 and 1, i chose that option. Bernoulli - Deals

# With 0 and 1 vector
from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB().fit(X_train, y_train)

#print the predictions

print(classifier.predict(X_train))

print(y_train.values)



# This will check the actual values first from the latest data. We theen proceed to the prediction and analysis



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred= classifier.predict(X_train)

print(classification_report(y_train,pred))

print()



print('Confusion matrix : \n', confusion_matrix (y_train,pred))

print()

print('Accuracy : ', accuracy_score(y_train, pred))
print(classifier.predict(X_test))

print(y_test.values)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred= classifier.predict(X_test)

print(classification_report(y_test,pred))

print()
print('Confusion matrix : \n', confusion_matrix (y_test,pred))

print()

print('Accuracy : ', accuracy_score(y_test,pred))