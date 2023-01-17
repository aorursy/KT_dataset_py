import numpy as np

import pandas as pd

import re



wine = pd.read_csv("../input/winemag-data-130k-v2.csv")
# review wine dataset

wine.head()
# make sure that each potential input / output does have values 

# drop every NaN / Null which occur in the necessary data. 

wine.info()
# remove duplicate descriptions 

wine.drop_duplicates(subset = ['description', 'title'], inplace = True)
wine_title_count = wine['title'].value_counts().reset_index().rename(columns = {'index':'title', 'title':'count'})

boolWines = wine_title_count[wine_title_count['count'] > 1]

wine_titles = list(boolWines['title'])
# Drop all indexes which do not have enough descriptions

wine_cleaned = pd.DataFrame(columns = wine.columns.values)

for title in wine_titles:

    boolTitle = wine['title'] == title

    wine_cleaned = wine_cleaned.append(wine[boolTitle == True], ignore_index= True)

            

len(wine_cleaned.index)
len(wine_cleaned['title'].value_counts())
# create input list and output list 

questions = list(wine_cleaned['description'])

answers = list(wine_cleaned['title'])
import nltk 

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

porterStemmer = PorterStemmer()
def clean_text(text):

    # put everything into lower_case

    text = text.lower()

    

    # sub everything out except of the words of alphabet in lower case. The description of the

    # wines do allow us to only keep that lower case alphabet - depending on some text analyzing

    # e.g. technical data analysis -- having the value (numbers) might be quite important

    text = re.sub('[^a-z]', ' ', text)

    

    # split the whole description in single words

    text = text.split()

    # reduce the number of words to only necessary words - by using stopwords from nltk

    # we can easily kick out words such as "is" "and" to reduce the input on our model later on and

    # keep focus on relevant words

    # also we will rewrite some words which may be written in plural to - e.g. "loved" to "love"

    text = [porterStemmer.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    return text
questions_nlp = []

for question in questions:

    questions_nlp.append(clean_text(question))
questions[0:4]
questions_nlp[0:4]
# create a dataframe to easilier prepare the correct sets

questions_and_answers = pd.DataFrame(columns = ['questions', 'answers'])

questions_and_answers['questions'] = questions_nlp

questions_and_answers['answers'] = answers

questions_and_answers = questions_and_answers.sort_values(by = 'answers')
questions_and_answers.head(n = 10)
questions_final = questions_and_answers['questions']

answers_final = questions_and_answers['answers']
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
questions_ML = cv.fit_transform(questions_final).toarray()

answers_ML = answers_final

len(questions_ML[0])
len(questions_ML)
len(answers_ML)
# number of unique classes

answers_ML.nunique()
# Split the data into train and test set

questions_train = []

questions_test = []

answers_train = []

answers_test = []

 # dividing the whole set by 50% - since the answers were sorted - 

 # we will have each title at least once in a test and train set

for row in range(0, len(answers_ML)):

    if row % 2 == 0:

        answers_train.append(answers_ML[row])

        questions_train.append(questions_ML[row])

    else:

        answers_test.append(answers_ML[row])

        questions_test.append(questions_ML[row])

        

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(questions_train, answers_train)
y_pred = classifier.predict(questions_test)
Evaluation = pd.DataFrame(columns = ['prediction', 'testvalues'])

Evaluation['prediction'] = y_pred

Evaluation['testvalues'] = answers_test
Evaluation.head(n = 15)
Evaluation['prediction_output'] = Evaluation['prediction'] == Evaluation['testvalues']
Evaluation['prediction_output'].value_counts()