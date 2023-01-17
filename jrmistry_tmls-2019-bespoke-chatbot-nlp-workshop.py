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
####################################

# Bespoke Chatbot NLP Workshop 2019-07-22

# Bespoke speaker Christine Gerpheide

# Organized by Toronto Machine Learning Summit (TMLS)

# Hosted by WeCloudData

####################################



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import precision_recall_fscore_support
# Questions and answers



answers = {

    'when_is_check_in' : 'Check in is at 3pm! :)',

    'where_is_the_front_desk' : 'The front desk is located on the 2nd floor.',

}
# Training dataset



training_phrases = {

    'when_is_check_in' : ' '.join([

        'when is check-in',

        'When can I check in?',

        'whens checkin'

    ]),

    'where_is_the_front_desk' : ' '.join([

        'Where is the front desk?',

        'what is the location of the front desk?'

    ])

}
# Train



training_documents = list(training_phrases.values())

labels = list(training_phrases.keys())



vectorizer = CountVectorizer() # stop_words

X = vectorizer.fit_transform(training_documents)



classifier = MultinomialNB()

classifier.fit(X, labels)
# Predict



def predict(raw_queries):

    queries = vectorizer.transform(raw_queries)

    return classifier.predict(queries)



raw_queries = ["where location",

               "when is",

               "where is check in location"]



predicted = predict(raw_queries)
# Evaluate



expected = ["where_is_the_front_desk",

            "when_is_check_in",

            "where_is_the_front_desk"]



evaluation = precision_recall_fscore_support(expected, predicted, average='micro')



metrics = {}

(metrics['p'], metrics['r'], metrics['f1'], _) = evaluation



print("Evaluation metrics: ", metrics)
# Run



user_query = 'where is my hotel please?'



predicted = predict([user_query])



print("Predicted intent: ", predicted[0])
# Testing some more



print("Predicted intent: ", predict(["Where is the front desk man ???"]))

print("Predicted intent: ", predict(["Hi, what is the check in time again?"]))

print("Predicted intent: ", predict(["Can I check in now? :)"]))
# Implement simple Chatbot



def chat_bot(request):

    print("Request:  ", request)

    print("Response: ", answers[predict([request])[0]])
chat_bot("I am lost, where is the front desk again?")
chat_bot("When was check in again :)")