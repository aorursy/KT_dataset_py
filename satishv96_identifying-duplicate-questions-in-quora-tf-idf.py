import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



print("hello")
df = pd.read_csv("../input/questions.csv")
# use a fraction if it time out in ur machine.

df = df[0:40000]

df.head(10)
#df.head(-1)
#Checks null column values 

df.isnull().sum()
df.is_duplicate.value_counts()
# In this case, 62.8% will be our baseline for accuracy.

25109/len(df)
# Take a look at some of the question pairs.

print("Not duplicate:")

print(df.question1[0])

print(df.question2[0])

print()

print("Not duplicate:")

print(df.question1[1])

print(df.question2[1])

print()

print("Is duplicate:")

print(df.question1[5])

print(df.question2[5])
#This task looks like it will be a little difficult since the first pair of questions have very similar wordings but different meanings, and the third pair have less similar wordings but the same meaning.



import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

import re



def review_to_wordlist(review, remove_stopwords=True):

    # Clean the text, with the option to remove stopwords.

    

    # Convert words to lower case and split them

    words = review.lower().split()



    # Optionally remove stop words (true by default)

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]

    

    review_text = " ".join(words)



    # Clean the text

    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)

    review_text = re.sub(r"\'s", " 's ", review_text)

    review_text = re.sub(r"\'ve", " 've ", review_text)

    review_text = re.sub(r"n\'t", " 't ", review_text)

    review_text = re.sub(r"\'re", " 're ", review_text)

    review_text = re.sub(r"\'d", " 'd ", review_text)

    review_text = re.sub(r"\'ll", " 'll ", review_text)

    review_text = re.sub(r",", " ", review_text)

    review_text = re.sub(r"\.", " ", review_text)

    review_text = re.sub(r"!", " ", review_text)

    review_text = re.sub(r"\(", " ( ", review_text)

    review_text = re.sub(r"\)", " ) ", review_text)

    review_text = re.sub(r"\?", " ", review_text)

    review_text = re.sub(r"\s{2,}", " ", review_text)

    

    words = review_text.split()

    

    # Shorten words to their stems

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in words]

    

    review_text = " ".join(stemmed_words)

    

    # Return a list of words

    return(review_text)
def process_questions(question_list, questions, question_list_name):

# function to transform questions and display progress

    for question in questions:

        question_list.append(review_to_wordlist(question))

        if len(question_list) % 10000 == 0:

            progress = len(question_list)/len(df) * 100

            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))
questions1 = []     

process_questions(questions1, df.question1, "questions1")

print()

questions2 = []     

process_questions(questions2, df.question2, "questions2")
# Take a look at some of the processed questions.

for i in range(5):

    print(questions1[i])

    print(questions2[i])

    print()
# Stores the indices of unusable questions

invalid_questions = []

for i in range(len(questions1)):

    # questions need to contain a vowel (which should be part of a full word) to be valid

    if not re.search('[aeiouyAEIOUY]', questions1[i]) or not re.search('[aeiouyAEIOUY]', questions2[i]):

    # Need to subtract 'len(invalid_questions)' to adjust for the changing index values as questions are removed.

        invalid_questions.append(i-len(invalid_questions))

print(len(invalid_questions))
# list of invalid questions

invalid_questions
#These questions look pretty unusable, so it should be okay to remove them. 

#Plus, we are only removing less than 0.09% of all of the questions.



# Remove the invalid questions

for index in invalid_questions:

    df = df[df.id != index]

    questions1.pop(index)

    questions2.pop(index)



# These questions are also unusable, but were not detected initially.

# They were found when the function 'cosine_sim' stopped due to an error.

unexpected_invalid_questions = [36460]#,42273,65937,304867,306828,353918] 

for index in unexpected_invalid_questions:

    df = df[df.id != index]

    questions1.pop(index)

    questions2.pop(index)
# Use TfidfVectorizer() to transform the questions into vectors,

# then compute their cosine similarity.



from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt





#***** Need to check the .fit-transform

vectorizer = TfidfVectorizer()

def cosine_sim(text1, text2):

    try:

        tfidf = vectorizer.fit_transform([text1, text2])

        #print(text1)

        #print("-------------------")

        #print(text2)

    except Exception:

        print(text1)

        print("-------------------")

        print(text2)

    

    return ((tfidf * tfidf.T).A)[0,1]
Tfidf_scores = []

for i in range(len(questions1)):

    try:

        score = cosine_sim(questions1[i], questions2[i])

    except Exception:

        print("error---")

        continue

        

    Tfidf_scores.append(score)

    if i % 10000 == 0:

        progress = i/len(questions1) * 100

        print("Similarity Scores is {}% complete.".format(round(progress,2)))

# Plot the scores

plt.figure(figsize=(12,4))

plt.hist(Tfidf_scores, bins = 200)

plt.xlim(0,1)

plt.show()
df.is_duplicate.count()
len(Tfidf_scores)
# Function to report the quality of the model

def performance_report(value, score_list):

    # the value (0-1) is the cosine similarity score to determine if a pair of questions

    # have the same meaning or not.

    scores = []

    for score in score_list:

        if score >= value:

            scores.append(1)

        else:

            scores.append(0)



    accuracy = accuracy_score(df.is_duplicate[:39952], scores) * 100

    print("Accuracy score is {}%.".format(round(accuracy),1))

    print()

    print("Confusion Matrix:")

    print(confusion_matrix(df.is_duplicate[:39952], scores))

    print()

    print("Classification Report:")

    print(classification_report(df.is_duplicate[:39952], scores))
#Tfidf_scores
performance_report(0.52, Tfidf_scores)
#Method 2: Doc2Vec



# Reset index to match the index values of questions1 and questions2

df = df.reset_index()