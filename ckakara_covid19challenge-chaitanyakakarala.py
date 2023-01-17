# Importing required Libraries

import os
import pandas as pd
import json
import numpy as np
import spacy
import re
from spacy.matcher import PhraseMatcher
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier


def dataExtraction(keywordList):
    keywords=keywordList
    paragraphs=[]

    for dirpath,dirnames,filenames in os.walk('/kaggle/input'):
        for file in filenames:
            if(file.endswith(".json")):
                with open(os.path.join(dirpath, file)) as json_file:
                    data = json.load(json_file)
                    for paragraph in data["body_text"]:
                        paragraph["text"] = re.sub(r'[-]',' ',paragraph["text"])
                        for word in keywords:
                            word_reg="\\b"+word+"\\b"
                            words_re = re.compile(word_reg,re.IGNORECASE)
                            if (words_re.search(paragraph["text"] )):
                                paragraphs.append(paragraph["text"])
                        break;

    #print(len(paragraphs)) 
    return paragraphs


def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]
def Tfidf_frequent_words(paragraphs):
    Tfidf_vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer, min_df=2)
    Tfidf_data_vectorized = Tfidf_vectorizer.fit_transform(paragraphs)
    
    word_count = pd.DataFrame({'word': Tfidf_vectorizer.get_feature_names(), 'sum of tf-idf': np.asarray(Tfidf_data_vectorized.sum(axis=0))[0]})

    word_count.sort_values('sum of tf-idf', ascending=False).set_index('word')[:20].sort_values('sum of tf-idf', ascending=True).plot(kind='barh')
    return word_count
    
def cow_frequent_words(paragraphs):
    cow_vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, min_df=2)
    cow_data_vectorized = cow_vectorizer.fit_transform(paragraphs)
    
    word_count = pd.DataFrame({'word': cow_vectorizer.get_feature_names(), 'sum of cow': np.asarray(cow_data_vectorized.sum(axis=0))[0]})

    word_count.sort_values('sum of cow', ascending=False).set_index('word')[:20].sort_values('sum of cow', ascending=True).plot(kind='barh')
    return word_count
    
def gen_wordcloud(paragraphs):
    
    wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black',stopwords = STOPWORDS).generate((" ").join(paragraphs))
    fig = plt.figure(figsize = (40, 30),facecolor = 'k',edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
def extractRelations(doc):
    spans=spacy.util.filter_spans(list(doc.ents) + list(doc.noun_chunks))
    relations = []
    nouns=[]

    for sent in doc.sents: # For every sentence in the doc
        tags=[]
        text= re.sub(r'-',' ',sent.text.lower())
        #print(sent.text)
        for word in keywords: # For every Keyword
                word_reg="\\b"+word+"\\b"
                words_re = re.compile(word_reg,re.IGNORECASE)
                if (words_re.search(text)): # if keyword present in the sentence
                    for t in sent:
                        if(t.tag_ in['NN']):
                            nouns.append(t)
                    for t in sent:
                        for noun in nouns:
                            if noun.text.lower() in t.text.lower():
                                #print(t.dep_,t.head.dep_)
                                #print(list(t.head.head.children))
                                if t.dep_ == "attr":
                                    n1 = [w for w in t.head.children if (w.dep_ == "nsubj")]
                                    n2 = t
                                    if len(n1) > 0:
                                        n1 = n1[0]
                                        n1_subtree_span = doc[n1.left_edge.i : n1.right_edge.i + 1]
                                        n2_subtree_span = doc[n2.left_edge.i : n2.right_edge.i + 1]
                                        tags.append(n2_subtree_span)  
                                if t.dep_ == "pobj" and t.head.dep_ == "prep":
                                    #print(list(t.head.head.children))
                                    #X= [w.dep_ for w in t.head.head.children]
                                    #print(X)
                                    n1 = [w for w in t.head.head.children]
                                    n2 = t
                                    if len(n1) > 0:
                                        n1 = n1[0]
                                        n1_subtree_span = doc[n1.left_edge.i : n1.right_edge.i + 1]
                                        n2_subtree_span = doc[n2.left_edge.i : n2.right_edge.i + 1]
                                        tags.append(n2_subtree_span) 
                    relations.append([tags,sent])
                    break
    return relations
        
from __future__ import unicode_literals, print_function
# Load Model
model="en_core_web_sm"
nlp = spacy.load(model)
# Set the keywords based on the question
keywords = ["naproxen","clarithromycin","minocycline"]
# call dataExtraction to find all the pargraphs containing atleast one of the keyword.
paragraphs=dataExtraction(keywords)
#Pass the Paragraphs to NLP Pipe
docs = list(nlp.pipe(paragraphs))
# Identify Most Frequent Words
Tfidf_word_cnt=Tfidf_frequent_words(paragraphs)
cow_word_cnt=cow_frequent_words(paragraphs)

# wordcloud on Paragraphs
gen_wordcloud(paragraphs)

factors=[]
# Find the relations of Noin phrases in each sentence by calling extarctRelations
for doc in docs:
    relations = extractRelations(doc)
    if len(relations) > 0:
        for relation in relations:
            factors.append(relation)

Q1Answer=pd.DataFrame(factors,columns=["Tags","Sentence"])
Q1Answer["Sentence"].drop_duplicates()
Q1Answer.to_csv("Q1Answer.csv")
pd.set_option('display.max_colwidth', -1)
Q1Answer.head(10)
from __future__ import unicode_literals, print_function
# Load Model
model="en_core_web_sm"
nlp = spacy.load(model)
# Set the keywords based on teh question
keywords = ["ADE","antibody dependent enhancement"]
# call dataExtraction to find all the pargraphs containing atleast one of the keyword.
paragraphs=dataExtraction(keywords)
#Pass the Paragraphs to NLP Pipe
docs = list(nlp.pipe(paragraphs))
# Identify Most Frequent Words
Tfidf_word_cnt=Tfidf_frequent_words(paragraphs)
cow_word_cnt=cow_frequent_words(paragraphs)

# wordcloud on Paragraphs
gen_wordcloud(paragraphs)

factors=[]
# Find the relations of Noin phrases in each sentence by calling extarctRelations
for doc in docs:
    relations = extractRelations(doc)
    if len(relations) > 0:
        for relation in relations:
            factors.append(relation)

Q2Answer=pd.DataFrame(factors,columns=["Tags","Sentence"])
Q2Answer["Sentence"].drop_duplicates()
Q2Answer.to_csv("Q2Answer.csv")
Q2Answer.head(10)
from __future__ import unicode_literals, print_function
# Load Model
model="en_core_web_sm"
nlp = spacy.load(model)
# Set the keywords based on teh question
keywords = ["animal models"]
# call dataExtraction to find all the pargraphs containing atleast one of the keyword.
paragraphs=dataExtraction(keywords)
#Pass the Paragraphs to NLP Pipe
docs = list(nlp.pipe(paragraphs))
# Identify Most Frequent Words
Tfidf_word_cnt=Tfidf_frequent_words(paragraphs)
cow_word_cnt=cow_frequent_words(paragraphs)

# wordcloud on Paragraphs
gen_wordcloud(paragraphs)

factors=[]
# Find the relations of Noin phrases in each sentence by calling extarctRelations
for doc in docs:
    relations = extractRelations(doc)
    if len(relations) > 0:
        for relation in relations:
            factors.append(relation)


Q3Answer=pd.DataFrame(factors,columns=["Tags","Sentence"])
Q3Answer["Sentence"].drop_duplicates()
Q3Answer.to_csv("Q3Answer.csv")
Q3Answer.head(10)
from __future__ import unicode_literals, print_function
# Load Model
model="en_core_web_sm"
nlp = spacy.load(model)
# Set the keywords based on teh question
keywords = ["therapeutic"]
# call dataExtraction to find all the pargraphs containing atleast one of the keyword.
paragraphs=dataExtraction(keywords)
#Pass the Paragraphs to NLP Pipe
docs = list(nlp.pipe(paragraphs))
# Identify Most Frequent Words
Tfidf_word_cnt=Tfidf_frequent_words(paragraphs)
cow_word_cnt=cow_frequent_words(paragraphs)

# wordcloud on Paragraphs
gen_wordcloud(paragraphs)

factors=[]
# Find the relations of Noin phrases in each sentence by calling extarctRelations
for doc in docs:
    relations = extractRelations(doc)
    if len(relations) > 0:
        for relation in relations:
            factors.append(relation)


Q4Answer=pd.DataFrame(factors,columns=["Tags","Sentence"])
Q4Answer["Sentence"].drop_duplicates()
Q4Answer.to_csv("Q4Answer.csv")
Q4Answer.head(10)
from __future__ import unicode_literals, print_function
# Load Model
model="en_core_web_sm"
nlp = spacy.load(model)
# Set the keywords based on teh question
keywords = ["corona vaccine","universal vaccine"]
# call dataExtraction to find all the pargraphs containing atleast one of the keyword.
paragraphs=dataExtraction(keywords)
#Pass the Paragraphs to NLP Pipe
docs = list(nlp.pipe(paragraphs))
# Identify Most Frequent Words
Tfidf_word_cnt=Tfidf_frequent_words(paragraphs)
cow_word_cnt=cow_frequent_words(paragraphs)

# wordcloud on Paragraphs
gen_wordcloud(paragraphs)

factors=[]
# Find the relations of Noin phrases in each sentence by calling extarctRelations
for doc in docs:
    relations = extractRelations(doc)
    if len(relations) > 0:
        for relation in relations:
            factors.append(relation)


Q5Answer=pd.DataFrame(factors,columns=["Tags","Sentence"])
Q5Answer["Sentence"].drop_duplicates()
Q5Answer.to_csv("Q5Answer.csv")
Q5Answer.head(10)
from __future__ import unicode_literals, print_function
# Load Model
model="en_core_web_sm"
nlp = spacy.load(model)
# Set the keywords based on teh question
keywords = ["prophylaxis"]
# call dataExtraction to find all the pargraphs containing atleast one of the keyword.
paragraphs=dataExtraction(keywords)
#Pass the Paragraphs to NLP Pipe
docs = list(nlp.pipe(paragraphs))
# Identify Most Frequent Words
Tfidf_word_cnt=Tfidf_frequent_words(paragraphs)
cow_word_cnt=cow_frequent_words(paragraphs)

# wordcloud on Paragraphs
gen_wordcloud(paragraphs)

factors=[]
# Find the relations of Noin phrases in each sentence by calling extarctRelations
for doc in docs:
    relations = extractRelations(doc)
    if len(relations) > 0:
        for relation in relations:
            factors.append(relation)


Q6Answer=pd.DataFrame(factors,columns=["Tags","Sentence"])
Q6Answer["Sentence"].drop_duplicates()
Q6Answer.to_csv("Q6Answer.csv")
Q6Answer.head(10)
sent_list = []
tag_list = []
for sent in Q1Answer["Sentence"]:
    sent_list.append(sent.text)
    tag_list.append("Inhibitors")
learningDf1 = pd.DataFrame({'Sentence': sent_list, 'Tag': tag_list})
del sent_list, tag_list

sent_list = []
tag_list = []
for sent in Q2Answer["Sentence"]:
    sent_list.append(sent.text)
    tag_list.append("ADE")
learningDf2 = pd.DataFrame({'Sentence': sent_list, 'Tag': tag_list})
del sent_list, tag_list

sent_list = []
tag_list = []
for sent in Q3Answer["Sentence"]:
    sent_list.append(sent.text)
    tag_list.append("Animal Models")
learningDf3 = pd.DataFrame({'Sentence': sent_list, 'Tag': tag_list})
del sent_list, tag_list

sent_list = []
tag_list = []
for sent in Q4Answer["Sentence"]:
    sent_list.append(sent.text)
    tag_list.append("Therapeutic")
learningDf4 = pd.DataFrame({'Sentence': sent_list, 'Tag': tag_list})
del sent_list, tag_list

sent_list = []
tag_list = []
for sent in Q5Answer["Sentence"]:
    sent_list.append(sent.text)
    tag_list.append("Vaccine")
learningDf5 = pd.DataFrame({'Sentence': sent_list, 'Tag': tag_list})
del sent_list, tag_list

sent_list = []
tag_list = []
for sent in Q6Answer["Sentence"]:
    sent_list.append(sent.text)
    tag_list.append("Prophylaxis")
learningDf6 = pd.DataFrame({'Sentence': sent_list, 'Tag': tag_list})
del sent_list, tag_list

data = pd.DataFrame([])
learningDf = pd.concat([learningDf1,learningDf2,learningDf3,learningDf4,learningDf5,learningDf6])
X = learningDf.Sentence
y = learningDf.Tag
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=123, shuffle=True)
tagNames=["Inhibitors","ADE","Animal Models","Therapeutic","Vaccine","Prophylaxis"]
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=tagNames,zero_division=1))
nb = Pipeline([('vect', TfidfVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=tagNames,zero_division=1))
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=tagNames,zero_division=1))
sgd = Pipeline([('vect', TfidfVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=tagNames,zero_division=1))