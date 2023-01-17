# Importing the dependencies that we need for this project



import json

import pandas as pd



from os import listdir

from os.path import isfile, join



from tqdm import tqdm



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



import re



import gensim
# Download NLTK copuses



#nltk.download('stopwords')
# Getting a list of folders present in the dataset



path = "/kaggle/input/CORD-19-research-challenge"



datasets = []



for d in listdir(path):

    if not isfile(join(path, d)):

        datasets.append(d)



print(datasets)
# Making a list of JSON files

json_files = []



for folder in datasets:

    files_path = join(path, folder, folder)

    

    files = listdir(files_path)

    

    for file in files:

        json_files.append(join(files_path, file))

        

print(json_files[0:10])
def word_tokenizer(sentence):

    return word_tokenize(sentence) 



def remove_stop_words(tokenized_sentences):

    stop_words = set(stopwords.words('english'))

    return [t for t in tokenized_sentences if not t in stop_words] 



def remove_unchars(doc):

    doc = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', doc, flags=re.MULTILINE)

    doc = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', doc)

    doc = re.sub(r'\b[0-9]+\b\s*', '', doc)



    return doc



def preprocessing_text(text):

    text = text.lower()

    

    text = remove_unchars(text)

    

    words = word_tokenizer(text)

    

    cleaned_words = remove_stop_words(words)

    

    return cleaned_words
# Reading all the json file and extracting the text of the paper



texts = []

cleaned_texts = []



for file in tqdm(json_files):

    data = json.load(open(file, "rb"))

    

        

    text = []

    for t in data["body_text"]:

        texts.append(t["text"])

        cleaned_texts.append(preprocessing_text(t["text"]))
len(texts)
# Training a Gensim Word2Vec model



model = gensim.models.Word2Vec(sentences=cleaned_texts)
# Saving the model



model.wv.save_word2vec_format("model.bin", binary=True)
# Methods for finding same sentences based on keywords



def find_similar_words(word):

    words = [*word]

    

    for word, _ in model.wv.most_similar(word):

        if len(word) > 2:

            words.append(word)

        

    return words



def find_sentences_rank(words, keywords):

    return len(set(words).intersection(set(keywords)))



def find_similar_sentences(cleaned_sentences, original_sentences, keywords):

    try:

        keywords = find_similar_words(keywords)

        

        similar_sentences = []

        similar_sentences_processed = []



        for i in range(len(cleaned_sentences)):

            k = find_sentences_rank(cleaned_sentences[i], keywords)

            if k > 2:

                similar_sentences.append(original_sentences[i])



        return similar_sentences

    except  Exception as ex:

        print("ERROR: ", ex)
# Finding similar sentences based on the word incubation



similar_sentences = find_similar_sentences(cleaned_texts, texts, ['incubation'])
for similar in similar_sentences:

    print(similar)

    print()
print(len(similar_sentences))