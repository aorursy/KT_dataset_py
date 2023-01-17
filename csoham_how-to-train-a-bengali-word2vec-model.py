import json

import os

import re

import string

import numpy as np



from gensim.models import Word2Vec
def extract_text(filename, field):

    

    extracted_field=[]

    

    with open(os.path.join(filename), 'r') as f:

        articles=json.load(f)

    

    for article in articles['articles']:

        extracted_field.append(article[field].strip())

    

    return extracted_field
def replace_strings(texts, replace):

    new_texts=[]

    

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    english_pattern=re.compile('[a-zA-Z0-9]+', flags=re.I)

    

    for text in texts:

        for r in replace:

            text=text.replace(r[0], r[1])

        text=emoji_pattern.sub(r'', text)

        text=english_pattern.sub(r'', text)

        text=re.sub(r'\s+', ' ', text).strip()

        new_texts.append(text)



    return new_texts
def remove_punc(sentences):

    # import ipdb; ipdb.set_trace()

    new_sentences=[]

    exclude = list(set(string.punctuation))

    exclude.extend(["’", "‘", "—"])

    for sentence in sentences:

        s = ''.join(ch for ch in sentence if ch not in exclude)

        new_sentences.append(s)

    

    return new_sentences
ebala_body=extract_text('../input/news articles/News Articles/ebala_articles.txt', 'body')



print("\x1b[31mCrawled Unprocessed Text\x1b[0m")

print(ebala_body[43])



replace=[('\u200c', ' '),

         ('\u200d', ' '),

        ('\xa0', ' '),

        ('\n', ' '),

        ('\r', ' ')]



ebala_body=remove_punc(ebala_body)



print("\x1b[31mSentences after removing all punctuations\x1b[0m")

print(ebala_body[43])



ebala_body=replace_strings(ebala_body, replace)



print("\x1b[31mSentences after replacing strings\x1b[0m")

print(ebala_body[43])
abz_body=extract_text('../input/news articles/News Articles/anandabazar_articles.txt', 'body')



abz_body=remove_punc(abz_body)

abz_body=replace_strings(abz_body, replace)
zee_body=extract_text('../input/news articles/News Articles/zeenews_articles.txt', 'body')



zee_body=remove_punc(zee_body)

zee_body=replace_strings(zee_body, replace)
body=[]

body.extend(zee_body)

body.extend(abz_body)

body.extend(ebala_body)



print(f"Total Number of training data: {len(body)}")
body=[article.split('।') for article in body]

body=[item for sublist in body for item in sublist]

body=[item.strip() for item in body if len(item.split())>1]



body=[item.split() for item in body]



print(body[:10])
model = Word2Vec(body, size=200, window=5, min_count=1)
print("What are the words most similar to chele")

girl=model.wv.most_similar('ছেলে', topn=5)[0][0]

print(model.wv.most_similar('ছেলে', topn=5)[0][0])
print("What is Father + Girl - Boy =?")

model.wv.most_similar(positive=['বাবা', girl], negative=['ছেলে'], topn=5)
print('Find the odd one out')

model.wv.doesnt_match("কলকাতা চেন্নাই দিল্লি রবীন্দ্রনাথ".split())
print("How similar are bengali and sweet?")

model.wv.similarity('বাঙালি', 'মিষ্টি')
print("What about Bihari and Sweets?")

model.wv.similarity('বিহারি', 'মিষ্টি')