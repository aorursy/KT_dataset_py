!pip install git+https://github.com/LIAAD/yake
import yake
import numpy as np

import pandas as pd
text="Apologies for the brief pause in #QuarantineContent Turns out I have exactly 1 month to move EVERYTHING into the new studio So I'm gonna spend my isolation time working hard on that Because it's clear there's a bunch of tech around the corner I need to be ready for!"
simple_kwextractor=yake.KeywordExtractor()

keywords=simple_kwextractor.extract_keywords(text)
for kw in keywords:

    print(kw)
forum_posts=pd.read_csv("../input/meta-kaggle/ForumMessages.csv")
forum_posts.shape
forum_posts['Message'].head()
simple_kwextractor.extract_keywords(forum_posts['Message'][0])
test_post=simple_kwextractor.extract_keywords(forum_posts['Message'][1])
type(simple_kwextractor.extract_keywords(forum_posts['Message'][61513]))
sentences=[]



for post in forum_posts.Message[:20]:

    

    post_keywords=simple_kwextractor.extract_keywords(post)

    

    sentence_output=''

    for word,num in post_keywords:

        sentence_output=sentence_output+' '+word

    

    sentences.append(sentence_output)
sentences