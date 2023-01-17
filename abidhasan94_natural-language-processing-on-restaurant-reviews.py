import numpy as np 
import pandas as pd
import spacy

data = pd.read_csv("../input/Restaurant_Reviews.tsv",delimiter="\t")
nlp = spacy.load('en_core_web_sm')

Liked_list = []
NLiked_list = []
for i in range(1000):
    review = data['Review'][i]
    review = nlp(review)
    for token in review:
        
        if((token.pos_ =='ADJ' or token.pos=='INTJ') and data['Liked'][i]):
            Liked_list.extend([token.lemma_])
        elif(token.pos_ =='ADJ' or token.pos=='INTJ'):
            NLiked_list.extend([token.lemma_])
    
NLiked_list = pd.DataFrame(NLiked_list)
NLiked_list.columns = ['keys']
Liked_list = pd.DataFrame(Liked_list)
Liked_list.columns = ['keys']
NLiked_list= NLiked_list[NLiked_list!='-PRON-']
Liked_list= Liked_list[Liked_list!='-PRON-']
Liked_list.groupby('keys')['keys'].count().sort_values(ascending = False)
NLiked_list.groupby('keys')['keys'].count().sort_values(ascending = False)

