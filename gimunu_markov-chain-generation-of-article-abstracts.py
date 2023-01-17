import numpy as np
import pandas as pd
import markovify #ready-to-use text Markov chain
df = pd.read_json("../input/arxivData.json")

#concatenating titles and abstract in new column
df["all_text"] = df["title"] + ". " + df["summary"]
df["all_text"] = df["all_text"].map(lambda x : x.replace("\n", " "))
df.head(5)
#number of words defining a state in the text Markov chain
STATE_SIZE = 2

#generating a model for all the text and one only for titles
text_model = markovify.Text( df["all_text"], state_size=STATE_SIZE)
title_model = markovify.Text( df["title"], state_size=STATE_SIZE)
def findnth( str, char=" ", n=2):
    """
    Returns position of n-th occurence of pattern in a string
    """
    
    index_from_beg = 0
    while n >= 1:
        index = str.find( char)
        str = str[index+1:]
        index_from_beg += index + len(char)
        n -= 1
    return index_from_beg

sample_size = 7
successes = 0
while successes < sample_size:
    try: #some make_sentence calls raise a KeyError exception for misunderstood reasons
        #first generating a title
        _title = title_model.make_sentence()
        _end_of_title = " ".join( _title.split()[-STATE_SIZE:])

        #generating abstract from the end of the tile
        _abstract = text_model.make_sentence_with_start( _end_of_title)
        
        #concatenating both
        index = findnth( _abstract, " ", 2)
        _abstract = _abstract[index:]
        _full_article_description = _title + " " + _abstract
        print( _full_article_description, end="\n\n")
        successes += 1

    except:
        pass