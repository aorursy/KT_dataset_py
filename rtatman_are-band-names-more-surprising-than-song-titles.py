# libraries we're going to use

import pandas as pd

import collections, nltk

from sklearn.model_selection import train_test_split

import csv



# read in & subset our lyrics into testing & training sets

songs = pd.read_csv("../input/Lyrics1.csv", nrows = 10000)

train, test = train_test_split(songs.Lyrics, test_size=0.2)



# save out test set

test.to_csv('test_data.csv')
# code taken from https://github.com/luochuwei/Perplexity_calculate/tree/fa601a0c95423ddb69b124c1f0547bdb36e20584

corpus = ' '.join(train.tolist())



# we first tokenize the text corpus

tokens = nltk.word_tokenize(corpus)



#here you construct the unigram language model 

def unigram(tokens_for_unigram):    

    model = collections.defaultdict(lambda: 0.01)

    for f in tokens_for_unigram:

        try:

            model[f] += 1

        except KeyError:

            model[f] = 1

            continue

    for word in model:

        model[word] = model[word]/float(len(model))

    return model



#computes perplexity of the unigram model on a testset  

def perplexity(testset, model):

    testset = testset.split()

    perplexity = 1

    N = 0

    for word in testset:

        N += 1

        perplexity = perplexity * (1/model[word])

    if N != 0:

        perplexity = pow(perplexity, 1/float(N))

    else:

        perplexity = "inf"

    return perplexity
# create unigram model with our training data

model = unigram(tokens)
# empty list for our lyrics plexity

lyrics_plexity = []



# get the plexity for each set of test lyrics

for i in test:

    lyrics_plexity.append((perplexity(i, model)))

    

# print first ten

lyrics_plexity[0:10]
# save out the lyrics info

with open('lyrics_plexity.csv','w') as output_file:

    output_file.write("perplexity\n")

    for item in lyrics_plexity:

        output_file.write("%s\n" % item)
# read in band names

bandNames = pd.read_csv("../input/ArtistUrl.csv")



# remove duplicates

unique_bandNames = bandNames.Artist.unique()



# save list of unique bandnames

with open('bandNames.csv','w') as output_file:

    output_file.write("band_name\n")

    for item in unique_bandNames:

        output_file.write("%s\n" % item)
# get the band name perplexity

# empty list for our band name plexity

band_plexity = []



# get the plexity for each band name

for i in unique_bandNames:

    band_plexity.append((perplexity(i, model)))



# print first ten

band_plexity[0:10]
# save our perplexity info! 

with open('band_plexity.csv','w') as output_file:

    output_file.write("perplexity\n")

    for item in band_plexity:

        output_file.write("%s\n" % item)