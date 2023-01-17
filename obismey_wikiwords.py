import re

import nltk

import sys



resultlist = list()

pagetitlelist = list()



pattern = re.compile("(?:was|is|like|are|were|means) (.*)")

nns = nltk.RegexpParser("3NN: {<NN|NNS><NN|NNS>?<NN|NNS>?}")

fichier = "../input/wikipedia-first.txt"



maxoutput = 5000

pagetitlecount = 0

resultcount = 0



"DONE"
with open(fichier, encoding="utf8") as file:

    pageTitle=""

    

    for line in file:

        

        if pageTitle=="":

            pagetitlecount = pagetitlecount+1

            pageTitle=line[:-1]

            pagetitlelist.append(pageTitle)

            continue

        if line=="\n":

            pageTitle=""

            continue

        

        #Controle du nombre de sortie

        if  resultcount > maxoutput: break 

    

        match=re.search(pattern, line)

        if match != None: 

            grp=match.group(1)

            grp=grp.replace("type of", "").replace("style of", "").replace("sort of", "").replace("Asteroid","asteroid")

            grp=grp.replace("Prime Minister", "prime minister").replace("President","president")

            grp=grp.replace("King", "king").replace("Main Belt", "main belt")

            words = nltk.word_tokenize(grp)

            tags = nltk.pos_tag(words)



            nntags =next(  nns.parse(tags).subtrees(lambda x: x.label() == "3NN" ), None )



            nntagschoix =""

            if nntags is not None : 

                resultcount = resultcount + 1                             

                sys.stdout.write("\rINFO-{0:06d}-{1:06d}".format(pagetitlecount, resultcount) )

                sys.stdout.flush()

                tmp =nntags.pos()

                nntagschoix = tmp[len(tmp)-1][0][0]

                resultlist.append([pageTitle, nntagschoix, grp])

                

import numpy as np

import pandas as pd

from pandas import DataFrame



resultarray = pd.DataFrame( np.array(resultlist))

resultarray.columns = ["Entity", "Type", "Regex"] 

resultarray = resultarray.set_index('Entity').sort_index()



pagetitlearray =pd.DataFrame( np.array(pagetitlelist))

pagetitlearray.columns = ["Entity"]

pagetitlearray = pagetitlearray.set_index('Entity').sort_index()



"DONE"
from wordcloud import WordCloud

wordcloud =WordCloud()

worddata=resultarray.groupby("Type").count()["Regex"].to_dict()

wordcloud=wordcloud.generate_from_frequencies(worddata)

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()