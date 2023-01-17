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
animeDataFrame = pd.read_csv("../input/myanimelist-comment-dataset/animeListGenres.csv", error_bad_lines=False)

animeDataFrame.head()

#print(animeDataFrame.loc[[3573]])
animeDataFrame.columns
animeDataFrameAux = animeDataFrame

animeDataFrameAux.genres = animeDataFrame.genres.str.split(", ")

animeDataFrameAux.head()



#Transformar a lista de listas de generos e apenas uma lista

flat_list = []

for sublist in animeDataFrameAux.genres:

    for item in sublist:

        flat_list.append(item)    

        

#rCriar lista com generos sem repetir

generosUnicos = []

for x in flat_list: 

    # check if exists in unique_list or not 

    if x not in generosUnicos: 

        generosUnicos.append(x)



# print list 

print(generosUnicos)

"""

for sublist in animeDataFrameAux.genres:

    for item in sublist:

        for item2 in generosUnicos:

            if item == item2:

                animeDataFrameAux[item]= 1

            else: 

                animeDataFrameAux[item]= 0

"""
#algo errado não está certo nessa parte



AnimesFinal = pd.DataFrame()



for index, row in animeDataFrameAux.iterrows():

    for item in generosUnicos:

        if item in row.genres:

            row[item] = 1

        else:

            row[item] = 0

        

        AnimesFinal.append(row)
AnimesFinal.head()