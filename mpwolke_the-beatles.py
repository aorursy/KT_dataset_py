#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoCi2-W71AWrWH7yn2taf8B9lpOfAC2pMfFx1DV52x5qAQmrSK&s',width=400,height=400)
beatles_file = '../input/poetry/beatles.txt'

with open(beatles_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

def plotWordFrequency(input):

    f = open(beatles_file,'r')

    words = [x for y in [l.split() for l in f.readlines()] for x in y]

    data = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)[:40] 

    most_words = [x[0] for x in data]

    times_used = [int(x[1]) for x in data]

    plt.figure(figsize=(20,10))

    plt.bar(x=sorted(most_words), height=times_used, color = 'grey', edgecolor = 'black',  width=.5)

    plt.xticks(rotation=45, fontsize=18)

    plt.yticks(rotation=0, fontsize=18)

    plt.xlabel('Most Common Words:', fontsize=18)

    plt.ylabel('Number of Occurences:', fontsize=18)

    plt.title('Most Commonly Used Words: %s' % (beatles_file), fontsize=24)

    plt.show()
beatles_file = '../input/poetry/beatles.txt'

plotWordFrequency(beatles_file)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRursYTNfVXcyao1fqpUzN935goZFrZtG1dSph6n5E35VywgYqv&s',width=400,height=400)
import pronouncing

import markovify

import re

import random

import numpy as np

import os

import keras

from keras.models import Sequential

from keras.layers import LSTM 

from keras.layers.core import Dense
def create_network(depth):

    model = Sequential()

    model.add(LSTM(4, input_shape=(2, 2), return_sequences=True))

    for i in range(depth):

        model.add(LSTM(8, return_sequences=True))

    model.add(LSTM(2, return_sequences=True))

    model.summary()

    model.compile(optimizer='rmsprop',

              loss='mse')

    if artist + ".rap" in os.listdir(".") and train_mode == False:

        model.load_weights(str(artist + ".rap"))

        print("loading saved network: " + str(artist) + ".rap") 

    return model
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpKnb5_peEj54gqYmF04NWK8hT5m_Py4ErKEUV-XqgxIWB2jBxuA&s',width=400,height=400)
def markov(text_file):

    ######

    read = open(text_file, "r", encoding='utf-8').read()

    text_model = markovify.NewlineText(read)

    return text_model
def syllables(line):

    count = 0

    for word in line.split(" "):

        vowels = 'aeiouy'

#       word = word.lower().strip("!@#$%^&*()_+-={}[];:,.<>/?")

        word = word.lower().strip(".:;?!")

        if word[0] in vowels:

            count +=1

        for index in range(1,len(word)):

            if word[index] in vowels and word[index-1] not in vowels:

                count +=1

        if word.endswith('e'):

            count -= 1

        if word.endswith('le'):

            count+=1

        if count == 0:

            count +=1

    return count / maxsyllables
def rhymeindex(lyrics):

        if str(artist) + ".rhymes" in os.listdir(".") and train_mode == False:

            print ("loading saved rhymes from " + str(artist) + ".rhymes")

            return open(str(artist) + ".rhymes", "r",encoding='utf-8').read().split("\n")

        else:

            rhyme_master_list = []

            print ("Building list of rhymes:")

            for i in lyrics:

                    word = re.sub(r"\W+", '', i.split(" ")[-1]).lower()

                    rhymeslist = pronouncing.rhymes(word)

                    rhymeslistends = []      

                    for i in rhymeslist:

                            rhymeslistends.append(i[-2:])

                    try:

                            rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)

                    except Exception:

                            rhymescheme = word[-2:]

                    rhyme_master_list.append(rhymescheme)

        rhyme_master_list = list(set(rhyme_master_list))

        reverselist = [x[::-1] for x in rhyme_master_list]

        reverselist = sorted(reverselist)

        rhymelist = [x[::-1] for x in reverselist]

        print("List of Sorted 2-Letter Rhyme Ends:")

        print(rhymelist)

        f = open(str(artist) + ".rhymes", "w", encoding='utf-8')

        f.write("\n".join(rhymelist))

        f.close()

        return rhymelist
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
def rhyme(line, rhyme_list):

    word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()

    rhymeslist = pronouncing.rhymes(word)

    rhymeslistends = []

    for i in rhymeslist:

        rhymeslistends.append(i[-2:])

    try:

        rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)

    except Exception:

        rhymescheme = word[-2:]

    try:

        float_rhyme = rhyme_list.index(rhymescheme)

        float_rhyme = float_rhyme / float(len(rhyme_list))

        return float_rhyme

    except Exception:

        float_rhyme = None

        return float_rhyme
def split_lyrics_file(text_file):

    text = open(text_file, encoding='utf-8').read()

    text = text.split("\n")

    while "" in text:

        text.remove("")

    return text
def generate_lyrics(text_model, text_file):

    bars = []

    last_words = []

    lyriclength = len(open(text_file,encoding='utf-8').read().split("\n"))

    count = 0

    markov_model = markov(text_file)

    

    while len(bars) < lyriclength / 9 and count < lyriclength * 2:

        bar = markov_model.make_sentence(max_overlap_ratio = .49, tries=100)

        if type(bar) != type(None) and syllables(bar) < 1:

            def get_last_word(bar):

                last_word = bar.split(" ")[-1]

                if last_word[-1] in "!.?,":

                    last_word = last_word[:-1]

                return last_word

            last_word = get_last_word(bar)

            if bar not in bars and last_words.count(last_word) < 3:

                bars.append(bar)

                last_words.append(last_word)

                count += 1

    return bars
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://miro.medium.com/max/1448/1*IUv42UtorfHGmx9aND3whA.png',width=400,height=400)
def build_dataset(lines, rhyme_list):

    dataset = []

    line_list = []

    for line in lines:

        line_list = [line, syllables(line), rhyme(line, rhyme_list)]

        dataset.append(line_list)

    x_data = []

    y_data = []

    for i in range(len(dataset) - 3):

        line1 = dataset[i    ][1:]

        line2 = dataset[i + 1][1:]

        line3 = dataset[i + 2][1:]

        line4 = dataset[i + 3][1:]

        x = [line1[0], line1[1], line2[0], line2[1]]

        x = np.array(x)

        x = x.reshape(2,2)

        x_data.append(x)

        y = [line3[0], line3[1], line4[0], line4[1]]

        y = np.array(y)

        y = y.reshape(2,2)

        y_data.append(y)

    x_data = np.array(x_data)

    y_data = np.array(y_data)

    return x_data, y_data
def compose_rock(lines, rhyme_list, lyrics_file, model):

    rock_vectors = []

    human_lyrics = split_lyrics_file(lyrics_file)

    initial_index = random.choice(range(len(human_lyrics) - 1))

    initial_lines = human_lyrics[initial_index:initial_index + 2]

    starting_input = []

    for line in initial_lines:

        starting_input.append([syllables(line), rhyme(line, rhyme_list)])

    starting_vectors = model.predict(np.array([starting_input]).flatten().reshape(1, 2, 2))

    rap_vectors.append(starting_vectors)

    for i in range(100):

        rock_vectors.append(model.predict(np.array([rock_vectors[-1]]).flatten().reshape(1, 2, 2)))

    return rock_vectors
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://miro.medium.com/max/1780/1*_0kQ6xWdHS4cI4d9xnaBNw.png',width=400,height=400)
def vectors_into_song(vectors, generated_lyrics, rhyme_list):

    print ("\n\n")

    print ("Writing verse:")

    print ("\n\n")

    def last_word_compare(rock, line2):

        penalty = 0 

        for line1 in rock:

            word1 = line1.split(" ")[-1]

            word2 = line2.split(" ")[-1]

            while word1[-1] in "?!,. ":

                word1 = word1[:-1]

            while word2[-1] in "?!,. ":

                word2 = word2[:-1]

            if word1 == word2:

                penalty += 0.2

        return penalty

    def calculate_score(vector_half, syllables, rhyme, penalty):

        desired_syllables = vector_half[0]

        desired_rhyme = vector_half[1]

        desired_syllables = desired_syllables * maxsyllables

        desired_rhyme = desired_rhyme * len(rhyme_list)

        score = 1.0 - abs(float(desired_syllables) - float(syllables)) + abs(float(desired_rhyme) - float(rhyme)) - penalty

        return score

    dataset = []

    for line in generated_lyrics:

        line_list = [line, syllables(line), rhyme(line, rhyme_list)]

        dataset.append(line_list)

    rock = []

    vector_halves = []

    for vector in vectors:

        vector_halves.append(list(vector[0][0])) 

        vector_halves.append(list(vector[0][1]))

    for vector in vector_halves:

        scorelist = []

        for item in dataset:

            line = item[0]

            if len(rap) != 0:

                penalty = last_word_compare(rap, line)

            else:

                penalty = 0

            total_score = calculate_score(vector, item[1], item[2], penalty)

            score_entry = [line, total_score]

            scorelist.append(score_entry)

        fixed_score_list = [0]

        for score in scorelist:

            fixed_score_list.append(float(score[1]))

        max_score = max(fixed_score_list)

        for item in scorelist:

            if item[1] == max_score:

                rock.append(item[0])

                print (str(item[0]))

                for i in dataset:

                    if item[0] == i[0]:

                        dataset.remove(i)

                        break

                break     

    return rock
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://miro.medium.com/max/1895/1*fV6LtafEA0ZC3SB7QErxeg.png',width=400,height=400)
def train(x_data, y_data, model):

    model.fit(np.array(x_data), np.array(y_data),

              batch_size=2,

              epochs=5,

              verbose=1)

    model.save_weights(artist + ".rock")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://miro.medium.com/max/1380/1*NxiBxeASQIej_qx5Eu_qYA.png',width=400,height=400)
def main(depth, train_mode):

    model = create_network(depth)

    text_model = markov(text_file)

    if train_mode == True:

        bars = split_lyrics_file(text_file)

    if train_mode == False:

        bars = generate_lyrics(text_model, text_file)

    rhyme_list = rhymeindex(bars)

    if train_mode == True:

        x_data, y_data = build_dataset(bars, rhyme_list)

        train(x_data, y_data, model)

    if train_mode == False:

        vectors = compose_rock(bars, rhyme_list, text_file, model)

        rock = vectors_into_song(vectors, bars, rhyme_list)

        f = open(rap_file, "w", encoding='utf-8')

        for bar in rock:

                f.write(bar)

                f.write("\n")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
depth = 4 

maxsyllables = 8

artist = "artist"

rock_file = "temporary_poem.txt"
maxsyllables = 8

text_file = "../input/poetry/beatles.txt"

train_mode = True        

#main(depth, train_mode)

train_mode = False

#main(depth, train_mode)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://miro.medium.com/max/2695/1*GKT1iUZhuQIHu8BfvIk6JQ.png',width=400,height=400)