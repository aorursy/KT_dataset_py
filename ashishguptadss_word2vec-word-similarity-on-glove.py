import numpy as np

import pandas as pd

import keras

import re

from keras.layers import Embedding, Flatten, Dense

from sklearn.metrics.pairwise import cosine_similarity

!pip install glove_python
from glove import Corpus, Glove
#Creating a corpus object

corpus = Corpus() 



lines=["Hello this is a tutorial on how to convert the word in an integer format","this is a beautiful day","Jack is going to office"]

new_lines=[] 

for line in lines: 

    new_lines.append(line.split(' ')) #new lines has the new format lines=new_lines

    

new_lines
# read data

data= pd.read_excel("../input/hindisongsexcelsupportedformat/processedSongs.xlsx" )



# preprocessing function for any song

def preprocessSong(song):

    dataset = ''

    listOfWords = re.split(r'[;,\s...\n()\'!?.]\s*',song) # gets me a list of words

    for word in listOfWords:

#         if word ==  '' || word=='(' || word==')' || word =='\'':

#             pass

#         else:

        word = word.lower()

        if word == 'x2':

            continue

        if word == 'x4':

            continue

        dataset+=' '+word

    return dataset
lyrics = list(data.songLyrics)

myProcessedSongs = []

for song in lyrics:

    mysong = preprocessSong(song)

    song_words = []

    for word in mysong.split(' '):

        if word!='':

            song_words.append(word)

    myProcessedSongs.append(song_words)

myProcessedSongs
#Training the corpus to generate the co occurence matrix which is used in GloVe

corpus.fit(myProcessedSongs, window=30)



glove = Glove(no_components=30, learning_rate=0.05) 

glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)

glove.add_dictionary(corpus.dictionary)

glove.save('glove_hindi_songs.model')



female_names=['ladki','girl','gori','lady','kudi','chhori','woman']

male_names=['ladka','munda','mundey','boy','chokra','chhora','man']



color =['sanwali','saanwala','pink','pinky','red','laal','kaala','kaali','gori','gora','white','black','yellow','brown']



softAttitude=['bholi','bhola','heeran','hirni','nadaan','beautiful','mastani','mastana','seedhi','seedha','sharmili','sharmeela','sohni','sohna','bhali','bhala']

strongAttitude=['kukkad','bigda','bigdi','khatra','khauf','handa','jungli','badmash','gussa']



cars=['car', 'gaddi','drive','lamborghini','jaguar','gaadi','motorcycle']

clothes=['jeans','skirt','shirt','lehnga','chunni','ainak','ghagra','kurta','pajama','jacket','choodi','jhumka','chasma','chashma','kangan','top']

food=['namkeen','mithi','tikhi','teekha','khatti','makkhan','sweet','nimbu','imli','mitthe','rasmalai','mirchi','mishti','naariyal']

alcohol=['daaru','whisky','daru','pila','botal','peg','shots','drink','peeta']

bodylooks=['choti','chota','cheeks','adayein','thumka','aankhen','aankhein','nazron','charming']





import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

 

# def cosine_similarity(np_array_1, np_array_2):

#     a = np_array_1

#     b = np_array_2

#     # manually compute cosine similarity

#     dot = np.dot(a, b)

#     norma = np.linalg.norm(a)

#     normb = np.linalg.norm(b)

#     cos = dot / (norma * normb)

#     # use library, operates on sets of vectors

#     return cos

    

def unit_vector(vec):

    """

    Returns unit vector

    """

    return vec / np.linalg.norm(vec)



def cosine_similarity(v1, v2):

    """

    Returns cosine of the angle between two vectors

    """

    v1_u = unit_vector(v1)

    v2_u = unit_vector(v2)

    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)

    



vec1 = glove.word_vectors[glove.dictionary['ladki']] 

vec2 = glove.word_vectors[glove.dictionary['ladka']] 



print(cosine_similarity((vec1), vec2))



def weat_test(target_one,target_two, target_one_words, attribute_one,attribute_two, attribute_one_words, target_two_words, attribute_two_words):

    cos=[]

    s=0

    s1=[]

    s2=[]

    S=[]

    n=0

        

    for i in range(0, len(target_one_words)):

            c1=[]

            c2=[]

            for k in range(0, len(attribute_one_words)):

                wt = target_one_words[i]

                at1 = attribute_one_words[k]

                try:

                    vec1 = glove.word_vectors[glove.dictionary[wt]]

                    vec2 = glove.word_vectors[glove.dictionary[at1]]

                    cos1 = cosine_similarity(vec1, vec2)

                    cos.append(cos1)

                    c1.append(cos1)

                except:

                    cos1=0

                    cos.append(cos1)

                    c1.append(cos1)

                    continue

            for k in range(0, len(attribute_two_words)):

                cos2=0

                wt = target_one_words[i]

                at2 = attribute_two_words[k]

                try:

                    vec1 = glove.word_vectors[glove.dictionary[wt]]

                    vec2 = glove.word_vectors[glove.dictionary[at2]]

                    cos2 = cosine_similarity(vec1, vec2)

                    cos.append(cos2)

                    c2.append(cos2)

                except:

                    cos2=0

                    cos.append(cos2)

                    c2.append(cos2)

                    continue

            s1.append((np.mean(c1)-np.mean(c2)))

            S.append((np.mean(c1)-np.mean(c2)))

            n=n+1

    for i in range(0, len(target_two_words)):

            c1=[]

            c2=[]

            for k in range(0, len(attribute_one_words)):

                wt = target_two_words[i]

                at1 = attribute_one_words[k]

                try:

                    vec1 = glove.word_vectors[glove.dictionary[wt]]

                    vec2 = glove.word_vectors[glove.dictionary[at1]]

                    cos1 = cosine_similarity(vec1, vec2)

                    cos.append(cos1)

                    c1.append(cos1)

                except:

                    cos1=0

                    cos.append(cos1)

                    c1.append(cos1)

                    continue

            for k in range(0, len(attribute_two_words)):

                cos2=0

                wt = target_two_words[i]

                at2 = attribute_two_words[k]

                try:

                    vec1 = glove.word_vectors[glove.dictionary[wt]]

                    vec2 = glove.word_vectors[glove.dictionary[at2]]

                    cos2 = cosine_similarity(vec1, vec2)

                    cos.append(cos2)

                    c2.append(cos2)

                except:

                    cos2=0

                    cos.append(cos2)

                    c2.append(cos2)

                    continue

            s2.append((np.mean(c1)-np.mean(c2)))

            S.append((np.mean(c1)-np.mean(c2)))

    s=np.sum(s1)-np.sum(s2)

    stdev=np.std(S)

    print(target_one + ' vs ' + target_two  + ' , ' +attribute_one + ' vs ' + attribute_two +', d = ' + str(s/(stdev*n)))



def avg_similarity(target,attribute):

   

    S=[]

    

    for i in range(0, len(target)):

        

#         cos.append(model.similarity(target[i],attribute))

#             c1=[]

        maxlist=[]    

    

        vec1 = glove.word_vectors[glove.dictionary[target[i]]] 

        

        for k in range(0, len(attribute)):

            

            vec2 = glove.word_vectors[glove.dictionary[attribute[k]]] 

            maxlist.append(cosine_similarity(vec1, vec2))

            

        maxlist.sort(reverse=True)

        

        for j in range(0,4):

            S.append(maxlist[j])

            

        ans= np.array(S).sum()/(len(target)*4)

    return ans



print()

print()

print("Weat test results are ")

print(weat_test('female_names','male_names', female_names, 'softAttitude' ,'strongAttitude', softAttitude, male_names, strongAttitude))



att=['color','softAttitude','strongAttitude','cars','clothes','food','alcohol','bodylooks']

att1=[color,softAttitude,strongAttitude,cars, clothes,food,alcohol,bodylooks]

female_score=[]

male_score=[]



for i in att1:

    

    female_score.append(avg_similarity(female_names,i))

    male_score.append(avg_similarity(male_names,i))      

    

support1=pd.DataFrame({'attribute':att,'female_names':female_score,'male_names':male_score})



print(support1)
print(glove.word_vectors[glove.dictionary['ladki']])
glove.most_similar("ladki",number = 20)
support1
# np.loadtxt('../input/glove.6B.50d.txt')

import pandas as pd

def read_glove_vectors(trainedWeightsFile = "../input/glove.6B.50d.txt"):

    f = open(trainedWeightsFile)

    word_embeddings_matrix = {}

    for wordAndEmbeddings in f:

        line = wordAndEmbeddings.split()

        word= line[0]

        coefs = np.asarray(line[1:], dtype = 'float32')

        word_embeddings_matrix[word] = coefs

    f.close()

    return word_embeddings_matrix
embedding_matrix = read_glove_vectors()

# print(len(embedding_matrix))

word_embeddings_matrix = embedding_matrix

embedding_matrix['ashish']
apple = embedding_matrix['apple']

# apple = np.array(apple)
lion = embedding_matrix['lion']

# lion = np.array(lion)
orange = embedding_matrix['orange']

# orange = np.array(orange)

print(orange)
# print(orange)

print(cosine_similarity(apple.reshape(1,-1), orange.reshape(1,-1)))
def generate_similar_words(wrd1, wrd2, wrd3):

    w1 = np.array(embedding_matrix[wrd1]).reshape(1,-1)

    w2 = np.array(embedding_matrix[wrd2]).reshape(1,-1)

    w3 = np.array(embedding_matrix[wrd3]).reshape(1,-1)

    options = ['giraffe', 'tiger', 'monkey', 'chimpanzee', 'duck','bird', 'ducks', 'lion', 'goat']

    tar = ''

    max_sim = -10000

    for word in options:

        if word in (wrd1, wrd2, wrd3):

            continue

        word_tar = np.array(embedding_matrix[word]).reshape(1,-1)

        sim = cosine_similarity((w3-(w1-w2)).reshape(1,-1),word_tar.reshape(1,-1))

        if(sim>max_sim):

            max_sim = sim

            tar = word

    return tar
generate_similar_words('monkey', 'chimpanzee', 'duck')
# f = open('../input/glove.6B.50d.txt')

# for line in f:

#     print(line)
import pandas as pd

processedSongs = pd.read_csv("../input/processedSongs.csv")