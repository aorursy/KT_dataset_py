"""

1. Atis üzerindeki cümleler(dataframe_name: atis), wordnet 3.1 deki lemmaların ait oldukları synsetler(dataframe_name: lemma2synset) 

ve her synset için rastgele atanmış (x,y,z) tipinde koordinat bilgileri(dataframe_name: vectors) aşağıdaki kodda bulunmaktadır.



2. Elimdeki bu verileri kullanarak, öncelikle atisteki cümlelerden rastgele bir tanesi seçerek disambigution yapma hedefledim.

    2.1 Rastgele seçilen cümle tokenlara ayrıldı.

    2.2 Bu tokenlardan stopwordler temizlendi.

    2.3 Geriye kalan tokenlar lemma haline getirildi.

    2.4 Daha sonra bu lemmalar üzerinden "lemma2synset" dataframe'inden lemmaların üyesi olduğu tüm synsetler bulundu.



3. Daha sonra anlattığınız gibi belirsizlik yaratan tüm lemmaların kombinasyonlarını buldum. 



4. Son olarak bu kombinasyonlar üzerinden shortest pathi bulmaya çalıştım.

    4.1 Tüm kombinasyonlardaki synsetleri koordinatlarını rastgele atadığım "vectors" dataframeni kullanarak (x,y,z) bilgilerini çıkardım.

    4.2 Daha sonra bu synset koordinatları arasında euclidean distance kullanarak aralarındaki uzaklığı bulup topladım.

    4.3 Ve tüm kombinasyonlar için elimde başlangıçtan bitiş synsetine kadar olan uzaklık bilgilerini hesapladım

    4.4 Ve bu bilgilerden enkısa olanı seçip consola yazdırdım.

"""

import pandas as pd

import random

from nltk.corpus import stopwords

# from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

import itertools

from scipy.spatial import distance

from nltk.tokenize import MWETokenizer
atis = pd.read_csv("../input/disambg/atis_intents.csv")             # Atis df includes the sentences from atis

lemma2synset = pd.read_csv("../input/disambg/lemma2synset.csv")     # Lemma2synset df includes the synsets of lemmas

vectors = pd.read_csv("../input/thesis2/vectorsReal.csv")               # Vectors df includes the random coordinates of synsets



# display(atis_intents)

# display(lemma2synset)

# display(vectors)
stop_words = set(stopwords.words('english'))  # For stopwords

lemmatizer = WordNetLemmatizer()              # For lemmatization
rnd = random.randint(0, 4977)

sentence = atis["vocab"].iloc[rnd]



# print("********     ", str, "      *********")



# or maybe you might want to use your own input



# sentence = 'show me the new flights from san diego to new york also st. peter'
tokenizer = MWETokenizer([('all', 'over'), ('all', 'right'), ('and', 'how'), ('arrival', 'time'), ('as', 'well')

                          , ('arrive', 'at'), ('at', 'least'), ('at', 'the', 'same', 'time'), ('at', 'times')

                          , ('at', 'will'), ('be', 'on'), ('belong', 'to'), ('bring', 'up'), ('car', 'rental')

                          , ('clock', 'in'), ('clock', 'on'), ('close', 'to'), ('coming', 'back'), ('coming', 'back')

                          , ('connecting', 'flight'), ('day', 'of', 'the', 'week'), ('day', 'return')

                          , ('departure', 'time'), ('direct', 'flight'), ('do', 'in'), ('early', 'on'), ('eat', 'on')

                          , ('economy', 'class'), ('equal', 'to'), ('find', 'out'), ('first', 'class'), ('fly', 'on')

                          , ('fort', 'worth'), ('get', 'down'), ('get', 'on'), ('get', 'to'), ('go', 'after')

                          , ('go', 'around'), ('go', 'for'), ('go', 'in'), ('go', 'into'), ('go', 'on')

                          , ('go', 'through'), ('go', 'to'), ('go', 'with'), ('have', 'on'), ('in', 'flight')

                          , ('in', 'for'), ('in', 'on'), ('kansas', 'city'), ('kind', 'of'), ('las', 'vegas')

                          , ('light', 'time'), ('live', 'in'), ('local', 'time'), ('lock', 'in'), ('long', 'beach')

                          , ('look', 'at'), ('look', 'like'), ('looking', 'for'), ('los', 'angeles'), ('many', 'a')

                          , ('more', 'than'), ('morning', 'time'), ('new', 'jersey'), ('new', 'york'), ('number', '1')

                          , ('new', 'york', 'city'), ('nonstop', 'flight'), ('north', 'carolina'), ('of', 'late')

                          , ('on', 'air'), ('on', 'that'), ('on', 'the', 'way'), ('on', 'time'), ('or', 'so')

                          , ('out', 'of'), ('per', 'se'), ('r', 'and', 'b'), ('ring', 'up'), ('round', 'trip')

                          , ('salt', 'lake', 'city'), ('san', 'diego'), ('san', 'francisco'), ('san', 'jose')

                          , ('seating', 'capacity'), ('show', 'business'), ('show', 'up'), ('some', 'other')

                          , ('sort', 'of'), ('st.', 'louis'), ('st.', 'paul'), ('st.', 'peter'), ('st.', 'petersburg')

                          , ('take', 'in'), ('stand', 'for'), ('stop', 'over'), ('take', 'ten'), ('take', 'to')

                          , ('thank', 'you'), ('the', 'city'), ('time', 'of', 'arrival'), ('time', 'zone')

                          , ('to', 'and', 'fro'), ('to', 'that'), ('to', 'wit'), ('travel', 'to'), ('turn', 'around')

                          , ('turn', 'in'), ('turn', 'to'), ('type', 'o'), ('up', 'on'), ('used', 'to'), ('very', 'much')

                          , ('very', 'well')], separator=' ')
word_tokens = tokenizer.tokenize(sentence.split())                          # Tokenization

print(word_tokens)

tokens = [w for w in word_tokens if not w in stop_words]   # Clearing stopwords

storage = []                                                    # It will store the synsets of lemmas

count = 0



for token in tokens:



    lemma = lemmatizer.lemmatize(token)                         # Lemmatization

    synsets = lemma2synset[lemma2synset['lemma'] == lemma]

    if synsets.empty: # For unknown synset

        continue

    else:

        storage.append([])

        print("The word " + token.upper() + " is ambiguous")



    for syn in synsets['synsetid']:

        storage[count].append(syn)

        

    print("Possible synsets")

    print(storage[count])

    print()

    count += 1



combinations = list(itertools.product(*storage))     # Produce the combinations

combinations = pd.DataFrame(combinations)



display(combinations)
summation = []                                      # It will store the total distance of each combination

cols = ['x', 'y', 'z']

c  = 0

for row, column in combinations.iterrows():

    a = []

    count = 0

    for element in column:

        a.append([])

        temp = vectors[vectors['SynsetID'] == element]

        temp = temp.reset_index()

        temp[cols] = temp[cols].apply(pd.to_numeric, errors='coerce', axis=1)

        a[count].append(temp['x'][0])

        a[count].append(temp['y'][0])

        a[count].append(temp['z'][0])

        count += 1



    sum = 0

    for d in range(len(a) - 1):

        dist = distance.euclidean(a[d+1], a[d])  # Compute the distance with euclidean formula for each synset

        sum = sum + dist

    summation.append(sum)

    print("The combination number ", c, " is ", combinations.iloc[c].tolist(), " and the total distance of the combination is ", summation[c])

    print()

    c += 1
index = summation.index(min(summation))     # Find the minimum value of the calculations

shortest_path = combinations.iloc[index]    # And take it from combinations dataframe



print(shortest_path.tolist(), "and its distance is", summation[index])  # Print the combination of shortest distance