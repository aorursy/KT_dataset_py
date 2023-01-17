#bu kod pycharm'da çalışırken burada hata vermektedir.
#Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space

#that measures the cosine of the angle between them.

#V1 and V2 : vectors  Cos(V1,V2) = (V1 * V2) / ||V1|| ∗ ||V2||
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
#kullanıcıdan iki tane string al

X = input("1: ").lower()

Y = input("2: ").lower()

#iki string içinde tokenlarını tutan listele oluştu

list_X = word_tokenize(X)

list_Y = word_tokenize(Y)

print(list_X)

print(list_Y)
sw = stopwords.words('english')

l1 = []

l2 = []

#stringlerin içinde stopwordlerle eşleşmeyen kelimeleri tutan kümeler oluştur, yani stopwordleri ayıkla.

set_X = {w for w in list_X if not w in sw}

set_Y = {w for w in list_Y if not w in sw}

print(set_X)

print(set_Y)
#iki kümeyi kesiştir ve vector oluştur.

vector = set_X.union(set_Y)

for w in vector:

    if w in set_X:

        l1.append(1)

    else:

        l1.append(0)

    if w in set_Y:

         l2.append(1)

    else:

         l2.append(0)

c = 0

#formülle kosinüs benzerliği hesapla.

for i in range(len(vector)):

    c+= l1[i]*l2[i]

cosine = c / float((sum(l1)*sum(l2))**0.5)

print("similarity: ", cosine)            