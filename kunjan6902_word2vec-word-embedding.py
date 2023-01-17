from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import inaugural



import warnings



warnings.filterwarnings(action = 'ignore')
import gensim 

from gensim.models import Word2Vec
s = inaugural.raw('1789-Washington.txt')

print(s)
f = s.replace("\n", " ")

#print(f)

data = []
# iterate through each sentence in the file 

for i in sent_tokenize(f): 

    temp = [] 

      

    # tokenize the sentence into words 

    for j in word_tokenize(i): 

        temp.append(j.lower()) 

  

    data.append(temp)

    

print(data)
# Create CBOW model 

model1 = gensim.models.Word2Vec(data, min_count = 2, size = 100, window = 5)



# Print results 

print("Cosine similarity between 'country' and 'citizens' - CBOW : ", model1.similarity('country', 'citizens'))

print("Cosine similarity between 'country' and 'government' - CBOW : ", model1.similarity('country', 'government'))
# Create Skip Gram model 

model2 = gensim.models.Word2Vec(data, min_count = 2, size = 100, window = 5, sg = 1)

  

# Print results 

print("Cosine similarity between 'country' and 'government' - Skip Gram : ", model2.similarity('country', 'government')) 

print("Cosine similarity between 'country' and 'states' - Skip Gram : ", model2.similarity('country', 'states')) 

#Finding Vectors for a Word from CBOW model



v1 = model1.wv['united']

print(v1)
vocabulary = model1.wv.vocab

print(len(vocabulary))
print(vocabulary)
#Finding some similar words

sim_words = model1.wv.most_similar('government')

print(sim_words)