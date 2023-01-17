#Import all the dependencies

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.tokenize import sent_tokenize, word_tokenize



from nltk.corpus import inaugural



inaugural.fileids()[0:5]
data2 = ["I love machine learning. Its awesome.",

        "I love coding in python",

        "I love building chatbots",

        "they chat amagingly well"]



tagged_data2 = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data2)]



print(tagged_data2)
data = [inaugural.raw('1789-Washington.txt'), 

        inaugural.raw('1793-Washington.txt'),

        inaugural.raw('2001-Bush.txt'), 

        inaugural.raw('2009-Obama.txt')]



sentences = [] 



# iterate through each sentence in the file

for para in data:

    for i in sent_tokenize(para): 

        sentences.append(i)



print(len(sentences))
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sentences)]
print(tagged_data[0:4])
max_epochs = 100

vec_size = 20

alpha = 0.025



model = Doc2Vec(size=vec_size,

                alpha=alpha, 

                min_alpha=0.00025,

                min_count=1,

                dm =1)



model.build_vocab(tagged_data)
for epoch in range(max_epochs):

    #print('iteration {0}'.format(epoch))

    model.train(tagged_data,

                total_examples=model.corpus_count,

                epochs=model.iter)

    

    #decrease the learning rate

    model.alpha -= 0.0002

    

    #fix the learning rate, no decay

    model.min_alpha = model.alpha



model.save("d2v.model")

print("Model Saved")
from gensim.models.doc2vec import Doc2Vec



model= Doc2Vec.load("d2v.model")



#to find the vector of a document which is not in training data

test_data = word_tokenize(inaugural.raw('2017-Trump.txt').lower())



v1 = model.infer_vector(test_data)

print("V1_infer", v1)
# to find most similar doc using tags

similar_doc = model.docvecs.most_similar('1')

print(similar_doc)