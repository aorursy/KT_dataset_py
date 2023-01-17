# install sent2vec
!pip install git+https://github.com/epfml/sent2vec
# install annoy
!pip install annoy

!pip freeze > kaggle_image_requirements.txt
!ls ../input
import re
import time

start = time.time()
english_sentences = []
with open("../input/brofo1/English_text.csv") as f:
    for line in f:
        english_sentences.append(re.sub(r'[\W\d]', " ",line.lower())) # clean and normalize
end = time.time()
print("Loading the english sentences took %d seconds"%(end-start))
print("A sample of the english sentences is:")
print(english_sentences[1])
print("The length of the list is:")
print(len(english_sentences))
import re
import time

start = time.time()
twi_sentences = []
with open("../input/whlugbe-twi/Twi_text.csv") as f:
    for line in f:
        twi_sentences.append(re.sub(r'[^a-zA-Z.ƆɔɛƐ!)?’]', " ",line.lower())) # clean and normalize

end = time.time()
print("Loading the twi sentences took %d seconds"%(end-start))
print("A sample of the twi sentences is:")
print(twi_sentences[1])
print("The length of the list is:")
print(len(twi_sentences))
import time
import sent2vec

model = sent2vec.Sent2vecModel()
start=time.time()
model.load_model('../input/sent2vec/wiki_unigrams.bin')
end = time.time()
print("Loading the sent2vec embedding took %d seconds"%(end-start))

import numpy as np
def assemble_embedding_vectors(data):
    out = None
    for item in data:
        vec = model.embed_sentence(item)
        if vec is not None:
            if out is not None:
                out = np.concatenate((out,vec),axis=0)
            else:
                out = vec                                            
        else:
            pass
        
    return out
maxsent = 189
start=time.time()
EmbeddingVectors = assemble_embedding_vectors(english_sentences[:maxsent])
end = time.time()
print("Computing all embeddings took %d seconds"%(end-start))
print(EmbeddingVectors)
print("The shape of embedding matrix:")
print(EmbeddingVectors.shape)
# Save embeddings for later use
np.save("english_sent2vec_vectors_jw.npy",EmbeddingVectors)
# Build and Test Index w/ Annoy for fast Neareast-Neighbor Retrieval

# First build the annoy index for the available English sent2vec vectors

from annoy import AnnoyIndex

start = time.time()
dimension = EmbeddingVectors.shape[1] # Length of item vector that will be indexed
english_NN_index = AnnoyIndex(dimension, 'angular')  
for i in range(EmbeddingVectors.shape[0]): # go through every embedding vector
    english_NN_index.add_item(i, EmbeddingVectors[i]) # add to index

english_NN_index.build(30) # 10 trees
english_NN_index.save('en_sent2vec_NN_index.ann') # save index
end = time.time()
print("Building the NN index took %d seconds"%(end-start))
test_english_NN_index = AnnoyIndex(dimension, 'angular')
test_english_NN_index.load('en_sent2vec_NN_index.ann') # super fast, will just mmap the file
translation_idx =1  # choose index of sentence to focus on in english_sentences/twi_sentences

annoy_out = test_english_NN_index.get_nns_by_item(translation_idx, 3) # will - nearest neighbors to the very first sentence
print(annoy_out)
print("- The sentence we are finding nearest neighbors of:\n")
print(english_sentences[annoy_out[1]])
print("\n\n- The 3 nearest neighbors found:\n")
for i in range(0,3):
    print(str(i) + ". "+ english_sentences[annoy_out[i]])
print("- In other words, if we were translating the english sentence:\n")
print(english_sentences[annoy_out[1]])
print("  where the known correct translation is:")
print(twi_sentences[annoy_out[1]])
print("\n\n- The 3 top translation suggested by our sparse retrieval system above are:\n")
for i in range(0,3):
    print(str(i) + ". "+ twi_sentences[annoy_out[i]])
print(len(english_sentences))
print(len(twi_sentences))