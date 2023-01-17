import numpy as np 
import pandas as pd 
import scipy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from scipy import linalg
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
np.set_printoptions(suppress=True)
from gensim.models.doc2vec import Doc2Vec 
#reading the data
df=pd.read_csv("../input/CORD-19-research-challenge/metadata.csv",engine='python',error_bad_lines=False) 
#observing the first 5 rows
df.head()
#I am interested in the abstract section
df['abstract'] #512397 abstracts
#replacing NaN values
df['abstract'].fillna('null',inplace=True)
# storing the abstracts in 'data'
data=np.array(df['abstract'])
len(data)
nltk.download('wordnet')
#using CountVectoriser to convert text into matrix form 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = CountVectorizer(stop_words='english',max_features=10000) # removing stop words
vectors = vectorizer.fit_transform(data).todense() #converting into a dense vector
# finding the vocab
vocab = np.array(vectorizer.get_feature_names())
print(vocab.shape)
# show topics method to 
num_top_words=10

def show_topics(a):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]
m,n=vectors.shape
d=10 # setting the number of topics to 10
clf = decomposition.NMF(n_components=d, random_state=1)
W1 = clf.fit_transform(vectors)
H1 = clf.components_
show_topics(H1)
#converting data into list format
list_data=data.tolist()
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(list_data)]
nltk.download('punkt')
#training the model on the corpus
#applying Doc2Vec to convert document into its vector form. Reference: https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
max_epochs = 50
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
    
model.save("d2v.model")
print("Model Saved")
#loding the model again for future reference
model= Doc2Vec.load("d2v.model")
list_data[4]
#finding document closest to the topic
def find_topic(text):
    topic_list=list(show_topics(H1))
    distances=[]
    test_data2=word_tokenize(list_data[n].lower())
    v2=model.infer_vector(test_data2)
    for i in range(len(topic_list)):
        test_data1=word_tokenize(text)
        v1=model.infer_vector(test_data1)
        distances.append(scipy.spatial.distance.cosine(v1,v2))

    min_ele = min(distances) 
    topic_no= [i for i, j in enumerate(distances) if j == min_ele] 
    print('The document probably belongs to category:',topic_no)    
    print('The category is:',show_topics(H1)[topic_no[0]])
find_topic(list_data[4])
def find_doc(topic):
    distances=[]
    topic_list=list(show_topics(H1))
    test_data1=word_tokenize(topic_list[topic].lower())
    v1=model.infer_vector(test_data1)
    for i in range(len(list_data)):
        test_data2=word_tokenize(list_data[i].lower())
        v2=model.infer_vector(test_data2)
        distances.append(scipy.spatial.distance.cosine(v1,v2))
        min_ele = min(distances) 
    for j in range(len(distances)):
        if distances[j]==min_ele:
            doc_no=j
    return(list_data[doc_no])
# Topic 6 basically is related to articles that dig into the i guess as to how the virus functions 
print('Topic 6:',show_topics(H1)[6])
doc=find_doc(6)
print('The document about topic 6 is:',doc)
