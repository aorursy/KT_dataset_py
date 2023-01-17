import pandas as pd

import numpy as np

from gensim.models.keyedvectors import KeyedVectors

from tensorflow.keras.preprocessing import text,sequence

from IPython.display import clear_output
data = pd.read_csv("/kaggle/input/nlp-paras-similarity/Text_Similarity_Dataset.csv")
word_vec_size = 300     #number of features

max_words = 650      # as max words in paragraph is 643

max_word_features = 25000      #number of words to consider(choosed by top frequency)



def transform_text(text, tokenizer):

    """

    Arguments: 

        text : Text to be tokenized(array|series of STRINGS)

        tokenizer: keras tokenizer object initialized with required parameters

    Returns:

        Embedded Text: Each word replaced with an index from vocabulary

    """

    text_emb = tokenizer.texts_to_sequences(text)

    text_emb = sequence.pad_sequences(text_emb, maxlen=max_words)

    return text_emb



para_tokenizer = text.Tokenizer(num_words=max_word_features)           # for preprocessing of text



para_tokenizer.fit_on_texts(data['text1'])                             # prepare tokenizer with new words

para_tokenizer.fit_on_texts(data['text2'])



para_embs_1 = transform_text(data['text1'].astype(str), para_tokenizer)           # transform text into embeddings

para_embs_2 = transform_text(data['text2'].astype(str), para_tokenizer)
embedding_file = "../input/word2vec-google/GoogleNews-vectors-negative300.bin"                     #Using Google's Word2Vec pretrained embeddings

print("Loading word vectors...")

word_vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=True)     #Load word2vec model parameters



print("Matching word vectors...")

EMBEDDING_DIM=300       #no. of features

word_index = para_tokenizer.word_index        # Dict mapping Index of words (same index from tokenizer used in preprocessing/transforming data)

vocabulary_size=min(len(word_index)+1,max_word_features)      # set vocabulary size

text_embs = np.zeros((vocabulary_size, EMBEDDING_DIM))        # initializing text embedding to zeros

for word, i in word_index.items():         # Update values in text-embedding (pick only words we need)

    if i>=max_word_features:         # a simple limiter for words

        continue

    try:

        embedding_vector = word_vectors[word]       # find word(if exists)

        text_embs[i] = embedding_vector             # update

    except KeyError:

        text_embs[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)      # random but controlled declaration of parameters for words not existing in pretrained model



del(word_vectors)
print(text_embs.shape,type(text_embs))

print(para_embs_1.shape,type(para_embs_1))

print(para_embs_2.shape)
def get_featured_vec(para):

    """

        Provides Word2Vec embedding for the every word in para

        

    Arguments:

        para : Array of Embedded text(Labelled Text with index integers)

        num_words : Integer __ Default = 25000

                    Number of words in para(length of para)

    Returns:

        Numpy array of Featured Vectors for every word in Para

    """

    num_words=25000

    para = para.reshape(-1)

    Para_features = []

    for word in para:

        temp = np.eye(num_words)[word]

        feature_vector = np.dot(temp,text_embs)

        Para_features.append(feature_vector)

    return np.array(Para_features)
Eye = np.eye(25000)
def get_featured_vec_vectorized(para):

    """

        Provides Word2Vec embedding for the every word in para

        

    Arguments:

        para : Array of Embedded text(Labelled Text with index integers)

        num_words : Integer __ Default = 25000

                    Number of words in para(length of para)

    Returns:

        Numpy array of Featured Vectors for every word in Para

    """

    num_words=25000

    para = para.reshape(-1)

    word_rep = np.array(list(map(lambda x:Eye[x],para)))

    Para_features =np.dot(word_rep,text_embs)

    return np.array(Para_features)


def cosine_similarity(u, v):        #for vectors only

    """

    Cosine similarity reflects the degree of similarity between u and v

        

    Arguments:

        u -- a word vector of shape (n,)          

        v -- a word vector of shape (n,)



    Returns:

        cosine_similarity -- the cosine similarity between u and v defined by the formula above.

    """

  

    dot = np.dot(u,v)

    

    norm_u = np.sqrt(np.sum(u**2))



    norm_v = np.sqrt(np.sum(v**2))

   

    cosine_similarity = dot/(norm_u*norm_v)

    

    return cosine_similarity
def cos_sim_matrix(matrix1,matrix2):

    """

        Provides Cosine similarity matrix for relation between every word in matrix1 to every word in matrix2

    

    Arguments:

        Two matrix with shape matrix1:(?,n) and matrix2:(?,n)

    

    Returns:

        Numpy Array: Cosine Similarity matrix(all possible combinations)

    

    """

    #Cosine Similarity Vectorized Implementation

    dot = np.dot(matrix1,matrix2.T)

    l2_norm = np.sum(matrix1**2,1)

    l2_norm_1 = np.sqrt(l2_norm)

    

    l2_norm = np.sum(matrix2**2,1)

    l2_norm_2 = np.sqrt(l2_norm)

    mul = (l2_norm_1*l2_norm_2)

    return (dot/mul)
def solve(features_a,features_b):

    """

    Computes the Cosine Similarity Matrix for feature_a and feature_b



    Arguments: Word2Vec representation of words in Paragraphs.

                Two featured representations features_a and features_b to compute similarity b/w them.

    

    Returns: Degree of Similarity between the Feature sets

    """

    def func(x):

        """

        To scale the output later

        """

        return (max(x-0.3,0))/0.7



    sim_matrix = cos_sim_matrix(features_a,features_b)         #compute cosine similarity

    

    # CLEANING PHASE

    k =np.nan_to_num(sim_matrix,nan=0)                         #remove NANs appearing due to paddings

    k[k<-1]=0                                                  #remove -infs and similar values due to paddings

    k[k>1e3]=0                                                 #remove +infs and similar values due to paddings

    

    #################

    #COMPUTING PHASE#

    #################

    

    #Using GREEDY ALGORITHM

    

    #for best matches

    temp = k.copy()                                            # a temporary matrix of Cosine Similarity

    best_match_list = []                                       # To store best matches of words

    for i in range(len(k)):                                    # Loop through the words in 1st matrix

        word = temp[:,i]                                       # select the word

        best_match = word.argmax()                             # choose best matching word from 2nd matrix

        best_match_list.append(word[best_match])               # store value of that index for the matched word

        temp= np.delete(temp,best_match,0)                     # remove the word from 2nd matrix



    best_match_list = np.array(best_match_list)

    sol_max =best_match_list[best_match_list != 0].mean()      # Remove all zeros and take the mean of best -

                                                               # -matched words FOR EVERY WORD IN 1st MATRIX

    #For worst matches

    temp = k.copy()

    best_match_list = []

    for i in range(len(k)):

        word = temp[:,i]

        best_match = word.argmin()

        best_match_list.append(word[best_match])

        temp= np.delete(temp,best_match,0)

    

    best_match_list = np.array(best_match_list)

    sol_min =best_match_list[best_match_list != 0].mean()

    

    print("max_sol: ",sol_max)

    print("min_sol: ",sol_min)

    print("Similarity :",func(sol_max))

    #scaled similarity returned

    # As using max() with greedy, we reduce the outcome by extending the lower limit by a factor of 0.3

    return func(sol_max)
def complete(string_a,string_b):

    """

    Computes similarity b/w two strings

    

    Arguments:

        Two strings string_a,string_b

    

    Returns:

        Scaled Cosine Similarity b/w them

    

    """

    tempa = transform_text([string_a],para_tokenizer)              # Embed

    tempa = get_featured_vec_vectorized(tempa)                                # Get Featurized vector

    tempb = transform_text([string_b],para_tokenizer)

    tempb = get_featured_vec_vectorized(tempb)

    sol = solve(tempa,tempb)                                       # Get similarity

    return sol
data["Similarity_Score"] = 0
data.loc[i,"Similarity_Score"] = complete(data["text1"][i],data["text2"][i])
for i in range(len(data)):

    print(i)

    data.loc[i,"Similarity_Score"] = complete(data["text1"][i],data["text2"][i])

    clear_output(wait=True)
temp = data[["Unique_ID","Similarity_Score"]]
k = temp["Similarity_Score"].max()
temp["Similarity_Score"] = temp["Similarity_Score"].apply(lambda x: x/k)
temp.to_csv("submission.csv",index=False)