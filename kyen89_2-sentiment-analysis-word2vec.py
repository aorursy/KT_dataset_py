

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Read the labeled training and test data
# Header = 0 indicates that the first line of the file contains column names, 
# delimiter = \t indicates that the fields are seperated by tabs, and 
# quoting = 3 tells python to ignore doubled quotes

train = pd.read_csv('../input/labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
test = pd.read_csv('../input/testData.tsv', header = 0, delimiter = '\t', quoting = 3)
unlabel_train = pd.read_csv('../input/unlabeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
'train dim:{}, unlabeled train dim:{}, test dim:{}'.format(train.shape, unlabel_train.shape, test.shape)

# Import the libraries for data cleaning.

from bs4 import BeautifulSoup
import re
import nltk

def preprocess_wordlist(data, stopwords = False):
    
    # Remove HTML tag
    review = BeautifulSoup(data,'html.parser').get_text()
    
    # Remove non-letters
    review = re.sub('[^a-zA-Z]', ' ', review)
    
    # Convert to lower case
    review = review.lower()
    
    # Tokenize
    word = nltk.word_tokenize(review)
    
    # Optional: Remove stop words (false by default)
    if stopwords:
        stops = set(nltk.corpus.stopwords.words("english"))
        
        words = [w for w in word if not w in stops]
    
    return word


def preprocess_sent(data, stopwords = False):
    
    # Split the paragraph into sentences
    
    #raw = tokenizer.tokenize(data.strip())
    raw = nltk.sent_tokenize(data.strip())
    
    # If the length of the sentence is greater than 0, plug the sentence in the function preprocess_wordlist (clean the sentence)
    sentences = [preprocess_wordlist(sent, stopwords) for sent in raw if len(sent) > 0]
    
    return sentences

sentence = []

# Append labeled reviews first
for review in train['review']:
    sentence += preprocess_sent(review)
    
# Append unlabeled reviews
for review in unlabel_train['review']:
    sentence += preprocess_sent(review)

print(len(sentence))
print()
print(sentence[:2])
train['review'][0]

from gensim.models import word2vec
num_features = 250 #400
min_count = 40
num_processor = 4
context = 10
downsampling = 0.001
# Plug in the sentence variable first.

model = word2vec.Word2Vec(sentence, workers = num_processor, 
                         size = num_features, min_count = min_count,
                         window = context, sample = downsampling)

# Unload unneccessary memory once the learning process is done.

model.init_sims(replace = True)
model_name = "250features_40minwords_20context"
model.save(model_name)
model.most_similar("king")

# Import libraries
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
# List of vocabulary.

vocab = list(model.wv.vocab)

# index vector values by corresponding vocab list

X = model[vocab]

print("Total Number of Vocab:", len(X))
print()
print(X[0][:10])

# Visualize only 100 words.

tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X[:100,:])
df = pd.DataFrame(X_tsne, index = vocab[:100], columns = ['X','Y'])
df.head()
fig = plt.figure()
fig.set_size_inches(30,20)

ax = fig.add_subplot(1,1,1)
ax.scatter(df['X'], df['Y'])

# Put the label on each point.
for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize = 30)
plt.show()

''' 

The purpose of this function is to combine all the word2vec vector values of each word in each review
if each review is given as input and divide by the total number of words.

Each word can be represented as number of feature dimension space vector. ex) model['flower'] = array([0.1, 0.2, ...]).
(You can think of it as extended xy coordinate.) Therefore, it enables vectorizing each review by 
combining all the words' vector values.

Illustration example:

'I' = [0.1, 0.2, ...]
'have' = [0.2, 0.3, ...]
'a car' = [0.1, 0.2, ...]
'I have a car' = [0.1 + 0.2 + 0.1, 0.2 + 0.3 + 0.2,  ...]


ex) review1 = ['he', 'has', 'a', 'cat']

First word : If the word 'he' is in the word2vec, index the vector values from word2vec model by model['he']
(the dimension of the matrix would be, in this case, (400,) ) and add them to predefined zero matrix.

Second word: Check if there is the word 'has' in the word2vec model and if there is, index the vector values and 
add them on top of the added vector values from the first word.

The rest: Iterate the above steps for the rest of words and lastly, divide by the total number of words. 

Illustration example: 

zero : [0,    0,   0,   ....]
word1: [0.2,  0.3, 0.4, ....]
word2: [0.1,  0.2, 0.3, ....]

word(1~2): [0.3, 0.5, 0.7, ....]

'''

def makeFeatureVec(review, model, num_features):
    
    featureVec = np.zeros((num_features,), dtype = "float32")
    
    # Unique word set
    word_index = set(model.wv.index2word)
    
    # For division we need to count the number of words
    nword = 0
    
    # Iterate words in a review and if the word is in the unique wordset, add the vector values for each word.
    for word in review:
        if word in word_index:
            nword += 1
            featureVec = np.add(featureVec, model[word])
    
    # Divide the sum of vector values by total number of word in a review.
    featureVec = np.divide(featureVec, nword)        
    
    return featureVec
''' 

While iterating over reviews, add the vector sums of each review from the function "makeFeatureVec" to 
the predefined vector whose size is the number of total reviews and the number of features in word2vec.
The working principle is basically same with "makeFeatureVec" but this is a review basis and 
makeFeatureVec is word basis (or each word's vector basis)


return matrix:

            'V1'    'V2'    'V3'     'V4'
review 1    0.1      0.2     0.1     0.5
review 2    0.5      0.4     0.05    0.05

'''

def getAvgFeatureVec(clean_reviews, model, num_features):
    
    # Keep track of the sequence of reviews, create the number "th" variable.
    review_th = 0
    
    # Row: number of total reviews, Column: number of vector spaces (num_features = 250 we set this in Word2Vec step).
    reviewFeatureVecs = np.zeros((len(clean_reviews), num_features), dtype = "float32")
    
    # Iterate over reviews and add the result of makeFeatureVec.
    for review in clean_reviews:
        reviewFeatureVecs[int(review_th)] = makeFeatureVec(review, model, num_features)
        
        # Once the vector values are added, increase the one for the review_th variable.
        review_th += 1
    
    return reviewFeatureVecs
clean_train_reviews = []

# Clean the reviews by preprocessing function with stopwords option "on".
for review in train["review"]:
    clean_train_reviews.append(preprocess_wordlist(review, stopwords = True))

# Apply "getAvgFeatureVec" function.
trainDataAvg = getAvgFeatureVec(clean_train_reviews, model, num_features)
    
    
# Same steps repeats as we did for train_set.    
clean_test_reviews = []

for review in test["review"]:
    clean_test_reviews.append(preprocess_wordlist(review, stopwords = True))

testDataAvg = getAvgFeatureVec(clean_test_reviews, model, num_features)

from sklearn.cluster import KMeans
import time

print(model.wv.syn0.shape)
num_clusters = model.wv.syn0.shape[0] // 5
start = time.time()

kmean = KMeans(n_clusters = num_clusters)
index = kmean.fit_predict(model.wv.syn0)

end = time.time()
print("Time taken for K-Means clustering: ", end - start, "seconds.")

index = list(index)
voca_list = model.wv.index2word

# dictionary format -  word : the cluster where the key word belongs.
voca_cluster = {voca_list[i]: index[i] for i in range(len(voca_list))}
# Check the first 10 clusters in voca_cluster we created.

# Loop from 0 to 9th cluster 
for cluster in range(10):
    
    word = []
    
    # Iterate over the number of total words. 
    for i in range(len(voca_cluster.values())):
        
        # If the cluster (0~10) corresponds to iterating i th voca_cluster values (cluster),
        if(list(voca_cluster.values())[i] == cluster):
            
            # Append the words.
            word.append(list(voca_cluster.keys())[i])
    
    print(word)

# Preprocess data for input as before

train_review = []

for review in train['review']:
    train_review.append(preprocess_wordlist(review, stopwords= True))

test_review = []

for review in test['review']:
    test_review.append(preprocess_wordlist(review, stopwords = True))
train_centroid = np.zeros((len(train['review']), num_clusters), dtype = 'float32')
test_centroid = np.zeros((len(test['review']), num_clusters), dtype = 'float32')
'''
The array that we are going to create looks like this:

cl1 cl2 cl3 cl4 ....
 3   10  5   30 ...

As usual we will be creating the empty array having the number of clusters dimension space.
While Iterating over words, if there is any word found in the voca_cluster, find the cluster where the word belongs to
and add one to the feature corresponding to the cluster.

( ex) if 'cat' assigned to cluster 10 then add one to 10th feature in the empty array. )  

'''

def create_boc(wordlist, voca_cluster):
    
    # The number of cluster == the maximum number of values in voca_cluster
    boc = np.zeros(max(voca_cluster.values()) + 1, dtype='float32')
    
    # Iterate over words and increase by one to the cluster if any word in the voca_cluster we created
    for word in wordlist:
        if word in voca_cluster:
            index = voca_cluster[word]
            boc[index] += 1
            
    return boc
    
# Transform the training and test set reviews into bags of centroid.

count = 0

for review in train_review:
    train_centroid[count] = create_boc(review, voca_cluster)
    count += 1
    
count = 0

for review in test_review:
    test_centroid[count] = create_boc(review, voca_cluster)
    count += 1
    
print("Train Dimension (avg):",trainDataAvg.shape,",", "Train Dimension (centroid):",train_centroid.shape)

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
kfold = StratifiedKFold(n_splits=5, random_state = 2018)

# LinearSVC

sv = LinearSVC(random_state=2018)

param_grid1 = {
    'loss':['squared_hinge'],
    'class_weight':[{1:2}],
    'C': [20],
    'penalty':['l2']
}

gs_sv = GridSearchCV(sv, param_grid = [param_grid1], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc' )
gs_sv.fit(trainDataAvg, train['sentiment'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)

# {'C': 20, 'class_weight': {1: 2}, 'loss': 'squared_hinge', 'penalty': 'l2'} - 86.30
y_submission1 = gs_sv.predict(testDataAvg)
print(gs_sv.best_score_)

bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha': [0.002],
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc')
gs_bnb.fit(trainDataAvg, train['sentiment'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)

# {'alpha': 0.002, 'binarize': 0.001} - 68.348
y_submission2 = gs_bnb.predict(testDataAvg)
print(gs_bnb.best_score_)

MLP = MLPClassifier(random_state = 2018)

mlp_param_grid = {
    'hidden_layer_sizes':[(10,10)],
    'activation':['tanh'],
    'solver':['adam'],
    'alpha':[0.01],
    'learning_rate':['constant'],
    'max_iter':[1000]
}

gsMLP = GridSearchCV(MLP, param_grid = mlp_param_grid, cv = kfold, scoring = 'roc_auc', n_jobs= 1, verbose = 1)
gsMLP.fit(trainDataAvg,train['sentiment'])
print(gsMLP.best_params_)
mlp_best0 = gsMLP.best_estimator_

# {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (1,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 87.012
# {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (2,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 86.960
# {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (5,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 87.020
# {'activation': 'tanh', 'alpha': 0.009, 'hidden_layer_sizes': (5,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 87.004
# {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (10, 10), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 87.108
y_submission3 = gsMLP.predict(testDataAvg)
print(gsMLP.best_score_)

lr = LogisticRegression(random_state = 2018)


lr_param2 = {
    'penalty':['l1'],
    'dual':[False],
    'C':[40],
    'class_weight':['balanced'],
    'solver':['saga']
    
}

lr_CV = GridSearchCV(lr, param_grid = [lr_param2], cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_CV.fit(trainDataAvg,train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_


# {'C': 100, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l1', 'solver': 'saga'} - 87.376
# {'C': 50, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l1', 'solver': 'saga'} - 87.380
# {'C': 40, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l1', 'solver': 'saga'} - 87.424
y_submission4 = lr_CV.predict(testDataAvg)
print(lr_CV.best_score_)

# LinearSVC

sv = LinearSVC(random_state=2018)

param_grid1 = {
    'loss':['squared_hinge'],
    'class_weight':['balanced'],
    'C': [0.001],
    'penalty':['l2']
}

gs_sv = GridSearchCV(sv, param_grid = [param_grid1], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc' )
gs_sv.fit(train_centroid, train['sentiment'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)

# {'C': 0.001, 'class_weight': 'balanced', 'loss': 'squared_hinge', 'penalty': 'l2'} - 87.256 
y_submission11 = gs_sv.predict(test_centroid)
print(gs_sv.best_score_)

bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha': [0.01],
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc')
gs_bnb.fit(train_centroid, train['sentiment'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)

# {'alpha': 0.01, 'binarize': 0.001} - 81.87
y_submission22 = gs_bnb.predict(test_centroid)
print(gs_bnb.best_score_)

MLP = MLPClassifier(random_state = 2018)

mlp_param_grid = {
    'hidden_layer_sizes':[(5,)],
    'activation':['tanh'],
    'solver':['sgd'],
    'alpha':[1],
    'learning_rate':['adaptive'],
    'max_iter':[1000]
}


gsMLP = GridSearchCV(MLP, param_grid = mlp_param_grid, cv = kfold, scoring = 'roc_auc', n_jobs= 1, verbose = 1)
gsMLP.fit(train_centroid, train['sentiment'])
print(gsMLP.best_params_)
mlp_best0 = gsMLP.best_estimator_

# {'activation': 'tanh', 'alpha': 1, 'hidden_layer_sizes': (1,), 'learning_rate': 'adaptive', 'max_iter': 1000, 'solver': 'sgd'} - 87.092
# {'activation': 'tanh', 'alpha': 1, 'hidden_layer_sizes': (5,), 'learning_rate': 'adaptive', 'max_iter': 1000, 'solver': 'sgd'} - 87.068
# {'activation': 'tanh', 'alpha': 1, 'hidden_layer_sizes': (5,), 'learning_rate': 'adaptive', 'max_iter': 1000, 'solver': 'sgd'} - 87.068
y_submission33 = gsMLP.predict(test_centroid)
print(gsMLP.best_score_)

lr = LogisticRegression(random_state = 2018)

lr_param2 = {
    'penalty':['l1'],
    'dual':[False],
    'class_weight':[{1:2}],
    'C': [100],
    'solver':['saga']
    
}

lr_CV = GridSearchCV(lr, param_grid = [lr_param2], cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_CV.fit(train_centroid,train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_


# {'C': 100, 'class_weight': {1: 2}, 'dual': False, 'penalty': 'l1', 'solver': 'saga'} - 85.968
y_submission44 = lr_CV.predict(test_centroid)
print(lr_CV.best_score_)

result = pd.DataFrame(data = {'id':test['id'], 'sentiment': y_submission4})
result.to_csv('submission_29.csv', index = False, quoting = 3)

