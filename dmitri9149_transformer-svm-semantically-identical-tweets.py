import time

start_time = time.time()



from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from tqdm import tqdm

import numpy as np

import pandas as pd

import re
!pip install tensorflow-text==2.0.0 --user
import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_text as textb
#print full tweet , not a part

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 310)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

train_copy = train.copy()

length_train = len(train.index)

length_train
index_hot = [4415, 4400, 4399,4403,4397,4396, 4394,4414, 4393,4392,4404,4407,4420,4412,4408,4391,4405]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [6840,6834,6837,6841,6816]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [6828,6831]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [591, 587]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [601,576,584,608,606,603,592,604]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [3667,3674,3688,3696]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [3913,3914,3936,3921,3941,3937,3938]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [3136, 3133]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [3930,3933,3924,3917]

train_example = train.copy()

train_example.loc[index_hot,:]
index_hot = [246,270,266,259,253,251,250,271]

train_example = train.copy()

train_example.loc[index_hot,:]
def new_line(text):

    text = re.sub(r'\t', ' ', text) # remove tabs

    text = re.sub(r'\n', ' ', text) # remove line jump

    return text
def url(text):

# quite many tweets are truncated like "Experts in France 

# begin examining airplane debris found on Reunion Island: French air 

# accident experts o... http://t.co/YVVPznZmXg #news" , the explanation is above

    text = re.sub(r' \w{1,3}\.{3,3} http\S{0,}', ' ', text)

    text = re.sub(r' \w{1,3}Û_ http\S{0,}', ' ', text)

# some symbols and words one space before 'http' are eliminated, it is assumed the words have no a 

# semantical meaning and predictive power in the position. 

    text = re.sub(r"mp3 http\S{0,}", r" ", text)

    text = re.sub(r"rar http\S{0,}", r" ", text)

    pattern = re.compile(r'( pin\:\d+ | via )http\S{0,}')

    text = pattern.sub(r' ', text)

# the pattern in tweet context have no a big meaning and the elimination of the words 

# unify the strings structure 

    pattern = re.compile(r'Full read by|Full read b|Full read|Full rea|Full re|Full r')

    text = pattern.sub(r' ', text)

    pattern = re.compile(r'Full story at|Full story a|Full story|Full stor|Full sto|Full st|Full s')

    text = pattern.sub(r' ', text)

    

    return text
def clean(text):    

    text = new_line(text)

# eliminate the pattern

    text = re.sub(r'(&amp;|&gt;|&lt;)', " ", text)

    text = re.sub(r"\s+", " ", text) # remove extra spaces

    text = url(text)

    

# the pattern is 'translated as 'USER'

# in https://www.kaggle.com/quentinsarrazin/tweets-preprocessing similar 'translation' is used

# in https://arxiv.org/ftp/arxiv/papers/1807/1807.07752.pdf similar pattern 

# is 'translated as 'USER_NAME'

    text = re.sub(r'@\S{0,}', ' USER ', text)

    text = re.sub(r"\s+", " ", text) # remove extra spaces  

# shrink multiple USER USER USER ... to USER

    text = re.sub(r'\b(USER)( \1\b)+', r'\1', text)

    

# multiple  letters repeats like in 'Oooooohhh' are truncated to 2 letters, not possible to truncate 

# to 1 letter, because it may generated false meaning like  'good' to 'god'

    text = re.sub(r'([a-zA-Z])\1{1,}', r'\1\1', text)

    

#  URLs , if not yet eliminated by url function are eliminated 

    text = re.sub(r"htt\S{0,}", " ", text)

    

# remove all characters if not in the list [a-zA-Z\d\s]

    text = re.sub(r"[^a-zA-Z\d\s]", " ", text)

    

# the digit(s) pattern is 'translated' to 'NUMBER'

# in https://www.kaggle.com/quentinsarrazin/tweets-preprocessing similar 'translation' is used

    text = re.sub(r'^\d\S{0,}| \d\S{0,}| \d\S{0,}$', ' NUMBER ', text)

    text = re.sub(r"\s+", " ", text) # remove extra spaces 

# shrink multiple NUMBER NUMBER  ... to NUMBER

    text = re.sub(r'\b(NUMBER)( \1\b)+', r'\1', text)

    

# remove digits if not eliminated above in 'NUMBER translation'

    text = re.sub(r"[0-9]", " ", text)

    

    text = text.strip() # remove spaces at the beginning and at the end of string    

# to reveal more equivalence classes the ' via USER' at the end of string is eliminated

    text = re.sub(r' via\s{1,}USER$', ' ', text)

    

    text = re.sub(r"\s+", " ", text) # remove extra spaces

    text = text.strip() # remove spaces at the beginning and at the end of string

    

    return text
train['text'][5450:5550] # train text before cleaning
train.text = train.text.apply(clean)

test.text = test.text.apply(clean)
max_length_tr = train.text.map(len).max()

max_length_te = test.text.map(len).max()

max_length = max(max_length_tr, max_length_te)



print("At the stage of text processing:")

print(f"...the size of longest text string in train set is  {max_length_tr}")

print(f"...the size of longest text string in test set is  {max_length_te}")

# the new max possible length will be (max_length - delta) , strings longer than new_max will be 

# decreased to new_max 

def cut(max_len, delta, x):

    new_max = max_len - delta

    length = len(x)

    if length <= new_max:

        return x 

    else:

        return x[:(new_max-length)]

    



delta = 25 

train.text = train.text.map(lambda x: cut(max_length, delta, x))

test.text = test.text.map(lambda x: cut(max_length, delta, x))



new_max_length_tr = train.text.map(len).max()

new_max_length_te = test.text.map(len).max()



print("After we cut tails of the longest tweets:")

print(f"...the size of longest text string in train set is  {new_max_length_tr}")

print(f"...the size of longest text string in test set is  {new_max_length_te}")
train['text'][5450:5550] # the 'text' after cleaning 
# the code in the cell is taken from 

# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert

df_mislabeled = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']

index_misl = df_mislabeled.index.tolist()



lenght = len(index_misl)



print(f"There are {lenght} equivalence classes with mislabelling")
index_misl # the list of strings (after cleaning/preprocessing) which represent the 85  classes
train_nu_target = train[train['text'].isin(index_misl)].sort_values(by = 'text')

#train_nu_target.head(60)

train_nu_target[0:309]
num_records = train_nu_target.shape[0]

length = len(index_misl)

print(f"There are {num_records} records in train set which are split in {lenght} equivalence classes (with mislabelling)") 
copy = train_nu_target.copy()

classes = copy.groupby('text').agg({'keyword':np.size, 'target':np.mean}).rename(columns={'keyword':'Number of records in train set', 'target':'Target mean'})



classes.sort_values('Number of records in train set', ascending=False).head(100)
majority_df = train_nu_target.groupby(['text'])['target'].mean()

len(majority_df.index)
def relabel(r, majority_index):

    ind = ''

    if r['text'] in majority_index:

        ind = r['text']

#        print(ind)

        if majority_df[ind] < 0.5:

            return 0

        else:

            return 1

    else: 

        return r['target']
train['target'] = train.apply( lambda row: relabel(row, majority_df.index), axis = 1)
new_df = train[train['text'].isin(majority_df.index)].sort_values(['target', 'text'], ascending = [False, True])

new_df.head(310)
# the code in the cell is taken from 

# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert

df_mislabeled = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']

index_misl = df_mislabeled.index.tolist()

#index_dupl[0:50]

len(index_misl)
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
X_train = []

for r in tqdm(train.text.values):

  emb = use(r)

  review_emb = tf.reshape(emb, [-1]).numpy()

  X_train.append(review_emb)



X_train = np.array(X_train)

y_train = train.target.values



X_test = []

for r in tqdm(test.text.values):

  emb = use(r)

  review_emb = tf.reshape(emb, [-1]).numpy()

  X_test.append(review_emb)



X_test = np.array(X_test)
train_arrays, test_arrays, train_labels, test_labels = train_test_split(X_train,

                                                                        y_train,

                                                                        random_state =42,

                                                                        test_size=0.20)
def svc_param_selection(X, y, nfolds):



#    Cs = [1.35, 1.40, 1.45]

#    gammas = [2.15, 2.20, 2.25, 2.30]    best params: {'C': 1.4, 'gamma': 2.25}

    

    Cs = [1.40]

    gammas = [2.25] 



    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=8)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search



model = svc_param_selection(train_arrays,train_labels, 10)
model.best_params_
pred = model.predict(test_arrays)
cm = confusion_matrix(test_labels,pred)

cm
accuracy = accuracy_score(test_labels,pred)

accuracy
test_pred = model.predict(X_test)

submission['target'] = test_pred.round().astype(int)

#submission.to_csv('submission.csv', index=False)
train_df_copy = train

train_df_copy = train_df_copy.fillna('None')

ag = train_df_copy.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'Count', 'target':'Disaster Probability'})



ag.sort_values('Disaster Probability', ascending=False).head(20)
count = 2

prob_disaster = 0.9

keyword_list_disaster = list(ag[(ag['Count']>count) & (ag['Disaster Probability']>=prob_disaster)].index)

#we print the list of keywords which will be used for prediction correction 

keyword_list_disaster
ids_disaster = test['id'][test.keyword.isin(keyword_list_disaster)].values

submission['target'][submission['id'].isin(ids_disaster)] = 1
submission.to_csv("submission.csv", index=False)

submission.head(10)
index_hot = [2700, 2695, 2713, 2698, 2692, 2686, 2685,2684]

train_example = train_copy.copy()

train_example.loc[index_hot,:]
index_hot = [6842,6821,6824,6828,6831,6843]

train_example = train_copy.copy()

train_example.loc[index_hot,:]
index_hot = [6113,6103,6097,6094,6091,6119,6123]

train_example = train_copy.copy()

train_example.loc[index_hot,:]
index_hot = [3670,3674,3688,3667,3696]

            

train_example = train_copy.copy()

train_example.loc[index_hot,:]
print("--- %s seconds ---" % (time.time() - start_time))