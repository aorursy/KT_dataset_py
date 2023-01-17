!pip install nmslib
print("Loading imports")

from collections import Counter

import os

print(os.listdir("../input"))



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import nmslib

from sklearn.feature_extraction.text import TfidfVectorizer





np.random.seed(0)

print("Finished loading imports")



# Load all CSVs here

answers_df = pd.read_csv("../input/answers.csv", usecols=['answers_author_id', 'answers_question_id','answers_body'])

questions_df = pd.read_csv("../input/questions.csv", usecols=['questions_id','questions_title', 'questions_body'])

questions_df['questions_title_and_body'] = questions_df['questions_title'] + ' ' + questions_df['questions_body']

questions_df.drop(['questions_title', 'questions_body'], inplace=True, axis=1)



print('loaded all CSVs here')
# Create base features and ground truth DF

QA_df = questions_df.merge(answers_df, how="left", left_on='questions_id', right_on='answers_question_id')

QA_df.dropna(inplace=True)

QA_df.drop(['answers_question_id'],inplace=True,axis=1)



# Space out each because of later .sum step

QA_df["questions_title_and_body"] = QA_df["questions_title_and_body"] + ' '

QA_df.head(3)
# Split data on question_id via hash

QA_df['split'] = QA_df['questions_id'].apply(lambda x: hash(x) % 5)

QA_df['split'].value_counts()
train_df = QA_df[QA_df['split']!=0]

test_df = QA_df[QA_df['split']==0]



print(QA_df.shape)

print(train_df.shape)

print(test_df.shape)
# Make sure there's only one author_id per question

# This will be knn with 1 neighbor

train_df = train_df.groupby("answers_author_id")["questions_title_and_body"].sum().reset_index()

train_df.head(3)
# Instantiate the vectorizer

word_vectorizer = TfidfVectorizer(

    encoding='utf8',

    use_idf=True, # T much better than F

    stop_words=None, #None is beter than 'english',

    lowercase=True, #True better than False

    min_df=2, #2 better than 1 or 3

    max_df=0.99, #Doesn't seem to matter much better 0.8,1.0, 

    smooth_idf=False, #False better than True

    sublinear_tf=False,  # keep sublinear_df as False, the alternative is much worse

    norm='l1', #better than l2,None,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{3,}',  #vectorize 3-character words or more

    ngram_range=(1, 2), #(1,3) worse

    max_features=None) #None better than 90000)



# fit and transform on it the training features

word_vectorizer.fit(train_df['questions_title_and_body'])

train_df_sparse = word_vectorizer.transform(train_df['questions_title_and_body'])



#transform the test features to sparse matrix

test_df_sparse = word_vectorizer.transform(test_df['questions_title_and_body'])



print(test_df_sparse.shape)

print(train_df_sparse.shape)

print('TFIDF done')
#https://github.com/nmslib/nmslib/blob/master/python_bindings/notebooks/search_sparse_cosine.ipynb

KNN_VAL = 1 # Needs to be set at 1 given authors are aggregated into one. 

NUM_THREADS = 4

# Set index parameters

index_time_params = {'M': 30, 

                     'indexThreadQty': NUM_THREADS, 

                     'efConstruction': 100,

                     'post' : 0}



index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)  #521

index.addDataPointBatch(train_df_sparse) 

index.createIndex(index_time_params) #index_time_params)# {'post': 2}, print_progress=True) #'post': 0

neighbours = index.knnQueryBatch(test_df_sparse, k=KNN_VAL, num_threads=NUM_THREADS)

print('knn done')
test_preds = []

for val in neighbours:

    mode_val = train_df.loc[val[0][0],'answers_author_id']

    #row_list = []

    #for j in range(KNN_VAL):

    #    row_list.append(train_df.loc[val[0][j],'answers_author_id'])

        

    #row_list = Counter(row_list)

    #mode_val = row_list.most_common(1)[0][0]

    #mode_val = max(set(row_list), key=row_list.count) # slower

    test_preds.append(mode_val)

print('done')
test_df['preds'] = test_preds

test_df.head(3)
numer = np.sum(test_df['preds']==test_df['answers_author_id'])

print(numer)
print(numer/float(test_df.shape[0]))
print('Using the most common author_id in test')

max_count = test_df['answers_author_id'].value_counts().max()

print(max_count)