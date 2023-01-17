import csv

import re

import math

from math import sqrt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import category_encoders as ce



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, train_test_split



from scipy.sparse import hstack, coo_matrix



import nltk

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk import pos_tag

df_train= pd.read_csv('../input/nlp-getting-started/train.csv')

df_test=pd.read_csv('../input/nlp-getting-started/test.csv')

df_train_all = df_train.append(df_test, sort=False)
# Preprocess for Neural Networks:

def preprocess_description_NN(line):

    stop_words = stopwords.words('english')

    pattern_url = 'http[^\s\n\t\r]+'

    pattern_user = '@[^\s\n\t\r]+'

    pattern_tag = '#([^\s\n\t\r]+)'

    

    if re.match(pattern_url, line):

        p = re.compile(pattern_url)

        l = p.sub(' URLS ', line)

        line = l

    '''   

    if re.match(pattern_user, line):

        p_user = re.compile(pattern_user)

        l_user = p_user.sub(' USERNAME ', line)

        line = l_user

        

    

    if re.match(pattern_tag, line):

        p_user = re.compile('#')

        l_user = p_user.sub(' tag_', line)

        line = l_user    

    '''        

    

    tokens_pos_all = nltk.pos_tag(word_tokenize(line))

    

    meaningful_pos_tokens = [x[0] for x in tokens_pos_all]

    meaningful_tokens = [x.lower() for x in meaningful_pos_tokens if x.isalpha()]

    

    result = " ".join(meaningful_tokens)

    return result



# Preprocess for other algorithms than neural networks:

def preprocess_description(line):

    stemmer = SnowballStemmer('english')

    stop_words = stopwords.words('english')

    

    pattern_url = 'http[^\s\n\t\r]+'

    pattern_user = '@[^\s\n\t\r]+'

    

    if re.match(pattern_url, line):

        p = re.compile(pattern_url)

        l = p.sub(' URLS ', line)

        line = l

    

    if re.match(pattern_user, line):

        p = re.compile(pattern_user)

        l_user = p.sub(' USERNAME ', line)

        line = l_user

    

    tokens_pos_all = nltk.pos_tag(word_tokenize(line))

    meaningful_pos_tokens = [x[0] for x in tokens_pos_all if re.match('(JJ|VB)', x[1]) or re.search('^(NN|NNS|NNP)$', x[1])]

    meaningful_tokens = [x.lower() for x in meaningful_pos_tokens if not x in stop_words and len(x)>3  and x.isalpha()]

    meaningful_stems = [stemmer.stem(x) for x in meaningful_tokens]

    result = " ".join(meaningful_stems)

    return result



def parse_train(df_train, df_train_all, df_test, algo_type, max_words, add_keyword, parse_pca, parse_ce):

    n_components = 0

    limit = 10000

    

    # Prepare the input training data, dataset and labels :

    df2 = df_train

    target = df2['target']

    Y = target.to_numpy()

    

    ids = df2['id']

    

    dfYid = pd.concat([ids, target], axis=1, sort=False)

    

    ar = np.reshape(dfYid.to_numpy(), (dfYid.shape[0], 2))

    

    # Check if the classes are unballanced and ballance if necessary:

    nb_docs_per_class = {cl:np.count_nonzero(ar[:,cl]) for cl in np.array([0,1])}

    nb_docs_per_class_zero = nb_docs_per_class[0] - nb_docs_per_class[1]

    nb_docs_per_class[0] = nb_docs_per_class_zero

    

    #if nb_docs_per_class[0] > nb_docs_per_class[1]:

    diff = nb_docs_per_class[0] - nb_docs_per_class[1]

    df_diff_1 = df_train[df_train['target']==1]

    df_diff = df_diff_1[df_diff_1['keyword']!=""].sample(n=diff)

    df_train = df_train.append(df_diff)

    

    # Category encoding

    parse_keyword = 1

    parse_kw_how = 1

    

    if add_keyword:

        kw = df_train['keyword']

        kw_all = df_train_all['keyword']

        

        if parse_kw_how == 0: # Target Encoder

            #cat_ce = TargetEncoder()

            pass

        elif parse_kw_how == 1: # Label Encoder

            kw = kw.fillna("0")

            kw_all = kw_all.fillna("0")

            le = LabelEncoder()

            le.fit(kw_all)

            kw_enc = le.transform(kw)

            kw_enc = np.reshape(kw_enc, (kw_enc.shape[0], 1))

        elif parse_kw_how == 2: # get_dummies    

            kw_enc = pd.get_dummies(kw).to_numpy()

        else: # OHE

            kw = kw.fillna("0")

            kw_all = kw_all.fillna("0")

            le = LabelEncoder()

            le.fit(kw_all)

            ohe = OneHotEncoder(handle_unknown='ignore')

            ohe.fit(kw_all)

            

            kw_lab = le.transform(kw)

            kw_lab = np.reshape(kw_lab, (kw_lab.shape[0], 1))

            kw_enc = ohe.transform(kw_lab)

            

        

    target = df_train['target']

    Y = target.to_numpy()

    

    descriptions = df_train['text']

    descriptions_all = df_train_all['text']

    

    if algo_type == 'nn':

        descriptions = descriptions.apply(lambda x : preprocess_description_NN(x)).to_numpy()

        descriptions_all = descriptions_all.apply(lambda x : preprocess_description_NN(x)).to_numpy()

    else:

        descriptions = descriptions.apply(lambda x : preprocess_description(x)).to_numpy()

        descriptions_all = descriptions_all.apply(lambda x : preprocess_description(x)).to_numpy()

    

    

    # Vectorize dataset with frequences and tfidf

    des = descriptions

    des_all = descriptions_all

    tf_vectorizer = CountVectorizer(max_df=0.95, max_features=max_words-1)

    tf_vectorizer1 = CountVectorizer(max_df=0.9, min_df=2, max_features=5932)

    tf_vectorizer.fit(des_all)

    

    X = tf_vectorizer.transform(des)

    nb_features = X.shape[1]

    #print(tf_vectorizer.vocabulary_)

    tf_transformer = TfidfTransformer().fit(X)

    X = tf_transformer.transform(X)

    

    if add_keyword:

        X = hstack((kw_enc, X))

    

    if parse_ce :

        cat_ce = ce.TargetEncoder()

        

        Xar = X.toarray()

        y_ce =  Xar[:,0]

        y1_ce = y_ce.reshape(y_ce.shape[0], 1)

        X = cat_ce.fit_transform(Xar, y1_ce)

        

        X = Xar

    elif parse_pca :

        pca = PCA(whiten=True)

        pca.fit(X.toarray())

        X = pca.transform(X.toarray())

    

    descriptions_test = df_test['text']

    ids_test = df_test['id']

    

    if algo_type == 'nn':

        descriptions_test = descriptions_test.apply(lambda x : preprocess_description_NN(x)).to_numpy()

    else:

        descriptions_test = descriptions_test.apply(lambda x : preprocess_description(x)).to_numpy()

    

    X_test = tf_vectorizer.transform(descriptions_test)

    X_test = tf_transformer.transform(X_test)

    

    if add_keyword:

        kw_test = df_test['keyword']

        if parse_kw_how == 1: # Label Encoder

            kw_test = kw_test.fillna("0")

            kw_enc_test = le.transform(kw_test)

            kw_enc_test = np.reshape(kw_enc_test, (kw_enc_test.shape[0], 1)) 

            

        elif parse_kw_how == 2: # get_dummies  

            kw_test = kw_test.fillna("0")

            kw_enc_test = pd.get_dummies(kw_test).to_numpy()

        else: # OHE

            kw_test = kw_test.fillna("0")

            kw_enc_lab = le.transform(kw_test)

            kw_enc_lab = np.reshape(kw_enc_lab, (kw_enc_lab.shape[0], 1)) 

            #ohe = OneHotEncoder(handle_unknown='ignore')

            kw_enc_test = ohe.transform(kw_enc_lab)  

            

        X_test = hstack((kw_enc_test, X_test))

    

    if parse_ce :

        cat_ce = ce.TargetEncoder()

        Xar = X_test.toarray()

        y_ce =  Xar[:,0]

        y1 = y_ce.reshape(y_ce.shape[0], 1)

        X_test = cat_ce.fit_transform(Xar, y1)

        

        X_test = Xar

    elif parse_pca:

        X_test = pca.transform(X_test.toarray())

    

    return (X, Y, X_test)





# Description; Ballance data for a pd.DataFrame, from scratch

# Input: df : pd.DataFrame without the Id column, containing the label column

# Output: X_res: pd.Dataframe for training ballanced containing the label column

def ballance_data_with_y(df):

    df_1 =  df[df["target"]==1]

    df_0 =  df[df["target"]==0]

    len1 = df_1.shape[0]

    len0 = df_0.shape[0]

    

    vmax = 0

    vmin = 1

    if len1 > len0:

        vmax = 1

        vmin = 0

        df_max = df_1

        df_min = df_0

    elif len1 < len0:

        vmax = 0

        vmin = 1

        df_max = df_0

        df_min = df_1

    else:

        return (df, Y)

    

    len_max = df_max.shape[0]

    len_min = df_min.shape[0]

    

    to_multiply = int(round(len_max/len_min))

    df_to_append = pd.concat([df_min] * to_multiply, ignore_index=True)

    

    len_append = df_to_append.shape[0]

    

    X_res = pd.concat([df_max, df_to_append], ignore_index=True)

    

    to_add = len_max - len_append

    if to_add > 0:

        df_to_add = df_min.sample(n=to_add, random_state=1)

        X_res = pd.concat([X_res, df_to_add], ignore_index=True)

    

    X_res = X_res.reset_index(drop=True)

    return X_res   



def print_into_file(y_pred, df_test, algo_name=None):

    l = []

    for myindex in range(y_pred.shape[0]):

        Y0 = y_pred[myindex]

        l.insert(myindex, Y0)

    

    df_result = df_test

    df_result = df_result.assign(target=pd.Series(l).values)

    df_result = df_result.drop(columns=['location'])

    df_result = df_result.drop(columns=['keyword'])

    df_result = df_result.drop(columns=['text'])

    

    f = open('submission.csv', 'w')

    r = df_result.to_csv(index=False, path_or_buf=f)

    f.close()
num_classes = 2

max_words = max_words = 701



add_keyword = True 

parse_pca = False

parse_ce = True



(X, Y, X_test) = parse_train(df_train, df_train_all, df_test, 'll', max_words, add_keyword, parse_pca, parse_ce)



x_train = X

y_train = Y

x_test = X_test



clf = RandomForestClassifier(max_depth=1000, n_estimators=5000, n_jobs=4, verbose=1)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)



print_into_file(y_pred, df_test)