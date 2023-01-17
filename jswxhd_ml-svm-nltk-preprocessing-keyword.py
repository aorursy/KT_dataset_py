import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

import re
##preprocess the texts using nltk

def text_preprocess(text):

    text = str(text)

    token_texts = nltk.word_tokenize(text,language = 'english')

    st = SnowballStemmer('english', ignore_stopwords = False)

    english_stopwords = stopwords.words("english")

    token_stem_stopword=[]

	

    ##remove punctuation and stopwords do the stemming

    for each in token_texts:

        if each not in english_stopwords and each.isalpha():

	        token_stem_stopword.append(st.stem(each))

		    

    new_texts = ' '.join(token_stem_stopword)

    

    ##remove url in the texts

    new_texts = re.sub(r'https?:\/\/t.co\/[A-Za-z0-9]+','',new_texts)

    #new_texts = re.sub(r'[a-z]*[:.]+\S+','',new_texts)

    

    return new_texts



#a=text_preprocess('forest fire http://t.co/Y8IcF89H6w http://t.co/t9MSnZV1Kb forest fire')

#print(a)





##preprocess the keyword using nltk

def text_preprocess2(text):

    

    if text is not np.nan:

        text = str(text)

        token_texts = text.split('%20')

        st = SnowballStemmer('english', ignore_stopwords = False)

        #st = PorterStemmer()

        english_stopwords = stopwords.words("english")

        token_stem_stopword=[]

        

        #remove punctuation and stopwords do the stemming

        for each in token_texts:

            if each not in english_stopwords:

                token_stem_stopword.append(st.stem(each))

        

        ##bomber will not be stemmed using SnowballStemmer so that add it manually

        token_stem_stopword_2 = ['bomb' if x=='bomber' else x for x in token_stem_stopword]

        new_texts = ' '.join(token_stem_stopword_2)



        return new_texts   

    

    else:

        return np.nan





#a=text_preprocess2('suicide%20bomber')

#print(a)

def fetch_data():

    raw_data = pd.read_csv('../input/nlp-getting-started/train.csv')

    raw_data = pd.DataFrame(raw_data, columns =['keyword','text','target'])    

    

    raw_data['keyword'] = raw_data['keyword'].apply(lambda x: text_preprocess2(x))

    raw_data['text'] = raw_data['text'].apply(lambda x: text_preprocess(x))

    keyword_list = list(set(list(raw_data['keyword'])))

    #print(keyword_list)

    

    ##Some texts without keyword actually contain keyword so that 

    ##I extract the keyword from these texts according to existed keyword list

    row, column = np.shape(raw_data)

    for i in range(row):

        for j in range(1,len(keyword_list)):

            if raw_data.loc[i,'text'].find(keyword_list[j]) != -1 and raw_data.loc[i,'keyword'] is np.nan:

                raw_data.loc[i,'keyword'] = keyword_list[j]                  

    #print(raw_data.isnull().sum())

    

    raw_data = raw_data.dropna(axis=0)

    

    ##Assign weight based on the average target of each keyword

    keyword_target = raw_data['target'].groupby(raw_data['keyword']).mean()

    keyword_weight = keyword_target.to_dict()

    #print(len(keyword_weight))

    raw_data['keyword'] = raw_data['keyword'].apply(lambda x: keyword_weight[x])

    

    

    features = pd.DataFrame(raw_data, columns =['keyword','text'])

    labels = pd.DataFrame(raw_data, columns =['target'])

    

    ##Also add words from texts in test data to the corpus

    predict_data = pd.read_csv('../input/nlp-getting-started/test.csv')

    predict_data = pd.DataFrame(predict_data, columns =['id','keyword','text'])

    predict_data['text'] = predict_data['text'].apply(lambda x: text_preprocess(x))

    corpus = list(features['text'])+list(predict_data['text'])

    

    

    ##Generate TF-IDF term matrix and use LSA to reduce dimension

    vectorizer = TfidfVectorizer()   

    vector = vectorizer.fit_transform(corpus)

    

    ##Recommending reserved number of feature is 100

    svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=10, random_state=1)

    result_vector = svd_model.fit_transform(vector) 

    #print(np.shape(result_vector))

    term_matrix = np.array(result_vector[:7597,:])

    keyword_matrix = np.array(features['keyword']).reshape(-1, 1)

    

    ##Concatenate the term matrix and keyword weight

    new_features = np.concatenate((term_matrix, keyword_matrix), axis=1)

    labels = np.array(labels)

    #print(np.shape(new_features))

    term_matrix_predict = np.array(result_vector[7597:,:])

    train_x, test_x, train_y, test_y = train_test_split(new_features,labels,test_size=0.2, random_state=1)

    #print(np.shape(train_x))

    return train_x, test_x, train_y, test_y, term_matrix_predict, predict_data, keyword_weight, new_features, labels



#fetch_data()
##trian on partial dataset

def SVM_classifier():

    train_x, test_x, train_y, test_y = fetch_data()[:4]

    train_y = np.array(train_y).ravel()

    test_y = np.array(test_y).ravel()

    #print(train_y)

    my_SVM = SVC(C =10, kernel='rbf',gamma ='scale', random_state=10)

    my_SVM = my_SVM.fit(train_x, train_y)

    result = my_SVM.score(test_x, test_y)

    print(result)

    score_train = my_SVM.score(train_x, train_y)

    print(score_train)



    return my_SVM 





SVM_classifier()





#train on whole dataset

def SVM_classifier_2():

    train_x, train_y = fetch_data()[7:9]

    train_y = np.array(train_y).ravel()

    #print(train_y)

    my_SVM = SVC(C =10, kernel='rbf',gamma ='scale', random_state=10)

    my_SVM = my_SVM.fit(train_x, train_y)

    score_train = my_SVM.score(train_x, train_y)

    print(score_train)



    return my_SVM
def SVM_predict_tweet():

    SVM = SVM_classifier_2()

    term_matrix_predict, predict_data, keyword_weight = fetch_data()[4:7]

    predict_data['keyword'] = predict_data['keyword'].apply(lambda x: text_preprocess2(x))

    

    ##Assign weight for each keyword according to keyword dict generated in fetch data

    row, column = np.shape(predict_data)

    for i in range(row):

        for key in keyword_weight.keys():

            if predict_data.loc[i,'text'].find(key) != -1 and predict_data.loc[i,'keyword'] is np.nan:

                predict_data.loc[i,'keyword'] = key

    #print(predict_data.isnull().sum())

    predict_data['keyword'] = predict_data['keyword'].apply(lambda x: keyword_weight[x] if x is not np.nan else x)

    predict_data = predict_data.fillna(int(0))

    #print(predict_data.isnull().sum())

    predict_keyword = np.array(predict_data['keyword']).reshape(-1, 1)

    predict_matrix = np.concatenate((term_matrix_predict, predict_keyword),axis =1)

    results = SVM.predict(predict_matrix)

    output = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

    output['target'] = results

    output.to_csv('submission.csv',index=False, header=True)





    

SVM_predict_tweet()