import re
import string
import jieba
import gensim
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
print(os.listdir("../input/spam identification/dict"))

def get_stop_words():
    stopword_list = []
    with open("../input/spam identification/dict/stop_words.utf8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            stopword_list.append(line)
    return stopword_list
def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens
def remove_stopwords(text):
    tokens = tokenize_text(text)
    stopword_list = get_stop_words()
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text
remove_stopwords("今天的天气很好，有点紧张")
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None,[re.sub(pattern=pattern,repl="",string=token) for token in tokens])
    filtered_text = ''.join(filtered_tokens)
    return filtered_text
remove_special_characters("今天的天气很好，有点紧张#()10")
def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
    return normalized_corpus
def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def get_data():
    '''
    获取数据
    :return: 文本数据，对应的labels
    '''
    with open("../input/spam identification/data/ham_data.txt", encoding="utf8") as ham_f, open("../input/spam identification/data/spam_data.txt", encoding="utf8") as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()
        
        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()
        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels
def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    '''
    :param corpus: 文本数据
    :param labels: label数据
    :param test_data_proportion:测试数据占比 
    :return: 训练数据,测试数据，训练label,测试label
    '''
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,
                                                        test_size=test_data_proportion, random_state=42)
    return train_X, test_X, train_Y, test_Y
def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels
def get_metrics(true_labels, predicted_labels):
    print('准确率:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        2))
    print('精度:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2))
    print('召回率:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2))
    print('F1得分:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2))
def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions
def main():
    corpus, labels = get_data()  # 获取数据集

    print("总的数据量:", len(labels))

    corpus, labels = remove_empty_docs(corpus, labels)

    print('样本之一:', corpus[10])
    print('样本的label:', labels[10])
    label_name_map = ["垃圾邮件", "正常邮件"]
    print('实际类型:', label_name_map[int(labels[10])], label_name_map[int(labels[5900])])
    
    #对数据进行划分
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                            labels,
                                                                            test_data_proportion=0.3)
    #进行归一化
    norm_train_corpus = normalize_corpus(train_corpus)
    norm_test_corpus = normalize_corpus(test_corpus)
    
    # 词袋模型特征
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    # tfidf 特征
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    # tokenize documents
    tokenized_train = [jieba.lcut(text)
                       for text in norm_train_corpus]
    #print(tokenized_train[2:10])
    tokenized_test = [jieba.lcut(text)
                      for text in norm_test_corpus]
    
    # build word2vec 模型
    model = gensim.models.Word2Vec(tokenized_train,
                                   size=500,
                                   window=100,
                                   min_count=30,
                                   sample=1e-3)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    mnb = MultinomialNB()
    svm = SGDClassifier(loss='hinge', n_iter=100)
    lr = LogisticRegression()

    print("\n")
    # 基于词袋模型的多项朴素贝叶斯
    print("基于词袋模型特征的贝叶斯分类器")
    mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)

    # 基于词袋模型特征的逻辑回归
    print("基于词袋模型特征的逻辑回归")
    lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                      train_features=bow_train_features,
                                                      train_labels=train_labels,
                                                      test_features=bow_test_features,
                                                      test_labels=test_labels)

    # 基于词袋模型的支持向量机方法
    print("基于词袋模型的支持向量机")
    svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)


    # 基于tfidf的多项式朴素贝叶斯模型
    print("基于tfidf的贝叶斯模型")
    mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)
    # 基于tfidf的逻辑回归模型
    print("基于tfidf的逻辑回归模型")
    lr_tfidf_predictions=train_predict_evaluate_model(classifier=lr,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)


    # 基于tfidf的支持向量机模型
    print("基于tfidf的支持向量机模型")
    svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)
    
    
    num = 0
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        print('邮件类型:', label_name_map[int(label)])
        print('预测的邮件类型:', label_name_map[int(predicted_label)])
        print('文本:-')
        print(re.sub('\n', ' ', document))
        print("\n")
        num+=1
        
        if num == 10:
            break
            
main()




