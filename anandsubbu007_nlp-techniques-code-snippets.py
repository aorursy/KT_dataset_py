#Remove HTML tags
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

#Remove extra whitespaces
def remove_extra_space(txt):
    import re
    return re.sub(' +',' ',txt)

#Language identification 
def language(txt):
    #pip install langdetect
    import langdetect
    lan = langdetect.detect(txt)
    if lan == "en":
        print("Uploaded Language is English")
        return (txt)
    else:
        return("Uploaded language is not in english")

#Remove non-ASCII characters from list of tokenized words   
def remove_non_ascii(words):
    import unicodedata
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
        rst = ''.join(new_words)
    return rst    
    
#Expand contractions
def expand_contraction(txt):
    #pip install contractions
    import contractions
    return contractions.fix(txt)

#Remove special characters
def remove_punctuation(txt):
    import string
    result = txt.translate(str.maketrans('','',string.punctuation))
    return result

#Lowercase all texts
def lower_text(txt):
    return txt.lower()

#Remove numbers
def remove_no(txt):
    import re
    return re.sub(r"\d+","",txt)

# convert number into words 
def convert_number(text): 
    #pip install inflect 
    # split string into list of words 
    temp_str = text.split() 
    # initialise empty list 
    new_string = [] 
    for word in temp_str: 
        # if word is a digit, convert the digit 
        # to numbers and append into the new_string list 
        if word.isdigit(): 
            temp = p.number_to_words(word) 
            new_string.append(temp) 
  
        # append the word as it is 
        else: 
            new_string.append(word) 
  
    # join the words of new_string to form a string 
    temp_str = ' '.join(new_string) 
    return temp_str 

#Text Normalization
def normalize(txt):
    words = remove_non_ascii(txt)
    words = lower_text(words)
    words = remove_punctuation(words)
    words = remove_no(words)
    return words
#Remove URL
def removeurl(txt):
    import re
    return re.sub(r'http?\S+|www\.\S+', '',txt)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#named entity recognition
def name_entity(text):
    import spacy
    sp = spacy.load('en_core_web_sm')
    sen = sp(text)
    txt = []
    entity = []
    for i in sen.ents:
            txt.append(i.text)
            entity.append(i.label_)
    import pandas as pd
    df = pd.DataFrame({"Text":txt,
                    "Entity":entity})
    return df

def name_entity_list(text_list):
    import spacy
    sp = spacy.load('en_core_web_sm')
    txt = []
    entity = []
    for i in text_list:
        sen = sp(i)
        for i in sen.ents:
            txt.append(i.text)
            entity.append(i.label_)
    import pandas as pd
    df = pd.DataFrame({"Text":txt,
                      "Entity":entity})
    return df

"/--------------------------------------Tokenization------------------------------------/"
#Tokenization
def token_words(txt):
    import nltk
    return nltk.word_tokenize(txt)

#Remove stop words
def Token_and_removestopword(txt):
    import nltk
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(txt)
    without_stop_words = []
    for word in words:
        if word not in stop_words:
            without_stop_words.append(word)
    return without_stop_words

#Lemmatization 
def lemmatize_word(tokens,pos="v"): 
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos =pos) for word in tokens] 
    return lemmas

#Stemming
def stem_words(tokens): 
    import nltk
    from nltk.stem.porter import PorterStemmer 
    stemmer = PorterStemmer()  
    stems = [stemmer.stem(word) for word in tokens] 
    return stems 

#Part of speech tagging 
def pos_tagging(word_tokens):
    import nltk
    from nltk import pos_tag 
    pos_tag(word_tokens)
    txt_gra = pos_tag(word_tokens)
    #chunking function 
    # Input from Pos_tagging
    text = []
    grammer = []
    for i in range(len(txt_gra)):
        txt = txt_gra[i][0]
        gram = txt_gra[i][1]
        text.append(txt)
        grammer.append(gram)
    return  text, grammer 

     
def correct_spellings(tokens):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    misspelled_words = spell.unknown(tokens)
    corrected_text = []
    for word in tokens:
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
#1-hot encoding model
def Onehotencoding(doc_list):
    from numpy import array
    from numpy import argmax
    import pandas as pd
    def toktotxt(txt_int):
        if isinstance(txt_int[0], list):
            text = txt_int.apply(lambda x :" ".join(str(i) for i in x))
        else:
            text = txt_int
        return text
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    values = array(toktotxt(doc_list))
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)    
    def unique(lst): 
        x = np.array(lst) 
        return np.unique(x) 
    col = unique(values)
    df = pd.DataFrame(onehot_encoded,index=values,columns=col)
    return onehot_encoded

#Bag-of-words model (Bow) [Countvectorizer] & N-grams language model
def countvectorizer(train_int,test_int=None,Ngram_min=1,Ngram_max=1):
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    def toktotxt(txt_int):
        if isinstance(txt_int[0], list):
            text = txt_int.apply(lambda x :" ".join(str(i) for i in x))
        else:
            text = txt_int
        return text
    train_txt = toktotxt(train_int)  
    vectorizer = CountVectorizer(ngram_range = (Ngram_min,Ngram_max))
    vectorizer.fit(train_txt)
    X = vectorizer.transform(train_txt)
    train = X.toarray()
    if test_int is None:
        out = train
    else:
        test_txt = toktotxt(test_int)
        Y = vectorizer.transform(test_txt)
        test = Y.toarray()
        out = train, test
    return out

#TF-IDF
def TFIDF(train_int,test_int=None,Ngram_min=1,Ngram_max=1):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    def toktotxt(txt_int):
        if isinstance(txt_int[0], list):
            text = txt_int.apply(lambda x :" ".join(str(i) for i in x))
        else:
            text = txt_int
        return text
    train_txt = toktotxt(train_int)  
    vectorizer = TfidfVectorizer(ngram_range = (Ngram_min,Ngram_max))
    vectorizer.fit(train_txt)
    X = vectorizer.transform(train_txt)
    train = X.toarray()
    if test_int is None:
        out = train
    else:
        test_txt = toktotxt(test_int)
        Y = vectorizer.transform(test_txt)
        test = Y.toarray()
        out = train, test
    return out

#Word2vec language model
def word2vector(txt_int,word,min_count=1):
    def texttotok(txt_int):
        import nltk
        if isinstance(txt_int[0], list):
            tokens = txt_int
        else:
            tokens = nltk.word_tokenize(txt_int)
        return tokens
    tokens = []
    for i in txt_int:
        tok = texttotok(i)
        tokens.append(tok)
    #print(tokens)
    from gensim.models import Word2Vec
    model = Word2Vec(tokens,min_count=min_count)
    index_lst = list(model.wv.vocab)
    ary = model[word]
    return ary

#Hash Vectorizer
def Hashvect(train_int,test_int=None,Ngram_min=1,Ngram_max=1):
    import pandas as pd
    from sklearn.feature_extraction.text import HashingVectorizer
    def toktotxt(txt_int):
        if isinstance(txt_int[0], list):
            text = txt_int.apply(lambda x :" ".join(str(i) for i in x))
        else:
            text = txt_int
        return text
    train_txt = toktotxt(train_int)  
    vectorizer = HashingVectorizer(ngram_range = (Ngram_min,Ngram_max))
    vectorizer.fit(train_txt)
    X = vectorizer.transform(train_txt)
    train = X.toarray()
    if test_int is None:
        out = train
    else:
        test_txt = toktotxt(test_int)
        Y = vectorizer.transform(test_txt)
        test = Y.toarray()
        out = train, test
    return out

#GloVe language model
#Component Analysis
#Principal Component Analysis (PCA)
def PCA(X_train,Y_train=None,X_test=None,n=10):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    X = pca.fit(X_train,y_train)
    train = pca.transform(X_train)
    if X_test is None:
        out = train
    else:
        test = pca.transform(X_test)
        out = train, test
    return out
#Independent Component Analysis (ICA)
def ICA(X_train,Y_train=None,X_test=None,n=100):
    from sklearn.decomposition import FastICA
    mod = FastICA(n_components=n)
    X = mod.fit(X_train,y_train)
    train = mod.transform(X_train)
    if X_test is None:
        out = train
    else:
        test = pca.transform(X_test)
        out = train, test
    return out
#Linear Discriminant Analysis (LDA)
def LDA(X_train,Y_train=None,X_test=None,n=100):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    LDA = LinearDiscriminantAnalysis(n_components=n)
    X = LDA.fit(X_train,y_train)
    test = LDA.transform(X_train)
    if X_test is None:
        out = train
    else:
        test = pca.transform(X_test)
        out = train, test
    return out

#Non-Negative Matrix Factorization (NMF)
def NMF(X_train,Y_train=None,X_test=None,n=100,init='random',random_state=0):
    from sklearn.decomposition import NMF
    mod = NMF(n_components=n, init=init, random_state=random_state)
    X = mod.fit(X_train,y_train)
    test = mod.transform(X_train)
    if X_test is None:
        out = train
    else:
        test = pca.transform(X_test)
        out = train, test
    return out

#Gaussian Random Projection
def gaurandpro(X_train,y_train=None,X_test=None):
    from sklearn import random_projection
    mod = random_projection.GaussianRandomProjection()
    X = mod.fit(X_train,y_train)
    test = mod.transform(X_train)
    if X_test is None:
        out = train
    else:
        test = pca.transform(X_test)
        out = train, test
    return out

#Sparse random projection
def sparandpro(X_train,y_train=None,X_test=None):
    from sklearn import random_projection
    mod = random_projection.SparseRandomProjection()
    X = mod.fit(X_train,y_train)
    test = mod.transform(X_train)
    if X_test is None:
        out = train
    else:
        test = pca.transform(X_test)
        out = train, test
    return out

#Johnson Lindenstrauss Lemma
def JLL(X_train,y_train=None,X_test=None,n=100,init='random'):
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    mod = NMF(n_components=n, init=init, random_state=0)
    X = mod.fit(X_train,y_train)
    test = mod.transform(X_train)
    if X_test is None:
        out = train
    else:
        test = pca.transform(X_test)
        out = train, test
    return out

#T- distributed Stochastic Neighbor Embedding (t-SNE)
def TSNE(X_train,y_train=None,X_test=None,n=100):
    from sklearn.manifold import TSNE
    mod = TSNE(n_components=n)
    X = mod.fit(X_train,y_train)
    test = mod.transform(X_train)
    if X_test is None:
        out = train
    else:
        test = pca.transform(X_test)
        out = train, test
    return out
#Rocchio Classification
def rocchio_clas(X_train, y_train,X_test):
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    model = NearestCentroid()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds
#Boosting
def boosting(X_train, y_train,X_test,n=20):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds
#Bagging
def Bagging(X_train, y_train,X_test,n=20):
    from sklearn.ensemble import BaggingClassifier
    model = BaggingClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds
#Logistic Regression
def logreg(X_train, y_train,X_test,n=20):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds
#Naive Bayes Text Classification
#MultinomialNB
def naivebiase_multinomial(X_train, y_train,X_test,n=20):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

#Gaussian
def naivebiase_gaussian(X_train, y_train,X_test,n=20):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

#Bernoullies
def naivebiase_gaussian(X_train, y_train,X_test,n=20):
    from sklearn.naive_bayes import BernoulliNB
    model = BernoulliNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

#K-Nearest Neighbors Algorithm (KNN)
def KNeighbors(X_train, y_train,X_test,n=2):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds
#Support Vector Machine (SVM)
def SVM(X_train, y_train,X_test):
    from sklearn.svm import LinearSVC
    model = LinearSVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds
#Decision Tree
def Decisiontree(X_train, y_train,X_test):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds
#Random Forests
def Randomforest(X_train, y_train,X_test):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

#Different input formate for crf
#Conditional Random Field (CRF)
def crf(X_train, y_train,X_test,alg='lbfgs'):
    import sklearn_crfsuite
    model = sklearn_crfsuite.CRF(algorithm=alg)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

#Deep Neural Networks (DNN)
#Recurrent Neural Network (RNN)
#Convolutional Neural Networks (CNN)
#Deep Belief Network (DBN)
#Hierarchical Attention Networks (HAN)
#Matthew correlation coefficient (MCC)
def MCC(y_true, y_pred):
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y_true, y_pred)

#Recall
def recall(y_true, y_pred):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred)

#Precision
def precision(y_true, y_pred):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred)

#Accuracy
def accuracy(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

#F Score
def f_score(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred)

#Receiver Operating Characteristics (ROC)
def ROC(y_true, y_pred):
    from sklearn.metrics import roc_curve
    return roc_curve(y_true, y_pred)
    
#Area Under ROC Curve (AUC)
def AUC(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)