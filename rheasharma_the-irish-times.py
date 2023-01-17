import pandas as pd

import numpy as np



from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize         



from matplotlib import pyplot as plt



from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.ensemble import RandomForestClassifier as RFClassi



from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit

from sklearn.linear_model import SGDClassifier as SGDC





from sklearn.feature_extraction.text import TfidfVectorizer as TVec

from sklearn.feature_extraction.text import CountVectorizer as CVec

from sklearn.preprocessing import MinMaxScaler as mmScaler

from sklearn.decomposition import TruncatedSVD



from sklearn.metrics import classification_report as cr

from sklearn.metrics import accuracy_score



from keras.models import Sequential

from keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils
path = r'../input/ireland-historical-news/irishtimes-date-text.csv'

df = pd.read_csv(path)



category_counts = df.headline_category.value_counts()

print("No of classes are: ", len(category_counts))

print(category_counts)

selected_category_counts = category_counts[category_counts > 3000].index.tolist()

df_small = df.loc[df['headline_category'].isin(selected_category_counts)]
%matplotlib inline

f, ax = plt.subplots(figsize=(30,30))

category_counts = category_counts.sort_values(ascending=False)

plt.barh(category_counts.index, category_counts)

plt.show()

#print(category_counts, category_counts.index)
stratSplit = StratifiedShuffleSplit(n_splits=3, test_size=0.25)

tr_idx, te_idx = next(stratSplit.split(np.zeros(len(df_small)),df_small['headline_category']))
class LemmaTokenizer(object):

    def __init__(self):

        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):

        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
def getSplit(te_idx, tr_idx):

    vec = CVec(ngram_range=(1,3), stop_words='english', tokenizer=LemmaTokenizer())

    lsa = TruncatedSVD(20, algorithm='arpack')

    mmS = mmScaler(feature_range=(0,1))



    countVec = vec.fit_transform(df_small.iloc[tr_idx]['headline_text'])

    countVec = countVec.astype(float)

    #print(len(countVec))

    dtm_lsa = lsa.fit_transform(countVec)

    X_train = mmS.fit_transform(dtm_lsa)



    countVec = vec.transform(df_small.iloc[te_idx]['headline_text'])

    countVec = countVec.astype(float)

    dtm_lsa = lsa.transform(countVec)

    X_test  = mmS.transform(dtm_lsa)



    x_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    x_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 



    enc = LabelEncoder()

    enc.fit(df_small.iloc[:]['headline_category'].astype(str))

    y_train = enc.transform(df_small.iloc[tr_idx]['headline_category'].astype(str))

    y_test = enc.transform(df_small.iloc[te_idx]['headline_category'].astype(str))



    y_train_c = np_utils.to_categorical(y_train)

    y_test_c = np_utils.to_categorical(y_test)



    return (X_train, y_train, X_test, y_test)
rfc = RFClassi(n_estimators=20)

mNB = MultinomialNB(alpha=.5)

gNB = GaussianNB()

bNB = BernoulliNB(alpha=.2)



sgdC = SGDC(n_jobs=-1, max_iter=1000, eta0=0.001)

gsCV_sgdClassifier = GridSearchCV(sgdC, {'loss':['hinge', 'squared_hinge',  'modified_huber', 'perceptron'], 

                                         'class_weight':['balanced',None], 'shuffle':[True, False], 'learning_rate':

                                        ['optimal', 'adaptive']})



models = [rfc, mNB, gNB, bNB, gsCV_sgdClassifier]
for model in models:    

    print("For model: ", model)

    acc = 0.0

    for tr_idx, te_idx in stratSplit.split(np.zeros(len(df_small)),df_small['headline_category']):

        (X_train, y_train, X_test, y_test) = getSplit(tr_idx, te_idx)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc += accuracy_score(y_test, y_pred)

    print("Classification Report is:\n", cr(y_test, y_pred))    

    print("Accuracy is: ", acc/3.0, "\n------------------------------------------------------------------------------------\n")

    
print(gsCV_sgdClassifier.best_params_, gsCV_sgdClassifier.best_score_)