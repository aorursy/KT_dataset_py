import pandas as pd

raw = pd.read_csv('../input/language-identification-datasst/dataset.csv')

raw
# Languages

languages = set(raw['language'])

print('Languages', languages)

print('========')

# Examples of multiple langs taken from heads and tails

print('Swedish & English:', raw['Text'][1])

print('Thai & English:', raw['Text'][2])

print('Chinese & English:', raw['Text'][21998])
import numpy as np

from sklearn.model_selection import train_test_split



X=raw['Text']

y=raw['language']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print(len(X_train))

print(len(X_test))

print(len(y_train))

print(len(y_test))
# Extract Unigrams

from sklearn.feature_extraction.text import CountVectorizer

unigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))

X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)

X_unigram_test_raw = unigramVectorizer.transform(X_test)





unigramFeatures = unigramVectorizer.get_feature_names()



print('Number of unigrams in training set:', len(unigramFeatures))
# Aggregate Unigrams per language

def train_lang_dict(X_raw_counts, y_train):

    lang_dict = {}

    for i in range(len(y_train)):

        lang = y_train[i]

        v = np.array(X_raw_counts[i])

        if not lang in lang_dict:

            lang_dict[lang] = v

        else:

            lang_dict[lang] += v

            

    # to relative

    for lang in lang_dict:

        v = lang_dict[lang]

        lang_dict[lang] = v / np.sum(v)

        

    return lang_dict



language_dict_unigram = train_lang_dict(X_unigram_train_raw.toarray(), y_train.values)



# Collect relevant chars per language

def getRelevantCharsPerLanguage(features, language_dict, significance=1e-5):

    relevantCharsPerLanguage = {}

    for lang in languages:

        chars = []

        relevantCharsPerLanguage[lang] = chars

        v = language_dict[lang]

        for i in range(len(v)):

            if v[i] > significance:

                chars.append(features[i])

    return relevantCharsPerLanguage



relevantCharsPerLanguage = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram)

    

# Print number of unigrams per language

for lang in languages:    

    print(lang, len(relevantCharsPerLanguage[lang]))
# get most common chars for a few European languages

europeanLanguages = ['Portugese', 'Spanish', 'Latin', 'English', 'Dutch', 'Swedish']

relevantChars_OnePercent = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram, 1e-2)



# collect and sort chars

europeanCharacters = []

for lang in europeanLanguages:

    europeanCharacters += relevantChars_OnePercent[lang]

europeanCharacters = list(set(europeanCharacters))

europeanCharacters.sort()



# build data

indices = [unigramFeatures.index(f) for f in europeanCharacters]

data = []

for lang in europeanLanguages:

    data.append(language_dict_unigram[lang][indices])



#build dataframe

df = pd.DataFrame(np.array(data).T, columns=europeanLanguages, index=europeanCharacters)

df.index.name = 'Characters'

df.columns.name = 'Languages'



# plot heatmap

import seaborn as sn

import matplotlib.pyplot as plt

sn.set(font_scale=0.8) # for label size

sn.set(rc={'figure.figsize':(10, 10)})

sn.heatmap(df, cmap="Greens", annot=True, annot_kws={"size": 12}, fmt='.0%')# font size

plt.show()
# number of bigrams

from sklearn.feature_extraction.text import CountVectorizer

bigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))

X_bigram_raw = bigramVectorizer.fit_transform(X_train)

bigramFeatures = bigramVectorizer.get_feature_names()

print('Number of bigrams', len(bigramFeatures))
# top bigrams (>1%) for Spanish, Italian (Latin), English, Dutch, Chinese, Japanese, Korean

language_dict_bigram = train_lang_dict(X_bigram_raw.toarray(), y_train.values)

relevantCharsPerLanguage = getRelevantCharsPerLanguage(bigramFeatures, language_dict_bigram, significance=1e-2)

print('Spanish', relevantCharsPerLanguage['Spanish'])

print('Italian (Latin)', relevantCharsPerLanguage['Latin'])

print('English', relevantCharsPerLanguage['English'])

print('Dutch', relevantCharsPerLanguage['Dutch'])

print('Chinese', relevantCharsPerLanguage['Chinese'])

print('Japanese', relevantCharsPerLanguage['Japanese'])
# Uni- & Bi-Gram Mixture CountVectorizer for top 1% features

from sklearn.feature_extraction.text import CountVectorizer



top1PrecentMixtureVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2), min_df=1e-2)

X_top1Percent_train_raw = top1PrecentMixtureVectorizer.fit_transform(X_train)

X_top1Percent_test_raw = top1PrecentMixtureVectorizer.transform(X_test)



language_dict_top1Percent = train_lang_dict(X_top1Percent_train_raw.toarray(), y_train.values)



top1PercentFeatures = top1PrecentMixtureVectorizer.get_feature_names()

print('Length of features', len(top1PercentFeatures))

print('')



#Unique features per language

relevantChars_Top1Percent = getRelevantCharsPerLanguage(top1PercentFeatures, language_dict_top1Percent, 1e-5)

for lang in relevantChars_Top1Percent:

    print("{}: {}".format(lang, len(relevantChars_Top1Percent[lang])))
def getRelevantGramsPerLanguage(features, language_dict, top=50):

    relevantGramsPerLanguage = {}

    for lang in languages:

        chars = []

        relevantGramsPerLanguage[lang] = chars

        v = language_dict[lang]

        sortIndex = (-v).argsort()[:top]

        for i in range(len(sortIndex)):

            chars.append(features[sortIndex[i]])

    return relevantGramsPerLanguage



top50PerLanguage_dict = getRelevantGramsPerLanguage(top1PercentFeatures, language_dict_top1Percent)



# top50

allTop50 = []

for lang in top50PerLanguage_dict:

    allTop50 += set(top50PerLanguage_dict[lang])



top50 = list(set(allTop50))

    

print('All items:', len(allTop50))

print('Unique items:', len(top50))
# getRelevantColumnIndices

def getRelevantColumnIndices(allFeatures, selectedFeatures):

    relevantColumns = []

    for feature in selectedFeatures:

        relevantColumns = np.append(relevantColumns, np.where(allFeatures==feature))

    return relevantColumns.astype(int)



relevantColumnIndices = getRelevantColumnIndices(np.array(top1PercentFeatures), top50)





X_top50_train_raw = np.array(X_top1Percent_train_raw.toarray()[:,relevantColumnIndices])

X_top50_test_raw = X_top1Percent_test_raw.toarray()[:,relevantColumnIndices] 



print('train shape', X_top50_train_raw.shape)

print('test shape', X_top50_test_raw.shape)
# Define some functions for our purpose



from sklearn.preprocessing import normalize

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix, f1_score

import seaborn as sn

import matplotlib.pyplot as plt

import scipy



# Utils for conversion of different sources into numpy array

def toNumpyArray(data):

    data_type = type(data)

    if data_type == np.ndarray:

        return data

    elif data_type == list:

        return np.array(data_type)

    elif data_type == scipy.sparse.csr.csr_matrix:

        return data.toarray()

    print(data_type)

    return None





def normalizeData(train, test):

    train_result = normalize(train, norm='l2', axis=1, copy=True, return_norm=False)

    test_result = normalize(test, norm='l2', axis=1, copy=True, return_norm=False)

    return train_result, test_result



def applyNaiveBayes(X_train, y_train, X_test):

    trainArray = toNumpyArray(X_train)

    testArray = toNumpyArray(X_test)

    

    clf = MultinomialNB()

    clf.fit(trainArray, y_train)

    y_predict = clf.predict(testArray)

    return y_predict



def plot_F_Scores(y_test, y_predict):

    f1_micro = f1_score(y_test, y_predict, average='micro')

    f1_macro = f1_score(y_test, y_predict, average='macro')

    f1_weighted = f1_score(y_test, y_predict, average='weighted')

    print("F1: {} (micro), {} (macro), {} (weighted)".format(f1_micro, f1_macro, f1_weighted))



def plot_Confusion_Matrix(y_test, y_predict, color="Blues"):

    allLabels = list(set(list(y_test) + list(y_predict)))

    allLabels.sort()

    confusionMatrix = confusion_matrix(y_test, y_predict, labels=allLabels)

    unqiueLabel = np.unique(allLabels)

    df_cm = pd.DataFrame(confusionMatrix, columns=unqiueLabel, index=unqiueLabel)

    df_cm.index.name = 'Actual'

    df_cm.columns.name = 'Predicted'



    sn.set(font_scale=0.8) # for label size

    sn.set(rc={'figure.figsize':(15, 15)})

    sn.heatmap(df_cm, cmap=color, annot=True, annot_kws={"size": 12}, fmt='g')# font size

    plt.show()



# Unigrams

X_unigram_train, X_unigram_test = normalizeData(X_unigram_train_raw, X_unigram_test_raw)

y_predict_nb_unigram = applyNaiveBayes(X_unigram_train, y_train, X_unigram_test)

plot_F_Scores(y_test, y_predict_nb_unigram)

plot_Confusion_Matrix(y_test, y_predict_nb_unigram, "Oranges")
# Top 1%

X_top1Percent_train, X_top1Percent_test = normalizeData(X_top1Percent_train_raw, X_top1Percent_test_raw)

y_predict_nb_top1Percent = applyNaiveBayes(X_top1Percent_train, y_train, X_top1Percent_test)

plot_F_Scores(y_test, y_predict_nb_top1Percent)

plot_Confusion_Matrix(y_test, y_predict_nb_top1Percent, "Reds")
# Top 50

X_top50_train, X_top50_test = normalizeData(X_top50_train_raw, X_top50_test_raw)

y_predict_nb_top50 = applyNaiveBayes(X_top50_train, y_train, X_top50_test)

plot_F_Scores(y_test, y_predict_nb_top50)

plot_Confusion_Matrix(y_test, y_predict_nb_top50, "Greens")
from sklearn.neighbors import KNeighborsClassifier



def applyNearestNeighbour(X_train, y_train, X_test):

    trainArray = toNumpyArray(X_train)

    testArray = toNumpyArray(X_test)

    

    clf = KNeighborsClassifier()

    clf.fit(trainArray, y_train)

    y_predict = clf.predict(testArray)

    return y_predict





## Unigrams

#y_predict_knn_unigram = applyNearestNeighbour(X_unigram_train, y_train, X_unigram_test)

#plot_F_Scores(y_test, y_predict_knn_unigram)

#plot_Confusion_Matrix(y_test, y_predict_knn_unigram, "Purples")



# Top 50

y_predict_knn_top50 = applyNearestNeighbour(X_top50_train, y_train, X_top50_test)

plot_F_Scores(y_test, y_predict_knn_top50)

plot_Confusion_Matrix(y_test, y_predict_knn_top50, "Blues")
# Train lang dict

def toRelative(X_test):

    return [v / np.sum(v) for v in X_test]



language_dict_top50 = train_lang_dict(X_top50_train, y_train.values)

X_top50_test_rel = toRelative(X_top50_test)
def ols_predict(language_dict, X_test):

    def calcSquareDifference(p, q):

        return np.sum((p-q)**2)

    

    def ols(language_dict, v, langs):

        olsVector = np.array([calcSquareDifference([language_dict[l]], v) for l in langs])

        index = np.argmin(olsVector)

        return langs[index]

    

    langs = [l for l in language_dict]

    return [ols(language_dict, v, langs) for v in X_test]





ols_predictions = ols_predict(language_dict_top50, X_top50_test_rel)



plot_F_Scores(y_test, ols_predictions)

plot_Confusion_Matrix(y_test, ols_predictions, 'cool')
# Kullback Leibler Divergence

from math import log





def kl_predict(language_dict, X_test):



    def kl_divergence(p, q):

        p_ = np.array(p) + 1e-200

        q_ = np.array(q) + 1e-200

        n = len(p)

        return np.sum([p_[i] * log(p_[i]/ (q_[i] )) for i in range(n)])

    

    def predict(language_dict, v, langs):

        divs = [kl_divergence(language_dict[l], v) for l in langs]

        index = np.argmin(divs)

        return langs[index]

    

    langs = [l for l in language_dict]

    return [predict(language_dict, v, langs) for v in X_test]





kl_predictions = kl_predict(language_dict_top50, X_top50_test_rel)



plot_F_Scores(y_test, kl_predictions)

plot_Confusion_Matrix(y_test, kl_predictions, 'plasma')
# Calculate average number of chars per Text

avgChars = np.mean(raw.apply(lambda x : len(x['Text']), axis=1))

print('avgChars', avgChars)
def ks_predict(language_dict, X_test, n, c_alpha=1.628):

    def calcMaxAbsDifference(p, q):

        return np.max(np.abs((p-q)))

    

    def scaleAlpha(n, m, c_alpha):

        factor = ((n + m) / n / m) ** 0.5

        return factor * c_alpha

    

    def ks(language_dict, v, langs):

        ksVector = np.array([calcMaxAbsDifference([language_dict[l]], v) for l in langs])

        index = np.argmin(ksVector)

        m = len(v)

        scaledAlpha = scaleAlpha(n, m, c_alpha)

        if ksVector[index] <= scaledAlpha:

            return langs[index]

        else:

            return '_N/A_'

    

    langs = [l for l in language_dict]

    return [ks(language_dict, v, langs) for v in X_test]





ks_predictions = ks_predict(language_dict_top50, X_top50_test_rel, 356*800)

print('None-Predicitons:', (np.array(ks_predictions)=='_N/A_').sum())

plot_F_Scores(y_test, ks_predictions)

plot_Confusion_Matrix(y_test, ks_predictions, 'summer')
! pip install langdetect

! pip install iso-639



from langdetect import detect

from iso639 import languages



def proofLanguage(text):

    twoLetterCode = detect(text)[:2] #consolidate Chinese 

    if twoLetterCode == 'it': # Italian -> Latin

        return 'Latin'

    elif twoLetterCode == 'pt': # Portuguese -> Portugese

        return 'Portugese'

    else:

        lang = languages.get(alpha2=twoLetterCode)

        langName = lang.name

        if not langName==None:

            return langName

        return twoLetterCode



y_test_proof = [proofLanguage(t) for t in X_test]
plot_F_Scores(y_test, y_test_proof)

plot_Confusion_Matrix(y_test, y_test_proof, 'cool')
def plotTopErrors(y_predict, top=5):

    ys = y_test.values

    Xs = X_test.values

    errorCount = 0

    

    for i in range(len(ys)):

        if not ys[i]==y_predict[i]:

            errorCount += 1

            print("#{}: Expected: {}, Predicted: {}".format(errorCount, ys[i], y_predict[i]))

            print("Text:", Xs[i])

            print("=================================================")

        if errorCount >= top:

            break
plotTopErrors(y_predict_nb_top50, top=5)
plotTopErrors(y_predict_nb_unigram, top=5)
plotTopErrors(ks_predictions, top=5)