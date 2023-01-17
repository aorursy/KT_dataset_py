#Importing necessary libraries:

# data analysis and wrangling
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import math
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
pd.options.mode.chained_assignment = None
dataFrameOriginal = pd.read_csv('../input/youtube-trending-and-not-trending-videos/Final Data/completeUSdata.csv', index_col=0)
dataFrameOriginal
dataFrameOriginal.info()
dataFrameOriginal['tags'] = dataFrameOriginal['tags'].replace('[none]', "")
dataFrameOriginal['description'] = dataFrameOriginal['description'].fillna("")
dataFrameOriginal.info()
dataFrameOriginal["description"].str.len()
for i in range(len(dataFrameOriginal["description"])):
    description = str(dataFrameOriginal["description"][i])
    if(len(description)>200):
        stringList = []
        for j in range(200):
            stringList.append(description[j])
        description = "".join(stringList)
        stringList = description.split()
        del stringList[-1]
        description = " ".join(stringList)
        dataFrameOriginal["description"][i] = description
dataFrameOriginal["description"].str.len()
dataFrameOriginal['trending'] = dataFrameOriginal['trending'].replace(True, int(1))
dataFrameOriginal['trending'] = dataFrameOriginal['trending'].replace(False, int(0))
dataFrameOriginal[['trending']]
dateColumn = dataFrameOriginal["publishedDateTime"]
for i, date in enumerate(dateColumn):
    date = dateutil.parser.parse(date)
    date = date.strftime('%w')
    dateColumn[i] = date
dataFrameOriginal = dataFrameOriginal.rename(columns={'publishedDateTime': 'dayOfWeek'})
dataFrameOriginal[["dayOfWeek"]]
uniqueChannelsNames = pd.DataFrame({'UniqueChannelsNames' : dataFrameOriginal["channel_title"].unique()})
uniqueChannelsNames.to_csv('./uniqueChannelsNames.csv', index=False)
uniqueChannelsNames
temp_df2 = pd.DataFrame({'channel_title': dataFrameOriginal["channel_title"].unique(),
                         'channel_title_new':range(len(dataFrameOriginal["channel_title"].unique()))})
dataFrameOriginal = dataFrameOriginal.merge(temp_df2, on='channel_title', how='left')
dataFrameOriginal = dataFrameOriginal.drop(['channel_title'], axis=1)
dataFrameOriginal = dataFrameOriginal.rename(columns={'channel_title_new': 'channel_title'})
dataFrameOriginal[["channel_title"]]
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def wordList(words):
    words = words.lower()
    words = re.sub("[^a-zA-Z0-9]", " ", words)
    words = words.split()
    filtered_sentence = []
    for word in words: 
        if (len(word)!=1) and (word not in stop_words):
            word = porter.stem(word)
            filtered_sentence.append(word)
    filtered_sentence = np.unique(filtered_sentence)
    return filtered_sentence

for i in range (len(dataFrameOriginal)):
    video_title = str(dataFrameOriginal["video_title"][i])
    dataFrameOriginal["video_title"][i] =  wordList(video_title)

    tags = str(dataFrameOriginal["tags"][i])
    dataFrameOriginal["tags"][i] =  wordList(tags)

    description = str(dataFrameOriginal["description"][i])
    dataFrameOriginal["description"][i] =  wordList(description)
dataFrameOriginal[["video_title", "tags", "description"]]
def getResult(number):
    if(number==0):
        number = 1
    number = len(dataFrameOriginal)/number
    number = math.log10(number)
    return number

def getFrequencyDatasets(column):
    totalList = []
    for i in range(len(dataFrameOriginal)):
        tempList = list(dataFrameOriginal[column][i])
        totalList = totalList + tempList
    totalList = np.unique(totalList)

    frequencyDfTrue = pd.DataFrame(columns = ['word','freq'])
    frequencyDfFalse = pd.DataFrame(columns = ['word','freq'])

    for word in totalList:
        t = 0
        f = 0
        for i in range(len(dataFrameOriginal)):
            if (word in dataFrameOriginal[column][i]):
                if (dataFrameOriginal["trending"][i]==True):
                    t = t+1
                if (dataFrameOriginal["trending"][i]==False):
                    f = f+1
        resultTrue = getResult(t)
        frequencyDfTrue.loc[len(frequencyDfTrue)] = [word, resultTrue]

        resultFalse = getResult(f)
        frequencyDfFalse.loc[len(frequencyDfFalse)] = [word, resultFalse]

    minTrue, maxTrue = [frequencyDfTrue['freq'].min(), frequencyDfTrue['freq'].max()]
    minFalse, maxFalse = [frequencyDfFalse['freq'].min(), frequencyDfFalse['freq'].max()]
    for i in range(len(frequencyDfTrue)):
        frequencyDfTrue["freq"][i] = 1-((frequencyDfTrue["freq"][i] - minTrue)/(maxTrue-minTrue))
        frequencyDfFalse["freq"][i] = 1-((frequencyDfFalse["freq"][i] - minFalse)/(maxFalse-minFalse))
    return frequencyDfTrue, frequencyDfFalse

video_titleFrequencyDfTrue, video_titleFrequencyDfFalse = getFrequencyDatasets("video_title")
tagsFrequencyDfTrue, tagsFrequencyDfFalse = getFrequencyDatasets("tags")
descriptionFrequencyDfTrue, descriptionFrequencyDfFalse = getFrequencyDatasets("description")
print("DONE")
video_titleFrequencyDfTrue = video_titleFrequencyDfTrue.sort_values(by='freq', ascending=False).reset_index(drop=True)
video_titleFrequencyDfFalse = video_titleFrequencyDfFalse.sort_values(by='freq', ascending=False).reset_index(drop=True)
tagsFrequencyDfTrue = tagsFrequencyDfTrue.sort_values(by='freq', ascending=False).reset_index(drop=True)
tagsFrequencyDfFalse = tagsFrequencyDfFalse.sort_values(by='freq', ascending=False).reset_index(drop=True)
descriptionFrequencyDfTrue = descriptionFrequencyDfTrue.sort_values(by='freq', ascending=False).reset_index(drop=True)
descriptionFrequencyDfFalse = descriptionFrequencyDfFalse.sort_values(by='freq', ascending=False).reset_index(drop=True)

video_titleFrequencyDfTrue.to_csv("../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/video_titleFrequencyDfTrue.csv")
video_titleFrequencyDfFalse.to_csv("../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/video_titleFrequencyDfFalse.csv")
tagsFrequencyDfTrue.to_csv("../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/tagsFrequencyDfTrue.csv")
tagsFrequencyDfFalse.to_csv("../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/tagsFrequencyDfFalse.csv")
descriptionFrequencyDfTrue.to_csv("../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/descriptionFrequencyDfTrue.csv")
descriptionFrequencyDfFalse.to_csv("../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/descriptionFrequencyDfFalse.csv")
video_titleFrequencyDfTrue = pd.read_csv('../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/video_titleFrequencyDfTrue.csv', index_col=0)
video_titleFrequencyDfFalse = pd.read_csv('../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/video_titleFrequencyDfFalse.csv', index_col=0)
tagsFrequencyDfTrue = pd.read_csv('../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/tagsFrequencyDfTrue.csv', index_col=0)
tagsFrequencyDfFalse = pd.read_csv('../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/tagsFrequencyDfFalse.csv', index_col=0)
descriptionFrequencyDfTrue = pd.read_csv('../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/descriptionFrequencyDfTrue.csv', index_col=0)
descriptionFrequencyDfFalse = pd.read_csv('../input/youtube-trending-and-not-trending-videos/Final Data/frequencies/descriptionFrequencyDfFalse.csv', index_col=0)

video_titleFrequencyDfTrue["word"] = video_titleFrequencyDfTrue["word"].fillna("nulll")
video_titleFrequencyDfFalse["word"] = video_titleFrequencyDfFalse["word"].fillna("nulll")
tagsFrequencyDfTrue["word"] = tagsFrequencyDfTrue["word"].fillna("nulll")
tagsFrequencyDfFalse["word"] = tagsFrequencyDfFalse["word"].fillna("nulll")
descriptionFrequencyDfTrue["word"] = descriptionFrequencyDfTrue["word"].fillna("nulll")
descriptionFrequencyDfFalse["word"] = descriptionFrequencyDfFalse["word"].fillna("nulll")
video_titleFrequencyDfTrue
index = descriptionFrequencyDfTrue[descriptionFrequencyDfTrue['word']=='nulll'].index[0]
descriptionFrequencyDfTrue = descriptionFrequencyDfTrue.drop(descriptionFrequencyDfTrue.index[index])
descriptionFrequencyDfTrue = descriptionFrequencyDfTrue.reset_index(drop=True)
descriptionFrequencyDfTrue.info()
index = descriptionFrequencyDfFalse[descriptionFrequencyDfFalse['word']=='nulll'].index[0]
descriptionFrequencyDfFalse = descriptionFrequencyDfFalse.drop(descriptionFrequencyDfFalse.index[index])
descriptionFrequencyDfFalse = descriptionFrequencyDfFalse.reset_index(drop=True)
descriptionFrequencyDfFalse.info()
def getFrequencyList(column, frequencyDF):
    freqList = []
    for record in dataFrameOriginal[column]:
        numberOfWords = len(record)
        if(numberOfWords!=0):
            totalFreq = 0
            for word in record:
                freq = frequencyDF[frequencyDF['word']==word]['freq'].astype(float)
                totalFreq = totalFreq + freq
            meanFreq = totalFreq/numberOfWords
        else:
            meanFreq = 0
        freqList.append(meanFreq)
    return freqList

dataFrameOriginal["video_titleFreqTrue"] = getFrequencyList("video_title", video_titleFrequencyDfTrue)
dataFrameOriginal["video_titleFreqFalse"] = getFrequencyList("video_title", video_titleFrequencyDfFalse)
dataFrameOriginal["tagsFreqTrue"] = getFrequencyList("tags", tagsFrequencyDfTrue)
dataFrameOriginal["tagsFreqFalse"] = getFrequencyList("tags", tagsFrequencyDfFalse)
dataFrameOriginal["descriptionFreqTrue"] = getFrequencyList("description", descriptionFrequencyDfTrue)
dataFrameOriginal["descriptionFreqFalse"] = getFrequencyList("description", descriptionFrequencyDfFalse)

dataFrameOriginal = dataFrameOriginal.drop(['video_title'], axis=1)
dataFrameOriginal = dataFrameOriginal.drop(['tags'], axis=1)
dataFrameOriginal = dataFrameOriginal.drop(['description'], axis=1)

dataFrameOriginal.to_csv('../input/youtube-trending-and-not-trending-videos/Final Data/dataFrameOriginalPREPROCESSED.csv')
dataFrameOriginal = pd.read_csv('../input/youtube-trending-and-not-trending-videos/Final Data/dataFrameOriginalPREPROCESSED.csv', index_col=0)
dataFrameOriginal
X = np.array(dataFrameOriginal.loc[:, dataFrameOriginal.columns != 'trending'])
y = np.array(dataFrameOriginal.loc[:, dataFrameOriginal.columns == 'trending'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
datFrameSMOTE = pd.DataFrame({'dayOfWeek': X_train_res[:, 0], 'category_id': X_train_res[:, 1], 'channel_title': X_train_res[:, 2], 
                               'video_titleFreqTrue': X_train_res[:, 3], 'video_titleFreqFalse': X_train_res[:, 4], 
                               'tagsFreqTrue': X_train_res[:, 5], 'tagsFreqFalse': X_train_res[:, 6],
                               'descriptionFreqTrue': X_train_res[:, 7], 'descriptionFreqFalse': X_train_res[:, 8], 'trending': y_train_res})
datFrameSMOTE.to_csv('./datFrameSMOTE.csv', index=False)
datFrameSMOTE = pd.read_csv('./datFrameSMOTE.csv')
datFrameSMOTE
g = sns.FacetGrid(datFrameSMOTE, col='trending', height=2, aspect=3)
g = g.map(plt.hist, 'dayOfWeek', bins = 7 )
g = sns.FacetGrid(datFrameSMOTE, col='trending')
g = g.map(plt.hist, 'category_id', bins = 15)
g = sns.FacetGrid(datFrameSMOTE, col='trending', height=6, aspect=1.5)
g = g.map(plt.hist, 'channel_title', bins = 40 )
g = sns.FacetGrid(datFrameSMOTE, col='trending', height=2, aspect=3)
g.map(sns.kdeplot, 'video_titleFreqTrue')
g = sns.FacetGrid(datFrameSMOTE, col='trending', height=2, aspect=3)
g.map(sns.kdeplot, 'video_titleFreqFalse')
g = sns.FacetGrid(datFrameSMOTE, col='trending', height=2, aspect=3)
g.map(sns.kdeplot, 'tagsFreqTrue')
g = sns.FacetGrid(datFrameSMOTE, col='trending', height=2, aspect=3)
g.map(sns.kdeplot, 'tagsFreqFalse')
g = sns.FacetGrid(datFrameSMOTE, col='trending', height=2, aspect=3)
g.map(sns.kdeplot, 'descriptionFreqTrue')
g = sns.FacetGrid(datFrameSMOTE, col='trending', height=2, aspect=3)
g.map(sns.kdeplot, 'descriptionFreqFalse')
# Logistic Regression

logreg = LogisticRegression(max_iter = 100)
logreg.fit(X_train_res, y_train_res)
Y_pred0 = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log 
coeff_df = pd.DataFrame(dataFrameOriginal.columns.delete(2))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines
svc = SVC()
svc.fit(X_train_res, y_train_res)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train_res, y_train_res) * 100, 2)
acc_svc
submission = pd.DataFrame({
        "Trending": Y_pred
    })
submission.to_csv('submission.csv', index=False)
submission