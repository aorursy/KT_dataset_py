# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing

import json

from pandas.io.json import json_normalize

import seaborn as sns # Beautiful plots



## Kfold for cross-validation

from sklearn.model_selection import KFold, StratifiedKFold



## Classifier of XGBosst

from   xgboost import XGBClassifier

##import xgboost as xgb



## Package used for fine tuning

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials



# Any results you write to the current directory are saved as output.

import os

print(os.listdir("../input/"))



import nltk

import string

from gensim.models import word2vec

from tqdm import tqdm

from keras.preprocessing.text import text_to_word_sequence

from nltk.corpus import stopwords
trainframe = pd.read_csv('../input/train/train.csv')

testframe = pd.read_csv('../input/test/test.csv')



train_id = trainframe['PetID']

test_id = testframe['PetID']



trainframe.AdoptionSpeed.value_counts()
## Feature engineering for simpler features



## Create a list of bins for the state variable

bins = trainframe["State"].value_counts().index.tolist()

bins.sort()

bins

trainframe['State_Binned'] = pd.cut(trainframe['State'], bins)

labels = list(range(0,len(bins)-1))

labels

trainframe['State_Binned'] = pd.cut(trainframe['State'], bins=bins, labels=labels)



## Clipping the amount of photos tail using a cut-off value of 5

maxVal = 5

trainframe['PhotoAmt_Clipped'] = trainframe['PhotoAmt'].where(trainframe['PhotoAmt'] <= maxVal, maxVal)



## Clipping the amount of videos to no video - 0 or one or more videos - 1

maxVal = 1

trainframe['VideoAmt_Clipped'] = trainframe['VideoAmt'].where(trainframe['VideoAmt'] <= maxVal, maxVal)



## Clipping the quantity of pets with a cut-off of 5

maxVal = 5

trainframe['Quantity_Clipped'] = trainframe['Quantity'].where(trainframe['Quantity'] <= maxVal, maxVal)



## Normalizing breed labels and fee column into a -1 to 1 distribution

trainframe["Breed1_Normalized"] = (trainframe["Breed1"] - trainframe["Breed1"].mean()) / (trainframe["Breed1"].max() - trainframe["Breed1"].min())

trainframe["Breed2_Normalized"] = (trainframe["Breed2"] - trainframe["Breed2"].mean()) / (trainframe["Breed2"].max() - trainframe["Breed2"].min())

trainframe["Fee_Normalized"] = (trainframe["Fee"] - trainframe["Fee"].mean()) / (trainframe["Fee"].max() - trainframe["Fee"].min())



## Transform the age feature into years to avoid large values

trainframe["Age_Years"] = (trainframe["Age"] / 12).round(1)



## Set PetID as Index



trainframe = trainframe.set_index("PetID")



trainframe.head(5)
## Feature engineering for simpler features

## For testing



## Create a list of bins for the state variable

bins = testframe["State"].value_counts().index.tolist()

bins.sort()

bins

testframe['State_Binned'] = pd.cut(testframe['State'], bins)

labels = list(range(0,len(bins)-1))

labels

testframe['State_Binned'] = pd.cut(testframe['State'], bins=bins, labels=labels)



## Clipping the amount of photos tail using a cut-off value of 5

maxVal = 5

testframe['PhotoAmt_Clipped'] = testframe['PhotoAmt'].where(testframe['PhotoAmt'] <= maxVal, maxVal)



## Clipping the amount of videos to no video - 0 or one or more videos - 1

maxVal = 1

testframe['VideoAmt_Clipped'] = testframe['VideoAmt'].where(testframe['VideoAmt'] <= maxVal, maxVal)



## Clipping the quantity of pets with a cut-off of 5

maxVal = 5

testframe['Quantity_Clipped'] = testframe['Quantity'].where(testframe['Quantity'] <= maxVal, maxVal)



## Normalizing breed labels and fee column into a -1 to 1 distribution

testframe["Breed1_Normalized"] = (testframe["Breed1"] - testframe["Breed1"].mean()) / (testframe["Breed1"].max() - testframe["Breed1"].min())

testframe["Breed2_Normalized"] = (testframe["Breed2"] - testframe["Breed2"].mean()) / (testframe["Breed2"].max() - testframe["Breed2"].min())

testframe["Fee_Normalized"] = (testframe["Fee"] - testframe["Fee"].mean()) / (testframe["Fee"].max() - testframe["Fee"].min())



## Transform the age feature into years to avoid large values

testframe["Age_Years"] = (testframe["Age"] / 12).round(1)



## Set PetID as Index



testframe = testframe.set_index("PetID")



testframe.head(5)
## Add image data - courtesy of Peter Hurford's Kernel found at

# https://www.kaggle.com/peterhurford/pets-lightgbm-baseline-with-all-the-data





vertex_xs = []

vertex_ys = []

bounding_confidences = []

bounding_importance_fracs = []

dominant_blues = []

dominant_greens = []

dominant_reds = []

dominant_pixel_fracs = []

dominant_scores = []

label_descriptions = []

label_scores = []

nf_count = 0

nl_count = 0

for pet in train_id:

    try:

        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:

            data = json.load(f)

        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

        vertex_xs.append(vertex_x)

        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

        vertex_ys.append(vertex_y)

        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

        bounding_confidences.append(bounding_confidence)

        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

        bounding_importance_fracs.append(bounding_importance_frac)

        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

        dominant_blues.append(dominant_blue)

        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

        dominant_greens.append(dominant_green)

        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

        dominant_reds.append(dominant_red)

        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

        dominant_pixel_fracs.append(dominant_pixel_frac)

        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

        dominant_scores.append(dominant_score)

        if data.get('labelAnnotations'):

            label_description = data['labelAnnotations'][0]['description']

            label_descriptions.append(label_description)

            label_score = data['labelAnnotations'][0]['score']

            label_scores.append(label_score)

        else:

            nl_count += 1

            label_descriptions.append('nothing')

            label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)



print(nf_count)

print(nl_count)

trainframe.loc[:, 'vertex_x'] = vertex_xs

trainframe.loc[:, 'vertex_y'] = vertex_ys

trainframe.loc[:, 'bounding_confidence'] = bounding_confidences

trainframe.loc[:, 'bounding_importance'] = bounding_importance_fracs

trainframe.loc[:, 'dominant_blue'] = dominant_blues

trainframe.loc[:, 'dominant_green'] = dominant_greens

trainframe.loc[:, 'dominant_red'] = dominant_reds

trainframe.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

trainframe.loc[:, 'dominant_score'] = dominant_scores

trainframe.loc[:, 'label_description'] = label_descriptions

trainframe.loc[:, 'label_score'] = label_scores





vertex_xs = []

vertex_ys = []

bounding_confidences = []

bounding_importance_fracs = []

dominant_blues = []

dominant_greens = []

dominant_reds = []

dominant_pixel_fracs = []

dominant_scores = []

label_descriptions = []

label_scores = []

nf_count = 0

nl_count = 0

for pet in test_id:

    try:

        with open('../input/test_metadata/' + pet + '-1.json', 'r') as f:

            data = json.load(f)

        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

        vertex_xs.append(vertex_x)

        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

        vertex_ys.append(vertex_y)

        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

        bounding_confidences.append(bounding_confidence)

        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

        bounding_importance_fracs.append(bounding_importance_frac)

        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

        dominant_blues.append(dominant_blue)

        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

        dominant_greens.append(dominant_green)

        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

        dominant_reds.append(dominant_red)

        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

        dominant_pixel_fracs.append(dominant_pixel_frac)

        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

        dominant_scores.append(dominant_score)

        if data.get('labelAnnotations'):

            label_description = data['labelAnnotations'][0]['description']

            label_descriptions.append(label_description)

            label_score = data['labelAnnotations'][0]['score']

            label_scores.append(label_score)

        else:

            nl_count += 1

            label_descriptions.append('nothing')

            label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)



print(nf_count)

testframe.loc[:, 'vertex_x'] = vertex_xs

testframe.loc[:, 'vertex_y'] = vertex_ys

testframe.loc[:, 'bounding_confidence'] = bounding_confidences

testframe.loc[:, 'bounding_importance'] = bounding_importance_fracs

testframe.loc[:, 'dominant_blue'] = dominant_blues

testframe.loc[:, 'dominant_green'] = dominant_greens

testframe.loc[:, 'dominant_red'] = dominant_reds

testframe.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

testframe.loc[:, 'dominant_score'] = dominant_scores

testframe.loc[:, 'label_description'] = label_descriptions

testframe.loc[:, 'label_score'] = label_scores



trainframe.columns
## Create a dataframe with total sentiment scores for all available descriptions; total score for each description = total description score * total description magnitude



trainlist = []

for i in os.listdir("../input/train_sentiment/"):

    with open('../input/train_sentiment/' + str(i)) as f:

        data = json.load(f)

        score = data.get('documentSentiment').get('magnitude') * data.get('documentSentiment').get('score')

        trainlist.append((str(i).replace('.json',''), score))



sentiment_scores = pd.DataFrame(trainlist, columns=['PetID','Total_Sentiment_Score']).set_index('PetID')



## Add sentiment scores into the dataframe

trainframe = sentiment_scores.join(trainframe, how='outer')

## Replace missing sentiment scores with neutral 0s

trainframe.fillna(0, inplace=True)
## Create a dataframe with total sentiment scores for all available descriptions; total score for each description = total description score * total description magnitude

## For testing



testlist = []

for i in os.listdir("../input/test_sentiment/"):

    with open('../input/test_sentiment/' + str(i)) as f:

        data = json.load(f)

        score = data.get('documentSentiment').get('magnitude') * data.get('documentSentiment').get('score')

        testlist.append((str(i).replace('.json',''), score))



sentiment_scores = pd.DataFrame(testlist, columns=['PetID','Total_Sentiment_Score']).set_index('PetID')



## Add sentiment scores into the dataframe

testframe = sentiment_scores.join(testframe, how='outer')

## Replace missing sentiment scores with neutral 0s

testframe.fillna(0, inplace=True)



testframe.head(5)
## Normalize numeric features into a single normalized frame



## Map numeric features + engineered features

trainframe_all = trainframe[['Total_Sentiment_Score','Type','Age_Years','Breed1_Normalized','Breed2_Normalized',

                             'Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health',

                             'Quantity_Clipped','Fee','Fee_Normalized','PhotoAmt_Clipped','AdoptionSpeed',

                             'vertex_x','vertex_y','bounding_confidence', 'bounding_importance',

                             'dominant_blue', 'dominant_green', 'dominant_red','dominant_pixel_frac',

                             'dominant_score','label_score', 'Breed1', 'Breed2']]



## Map labels into a separate dataset

train_labels = pd.DataFrame(trainframe, columns=['AdoptionSpeed'])



## Reset index for training features to prepare for normalization and re-indexation

trainframe_all = trainframe_all.reset_index()



## Create new normalized frames for normalization

normalized_frame_all = pd.DataFrame()

normalized_frame_all = trainframe_all





##Run normalization on both training features excluding non-numeric IDs



##Add column names

#columnNames = list(trainframe_all.head(0))



#for i in columnNames:

   # if i != 'PetID' and i != 'AdoptionSpeed':

      #  normalized_frame_all[i] = preprocessing.scale(trainframe_all[i].astype('float64'))



## Add IDs and re-index both normalized feature sets to prepare for label merging



#normalized_frame_all['PetID'] = trainframe_all['PetID']

#normalized_frame_all = normalized_frame_all.set_index('PetID')



#normalized_frame_all.head(5)
## Engineer feature crosses for the train set



normalized_frame_all['Breed1 x MaturitySize'] = normalized_frame_all['Breed1'] *  normalized_frame_all['MaturitySize']

normalized_frame_all['Breed1 x Gender'] = normalized_frame_all['Breed1'] *  normalized_frame_all['Gender']

normalized_frame_all['MaturitySize x Gender'] = normalized_frame_all['MaturitySize'] *  normalized_frame_all['Gender']

normalized_frame_all['Type x Gender'] = normalized_frame_all['Type'] *  normalized_frame_all['Gender']

normalized_frame_all['Type x MaturitySize'] = normalized_frame_all['Type'] *  normalized_frame_all['MaturitySize']

normalized_frame_all['Type x FurLength'] = normalized_frame_all['Type'] *  normalized_frame_all['FurLength']

normalized_frame_all['Gender x FurLength'] = normalized_frame_all['Gender'] *  normalized_frame_all['FurLength']

normalized_frame_all['Type x Health'] = normalized_frame_all['Type'] *  normalized_frame_all['Health']

normalized_frame_all['Fee x Health'] = normalized_frame_all['Fee'] *  normalized_frame_all['Health']

normalized_frame_all['Vaccinated x Dewormed x Sterilized'] = normalized_frame_all['Vaccinated'] *  normalized_frame_all['Dewormed'] * normalized_frame_all['Sterilized']

normalized_frame_all['Type x Color1'] = normalized_frame_all['Type'] *  normalized_frame_all['Color1']

normalized_frame_all['Breed1 x Color1'] = normalized_frame_all['Breed1'] *  normalized_frame_all['Color1']

normalized_frame_all['Type x Age_Years'] = normalized_frame_all['Type'] *  normalized_frame_all['Age_Years']

normalized_frame_all['Health x Age_Years'] = normalized_frame_all['Health'] * normalized_frame_all['Age_Years']

normalized_frame_all['Fee x Age_Years'] = normalized_frame_all['Fee'] * normalized_frame_all['Age_Years']



##Round 2 of engineering

normalized_frame_all['Total_Sentiment_Score x Health'] = normalized_frame_all['Total_Sentiment_Score'] * normalized_frame_all['Health']

normalized_frame_all['Total_Sentiment_Score x Gender'] = normalized_frame_all['Total_Sentiment_Score'] * normalized_frame_all['Gender']

normalized_frame_all['Total_Sentiment_Score x Breed1'] = normalized_frame_all['Total_Sentiment_Score'] * normalized_frame_all['Breed1']

normalized_frame_all['Total_Sentiment_Score x PhotoAmt_Clipped'] = normalized_frame_all['Total_Sentiment_Score'] * normalized_frame_all['PhotoAmt_Clipped']

normalized_frame_all['Total_Sentiment_Score x Fee'] = normalized_frame_all['Total_Sentiment_Score'] * normalized_frame_all['Fee']

normalized_frame_all['PhotoAmt_Clipped x Health'] = normalized_frame_all['PhotoAmt_Clipped'] * normalized_frame_all['Health']

normalized_frame_all['PhotoAmt_Clipped x Breed1'] = normalized_frame_all['PhotoAmt_Clipped'] * normalized_frame_all['Breed1']

normalized_frame_all['PhotoAmt_Clipped x MaturitySize'] = normalized_frame_all['PhotoAmt_Clipped'] * normalized_frame_all['MaturitySize']

normalized_frame_all['PhotoAmt_Clipped x Gender'] = normalized_frame_all['PhotoAmt_Clipped'] * normalized_frame_all['Gender']

normalized_frame_all['Fee x PhotoAmt_Clipped'] = normalized_frame_all['Fee'] * normalized_frame_all['PhotoAmt_Clipped']





normalized_frame_all.head(5)
## Normalize numeric features into a single normalized frame

## For testing



## Map numeric features + engineered features

testframe_all = testframe[['Total_Sentiment_Score','Type','Age_Years','Breed1_Normalized','Breed2_Normalized',

                             'Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health',

                             'Quantity_Clipped','Fee','Fee_Normalized','PhotoAmt_Clipped',

                             'vertex_x','vertex_y','bounding_confidence', 'bounding_importance',

                             'dominant_blue', 'dominant_green', 'dominant_red','dominant_pixel_frac',

                             'dominant_score','label_score', 'Breed1', 'Breed2']]





## Reset index for testing features to prepare for normalization and re-indexation

testframe_all = testframe_all.reset_index()



## Create new normalized frames for normalization

normalized_frame_test_all = pd.DataFrame()

normalized_frame_test_all = testframe_all



##Add column names

#columnNames = list(testframe_all.head(0))



##Run normalization on both testing features excluding non-numeric IDs

#for i in columnNames:

    #if i != 'PetID':

      #  normalized_frame_test_all[i] = preprocessing.scale(testframe_all[i].astype('float64'))





## Add IDs and re-index both normalized feature sets to prepare for label merging



#normalized_frame_test_all['PetID'] = testframe_all['PetID']

#normalized_frame_test_all = normalized_frame_test_all.set_index('PetID')







#normalized_frame_test_all.head(5)
## Engineer feature crosses for the test set



normalized_frame_test_all['Breed1 x MaturitySize'] = normalized_frame_test_all['Breed1'] *  normalized_frame_test_all['MaturitySize']

normalized_frame_test_all['Breed1 x Gender'] = normalized_frame_test_all['Breed1'] *  normalized_frame_test_all['Gender']

normalized_frame_test_all['MaturitySize x Gender'] = normalized_frame_test_all['MaturitySize'] *  normalized_frame_test_all['Gender']

normalized_frame_test_all['Type x Gender'] = normalized_frame_test_all['Type'] *  normalized_frame_test_all['Gender']

normalized_frame_test_all['Type x MaturitySize'] = normalized_frame_test_all['Type'] *  normalized_frame_test_all['MaturitySize']

normalized_frame_test_all['Type x FurLength'] = normalized_frame_test_all['Type'] *  normalized_frame_test_all['FurLength']

normalized_frame_test_all['Gender x FurLength'] = normalized_frame_test_all['Gender'] *  normalized_frame_test_all['FurLength']

normalized_frame_test_all['Type x Health'] = normalized_frame_test_all['Type'] *  normalized_frame_test_all['Health']

normalized_frame_test_all['Fee x Health'] = normalized_frame_test_all['Fee'] *  normalized_frame_test_all['Health']

normalized_frame_test_all['Vaccinated x Dewormed x Sterilized'] = normalized_frame_test_all['Vaccinated'] *  normalized_frame_test_all['Dewormed'] * normalized_frame_test_all['Sterilized']

normalized_frame_test_all['Type x Color1'] = normalized_frame_test_all['Type'] *  normalized_frame_test_all['Color1']

normalized_frame_test_all['Breed1 x Color1'] = normalized_frame_test_all['Breed1'] *  normalized_frame_test_all['Color1']

normalized_frame_test_all['Type x Age_Years'] = normalized_frame_test_all['Type'] *  normalized_frame_test_all['Age_Years']

normalized_frame_test_all['Health x Age_Years'] = normalized_frame_test_all['Health'] * normalized_frame_test_all['Age_Years']

normalized_frame_test_all['Fee x Age_Years'] = normalized_frame_test_all['Fee'] * normalized_frame_test_all['Age_Years']



##Round 2 of engineering

normalized_frame_test_all['Total_Sentiment_Score x Health'] = normalized_frame_test_all['Total_Sentiment_Score'] * normalized_frame_test_all['Health']

normalized_frame_test_all['Total_Sentiment_Score x Gender'] = normalized_frame_test_all['Total_Sentiment_Score'] * normalized_frame_test_all['Gender']

normalized_frame_test_all['Total_Sentiment_Score x Breed1'] = normalized_frame_test_all['Total_Sentiment_Score'] * normalized_frame_test_all['Breed1']

normalized_frame_test_all['Total_Sentiment_Score x PhotoAmt_Clipped'] = normalized_frame_test_all['Total_Sentiment_Score'] * normalized_frame_test_all['PhotoAmt_Clipped']

normalized_frame_test_all['Total_Sentiment_Score x Fee'] = normalized_frame_test_all['Total_Sentiment_Score'] * normalized_frame_test_all['Fee']

normalized_frame_test_all['PhotoAmt_Clipped x Health'] = normalized_frame_test_all['PhotoAmt_Clipped'] * normalized_frame_test_all['Health']

normalized_frame_test_all['PhotoAmt_Clipped x Breed1'] = normalized_frame_test_all['PhotoAmt_Clipped'] * normalized_frame_test_all['Breed1']

normalized_frame_test_all['PhotoAmt_Clipped x MaturitySize'] = normalized_frame_test_all['PhotoAmt_Clipped'] * normalized_frame_test_all['MaturitySize']

normalized_frame_test_all['PhotoAmt_Clipped x Gender'] = normalized_frame_test_all['PhotoAmt_Clipped'] * normalized_frame_test_all['Gender']

normalized_frame_test_all['Fee x PhotoAmt_Clipped'] = normalized_frame_test_all['Fee'] * normalized_frame_test_all['PhotoAmt_Clipped']





normalized_frame_test_all.head(5)
## Converting description data into word2vec representation - with the assistance of takuok's code, found at https://www.kaggle.com/takuok/word2vec



eng_stopwords = set(stopwords.words("english"))

remove_punctuation_map = dict((ord(char), ' ') for char in string.punctuation)

#stemmer = nltk.stem.snowball.SnowballStemmer('english')

stemmer = nltk.stem.porter.PorterStemmer()



def stem_tokens(tokens):

    lst = [stemmer.stem(item) for item in tokens]

    return ' '.join(lst)



def get_textfeats(df, col, flag=True):

    df[col] = df[col].fillna('none').astype(str)

    df[col] = df[col].str.lower()

    df[col] = df[col].apply(lambda x: stem_tokens(nltk.word_tokenize(x.translate(remove_punctuation_map))))

    

    return df



def load_text(train, test):

    train = get_textfeats(train, "Description")

    test = get_textfeats(test, "Description")

    train_desc = train['Description'].values

    test_desc = test['Description'].values



    train_corpus = [text_to_word_sequence(text) for text in tqdm(train_desc)]

    test_corpus = [text_to_word_sequence(text) for text in tqdm(test_desc)]

    

    return train_corpus, test_corpus



def get_result(corpus, model):

    result = []

    for text in corpus:

        n_skip = 0

        for n_w, word in enumerate(text):

            try:

                vec_ = model.wv[word]

            except:

                n_skip += 1

                continue

            if n_w == 0:

                vec = vec_

            else:

                vec = vec + vec_

        vec = vec / (n_w - n_skip + 1)

        result.append(vec)

        

    return result



train_corpus, test_corpus = load_text(trainframe, testframe)

model = word2vec.Word2Vec(train_corpus+test_corpus, size=200, window=10, max_vocab_size=50000, seed=0)

train_result = get_result(train_corpus, model)

test_result = get_result(test_corpus, model)



w2v_cols = ["wv{}".format(i) for i in range(1, 201)]

train_result = pd.DataFrame(train_result)

train_result.columns = w2v_cols

test_result = pd.DataFrame(test_result)

test_result.columns = w2v_cols



normalized_frame_all = pd.concat((normalized_frame_all, train_result), axis=1)

normalized_frame_test_all = pd.concat((normalized_frame_test_all, test_result), axis=1)



## Select final features and targets for training and validation



feature_selection = []



columnNames = list(normalized_frame_all.head(0))



for i in columnNames:

    if i != 'PetID' and i != 'AdoptionSpeed':

        feature_selection.append(i)

                        

target_train = trainframe_all['AdoptionSpeed'].values

train  = np.array(normalized_frame_all[feature_selection])



## Prepare cross-validation



X = train

y = target_train

K = 10

kf = StratifiedKFold(n_splits = K, random_state = 3228, shuffle = True)
## Kappa calculation taken from ulissesdias https://www.kaggle.com/ulissesdias/xgboost-all-data-hyperopt-parameter-tuning/notebook competition entry by the way of

## Hamner's github repository

# https://github.com/benhamner/Metrics





def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(rater_a + rater_b)

    if max_rating is None:

        max_rating = max(rater_a + rater_b)

    num_ratings = int(max_rating - min_rating + 1)

    conf_mat = [[0 for i in range(num_ratings)]

                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):

        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat





def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """

    if min_rating is None:

        min_rating = min(ratings)

    if max_rating is None:

        max_rating = max(ratings)

    num_ratings = int(max_rating - min_rating + 1)

    hist_ratings = [0 for x in range(num_ratings)]

    for r in ratings:

        hist_ratings[r - min_rating] += 1

    return hist_ratings





def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return (1.0 - numerator / denominator)
## Modified version of the classifier from ulissesdias

## https://www.kaggle.com/ulissesdias/xgboost-all-data-hyperopt-parameter-tuning/notebook

## Including regularization and 10 strat k-fold validation



## Set hyperopt parameter space including L1 and L2 reg



space ={

    #'max_depth'      : 23,

    'max_depth'       : hp.quniform('max_depth', 12, 25, 0.5),

    'min_child_weight': 21,

    #'min_child_weight': hp.quniform('min_child_weight', 18,30,0.5),

    'subsample'       : 0.6000000000000001,

    #'subsample'       : hp.quniform('subsample', 0.2, 0.9, 0.05),

    #'colsample_bytree': hp.quniform('colsample_bytree', 0, 0.99, 0.05),

    'colsample_bytree': 0.65,

   # 'alpha': hp.quniform('alpha', 0, 20, 0.5),

     #'alpha': 7.3,

    #'lambda': 15,

    #'lambda': hp.quniform('lambda', 10, 15, 0.5),

    #'gamma': hp.quniform('gamma', 15,25,0.5)

    #'gamma': 22.5

    'seed': 1337,

    'eta': 0.0123

    }



## Create summary list of training outcomes

outcomes = []

bestkappa = []

bestparams = []



def print_feature_importance(clf) :

    sorted_idx = np.argsort(clf.feature_importances_)[::-1]

    importance = "Importance = ["

    for index in sorted_idx[:15] :

        importance += feature_selection[index] + ","

        #print([features[index], clf.feature_importances_[index]])

    print(importance + "]")

    return importance

    



## Objective function - XGBClassifier with 10 strat k-fold validation



def objective(space):

    

    clf = XGBClassifier(

        nthread          = 40,

            #n_estimators     = 10000,

            #objective = 'multi:softmax', 

            #num_class = 5, 

        max_depth        = int(space['max_depth']),

        min_child_weight = float(space['min_child_weight']),

        subsample        = float(space['subsample']),

        colsample_bytree = float(space['colsample_bytree']),

        seed = int(space['seed']),

        eta = int(space['eta'])

                )

    

        ## Apply cross-validation

    for train_index, test_index in kf.split(X, y):

        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

    

        train_features = pd.DataFrame(X_train)

        train_targets = pd.DataFrame(y_train).values

        valid_features = pd.DataFrame(X_test)

        valid_targets = pd.DataFrame(y_test).values



        eval_set  = [(train_features, train_targets.ravel()), ( valid_features, valid_targets.ravel())]

        clf.fit(train_features, train_targets.ravel(), eval_set = eval_set, eval_metric="merror", early_stopping_rounds=50,verbose = False)

        

    importance = print_feature_importance(clf)

    prediction_train = clf.predict(train_features)

    

    kappa_valid_frame = normalized_frame_all

    kappa_valid_frame = kappa_valid_frame.reset_index()

    kappa_valid_frame['AdoptionSpeed'] = trainframe_all['AdoptionSpeed']

    kappa_valid_frame = kappa_valid_frame.sample(6000)



    kappa_valid_features = pd.DataFrame(np.array(kappa_valid_frame[feature_selection]))

    kappa_valid_targets = kappa_valid_frame['AdoptionSpeed'].values



    prediction_valid = clf.predict(valid_features)

    kappa_train = quadratic_weighted_kappa(train_targets.ravel(), prediction_train)

    kappa_valid = quadratic_weighted_kappa(valid_targets.ravel(), prediction_valid)

    

    if not bestkappa:

        bestkappa.append(kappa_valid)

        bestparams.append(space['max_depth'])

        bestparams.append(space['max_depth'])

        bestparams.append(space['max_depth'])

    elif bestkappa[0] < kappa_valid:

        bestkappa[0] = kappa_valid

        bestparams[0] = space['max_depth']

        bestparams[1] = space['max_depth']

        bestparams[2] = space['max_depth']

    

    print("space: %s, Kappa Train: %.3f, Kappa Valid: %.3f, Feature Importance: %s" % (str(space), kappa_train, kappa_valid, importance))

    print(importance)

    print("")

    outcomes.append(str("space: %s, Kappa Train: %.3f, Kappa Valid: %.3f, Feature Importance: %s, Best Kappa: %.3f" % (str(space), kappa_train, kappa_valid, importance, int(bestkappa[0]))))

    print(bestkappa)

    print(bestparams)

    return{'loss':1-kappa_valid, 'status': STATUS_OK }



trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=18,

            trials=trials)





## Write hyperparameter performance in a csv file

outcomereport = pd.DataFrame()

outcomereport['Outcomes'] = outcomes

outcomereport



outcomereport.to_csv('outcomes.csv',index=False)
## Optimal hyperparameters from the tuning iterations



space ={

    #'max_depth'      : 17.5,

    'max_depth'       : bestparams[0],

    'min_child_weight': 21,

    #'min_child_weight': hp.quniform('min_child_weight', 15, 25,0.5),

    'subsample'       : 0.6000000000000001,

    #'subsample'       : hp.quniform('subsample', 0, 0.99, 0.05),

    #'colsample_bytree': hp.quniform('colsample_bytree', 0, 0.99, 0.05),

    'colsample_bytree': 0.65,

    #'alpha': hp.quniform('alpha', 0, 20, 0.5),

    #'alpha': bestparams[1],

    #'lambda': bestparams[1],

    #'lambda': 15,

   # 'gamma': hp.quniform('gamma', 0, 20,0.5)

    #'gamma': 22.5

    'seed': 1337,

    'eta': 0.0123

    }



## Objective function - XGBClassifier with 10 strat k-fold validation



clf = XGBClassifier(

    thread          = 40,

    #objective = 'multi:softmax', 

    #num_class = 5,

    max_depth = int(space['max_depth']),

    min_child_weight = float(space['min_child_weight']),

    subsample = float(space['subsample']),

    colsample_bytree = float(space['colsample_bytree']),

    seed = int(space['seed']),

    eta = int(space['eta'])

            )

    ## Apply cross-validation



for train_index, test_index in kf.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    train_features = pd.DataFrame(X_train)

    train_targets = pd.DataFrame(y_train).values

    valid_features = pd.DataFrame(X_test)

    valid_targets = pd.DataFrame(y_test).values



    eval_set  = [(train_features, train_targets.ravel()), ( valid_features, valid_targets.ravel())]

    clf.fit(train_features, train_targets.ravel(), eval_set = eval_set, eval_metric="merror", early_stopping_rounds=50,verbose = False)

    

print_feature_importance(clf)

prediction_train = clf.predict(train_features)

prediction_valid = clf.predict(valid_features)

    

kappa_train = quadratic_weighted_kappa(train_targets.ravel(), prediction_train)

kappa_valid = quadratic_weighted_kappa(valid_targets.ravel(), prediction_valid)

    

print("space: %s, Kappa Train: %.3f, Kappa Valid: %.3f" % (str(space), kappa_train, kappa_valid))

print("")

## Creaate test set for prediction and submission

test  = pd.DataFrame(np.array(normalized_frame_test_all[feature_selection]))

normalized_frame_test_all[feature_selection].head(5)
normalized_frame_all[feature_selection].head(5)
## Creat submission and save to csv



prediction_test = clf.predict(test)



submission = pd.DataFrame(

    { 

        'PetID'         : testframe_all.PetID, 

        'AdoptionSpeed' : prediction_test

    }

)



submission.to_csv('submission.csv',index=False)



testframe.describe()
trainframe.describe()