# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

print(os.listdir("../input/train/"))
# print(os.listdir("../input/train_metadata/"))
# print(os.listdir("../input/train_sentiment/"))
# print(os.listdir("../input/train_images/"))

print(os.listdir("../input/test/"))
# print(os.listdir("../input/test_sentiment/"))
# print(os.listdir("../input/test_metadata/"))
# print(os.listdir("../input/test_images/"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train/train.csv")
df_train.shape
df_train.head()
df_train.hist(figsize=(15,15))
# let's create some convenience constants
DISCREET_COLS = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', ]
SCALAR_COLS = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', ]
TEXT_COLS = ['Name', 'RescuerID', 'Description', 'PetID', ]
TARGET_COL = 'AdoptionSpeed'
corr = df_train.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
for c in DISCREET_COLS:
    df_g = df_train[[c, 'AdoptionSpeed', ]].groupby([c, 'AdoptionSpeed', ]).size().unstack()
    df_g.div(df_g.sum(1), axis=0).plot.bar(figsize=(12,5), colormap='tab20', stacked=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))
for c in SCALAR_COLS:
    plt.figure()
    df_g = df_train[[c, 'AdoptionSpeed', ]].groupby([c, 'AdoptionSpeed', ]).size().unstack().fillna(0)
    df_g.apply(lambda x: np.average(range(5), weights=x), axis=1).plot.line(figsize=(12,5), legend=False)
df_test = pd.read_csv("../input/test/test.csv")
df_test.shape
df_test.head()
def compare_discreet_column(col, train, test):
    print('\nComparing column {0}'.format(col))
    s_trn = train[col]
    s_tst = test[col]
    # check for extra values
    train_vals = set(s_trn.unique())
    test_vals = set(s_tst.unique())
    print("extra values in train: {0}".format(train_vals - test_vals))
    print("extra values in test: {0}".format(test_vals - train_vals))
    # check for major changes in representation
    if len(train_vals) < 10:
        trn_rep = s_trn.groupby(s_trn).size() / len(s_trn)
        tst_rep = s_tst.groupby(s_tst).size() / len(s_tst)
        df_temp = pd.concat([trn_rep, tst_rep], axis=1)
        df_temp.columns = ['Train', 'Test']
        print(df_temp)
    else:
        print("too many values to compare representation")
for c in DISCREET_COLS:
    compare_discreet_column(c, df_train, df_test)
extra_breed_1 = set(df_test['Breed1'].unique()) - set(df_train['Breed1'].unique())
print("% of test set with Breed1 not in train: {0:0.2f}%".format(len(df_test[df_test['Breed1'].isin(extra_breed_1)])/len(df_test)*100))
extra_breed_2 = set(df_test['Breed2'].unique()) - set(df_train['Breed2'].unique())
print("% of test set with Breed2 not in train: {0:0.2f}%".format(len(df_test[df_test['Breed2'].isin(extra_breed_1)])/len(df_test)*100))
fpatt = re.compile("([0-9a-zA-Z]+)-(\d+)\.(.+)")
FACE_ANNOTATION_FIELDS = ['angerLikelihood','blurredLikelihood','detectionConfidence','joyLikelihood','sorrowLikelihood','surpriseLikelihood','underExposedLikelihood']
def get_image_info(image_path, meta_path):
    image_info = {}
    # create a dictionary to represent the image data
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg"): 
            res = fpatt.match(filename)
            if res.group(1) in image_info:
                image_info[res.group(1)][res.group(2)] = {}
            else:
                image_info[res.group(1)] = {}
                image_info[res.group(1)][res.group(2)] = {}
    
    for filename in os.listdir(meta_path):
        if filename.endswith(".json"): 
            with open(os.path.join(meta_path, filename)) as json_file:  
                jsond = json.load(json_file)
                res = fpatt.match(filename)
                assert res.group(1) in image_info
                pet_record = image_info[res.group(1)]
                if res.group(2) in pet_record:
                    img_record = pet_record[res.group(2)]
                    # face annotation
                    if 'faceAnnotations' in jsond:
                        img_record['face'] = {}
                        for f in FACE_ANNOTATION_FIELDS:
                            img_record['face'][f] = jsond['faceAnnotations'][0][f]
                    else:
                        img_record['face'] = None
                    # label annotations
                    if 'labelAnnotations' in jsond:
                        img_record['labels'] = [x['description'] for x in jsond['labelAnnotations']]
                    else:
                        img_record['labels'] = []
                else:
                    print("DIAG: missing {0} for {1}".format(res.group(2), res.group(1)))
                    break
                
    return image_info

train_image_info = get_image_info('../input/train_images/', '../input/train_metadata/')
test_image_info = get_image_info('../input/test_images/', '../input/test_metadata/')
s_img = df_train['PetID'].isin(train_image_info.keys())
print("Train set image prevalence:\n{0}".format(s_img.groupby(s_img).size() / len(s_img)))
s_img = df_test['PetID'].isin(test_image_info.keys())
print("Test set image prevalence:\n{0}".format(s_img.groupby(s_img).size() / len(s_img)))
df_train_pix = pd.DataFrame(index=df_train.index, columns=['angerLikelihood','blurredLikelihood','detectionConfidence','joyLikelihood','sorrowLikelihood','surpriseLikelihood','underExposedLikelihood','rec_cat','rec_dog'])
df_train_pix['PetID'] = df_train['PetID']
for idx, row in df_train_pix.iterrows():
    if row['PetID'] in train_image_info:
        pet_record = train_image_info[row['PetID']]
        image_record = pet_record["1"]
        data = []
        if image_record['face']:
            for faf in FACE_ANNOTATION_FIELDS:
                data.append(image_record['face'][faf])
        else:
            for faf in FACE_ANNOTATION_FIELDS:
                data.append(None)

        if 'cat' in image_record['labels']:
            data.append(True)
        else:
            data.append(False)
        
        if 'dog' in image_record['labels']:
            data.append(True)
        else:
            data.append(False)
        
        df_train_pix.loc[idx,0:9] = data
        
df_train_pix[TARGET_COL] = df_train[TARGET_COL]
df_train_pix['Type'] = df_train['Type']
for c in FACE_ANNOTATION_FIELDS:
    print("{0}:{1}".format(c, df_train_pix[c].unique()))
print('Misrec dogs: {0}'.format(len(df_train_pix[(df_train_pix['Type'] == 1)&(df_train_pix['rec_dog'] == False)])))
print('Misrec cats: {0}'.format(len(df_train_pix[(df_train_pix['Type'] == 2)&(df_train_pix['rec_cat'] == False)])))

print("Mean adoption time: {0:0.2f} vs misrec dog pix adotion time: {1:0.2f}".format(
    df_train[df_train_pix['Type'] == 1][TARGET_COL].mean(),
    df_train[(df_train_pix['Type'] == 1)&(df_train_pix['rec_dog'] == False)][TARGET_COL].mean()
))

print("Mean adoption time: {0:0.2f} vs misrec cat pix adotion time: {1:0.2f}".format(
    df_train[df_train_pix['Type'] == 2][TARGET_COL].mean(),
    df_train[(df_train_pix['Type'] == 2)&(df_train_pix['rec_cat'] == False)][TARGET_COL].mean()
))
fspatt = re.compile("([0-9a-zA-Z]+)\.(.+)")
def get_sentiment_info(sentiment_path):
    sentiment_info = {}
    for filename in os.listdir(sentiment_path):
        if filename.endswith(".json"): 
            res = fspatt.match(filename)
            sentiment_info[res.group(1)] = {}    
            with open(os.path.join(sentiment_path, filename)) as json_file:  
                jsond = json.load(json_file)
                sentiment_info[res.group(1)]['score'] = jsond['documentSentiment']['score']
                sentiment_info[res.group(1)]['magnitude'] = jsond['documentSentiment']['magnitude']     
    return sentiment_info

train_sentiment_info = get_sentiment_info('../input/train_sentiment/')
test_sentiment_info = get_sentiment_info('../input/test_sentiment/')
s_sen = df_train['PetID'].isin(train_sentiment_info.keys())
print("Train set image prevalence:\n{0}".format(s_sen.groupby(s_sen).size() / len(s_sen)))
s_sen = df_test['PetID'].isin(test_sentiment_info.keys())
print("Test set image prevalence:\n{0}".format(s_sen.groupby(s_sen).size() / len(s_sen)))
df_train_sentiment = pd.DataFrame(index=df_train.index, columns=['score','magnitude'])
df_train_sentiment['PetID'] = df_train['PetID']
for idx, row in df_train_sentiment.iterrows():
    if row['PetID'] in train_sentiment_info:
        pet_record = train_sentiment_info[row['PetID']]
        data = []
        data.append(pet_record['score'])
        data.append(pet_record['magnitude'])
        df_train_sentiment.loc[idx,0:2] = data
df_train_sentiment.fillna(0.0, inplace=True)
df_train_sentiment[TARGET_COL] = df_train[TARGET_COL]
df_train_sentiment['Type'] = df_train['Type']

df_train_sentiment[['score', 'AdoptionSpeed']].corr()
def calc_full_sentiment(score, magnitude):
    if (score < 0 ) and (magnitude > 0.25):
        return -2
    elif (score < 0 ):
        return -1
    elif score == 0:
        return 0
    elif score > 0 and magnitude > 0.25:
        return 2
    else:
        return 1
df_train_sentiment['sent_agg'] = df_train_sentiment.apply(lambda x: calc_full_sentiment(x['score'],x['magnitude']), axis=1)
df_g = df_train_sentiment[['sent_agg', 'AdoptionSpeed', ]].groupby(['sent_agg', 'AdoptionSpeed', ]).size().unstack()
df_g.div(df_g.sum(1), axis=0).plot.bar(figsize=(12,5), colormap='tab20', stacked=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))
