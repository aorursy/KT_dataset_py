# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

import json



import xgboost as xgb

import lightgbm as lgb



from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from keras.models import Sequential

from keras.layers import Dense, Activation

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train/train.csv")

test = pd.read_csv("../input/test/test.csv")

breeds = pd.read_csv("../input/breed_labels.csv")

colors = pd.read_csv("../input/color_labels.csv")

states = pd.read_csv("../input/state_labels.csv")
train['Name'] = train['Name'].fillna('Unnamed')

test['Name'] = test['Name'].fillna('Unnamed')
train['No_name'] = 0

train.loc[train['Name'] == 'Unnamed', 'No_name'] = 1

train.loc[train['Name'] == 'No Name Yet', 'No_name'] = 1

test['No_name'] = 0

test.loc[test['Name'] == 'Unnamed', 'No_name'] = 1

test.loc[train['Name'] == 'No Name Yet', 'No_name'] = 1
train.head()
train['Out_Age'] = 0

train.loc[train['Age'] > 20, 'Out_Age'] = 1

test['Out_Age'] = 0

test.loc[test['Age'] > 20, 'Out_Age'] = 1
train['Pure_breed'] = 0

train.loc[train['Breed2'] == 0, 'Pure_breed'] = 1

test['Pure_breed'] = 0

test.loc[test['Breed2'] == 0, 'Pure_breed'] = 1
train['BadName'] = 0

train.loc[train['Name'].apply(lambda x: len(str(x))) < 3,'BadName']=1
test['BadName'] = 0

test.loc[test['Name'].apply(lambda x: len(str(x))) < 3,'BadName']=1
train_id = train['PetID']

doc_sent_mag = []

doc_sent_score = []

nf_count = 0

for pet in train_id:

    try:

        with open('../input/train_sentiment/' + pet + '.json', 'r') as f:

            sentiment = json.load(f)

        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])

        doc_sent_score.append(sentiment['documentSentiment']['score'])

    except FileNotFoundError:

        nf_count += 1

        doc_sent_mag.append(-1)

        doc_sent_score.append(-1)

train.loc[:, 'doc_sent_mag'] = doc_sent_mag

train.loc[:, 'doc_sent_score'] = doc_sent_score
test_id = test['PetID']

doc_sent_mag = []

doc_sent_score = []

nf_count = 0

for pet in test_id:

    try:

        with open('../input/test_sentiment/' + pet + '.json', 'r') as f:

            sentiment = json.load(f)

        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])

        doc_sent_score.append(sentiment['documentSentiment']['score'])

    except FileNotFoundError:

        nf_count += 1

        doc_sent_mag.append(-1)

        doc_sent_score.append(-1)



test.loc[:, 'doc_sent_mag'] = doc_sent_mag

test.loc[:, 'doc_sent_score'] = doc_sent_score
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

train.loc[:, 'vertex_x'] = vertex_xs

train.loc[:, 'vertex_y'] = vertex_ys

train.loc[:, 'bounding_confidence'] = bounding_confidences

train.loc[:, 'bounding_importance'] = bounding_importance_fracs

train.loc[:, 'dominant_blue'] = dominant_blues

train.loc[:, 'dominant_green'] = dominant_greens

train.loc[:, 'dominant_red'] = dominant_reds

train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

train.loc[:, 'dominant_score'] = dominant_scores

train.loc[:, 'label_description'] = label_descriptions

train.loc[:, 'label_score'] = label_scores





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

test.loc[:, 'vertex_x'] = vertex_xs

test.loc[:, 'vertex_y'] = vertex_ys

test.loc[:, 'bounding_confidence'] = bounding_confidences

test.loc[:, 'bounding_importance'] = bounding_importance_fracs

test.loc[:, 'dominant_blue'] = dominant_blues

test.loc[:, 'dominant_green'] = dominant_greens

test.loc[:, 'dominant_red'] = dominant_reds

test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

test.loc[:, 'dominant_score'] = dominant_scores

test.loc[:, 'label_description'] = label_descriptions

test.loc[:, 'label_score'] = label_scores
target=train['AdoptionSpeed']

train=train.drop(['AdoptionSpeed'],axis=1)
df_train=train.drop(['Name','RescuerID','Description','PetID','label_description'],axis=1)

df_test=test.drop(['Name','RescuerID','Description','PetID','label_description'],axis=1)
len(df_train)
df_train.dtypes
#d=pd.concat([df_train,df_test])

#d=pd.get_dummies(d)
#df_train=d.iloc[:14993]

#df_test=d.iloc[14993:]
df_train.shape
df_test.shape
xg=xgb.XGBClassifier()

xg.fit(df_train,target)

pred_xgb=xg.predict(df_test)
rf=RandomForestClassifier()

rf.fit(df_train,target)

pred_rf=rf.predict(df_test)
gb=lgb.LGBMClassifier()

gb.fit(df_train,target)

pred_gb=gb.predict(df_test)
abc=AdaBoostClassifier()

abc.fit(df_train,target)

pred_abc=abc.predict(df_test)
gbc=GradientBoostingClassifier()

gbc.fit(df_train,target)

pred_gbc=gbc.predict(df_test)
etc=ExtraTreesClassifier()

etc.fit(df_train,target)

pred_etc=etc.predict(df_test)
svc=SVC()

svc.fit(df_train,target)

pred_svc=svc.predict(df_test)
df=pd.DataFrame()

df['abc']=pred_abc

df['etc']=pred_etc

df['gb']=pred_gb

df['gbc']=pred_gbc

df['rf']=pred_rf

df['xgb']=pred_xgb

df['svc']=pred_svc

df.head()
df.mode(axis=1)
submit=pd.DataFrame()

submit['PetID']=test['PetID']

submit['AdoptionSpeed']=df.mode(axis=1)[0].astype(int)

submit.to_csv('submission.csv',index=False)