# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

train = pd.read_csv('../input/pet-data/pets_train.csv')
test = pd.read_csv('../input/pet-data/pets_test.csv')
cats = pd.read_csv('../input/catinfo/cat_info.csv')
for index, row in train.iterrows():
    if row['Type'] == 2:
        for i, r in cats.iterrows():
            if row['Breed1'] == r['BreedID']:
                train.set_value(index, 'cat_Cute', r['Cute'])
                train.set_value(index, 'cat_Hypo', r['Hypo'])
                break
    else:
        train.set_value(index, 'cat_Cute', -1)
        train.set_value(index, 'cat_Hypo', -1)
train.cat_Cute.astype('category', inplace=True)
train.cat_Hypo.astype('category', inplace=True)
for index, row in test.iterrows():
    if row['Type'] == 2:
        for i, r in cats.iterrows():
            if row['Breed1'] == r['BreedID']:
                test.set_value(index, 'cat_Cute', r['Cute'])
                test.set_value(index, 'cat_Hypo', r['Hypo'])
                break
    else:
        test.set_value(index, 'cat_Cute', -1)
        test.set_value(index, 'cat_Hypo', -1)
test.cat_Cute.astype('category', inplace=True)
test.cat_Hypo.astype('category', inplace=True)
train.Name.fillna('None', inplace=True)
train.Description.fillna('None', inplace=True)
train = train.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)

test.Name.fillna('None', inplace=True)
test.Description.fillna('None', inplace=True)
test = test.drop(['Unnamed: 0', 'Unnamed: 0.1','Unnamed: 0.1.1'], axis=1)
test.Group.fillna('Cat', inplace=True)
for index, row in train.iterrows():
    file = '../input/petfinder-adoption-prediction/train_metadata/' + row['PetID'] + '-1.json'
    if os.path.exists(file):
        data = json.load(open(file))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        train.set_value(index, 'vertex_x', vertex_x)
        train.set_value(index, 'vertex_y', vertex_y)
        train.set_value(index, 'bounding_conf', bounding_confidence)                
        train.set_value(index, 'bounding_imp', bounding_importance_frac)                
        train.set_value(index, 'dom_blue', dominant_blue)                
        train.set_value(index, 'dom_green', dominant_green)                
        train.set_value(index, 'dom_red', dominant_red)   
        train.set_value(index, 'pixel_frac', dominant_pixel_frac)                
        train.set_value(index, 'score', dominant_score)
    else:
        train.set_value(index, 'vertex_x', -1)
        train.set_value(index, 'vertex_y', -1)
        train.set_value(index, 'bounding_conf', -1)                
        train.set_value(index, 'bounding_imp', -1)                
        train.set_value(index, 'dom_blue', -1)                
        train.set_value(index, 'dom_green', -1)                
        train.set_value(index, 'dom_red', -1)   
        train.set_value(index, 'pixel_frac', -1)                
        train.set_value(index, 'score', -1)

for index, row in train.iterrows():
    file = '../input/petfinder-adoption-prediction/train_metadata/' + row['PetID'] + '-2.json'
    if os.path.exists(file):
        data = json.load(open(file))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        try:
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        except:
            dominant_blue = -1
        try:
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        except:
            dominant_green = -1
        try:
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        except:
            dominant_red = -1
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        train.set_value(index, 'vertex_x2', vertex_x)
        train.set_value(index, 'vertex_y2', vertex_y)
        train.set_value(index, 'bounding_conf2', bounding_confidence)                
        train.set_value(index, 'bounding_imp2', bounding_importance_frac)                
        train.set_value(index, 'dom_blue2', dominant_blue)                
        train.set_value(index, 'dom_green2', dominant_green)                
        train.set_value(index, 'dom_red2', dominant_red)   
        train.set_value(index, 'pixel_frac2', dominant_pixel_frac)                
        train.set_value(index, 'score2', dominant_score)
    else:
        train.set_value(index, 'vertex_x2', -1)
        train.set_value(index, 'vertex_y2', -1)
        train.set_value(index, 'bounding_conf2', -1)                
        train.set_value(index, 'bounding_imp2', -1)                
        train.set_value(index, 'dom_blue2', -1)                
        train.set_value(index, 'dom_green2', -1)                
        train.set_value(index, 'dom_red2', -1)   
        train.set_value(index, 'pixel_frac2', -1)                
        train.set_value(index, 'score2', -1)
for index, row in test.iterrows():
    file = '../input/petfinder-adoption-prediction/test_metadata/' + row['PetID'] + '-2.json'
    if os.path.exists(file):
        data = json.load(open(file))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        test.set_value(index, 'vertex_x2', vertex_x)
        test.set_value(index, 'vertex_y2', vertex_y)
        test.set_value(index, 'bounding_conf2', bounding_confidence)                
        test.set_value(index, 'bounding_imp2', bounding_importance_frac)                
        test.set_value(index, 'dom_blue2', dominant_blue)                
        test.set_value(index, 'dom_green2', dominant_green)                
        test.set_value(index, 'dom_red2', dominant_red)   
        test.set_value(index, 'pixel_frac2', dominant_pixel_frac)                
        test.set_value(index, 'score2', dominant_score)
    else:
        test.set_value(index, 'vertex_x2', -1)
        test.set_value(index, 'vertex_y2', -1)
        test.set_value(index, 'bounding_conf2', -1)                
        test.set_value(index, 'bounding_imp2', -1)                
        test.set_value(index, 'dom_blue2', -1)                
        test.set_value(index, 'dom_green2', -1)                
        test.set_value(index, 'dom_red2', -1)   
        test.set_value(index, 'pixel_frac2', -1)                
        test.set_value(index, 'score2', -1)

for index, row in test.iterrows():
    file = '../input/petfinder-adoption-prediction/test_metadata/' + row['PetID'] + '-1.json'
    if os.path.exists(file):
        data = json.load(open(file))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        test.set_value(index, 'vertex_x', vertex_x)
        test.set_value(index, 'vertex_y', vertex_y)
        test.set_value(index, 'bounding_conf', bounding_confidence)                
        test.set_value(index, 'bounding_imp', bounding_importance_frac)                
        test.set_value(index, 'dom_blue', dominant_blue)                
        test.set_value(index, 'dom_green', dominant_green)                
        test.set_value(index, 'dom_red', dominant_red)   
        test.set_value(index, 'pixel_frac', dominant_pixel_frac)                
        test.set_value(index, 'score', dominant_score)
    else:
        test.set_value(index, 'vertex_x', -1)
        test.set_value(index, 'vertex_y', -1)
        test.set_value(index, 'bounding_conf', -1)                
        test.set_value(index, 'bounding_imp', -1)                
        test.set_value(index, 'dom_blue', -1)                
        test.set_value(index, 'dom_green', -1)                
        test.set_value(index, 'dom_red', -1)   
        test.set_value(index, 'pixel_frac', -1)                
        test.set_value(index, 'score', -1)

labels = pd.get_dummies(train, columns = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                                 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
                                 'State', 'Type', 'Group'
                                ])
test_labels = pd.get_dummies(test, columns = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                                 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
                                 'State', 'Type', 'Group'
                                ])
diff_columns = set(labels.columns).difference(set(test_labels.columns))
for i in diff_columns:
    test_labels[i] = test_labels.apply(lambda _: 0, axis=1)
diff_columns2 = set(test_labels.columns).difference(set(labels.columns))
for i in diff_columns2:
    labels[i] = labels.apply(lambda _: 0, axis=1)
target = train['AdoptionSpeed']
labels['NameLength'] = train['Name'].map(lambda x: 0 if x == 'None' else len(x)).astype('int')
labels['DescLength'] = train['Description'].map(lambda x: len(x)).astype('int')
labels['Cute'] = train['Description'].map(lambda x: 1 if 'CUTE' in x.upper() else 0).astype('int')
test_labels['Cute'] = test['Description'].map(lambda x: 1 if 'CUTE' in x.upper() else 0).astype('int')
test_labels['NameLength'] = test['Name'].map(lambda x: 0 if x == 'None' else len(x)).astype('int')
test_labels['DescLength'] = test['Description'].map(lambda x: len(x)).astype('int')
labels = labels.drop(['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'], axis=1)
test_labels = test_labels.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)
clf = xgb.XGBClassifier()
clf.fit(labels, target)
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(12,18))
plot_importance(clf, max_num_features=50, height=0.8, ax=ax)
plt.show()
test_labels = test_labels[labels.columns]
pred = pd.DataFrame()
pred['PetID'] = test['PetID']
pred['AdoptionSpeed'] = clf.predict(test_labels)
pred.set_index('PetID').to_csv("submission.csv", index=True)
