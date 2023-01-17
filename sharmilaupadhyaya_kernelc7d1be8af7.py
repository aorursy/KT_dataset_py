# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import re

print(os.listdir("../input"))



import spacy

import networkx as nx



import zipfile



sample_submission = pd.read_csv("../input/gendered-pronoun-resolution/sample_submission_stage_1.csv")

final_test = pd.read_csv("../input/gendered-pronoun-resolution/test_stage_2.tsv", sep = "\t")

nlp = spacy.load('en_core_web_sm')

dep = ["ACL", "ACOMP", "ADVCL", "ADVMOD", "AGENT", "AMOD", "APPOS", "ATTR", "AUX", "AUXPASS",

       "CASE", "CC", "CCOMP", "COMPOUND", "CONJ", "CSUBJ", "CSUBJPASS", "DATIVE", "DEP", "DET", "DOBJ"

     , "EXPL", "INTJ", "MARK", "META", "NEG", "NOUNMOD", "NPMOD", "NSUBJ", "NSUBJPASS", "NUMMOD"

     , "OPRD", "PARATAXIS", "PCOMP", "POBJ", "POSS", "PRECONJ", "PREDET", "PREP", "PRT", "PUNCT", "QUANTMOD",

       "RELCL", "ROOT", "XCOMP", "COMPLM","INFMOD","PARTMOD","HMOD","HYPH","IOBJ","NUM",

       "NUMBER","NMOD","NN","NPADVMOD","POSSESSIVE","RCMOD","SUBTOK"]



# Any results you write to the current directory are saved as output.
final_test.shape
import tensorflow as tf
# downloading test, train and validation data from github

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv

!ls

train_data = pd.read_csv("gap-development.tsv", sep = "\t")

validation_data = pd.read_csv("gap-validation.tsv", sep = "\t")

test_data = pd.read_csv("gap-test.tsv", sep = "\t")


merge_data = pd.concat([train_data,validation_data]).reset_index(drop = True)

merge_data = pd.concat([merge_data,train_data]).reset_index(drop = True)

count = 0




def name_replace(s, r1, r2):

    s = str(s).replace(r1,r2)

    for r3 in r1.split(' '):

        s = str(s).replace(r3,r2)

    return s

def shortest_dependency_path(doc, e1=None, e2=None):

    

    edges = []

    for token in doc:

        for child in token.children:

            edges.append(('{0}'.format(token),

                          '{0}'.format(child)))

    graph = nx.Graph(edges)

    try:

        shortest_path = nx.shortest_path(graph, source=e1, target=e2)

    except Exception as e:

        shortest_path = [e1, e2]

        print(e)

        print(doc, e1, e2)



    return shortest_path



def dependency_vector(doc, pronoun, word):

    

    vector = [0] * 59

#     for token in doc:

#         if token.text == pronoun:

#             pi = token.i

#         elif token.text == word:

#             wi = token.i

#     if pi>wi:

#         for token in doc[wi:pi+1]:

#             index = dep.index(token.dep_.upper())

#             vector[index] = 1



#     else:

#         for token in doc[pi:wi+1]:

#             index = dep.index(token.dep_.upper())

#             vector[index] = 1

                

#     return vector

         

    x = shortest_dependency_path(doc, pronoun, word)

    for token in doc:

        if token.text in x:

            val = (x.index(str(token)) + 1) / len(x)

            try:

                index = dep.index(token.dep_.upper())

                vector[index] = val

            except:

                pass

    return vector

def get_features(df):

    

    df['A-offset2'] = df['A-offset'] + df['A'].map(len)

    df['B-offset2'] = df['B-offset'] + df['B'].map(len)

    df["Text"] =  df.apply(lambda row: name_replace(row["Text"], row["A"], "Noun_1"), axis = 1)

    df["Text"] =  df.apply(lambda row: name_replace(row["Text"], row["B"], "Noun_2"), axis = 1)

    new_df = pd.DataFrame([])

    new_df["Pronoun-offset"] = df["Pronoun-offset"]

    new_df['A-offset'] = df["A-offset"]

    new_df["B-offset"] = df["B-offset"]

    new_df['A-offset2'] = df['A-offset2']

    new_df['B-offset2'] = df['B-offset2']

    new_df['A_dist'] = (df['Pronoun-offset'] - df['A-offset']).abs()

    new_df['B_dist'] = (df['Pronoun-offset'] - df['B-offset']).abs()

    df["Text"] = df.Text.apply(lambda row: " and ".join(row.split(". ")))

    vectors_A = df.apply(lambda row: dependency_vector(nlp(row["Text"]), row["Pronoun"],"Noun_1") + dependency_vector(nlp(row["Text"]), row["Pronoun"],"Noun_2"), axis = 1)

    print(count)

    new_df_2 = pd.DataFrame(vectors_A.tolist())

    new_df = pd.concat([new_df, new_df_2], axis = 1)    

    return new_df

    

    
feature = get_features(merge_data)





feature


Y = merge_data[["A-coref", "B-coref"]]

Y.columns = ["A","B"]

Y["A"] = Y["A"].astype(int)

Y["B"] = Y["B"].astype(int)

Y["NEITHER"] = 1- (Y["A"] + Y["B"])
from sklearn import *

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

x1, x2, y1, y2 = model_selection.train_test_split(feature.fillna(-1), Y, test_size=0.2, random_state=1)

x1.head()

x2.head()

y2
scaler = StandardScaler()

x1 = scaler.fit_transform(x1)

x2 = scaler.transform(x2)

model = multiclass.OneVsRestClassifier(ensemble.RandomForestClassifier(max_depth = 7, n_estimators=1000, random_state=33))

# model = multiclass.OneVsRestClassifier(ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=100, random_state=33))



# param_dist = {'objective': 'binary:logistic', 'max_depth': 1, 'n_estimators':1000, 'num_round':1000, 'eval_metric': 'logloss'}

# model = multiclass.OneVsRestClassifier(xgb.XGBClassifier(**param_dist))



model.fit(x1, y1)

print('log_loss', metrics.log_loss(y2, model.predict_proba(x2)))
final_test = pd.read_csv("../input/gendered-pronoun-resolution/test_stage_2.tsv", sep = "\t")

feature = get_features(final_test)

print(feature)





feature = feature.fillna(-1)

# feature = scaler.transform(feature)

print(feature)



Y = pd.DataFrame(model.predict_proba(feature).tolist(), columns=["A","B", "NEITHER"])

r = final_test[["ID"]]

submission = pd.concat([r,Y], axis = 1)
print(submission)

submission.to_csv('submission.csv', index=False)