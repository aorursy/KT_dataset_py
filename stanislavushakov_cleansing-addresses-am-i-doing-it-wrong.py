# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder



X = pd.read_json("../input/train.json")

X_test = pd.read_json("../input/test.json")
street_encoder = LabelEncoder()

street_encoder.fit(list(X['display_address']) + list(X_test['display_address']))
X['display_address'].head(15)
def normalize_address(X, column):

    print("Before: {0}".format(len(X[column].unique())))

    substitution = [('west', 'w'), ('east', 'e'), ('south', 's'), ('north', 'n'),

                    ('1st', '1'), ('1th', '1'), ('2nd', '2'), ('2th', '2'),

                    ('3rd', '3'), ('3th', '3'), ('4th', '4'), ('5th', '5'),

                    ('6th', '6'), ('7th', '7'), ('8th', '8'), ('9th', '9'),

                    ('0th', '0'),

                    ('street', 'st'), ('str', 'st'),

                    ('avenue', 'av'), ('ave', 'av'),

                    ('place', 'pl'), ('boulevard', 'blvd'), ('road', 'rd'),

                    ('first', '1'), ('second', '2'), ('third', '3'),

                    ('fourth', '4'), ('fifth', '5'), ('sixth', '6'),

                    ('seventh', '7'), ('eighth', '8'), ('nineth', '9'),

                    ('tenth', '10'),                    

                    (',', ''), ('.', '')]

    

    def apply_normalization(s):

        for subst in substitution:

            s = s.lower().replace(subst[0], subst[1])

        s = s.strip()

        

        return s

        

    X[column] = X[column].apply(apply_normalization)

    print("After: {0}".format(len(X[column].unique())))
normalize_address(X, 'display_address')   

normalize_address(X_test, 'display_address')

normalize_address(X, 'street_address')   

normalize_address(X_test, 'street_address')
X['display_address'].head(15)