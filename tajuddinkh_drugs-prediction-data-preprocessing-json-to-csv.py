# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.offline as ply

import plotly.graph_objs as go

import seaborn as sns

import sklearn.metrics.base

import warnings

from collections import Counter

from pandas.io.json import json_normalize

from scipy.sparse import csr_matrix

from sklearn.decomposition import LatentDirichletAllocation, PCA

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, f1_score

from sklearn.model_selection import (StratifiedKFold, GridSearchCV, train_test_split,

                                     cross_val_predict)

ply.init_notebook_mode(connected=True)

%matplotlib inline
unfiltered_data = pd.read_json('/kaggle/input/prescriptionbasedprediction/roam_prescription_based_prediction.jsonl', lines=True)


data = unfiltered_data[unfiltered_data.cms_prescription_counts.apply(lambda x: len(x.keys())) >= 50]

print(data)
# Filter out rows for providers with <50 unique prescribed drugs. Then, remove providers that

# correspond to specialties with <50 providers in the filtered dataset.

data = unfiltered_data[unfiltered_data.cms_prescription_counts.apply(lambda x: len(x.keys())) >= 50]

specialty_counts = Counter(data.provider_variables.apply(lambda x: x['specialty']))

specialties_to_ignore = set(

    specialty for specialty, _ in filter(lambda x: x[1] < 50, specialty_counts.items()))

data = data[data.provider_variables.apply(lambda x: x['specialty'] not in specialties_to_ignore)]

data.head()
# prescribed drug names

cms = data.cms_prescription_counts.apply(lambda x: list(x.keys()));
# Expand the provider variables into a data frame.

# provider variables are doctors who prescribed it



provider_variables = json_normalize(data=data.provider_variables)
cms_head = cms.head()

prov_head = provider_variables.head()



#  taken only "speciality" and "years of practicce" features

prov_req_head = prov_head.iloc[:, [2,3]];

cms_req_head = cms_head.apply(lambda x: ', '.join(x)).to_frame()



# reset the index otherwise joining the dataframes of drugs and providers will show shifted data with NaN values

cms_req_head = cms_req_head.reset_index()



# removes the index column

cms_req_head = cms_req_head.iloc[:, 1]


print(prov_req_head)


print(cms_req_head)


# Join the drugs and providers data frames

df_records = pd.concat([prov_req_head, cms_req_head], axis=1).iloc[:,1:3]
df_records
unfiltered_cms = unfiltered_data.cms_prescription_counts.apply(lambda x: list(x.keys()))

unfiltered_cms = unfiltered_cms.apply(lambda x: ', '.join(x)).to_frame()

unfiltered_provider_variables = json_normalize(data=unfiltered_data.provider_variables)

medicine_prescription_records = pd.concat([unfiltered_provider_variables.iloc[:,  [2,3]], unfiltered_cms.reset_index().iloc[:, 1]], axis=1)
medicine_prescription_records