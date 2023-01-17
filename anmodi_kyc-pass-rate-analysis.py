from mpl_toolkits.mplot3d import Axes3D

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting



import seaborn as sns  # visualization tool

import json #for parse "properties" parameter



import warnings

warnings.filterwarnings('ignore')
# Function for maping datasets. 

#We will change all 'consider' value to 1, and 'clear' = 0 for **map_encode_docs** list 

def result_encode(features, dataset):

    for feature in features:

        dataset[feature] = dataset[feature].map(result_map)



# function for clean and prepare dataframe        

def prepare_data(dataframe):

    dataframe.created_at = pd.to_datetime(dataframe.created_at,

                                          errors='coerce',

                                          format='%Y-%m-%d %H:%M:%S')

    dataframe.fillna(np.nan)

    

map_encode_all = ['result_doc', 

                  "visual_authenticity_result_doc", 

                  "image_integrity_result",                   

                  'police_record_result', 

                  'compromised_document_result',

                  "face_detection_result", 

                  "image_quality_result",

                  "supported_document_result",

                  'conclusive_document_quality_result',

                  'colour_picture_result', 

                  'data_validation_result',

                  'data_consistency_result', 

                  'data_comparison_result',

                  'face_comparison_result', 

                  'facial_image_integrity_result',

                  'visual_authenticity_result_face',

                  'result_face']



result_map = {"clear": 0, 'unidentified': 1, "consider" : 1}
document = pd.read_csv('../input/kyc-challenge/doc_reports.csv', delimiter=',',index_col=0)

document.dataframeName = 'doc_reports.csv'

prepare_data(document)

document.head(2)
faces = pd.read_csv('../input/kyc-challenge/facial_similarity_reports.csv', delimiter=',',index_col=0)

faces.dataframeName = 'facial_similarity_reports.csv'

prepare_data(faces)

faces.head(2)
mb = pd.merge(document, faces, on='attempt_id', how='left', suffixes=('_doc', '_face'), validate='one_to_one')

mb = mb.drop([ 'user_id_face',  'created_at_face', 'sub_result', 'properties_face',], axis=1)
mb.head()
mb.result_doc.value_counts(normalize=True).plot.bar();
result_encode(map_encode_all, mb)

mb.fillna(0, inplace=True)
mb.head()
plt.rcParams['figure.figsize'] = (18, 7)

mb.groupby(pd.Grouper(key='created_at_doc', freq='D'))['user_id_doc'].count().plot();
data_corr  = mb.corr(method='pearson')

data_corr = data_corr.apply(lambda x: [y if y >= 0.3 else np.nan for y in x])

# correlation map plotting

f,ax = plt.subplots(figsize =(13,13))

sns.heatmap(data_corr, annot = True, linewidths = 5, fmt = '.3f', ax = ax, cmap='Reds', center=0.8)

plt.show()

print(data_corr.apply(lambda x: [y if y >= 0.3 else np.nan for y in x]))
data_corr
mb.groupby(pd.Grouper(key='created_at_doc', freq='D'))[map_encode_all].mean().plot();
suspected_params = ["image_integrity_result", 'facial_image_integrity_result', 'image_quality_result']
mb.groupby(pd.Grouper(key='created_at_doc', freq='D'))[suspected_params].mean().plot();


mb.properties_doc = mb.properties_doc.apply(lambda row: row.replace('None', "\"NaN\""))

mb['properties_doc'] = mb.properties_doc.apply(lambda x: x.strip("\'<>()").replace('\'', '\"'))

#cleaning

mb.properties_doc = mb.properties_doc.apply(lambda row: row.replace('None', "\"NaN\""))

mb['properties_doc'] = mb.properties_doc.apply(lambda x: x.strip("\'<>()").replace('\'', '\"'))

#loading

mb['properties_doc'] = mb['properties_doc'].apply(json.loads, strict=False)

#parsing

mb = mb.drop('properties_doc', 1).assign(**pd.DataFrame(mb.properties_doc.values.tolist()))

#and get dates from new columns

mb.date_of_expiry = pd.to_datetime(mb.date_of_expiry, errors='coerce', format='%Y-%m-%d')

mb.issuing_date = pd.to_datetime(mb.issuing_date, errors='coerce', format='%Y-%m')
mb = mb.set_index(keys='created_at_doc')

mb.head()
mb[(mb['result_doc'] == 1) & (mb['image_integrity_result'] == 1)]['issuing_country'].value_counts()[:10].plot(kind='bar');
mb[(mb['result_doc'] == 1) & (mb['image_integrity_result'] == 1)]['nationality'].value_counts()[:10].plot(kind='bar');
mb[(mb['result_doc'] == 1) & (mb['image_integrity_result'] == 1)]['document_type'].value_counts()[:10].plot(kind='bar');
attempts = mb.groupby(['user_id_doc'])['attempt_id'].count().reset_index(name='count')

attempts.groupby(['count'])['user_id_doc'].count()[:10].plot(kind='bar');

suspect_data = mb['2017-10-10':'2017-10-25']
suspect_data[suspect_data['image_integrity_result'] == 1]['issuing_country'].value_counts()[:10].plot(kind='bar');
suspect_data[suspect_data['image_integrity_result'] == 1]['nationality'].value_counts()[:10].plot(kind='bar');
suspect_data[suspect_data['image_integrity_result'] == 1]['document_type'].value_counts()[:10].plot(kind='bar');