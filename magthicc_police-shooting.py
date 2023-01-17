import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib inline
import pandas as pd
import numpy as np
import datetime
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree

gle = LabelEncoder()
data = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')
data = data.dropna()
data['date'] = pd.to_datetime(data['date'])
data.describe(include='all')
data['date'].dt.year.value_counts()
data['date'].dt.month.value_counts()
date_df = data['date'].dt.to_period('M').value_counts()
date_df = date_df.rename_axis('period').reset_index(name='counts')
date_df = date_df.sort_values(by=['period'])
date_df.plot('period','counts')
data['manner_of_death'].value_counts()
data['armed'].value_counts()
armed_df = data['armed'].value_counts()
armed_df = armed_df.rename_axis('armed').reset_index(name='counts')
armed_df[armed_df['armed'].str.contains('gun')]
data['age'].plot.hist()
data['gender'].value_counts()
data['race'].value_counts()
data['city'].value_counts()
data['state'].value_counts().plot()
data['signs_of_mental_illness'].value_counts()
data['threat_level'].value_counts()
data['flee'].value_counts()
data['body_camera'].value_counts()
gle = LabelEncoder()
data['tasered'] = data['manner_of_death'].apply(lambda x: '1' if x == 'shot and Tasered' else '0')
data['armed_gun'] = data['armed'].str.contains('gun').apply(lambda x: '1' if x == True else '0')
data['armed_vehicle'] = data['armed'].str.contains('vehicle').apply(lambda x: '1' if x == True else '0')
data['unarmed'] = data['armed'].apply(lambda x: '1' if x == 'unarmed' else '0')
data['male'] = data['gender'].apply(lambda x: '1' if x == 'M' else '0')
data['white'] = data['race'].apply(lambda x: '1' if x == 'W' else '0')
data['black'] = data['race'].apply(lambda x: '1' if x == 'B' else '0')
data['hispanic'] = data['race'].apply(lambda x: '1' if x == 'H' else '0')
data['asian'] = data['race'].apply(lambda x: '1' if x == 'A' else '0')
data['native'] = data['race'].apply(lambda x: '1' if x == 'N' else '0')
data['other'] = data['race'].apply(lambda x: '1' if x == 'O' else '0')
data['mentally_ill'] = data['signs_of_mental_illness'].apply(lambda x: '1' if x == True else '0')
data['attacking'] = data['threat_level'].apply(lambda x: '1' if x == 'attack' else '0')
data['flee_foot'] = data['flee'].apply(lambda x: '1' if x == 'Foot' else '0')
data['flee_car'] = data['flee'].apply(lambda x: '1' if x == 'Car' else '0')
data['bodycam'] = data['body_camera'].apply(lambda x: '1' if x == True else '0')

data['tasered'] = pd.to_numeric(data['tasered'], errors='coerce')
data['armed_gun'] = pd.to_numeric(data['armed_gun'], errors='coerce')
data['armed_vehicle'] = pd.to_numeric(data['armed_vehicle'], errors='coerce')
data['unarmed'] = pd.to_numeric(data['unarmed'], errors='coerce')
data['male'] = pd.to_numeric(data['male'], errors='coerce')
data['white'] = pd.to_numeric(data['white'], errors='coerce')
data['black'] = pd.to_numeric(data['black'], errors='coerce')
data['hispanic'] = pd.to_numeric(data['hispanic'], errors='coerce')
data['asian'] = pd.to_numeric(data['asian'], errors='coerce')
data['native'] = pd.to_numeric(data['native'], errors='coerce')
data['other'] = pd.to_numeric(data['other'], errors='coerce')
data['mentally_ill'] = pd.to_numeric(data['mentally_ill'], errors='coerce')
data['attacking'] = pd.to_numeric(data['attacking'], errors='coerce')
data['flee_foot'] = pd.to_numeric(data['flee_foot'], errors='coerce')
data['flee_car'] = pd.to_numeric(data['flee_car'], errors='coerce')
data['bodycam'] = pd.to_numeric(data['bodycam'], errors='coerce')

data
data_sub1 = data.drop(columns=[ 'id',
                                'name',
                                'manner_of_death',
                                'armed',
                                'gender',
                                'race',
                                'city',
                                'state',
                                'signs_of_mental_illness',
                                'threat_level',
                                'flee',
                                'body_camera'])

corrMatrix = data_sub1.corr()
corrMatrix.style.format("{:.0%}")
data_sub2 = data_sub1[['age','armed_gun','unarmed','male','black','mentally_ill','flee_foot']]

data_sub2
#define Decision Tree
dt = DecisionTreeClassifier(max_depth = 4)
#Define input vectors
#X is the features in this dataset
features = data[['age','armed_gun','unarmed','male','black','mentally_ill']]
X = np.array(features)
#Y is the vector with our Target Variables
Y = data['flee_foot'].values

dt.fit(X, Y)

feat_names = data[['age','armed_gun','unarmed','male','black','mentally_ill']].columns
targ_names = ['no run','run']

dt_visual = export_graphviz(dt,out_file=None,feature_names=feat_names,class_names=targ_names,   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dt_visual)
graph
#define Decision Tree
dt = DecisionTreeClassifier(max_depth = 4)
#Define input vectors
#X is the features in this dataset
features = data[['age','unarmed','mentally_ill','flee_foot','bodycam']]
X = np.array(features)
#Y is the vector with our Target Variables
Y = data['race'].values

dt.fit(X, Y)

feat_names = data[['age','unarmed','mentally_ill','flee_foot','bodycam']].columns
targ_names = ['A','B','H','N','O','W']

dt_visual = export_graphviz(dt,out_file=None,feature_names=feat_names,class_names=targ_names,   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dt_visual)
graph