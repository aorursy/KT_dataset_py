from IPython.display import YouTubeVideo
YouTubeVideo('hHkSn0r5mdM',width=640, height=480)   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import folium 
from folium import plugins
import gc
import os
print(os.listdir("../input"))
from sklearn import preprocessing
colormap = plt.cm.terrain
df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
df_2018=pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

df_2017_raw = df_2017.copy()
df_2018_raw = df_2018.copy()
for column in df_2017:
    df_2017[column] = df_2017[column].astype('category')
    df_2017[column] = df_2017[column].cat.codes
for column in df_2018:
    df_2018[column] = df_2018[column].astype('category')
    df_2018[column] = df_2018[column].cat.codes
print(df_2017.shape)
print(df_2018.shape)
df_2017_mov = df_2017.copy()
df_2018_mov = df_2018.copy()

f, ax = plt.subplots(figsize=(12,12)) 
ax.xaxis.label.set_color('black')
g = sns.heatmap(df_2017.corr(),  cmap=colormap,  annot=False)
ax.set_title('Kaggle ML & DS Survey Challenge 2017 Pearson')
plt.show()
f, ax = plt.subplots(figsize=(12,12)) 
ax.xaxis.label.set_color('black')
g = sns.heatmap(df_2018.corr(), cmap=colormap, annot=False)
ax.set_title('Kaggle ML & DS Survey Challenge 2018 Pearson')
plt.show()
df_2017_mov=df_2017.apply(pd.to_numeric)
df_2017_mov = df_2017_mov.dropna()
x = df_2017_mov.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_2017_mov = pd.DataFrame(x_scaled)

df_2018_mov=df_2018.apply(pd.to_numeric)
df_2018_mov = df_2018_mov.dropna()
x = df_2018_mov.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_2018_mov = pd.DataFrame(x_scaled)
f, ax = plt.subplots(figsize=(12,12)) 
ax.xaxis.label.set_color('black')
g = sns.heatmap(df_2017_mov.T,vmax=1,  cmap=colormap,  annot=False)
ax.set_xlabel('Responses')
ax.set_ylabel('Question nr.')
ax.set_title('Kaggle ML & DS Survey Challenge 2017 Heatmap')
plt.show()
f, ax = plt.subplots(figsize=(12,12)) 
ax.xaxis.label.set_color('black')
g = sns.heatmap(df_2018_mov.T,vmax=1,  cmap=colormap,  annot=False)
ax.set_xlabel('Responses')
ax.set_ylabel('Question nr.')
ax.set_title('Kaggle ML & DS Survey Challenge 2018 Heatmap')
plt.show()
df_2018 = df_2018.sort_values(by=['Time from Start to Finish (seconds)'])
df_2018_mov=df_2018.apply(pd.to_numeric)
df_2018_mov = df_2018_mov.dropna()
x = df_2018_mov.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_2018_mov = pd.DataFrame(x_scaled)
f, ax = plt.subplots(figsize=(12,12)) 
ax.xaxis.label.set_color('black')
g = sns.heatmap(df_2018_mov.T,vmax=1,  cmap=colormap,  annot=False)
ax.set_xlabel('Responses')
ax.set_ylabel('Question nr.')
ax.set_title('Kaggle ML & DS Survey Challenge 2018 Heatmap')
plt.show()
print(df_2018_raw.Q37.unique())
df_2018_DataCamp = df_2018_raw.loc[df_2018_raw['Q37'] == 'DataCamp']
print('DataCamp : ' + str(df_2018_DataCamp.shape))
df_2018_KaggleLearn = df_2018_raw.loc[df_2018_raw['Q37'] == 'Kaggle Learn']
print('Kaggle Learn : ' + str(df_2018_KaggleLearn.shape))
print('Sum: ' + str(df_2018_raw.shape))
for column in df_2018_DataCamp:
    df_2018_DataCamp[column] = df_2018_DataCamp[column].astype('category')
    df_2018_DataCamp[column] = df_2018_DataCamp[column].cat.codes
for column in df_2018_KaggleLearn:
    df_2018_KaggleLearn[column] = df_2018_KaggleLearn[column].astype('category')
    df_2018_KaggleLearn[column] = df_2018_KaggleLearn[column].cat.codes
    
df_2018_DataCamp = df_2018_DataCamp.apply(pd.to_numeric)
df_2018_DataCamp = df_2018_DataCamp.dropna()
x = df_2018_DataCamp.values 
min_max_scaler_D = preprocessing.MinMaxScaler()
x_scaled_D = min_max_scaler_D.fit_transform(x)
df_2018_DataCamp = pd.DataFrame(x_scaled_D)

df_2018_KaggleLearn = df_2018_KaggleLearn.apply(pd.to_numeric)
df_2018_KaggleLearn = df_2018_KaggleLearn.dropna()
x = df_2018_KaggleLearn.values 
min_max_scaler_K = preprocessing.MinMaxScaler()
x_scaled_K = min_max_scaler_K.fit_transform(x)
df_2018_KaggleLearn = pd.DataFrame(x_scaled_K)
f, ax = plt.subplots(figsize=(24, 12))

plt.subplot(1, 2, 1)
sns.heatmap(df_2018_DataCamp.T,annot=False,cmap=colormap) 
plt.title('DATACAMP', fontsize=18)

plt.subplot(1, 2, 2)
sns.heatmap(df_2018_KaggleLearn.T,annot=False,cmap=colormap) 
plt.title('KAGGLE LEARN', fontsize=18)

plt.show()
df_lang = df_2018_raw[['Time from Start to Finish (seconds)', 'Q1', 'Q3',
                       'Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 
                       'Q16_Part_4', 'Q16_Part_5', 'Q16_Part_6', 
                       'Q16_Part_7', 'Q16_Part_8', 'Q16_Part_9',
                       'Q16_Part_10', 'Q16_Part_11', 'Q16_Part_12',
                       'Q16_Part_13', 'Q16_Part_15']].copy()
df_lang.rename(columns={'Q1': 'Gender', 'Q3': 'Country',
                        'Q16_Part_1': 'Python', 'Q16_Part_2': 'R',
                        'Q16_Part_3': 'SQL', 'Q16_Part_4': 'Bash',
                        'Q16_Part_5': 'Java', 'Q16_Part_6': 'Javascript/Typescript',
                        'Q16_Part_7': 'Visual Basic/VBA', 'Q16_Part_8': 'C/C++',
                        'Q16_Part_9': 'MATLAB', 'Q16_Part_10': 'Scala',
                        'Q16_Part_11': 'Julia', 'Q16_Part_12': 'Go',
                        'Q16_Part_13': 'C#/.NET', 'Q16_Part_14': 'PHP',
                        'Q16_Part_15': 'Ruby', 'Q16_Part_16': 'SAS/STATA',
                                              
                       }, inplace=True)
df_lang.drop(df_lang.head(1).index, inplace=True)
df_lang.fillna(0, inplace=True)
df_lang['Time from Start to Finish (seconds)'] = pd.to_numeric(df_lang['Time from Start to Finish (seconds)'] )
train_df = df_lang
train_df['y'] = df_lang['Time from Start to Finish (seconds)']
for f in ['Gender','Country', 'Python', 'R', 'SQL', 'Bash','Java', 'Javascript/Typescript', 'Visual Basic/VBA', 'C/C++', 'MATLAB', 'Scala', 'Julia', 'Go','C#/.NET', 'Ruby']:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
train_df = train_df.drop(['Time from Start to Finish (seconds)'], axis=1)


f, ax = plt.subplots(figsize=(12,12)) 
ax.xaxis.label.set_color('black')
g = sns.heatmap(train_df.corr(), cmap=colormap, annot=False)
ax.set_title('What programming languages do you use on a regular basis?')
plt.show()