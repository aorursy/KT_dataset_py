!pip install -q imbalanced-learn

import numpy as np
import operator
import string
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN

from sklearn import manifold
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split

# Estimadores que vamos testar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# utilitários para plots

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

plt.style.use('ggplot')
sns.set_style("whitegrid")
sns.despine()

def plot_bar(df, title, filename):    
    p = (
        'Set2', 
        'Paired', 
        'colorblind', 
        'husl',
        'Set1', 
        'coolwarm', 
        'RdYlGn', 
        #'spectral'
    )
    color = sns.color_palette(np.random.choice(p), len(df))
    bar   = df.plot(kind='barh',
                    title=title,
                    fontsize=8,
                    figsize=(12,8),
                    stacked=False,
                    width=1,
                    color=color,
    )

    bar.figure.savefig(filename)

    plt.show()

def plot_top_crimes(df, column, title, fname, items=0):
    try:        
        by_col         = df.groupby(column)
        col_freq       = by_col.size()
        col_freq.index = col_freq.index.map(string.capwords)
        col_freq.sort_values(ascending=True, inplace=True)
        plot_bar(col_freq[slice(-1, - items, -1)], title, fname)
    except Exception:
        plot_bar(df, title, fname)
train_data = pd.read_csv("../input/train.csv", parse_dates =['Dates'])
test_data = pd.read_csv("../input/test.csv", parse_dates =['Dates'])
print('Shape dos dados de treino:',train_data.shape)
print('Shape dos dados de teste :',test_data.shape)
train_data.head(6)
# separa as datas em ano, mês, dia, hora, minuto e segundo.
# cada parte da data em uma coluna separada. Isto aumenta a quantidade de features

for x in [train_data, test_data]: 
    x['years'] = x['Dates'].dt.year
    x['months'] = x['Dates'].dt.month
    x['days'] = x['Dates'].dt.day
    x['hours'] = x['Dates'].dt.hour
    x['minutes'] = x['Dates'].dt.minute
train_data['XY'] = train_data.X * train_data.Y
test_data['XY'] = test_data.X * test_data.Y
plot_top_crimes(train_data, 'Category', 'Por categoria', 'category.png')
# quantidade de crimes associado à cada uma das categorias
print(train_data.Category.value_counts())
plot_top_crimes(train_data, 'Address', 'Principais localizações de ocorrências',  'location.png', items=50)
print(train_data.Address.value_counts())
plot_top_crimes(train_data, 'PdDistrict', 'Departamentos com mais atividades',  'police.png')
# quantidade de incidentes associada à cada distrito policial
print(train_data.PdDistrict.value_counts())
fig, ((axis1,axis2)) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(15,4)

sns.countplot(data=train_data, x='days', ax=axis1)
sns.countplot(data=train_data, x='hours', ax=axis2)
plt.show()
addr = train_data['Address'].apply(lambda x: ' '.join(x.split(' ')[-2:]))

year_count=addr.value_counts().reset_index().sort_values(by='index').head(10)
year_count.columns=['addr','Count']
# Create a trace
tag = (np.array(year_count.addr))
sizes = (np.array((year_count['Count'] / year_count['Count'].sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Endereços com mais incidentes')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Inncidentes")

data=[]
for i in range(2003,2015):
    year=train_data[train_data['years']==i]
    year_count=year['months'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['months','Count']
    trace = go.Scatter(
    x = year_count.months,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')
m = folium.Map(
    location=[train_data.Y.mean(), train_data.X.mean()],
    tiles='Cartodb Positron',
    zoom_start=13
)

marker_cluster = MarkerCluster(
    name='Locais de crimes em San Francisco',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(1000):
    location = train_data.Y.values[k], train_data.X.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = train_data.Address.values[k]
    #popup = train_data.Address.apply(lambda x: ' '.join(x.split(' ')[-2:])).values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("cluster.html")

m
new=train_data[train_data['Category']=='LARCENY/THEFT']
M= folium.Map(location=[train_data.Y.mean(), train_data.X.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 13) 

heat_data = [[[row['Y'],row['X']] 
                for index, row in new.head(1000).iterrows()] 
                 for i in range(0,11)]

hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(M)

hm.save('heatmap.html')

M
# correlações entre as variáveis
train_data.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('tab20c'), axis=1)
def street_addr(x):
    street=x.split(' ')
    return (''.join(street[-1]))

train_data['Address_Type'] = train_data['Address'].apply(lambda x:street_addr(x))
test_data['Address_Type'] = test_data['Address'].apply(lambda x:street_addr(x))

for x in [train_data,test_data]:
    x['is_street'] = (x['Address_Type'] == 'ST')
    x['is_avenue'] = (x['Address_Type'] == 'AV')

train_data['is_street'] = train_data['is_street'].apply(lambda x:int(x))
train_data['is_avenue'] = train_data['is_avenue'].apply(lambda x:int(x))

test_data['is_avenue'] = test_data['is_avenue'].apply(lambda x:int(x))
test_data['is_street'] = test_data['is_street'].apply(lambda x:int(x))
def is_block(x):
    if 'Block' in x:
        return 1
    else:
        return 0

train_data['is_block'] = train_data['Address'].apply(lambda x:is_block(x)) 
test_data['is_block'] = test_data['Address'].apply(lambda x:is_block(x)) 
train_data.head(20)
category = LabelEncoder()
train_data['Category'] = category.fit_transform(train_data.Category)
# codifica outras features categoricas, incluindo-as como novas colunas no dataframe
feature_cols =['DayOfWeek', 'PdDistrict']
train_data = pd.get_dummies(train_data, columns=feature_cols)
test_data = pd.get_dummies(test_data, columns=feature_cols)
# Nós não precisaremos das colunas abaixo, motivo pelo qual irems descartá-las.
train_data = train_data.drop(['Dates', 'Address', 'Address_Type', 'Resolution'], axis = 1)
train_data = train_data.drop(['Descript'], axis = 1)
test_data = test_data.drop(['Address','Address_Type', 'Dates'], axis = 1)
train_data.head(5)
test_data.head(5)
feature_cols = [x for x in train_data if x!='Category']
X = train_data[feature_cols]
y = train_data['Category']
X_train, x_test,y_train, y_test = train_test_split(X, y)
del X
del y

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(x_test)

normalizer = Normalizer()
X_train = normalizer.fit_transform(X_train)
x_test = normalizer.transform(x_test)

#ros = RandomOverSampler(random_state=42, sampling_strategy='not majority')
#X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

#sm = SMOTE(random_state=42, k_neighbors=3)
#X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

#ada = ADASYN(random_state=42, n_neighbors=4)
#X_resampled, y_resampled = ada.fit_resample(X_train, y_train)

#from collections import Counter

#print('shape do dataset original %s' % Counter(y_train))
#print('shape do dataset com oversampling %s' % Counter(y_resampled))

#print(sorted(Counter(y_resampled).items()))
random_forest = RandomForestClassifier(n_estimators=100, max_depth=23)
random_forest.fit(X_train, y_train.ravel())
#random_forest.fit(X_resampled, y_resampled)
pred = random_forest.predict(x_test)
print("accuracy_score: ", accuracy_score(pred,y_test))
print("f1_score_weighted", f1_score(pred,y_test, average='weighted'))
#X_test =test_data.drop(['Id'], axis = 1)
predicted_sub = random_forest.predict_proba(test_data.drop(['Id'], axis = 1))
submission_results = pd.DataFrame(predicted_sub, columns=category.classes_)
submission_results.to_csv('submission.csv', index_label = 'Id')