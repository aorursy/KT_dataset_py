import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/drive/My Drive/ml_data/US_Accidents_June20.csv')

df.info()

df['Source'].value_counts()
df.groupby(['Source'])['Severity'].value_counts()
delete = ['ID','End_Time','End_Lat','End_Lng','Description','Description','TMC','Weather_Timestamp','Weather_Condition']
df.drop(delete,axis=1,inplace=True)
df
print(df.columns)
category = ['County',             
   'State' ,                  
   'Zipcode' ,                
   'Country' ,                
   'Timezone',               
   'Airport_Code',            
   'Precipitation(in)', 'Amenity', 'Bump', 'Crossing',
       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
       'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
       'Astronomical_Twilight']
for i in category:  
  print(i,df[i].unique().size)
# удалим столбцы с одним значением

df1 = df.drop(['Country','Turning_Loop'],axis=1)
df1.info()
plt.figure(figsize=(30,20))
sns.heatmap(df1.isnull(), cbar=False)
plt.show()
def missing_percentage(df): 
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(df1)
import missingno as msno
msno.heatmap(df1)
drop_columns = ['Number','Precipitation(in)','Wind_Chill(F)']
df1.drop(df1[drop_columns], inplace=True, axis=1)
df1.columns
df1 = df1.dropna()
missing_percentage(df1)
df1.shape
df['Wind_Direction'].unique()
# необходимо преобразовать данные
# по другому преобразовать не получится (самое корректное), потом факторайзом все
df.loc[df['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
df.loc[(df['Wind_Direction']=='West')|(df['Wind_Direction']=='WSW')|(df['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
df.loc[(df['Wind_Direction']=='South')|(df['Wind_Direction']=='SSW')|(df['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
df.loc[(df['Wind_Direction']=='North')|(df['Wind_Direction']=='NNW')|(df['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
df.loc[(df['Wind_Direction']=='East')|(df['Wind_Direction']=='ESE')|(df['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
df.loc[df['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'
print("Wind Direction after simplification: ", df['Wind_Direction'].unique())
df1['year'] = df1['Start_Time'].str.split("-").str[0].astype('int64')
df1['day'] = df1['Start_Time'].str.split("-").str[1].astype('int64')
from datetime import datetime
df1['Start_Time'] = pd.to_datetime(df1['Start_Time'])
df1['month'] = df1['Start_Time'].dt.month
df1['hour'] = df1['Start_Time'].dt.hour
df1=df1.drop(['Start_Time'],axis=1)
df1.head(10)
def factorize (dataset):
    result = dataset.copy() 
    for cols in result.columns:
        if result.dtypes[cols] == np.bool:
            result[cols] = pd.factorize(result[cols],sort=True)[0]
    return result
df1 = factorize(df1)
df1.info()

print(df1['Sunrise_Sunset'].unique(),'\n'
,df1['Civil_Twilight'].unique(),'\n',
df1['Nautical_Twilight'].unique(),'\n',
df1['Astronomical_Twilight'].unique())
mapping = {'Day':1,'Night':0}
df1['Sunrise_Sunset'] = df1['Sunrise_Sunset'].map(mapping).astype('int64')
df1['Civil_Twilight'] = df1['Civil_Twilight'].map(mapping).astype('int64')
df1['Nautical_Twilight'] = df1['Nautical_Twilight'].map(mapping).astype('int64')
df1['Astronomical_Twilight'] = df1['Astronomical_Twilight'].map(mapping).astype('int64')
df1 = df1.loc[df1['Side'] != ' ']

mapping = {'R':1,'L':0}
df1['Side'] = df1['Side'].map(mapping).astype('int64')
print(df1['Side'].unique())
df1['Severity'] = df1['Severity'].map(lambda x: True if x == 4 else False)
mapping = {True:1,False:0}
df1['Severity'] = df1['Severity'].map(mapping).astype('int64')
df1['Severity'].value_counts()
df1.head()
plt.figure(figsize =(10,10))
sns.heatmap(df1.corr(),annot= True,cmap = 'rocket')
import plotly
import plotly.graph_objs as go
data_sever = df1.sample(n=10000)
# если брать больше 1000 значений то plotly не запускается и происходит переподключение
fig = go.Figure(data=go.Scattergeo(
        lon = data_sever['Start_Lng'],
        lat = data_sever['Start_Lat'],
        text = data_sever['City'],
        mode = 'markers',
        marker = dict(
            size = data_sever['Severity'],
            opacity = 0.8,
            reversescale = True,
            autocolorscale = True,
            symbol = 'circle',
            line = dict(
                width=0.5,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'Reds',
            cmin = data_sever['Severity'].max(),
        color = data_sever['Severity'],
        cmax = 1,
            colorbar_title="Severity"
        )

        ))

fig.update_layout(
        title = 'Тяжесть аварии',
        geo_scope='usa',
    )
fig.show()
sns.countplot(df1[df1['State'].isin(df1['State'].value_counts().head().index)]['State'])

# тут был цикл
feat = df1[[ 'Amenity' ,                 
   'Bump'       ,              
   'Crossing'   ,              
   'Junction'   ,              
   'Give_Way'   ,              
   'No_Exit'    ,              
   'Railway'    ,              
   'Roundabout' ,              
   'Station'    ,              
   'Stop'        ,             
   'Traffic_Calming'  ,        
   'Traffic_Signal'   ,        
   'Sunrise_Sunset'   ,        
   'Civil_Twilight'    ,       
   'Nautical_Twilight'  ,      
   'Astronomical_Twilight',
   'Distance(mi)',
   'Temperature(F)',
   'Humidity(%)','Pressure(in)',
   'Visibility(mi)',
   'Wind_Speed(mph)',
   'year',
'day',
'month',
'hour',
'Severity']]
sns.pairplot(feat);
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(13, 5))
sns.countplot(x='hour', hue='Severity', data=df1 ,palette="Set1")
plt.show()
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(13, 5))
sns.countplot(x='day', hue='Severity', data=df1 ,palette="Set1")
plt.show()
plt.figure(figsize=(25,5))
chart = sns.countplot(x='State', hue='Severity', 
                      data=df1 ,palette="Set2", order=df1['State'].value_counts().index)
plt.show()
df1 = df1.drop(['County','Zipcode','Timezone','Airport_Code','Street'],axis=1)
from sklearn import preprocessing

def into_integer (dataset):
    result = dataset.copy() #делаем копию
    encoders = {}
    for res in result.columns:  #пройдемся по колонкам
        if result.dtypes[res] == np.object:  #если тип колонки объект
            encoders[res] = preprocessing.LabelEncoder()  #сздаем кодировщик
            result[res] = encoders[res].fit_transform (result[res]) #применяем кодировщик к столбцу и перезаписываем столбец
    return result,encoders
df,encoders = into_integer(df1)
df.head()
df.info()
df['Source'].value_counts()
df['Source'].value_counts()
df_source = df.groupby(['Severity','Source']).size().reset_index().pivot(\
    columns='Severity', index='Source', values=0)
df_source.plot(kind='bar', stacked=True);
df_bing = df [df['Source'] == 0]
df_bing.head()
df_MapQuest = df [df['Source'] == 1]
df_MapQuest.head()
df_bing = df_bing.drop(['Source'],axis=1)
df_MapQuest = df_MapQuest.drop(['Source'],axis=1)
print(df_bing['Severity'].value_counts(),df_MapQuest['Severity'].value_counts())

print('Процент данных показывающих целевую группу Bing ', round(88242/len(df_bing['Severity']),2))
sns.countplot(df_bing['Severity'])
print('Процент данных показывающих целевую группу MapQuest ', round(6293/len(df_MapQuest['Severity']),3))
sns.countplot(df_MapQuest['Severity'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df_bing)

X_bing = df_bing.drop(['Severity'],axis=1)
y_bing = df_bing['Severity']

print(X_bing.shape,y_bing.shape)
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_bing, y_bing, test_size=0.2)
pd.Series(y_test).value_counts()
from imblearn.under_sampling import TomekLinks

augm = TomekLinks(sampling_strategy='majority')
X_train_augm, y_train_augm = augm.fit_resample(np.array(X_train), np.array(y_train))
print('С аугментацией',pd.Series(y_train_augm).value_counts(),'\n','Без аугментации',pd.Series(y_train).value_counts())
X_test = np.array(X_test)
y_test = np.array(y_test)

from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

model_tree = DecisionTreeClassifier(max_depth=7)
model_tree.fit(X_train_augm, y_train_augm) # обучение
a = model_tree.predict(X_test) # предсказание

print ("AUC-ROC (test) = ", roc_auc_score(y_test, a))
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(model_tree.fit(X_train_augm, y_train_augm), out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('crash.png')
Image(graph.create_png())

# далее - (X, y) - обучение, (X2, y2) - контроль
# модель - здесь (для контраста) рассмотрим регрессор
model =  RandomForestClassifier(n_estimators=10 ,
                               oob_score=True,
                               random_state=42)
model.fit(X_train_augm, y_train_augm) # обучение
a = model.predict(X_test) # предсказание

print ("AUC-ROC (test) = ", roc_auc_score(y_test, a))
feats = {}
for feature, importance in zip(df.columns, model.feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})
sns.set(font_scale = 5)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
fig, ax = plt.subplots()
fig.set_size_inches(30,15)
sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
plt.xlabel('Важность', fontsize=25, weight = 'bold')
plt.ylabel('Значения', fontsize=25, weight = 'bold')
plt.title('Важность признаков', fontsize=25, weight = 'bold')
display(plt.show())
display(importances)
from sklearn.decomposition import PCA
pca_test = PCA(n_components=20)
pca_test.fit(X_train_augm)
sns.set(style='whitegrid')
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('номер компонент')
plt.ylabel('размер объясненной дисперсии')
plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
display(plt.show())
evr = pca_test.explained_variance_ratio_
cvr = np.cumsum(pca_test.explained_variance_ratio_)
pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cvr
pca_df['Explained Variance Ratio'] = evr
display(pca_df.head(10))
pca = PCA(n_components=10)
pca.fit(X_train_augm)
X_train_scaled_pca = pca.transform(X_train_augm)
X_test_scaled_pca = pca.transform(X_test)
model = RandomForestClassifier(n_estimators=10 ,
                               oob_score=True,
                               random_state=42,n_jobs=-1)
model.fit(X_train_scaled_pca, y_train_augm)

a = model.predict(X_test_scaled_pca) # предсказание
print ("AUC-ROC (test) = ", roc_auc_score(y_test, a))
from sklearn.model_selection import RandomizedSearchCV 
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 30, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(start = 5, stop = 20,num=5)]
min_samples_split = [int(x) for x in np.linspace(start = 5, stop = 15, num = 5)]
min_samples_leaf = [int(x) for x in np.linspace(start = 5, stop = 15, num = 5)]
bootstrap = [True, False]
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
model = RandomForestClassifier()
rs = RandomizedSearchCV(model,param_dist,cv=3,n_jobs=-1,n_iter=10,random_state=42) 
#grid(RandomForestClassifier(),
                        #param_dist,
                        #cv = 3, 
                       # n_jobs=-1,
                    #random_state=42)
rs.fit(X_train_augm, y_train_augm)
rs.best_params_
rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)

rs_df.head(10)
fig, axs = plt.subplots(ncols=3, nrows=2)
sns.set(style="whitegrid", color_codes=True, font_scale = 2)
fig.set_size_inches(25,20)
sns.barplot(x='param_n_estimators', y='mean_test_score', data=rs_df, ax=axs[0,0], color='lightgrey')
axs[0,0].set_ylim([.83,.93]),axs[0,0].set_title(label = 'n_estimators', size=30, weight='bold')
sns.barplot(x='param_min_samples_split', y='mean_test_score', data=rs_df, ax=axs[0,1], color='coral')
axs[0,1].set_ylim([.85,.93]),axs[0,1].set_title(label = 'min_samples_split', size=30, weight='bold')
sns.barplot(x='param_min_samples_leaf', y='mean_test_score', data=rs_df, ax=axs[0,2], color='lightgreen')
axs[0,2].set_ylim([.80,.93]),axs[0,2].set_title(label = 'min_samples_leaf', size=30, weight='bold')
sns.barplot(x='param_max_features', y='mean_test_score', data=rs_df, ax=axs[1,0], color='wheat')
axs[1,0].set_ylim([.88,.92]),axs[1,0].set_title(label = 'max_features', size=30, weight='bold')
sns.barplot(x='param_max_depth', y='mean_test_score', data=rs_df, ax=axs[1,1], color='lightpink')
axs[1,1].set_ylim([.80,.93]),axs[1,1].set_title(label = 'max_depth', size=30, weight='bold')
sns.barplot(x='param_bootstrap',y='mean_test_score', data=rs_df, ax=axs[1,2], color='skyblue')
axs[1,2].set_ylim([.88,.92])
axs[1,2].set_title(label = 'bootstrap', size=30, weight='bold')
plt.show()
model = RandomForestClassifier(bootstrap = False,
 max_depth= 16,
 max_features= 'log2',
 min_samples_leaf= 12,
 min_samples_split= 10,
 n_estimators= 23)
model.fit(X_train_augm, y_train_augm)

a = model.predict(X_test) # предсказание
print ("AUC-ROC (test) = ", roc_auc_score(y_test, a))
