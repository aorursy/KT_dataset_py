# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

import plotly.graph_objects as go

import seaborn as sns

import plotly.express as px



# Load libraries



from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import  linear_model

from sklearn.model_selection import KFold



from plotly.offline import init_notebook_mode, plot, iplot

import plotly as py

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object



















# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

database = pd.read_csv("../input/earthquake-database/database.csv",encoding='ISO-8859-1')

print(database)
df=pd.DataFrame(database)

df
df.info()
df['Date']
df[(df['Magnitude']>5) & (df['Date'])]
df.sort_values('Magnitude', axis = 0, ascending = False)
df.groupby('Date').size()
df['Date'] = df['Date'].astype(str)

df.groupby('Date').size()
df['Magnitude'].mode()

df['Magnitude'].std()
df.cov()
df.plot(x='Date', y='Magnitude', style='-')
#Korelasyon Gösterim

import seaborn as sns

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
df.isnull().sum().sum()

#Toplam kaç hücrede eksik değer (NaN ya da None) var?
#Özniteliklerin değer almadığı kaç satır var?

df.isnull().sum()
#Eksik değer tablosu

def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(

    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son

  

eksik_deger_tablosu(df)
#%70 üzerinde null değer içeren kolonları sil

tr = len(df) * .3

df.dropna(thresh = tr, axis = 1, inplace = True)



df
#Apply fonksiyonu kullanarak büyüklüğü 6 dan yüksek depremleri yeni öznitelik olarak ekle

def deprem_durumu(Magnitude):

    return (Magnitude >= 6.0)



df['Greater than 6'] = df['Magnitude'].apply(deprem_durumu)

df
#6 dan büyük müdür? veri bilgisini 0 ve 1lere çevirdik.



from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 

df['Greater than 6_Encoded']= label_encoder.fit_transform(df['Greater than 6'])



df


#Magnitude özniteliğini ölçeklendirmek istiyoruz

x = df[['Magnitude']].values.astype(float)



#Ölçeklendirme için MinMaxScaler fonksiyonunu kullanıyoruz.

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df['Magnitude2'] = pd.DataFrame(x_scaled)



df
#Quartile (Kartiller) ve IQR ile Aykırı Değer Tespiti



import seaborn as sns

sns.boxplot(x=df['Magnitude'])
Q1 = df.Magnitude.quantile(0.25)

Q2 = df.Magnitude.quantile(0.5)

Q3 = df.Magnitude.quantile(0.75)

Q4 = df.Magnitude.quantile(1)

IQR = Q3 - Q1



print("Q1-->", Q1)

print("Q3-->", Q3)

print("Q2-->", Q2)

print("Q4-->", Q4)

print("IQR-->", IQR)

print("Alt sınır: Q1 - 1.5 * IQR--->", Q1 - 1.5 * IQR)

print("Üst sınır: Q3 + 1.5 * IQR--->", Q3 + 1.5 * IQR)
df = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

df
from mpl_toolkits.basemap import Basemap



m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')



longitudes = df["Longitude"].tolist()

latitudes = df["Latitude"].tolist()

#m = Basemap(width=12000000,height=9000000,projection='lcc',

            #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)

x,y = m(longitudes,latitudes)
fig = plt.figure(figsize=(12,10))

plt.title("All affected areas")

m.plot(x, y, "o", markersize = 2, color = 'blue')

m.drawcoastlines()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
import datetime



# Extract year for filtering purpose

year = []

for index, row in df.iterrows():

    try:

        date = row['Date']

        date_time_obj = datetime.datetime.strptime(date, '%m/%d/%Y')

        y = date_time_obj.date().year

        year.append(y)

    except:

        year.append(-1)

print (year[:10])
#'Year' adında yeni öznitelik ekledik

df.insert(loc=1, column='Year', value=year)

df.head(5)
#yıllara göre deprem sıklığını gösteren grafik



df.Year.value_counts().plot(kind = "bar" , color = "red" , figsize = (30,10),fontsize = 20)

plt.xlabel("Year",fontsize=18,color="blue")

plt.ylabel("Frequency",fontsize=18,color="blue")

plt.show()
#en yüksek  şiddetli depremin bilgileri



filtre=df.Magnitude==df.Magnitude.max()

df[filtre]



#derinliği en çok olan depremin bilgileri

filtre=df.Depth==df.Depth.max()

df[filtre]

#derinliği en az olan depremin bilgileri

filtre=df.Depth==df.Depth.min()

df[filtre]

#şiddeti en yüksek olan depremin yılı zamanı ve bölge bilgisi.

df = df[['Date', 'Time', 'Latitude', 'Longitude', 'Magnitude']]

filtre=df.Magnitude==df.Magnitude.max()

df[filtre]
# en şiddetli deprem hangi bölgede yaşanmıştır?

df = df[['Latitude', 'Longitude', 'Magnitude']]

filtre=df.Magnitude==df.Magnitude.max()

df[filtre]



#dünya haritasında görselleştirme

from mpl_toolkits.basemap import Basemap



m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')



longitudes = df["Longitude"].tolist()

latitudes = df["Latitude"].tolist()

#m = Basemap(width=12000000,height=9000000,projection='lcc',

            #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)

x,y = m(longitudes,latitudes)
fig = plt.figure(figsize=(12,10))

plt.title("Etkilenen Tüm Alanlar")

m.plot(x, y, "o", markersize = 2, color = 'blue')

m.drawcoastlines()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
latitude_list=[]

longitude_list=[]

for row in df.Latitude:

     latitude_list.append(row)

for row in df.Longitude:

    longitude_list.append(row)

    
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
%matplotlib inline
earthquake_map = Basemap(projection='robin', lat_0=-90,lon_0=130,resolution='c', area_thresh=1000.0)
earthquake_map.drawcoastlines()

earthquake_map.drawcountries()

earthquake_map.drawmapboundary()

earthquake_map.bluemarble()

earthquake_map.drawstates()

earthquake_map.drawmeridians(np.arange(0, 360, 30))

earthquake_map.drawparallels(np.arange(-90, 90, 30))



x,y = earthquake_map(longitude_list, latitude_list)

earthquake_map.plot(x, y, 'ro', markersize=1)

plt.title("1965 - 2016 yılları arasında EarthQuakes, Rock Bursts & Nükleer Patlamaların gerçekleştiği yerler")

 

plt.show()
plt.hist(df['Magnitude'])

plt.xlabel(' Deprem Şiddeti')

plt.ylabel('Oluşum Sayısı')
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap 

#import matplotlib.pyplot as plt

import numpy as np

import string

import matplotlib.cm as cm



areas = [

    { 'label': 'İtalya',

      'llcrnrlat': 35.57580,

      'llcrnrlon': 6.67969,

      'urcrnrlat': 47.55336,

      'urcrnrlon': 19.33594},

    { 'label': 'Yunanistan',

      'llcrnrlat': 33.62262,

      'llcrnrlon': 18.01758,

      'urcrnrlat': 42.33317,

      'urcrnrlon': 29.17969},

    { 'label': 'Japonya',

      'llcrnrlat': 29.65822,

      'llcrnrlon': 127.79297,

      'urcrnrlat': 46.41419,

      'urcrnrlon': 151.08398},

    { 'label': 'Güneydoğu Asya',

      'llcrnrlat': -11.90095,

      'llcrnrlon': 92.02148,

      'urcrnrlat': 19.02967,

      'urcrnrlon': 130.51758},

]



fig = plt.figure()

fig.set_figheight(15)

fig.set_figwidth(15)



for i, a in enumerate(areas):

    print(i, a)

    ax = fig.add_subplot(100*len(areas) + 20 + i+1)

    m = Basemap(projection='cyl',

                llcrnrlat=a['llcrnrlat'],

                llcrnrlon=a['llcrnrlon'],

                urcrnrlat=a['urcrnrlat'],

                urcrnrlon=a['urcrnrlon'],

                resolution='l')

    m.drawcountries()

    m.drawcoastlines()

    m.shadedrelief()



    m.scatter(df['Longitude'].values

              ,df['Latitude'].values

              ,s=df['Magnitude'].values*1

              ,marker="o"

              ,cmap=cm.seismic

              ,alpha=.5

              ,latlon=True)



    plt.title("%s Bölgesinde ki Sismik Olaylar" % a['label'])

#plt.tight_layout()



plt.show()
df.loc[df['Magnitude'] > 8, 'Sınıf'] = 'İyi'

df.loc[ (df['Magnitude'] >= 7) & (df['Magnitude'] < 7.9), 'Sınıf'] = 'Önemli'

df.loc[ (df['Magnitude'] >= 6) & (df['Magnitude'] < 6.9), 'Sınıf'] = 'Güçlü'

df.loc[ (df['Magnitude'] >= 5.5) & (df['Magnitude'] < 5.9), 'Sınıf'] = 'Ilımlı'
# Magnitude Class distribution



sns.countplot(x="Sınıf", data=df)

plt.ylabel('Sıklık')

plt.title('Büyüklük Sınıfı VS Sıklık')



df=pd.DataFrame(database)

df
import datetime



# Extract year for filtering purpose

year = []

for index, row in df.iterrows():

    try:

        date = row['Date']

        date_time_obj = datetime.datetime.strptime(date, '%m/%d/%Y')

        y = date_time_obj.date().year

        year.append(y)

    except:

        year.append(-1)

print (year[:10])
#'Year' adında yeni öznitelik ekledik

df.insert(loc=1, column='Year', value=year)

df.head(5)
import datetime

df['date']=df['Date'].apply(lambda x: pd.to_datetime(x))
df['year']=df['date'].apply(lambda x:str(x).split('-')[0])
plt.figure(figsize=(25,8))

sns.set(font_scale=1.0)

sns.countplot(x="year",data=df)

plt.ylabel('Deprem Sayısı')

plt.xlabel('Her Yıl Meydana Gelen Deprem Sayısı')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.basemap import Basemap

import matplotlib.animation as animation

from IPython.display import HTML

import warnings

warnings.filterwarnings('ignore')
df['Year']= df['Date'].str[6:]
fig = plt.figure(figsize=(10, 10))

fig.text(.8, .3, 'Soumitra', ha='right')

cmap = plt.get_cmap('coolwarm')



m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 1)

m.drawmapboundary(fill_color='lightblue')





START_YEAR = 1965

LAST_YEAR = 2016



points = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][df['Year']==str(START_YEAR)]



x, y= m(list(points['Longitude']), list(points['Latitude']))

scat = m.scatter(x, y, s = points['Magnitude']*points['Depth']*0.3, marker='o', alpha=0.3, zorder=10, cmap = cmap)

year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)

plt.title("Earthquake visualisation (1965 - 2016)")

plt.close()





def update(frame_number):

    current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))

    year_text.set_text(str(current_year))

    points = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][df['Year']==str(current_year)]

    x, y= m(list(points['Longitude']), list(points['Latitude']))

    color = points['Depth']*points['Magnitude'];

    scat.set_offsets(np.dstack((x, y)))

    scat.set_sizes(points['Magnitude']*points['Depth']*0.3)

    

ani = animation.FuncAnimation(fig, update, interval=750, frames=LAST_YEAR - START_YEAR + 1)

ani.save('animation.gif', writer='imagemagick', fps=5)

df = pd.read_csv("/kaggle/input/earthquake-database/database.csv")

df = df.drop([3378,7512,20650])

df["year"]= [int(each.split("/")[2]) for each in df.iloc[:,0]]

df.head()
df.Type.unique()
df = df.loc[:,["Date","Latitude","Longitude","Type","Depth","Magnitude","year"]]

years = [str(each) for each in list(df.year.unique())]  # str unique years

# make list of types

types = ['Earthquake', 'Nuclear Explosion', 'Explosion', 'Rock Burst']

custom_colors = {

    'Earthquake': 'rgb(189, 2, 21)',

    'Nuclear Explosion': 'rgb(52, 7, 250)',

    'Explosion': 'rgb(99, 110, 250)',

    'Rock Burst': 'rgb(0, 0, 0)'

}

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, 

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1965',

    'plotlycommand': 'animate',

    'values': years,

    'visible': True

}

figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]



sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Year:',

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 300, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}



# make data

year = 1695

for ty in types:

    dataset_by_year = df[df['year'] == year]

    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Type'] == ty]

    

    data_dict = dict(

    type='scattergeo',

    lon = df['Longitude'],

    lat = df['Latitude'],

    hoverinfo = 'text',

    text = ty,

    mode = 'markers',

    marker=dict(

        sizemode = 'area',

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "white"),

        color = custom_colors[ty],

        opacity = 0.7),

)

    figure['data'].append(data_dict)

    

# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for ty in types:

        dataset_by_year = df[df['year'] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Type'] == ty]



        data_dict = dict(

                type='scattergeo',

                lon = dataset_by_year_and_cont['Longitude'],

                lat = dataset_by_year_and_cont['Latitude'],

                hoverinfo = 'text',

                text = ty,

                mode = 'markers',

                marker=dict(

                    sizemode = 'area',

                    sizeref = 1,

                    size= 10 ,

                    line = dict(width=1,color = "white"),

                    color = custom_colors[ty],

                    opacity = 0.7),

                name = ty

            )

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)





figure["layout"]["autosize"]= True

figure["layout"]["title"] = "Earthquake"       



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)
df['Year']= df['Date'].str[6:]
fig = plt.figure(figsize=(10, 10))

fig.text(.8, .3, 'Soumitra', ha='right')

cmap = plt.get_cmap('coolwarm')



m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 1)

m.drawmapboundary(fill_color='lightblue')





START_YEAR = 1965

LAST_YEAR = 2016



points = df[['Date',  'Latitude', 'Longitude', 'Depth', 'Magnitude']][df['Year']==str(START_YEAR)]



x, y= m(list(points['Longitude']), list(points['Latitude']))

scat = m.scatter(x, y, s = points['Magnitude']*points['Depth']*0.3, marker='o', alpha=0.3, zorder=10, cmap = cmap)

year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)

plt.title("Earthquake visualisation (1965 - 2016)")

plt.close()





def update(frame_number):

    current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))

    year_text.set_text(str(current_year))

    points = df[['Date',  'Latitude', 'Longitude', 'Depth', 'Magnitude']][df['Year']==str(current_year)]

    x, y= m(list(points['Longitude']), list(points['Latitude']))

    color = points['Depth']*points['Magnitude'];

    scat.set_offsets(np.dstack((x, y)))

    scat.set_sizes(points['Magnitude']*points['Depth']*0.3)

    

ani = animation.FuncAnimation(fig, update, interval=750, frames=LAST_YEAR - START_YEAR + 1)

ani.save('animation.gif', writer='imagemagick', fps=5)
import io

import base64



filename = 'animation.gif'



video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
df
df.loc[df['Magnitude'] > 8, 'Sınıf'] = 'İyi'

df.loc[ (df['Magnitude'] >= 7) & (df['Magnitude'] < 7.9), 'Sınıf'] = 'Önemli'

df.loc[ (df['Magnitude'] >= 6) & (df['Magnitude'] < 6.9), 'Sınıf'] = 'Güçlü'

df.loc[ (df['Magnitude'] >= 5.5) & (df['Magnitude'] < 5.9), 'Sınıf'] = 'Ilımlı'
df.dropna(how="any",inplace=True) 
df
df =df.drop(columns ='year')


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 

df['Latitude_Encoded']= label_encoder.fit_transform(df['Latitude'])

df['Longitude_Encoded']= label_encoder.fit_transform(df['Longitude'])

df['Depth_Encoded']= label_encoder.fit_transform(df['Depth'])

df['Magnitude_Encoded']= label_encoder.fit_transform(df['Magnitude'])

df['Year_Encoded']= label_encoder.fit_transform(df['Year'])

df
df =df.drop(columns ='Date')

df =df.drop(columns ='Latitude')

df =df.drop(columns ='Longitude')

df =df.drop(columns ='Type')

df =df.drop(columns ='Depth')

df =df.drop(columns ='Magnitude')

df =df.drop(columns ='Year')





df
array = df.values

X = array[:,1:6]

y = array[:,0:1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)



print("Dataframe boyutu: ",df.shape)

print("Eğitim verisi boyutu: ",X_train.shape, Y_train.shape)

print("Test verisi boyutu: ",X_validation.shape, Y_validation.shape)
from sklearn import preprocessing

from sklearn import utils



from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

#Decision Trees

cellTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

print(cellTree) # it shows the default parameters

  #I fit the data with the training

cellTree.fit(X_train,Y_train)

  #now predictions

yhat_dt = cellTree.predict(X_validation)



  #Accuracy evaluation

acc = metrics.accuracy_score(Y_validation, yhat_dt)

print('karar agaci icin accuracy: ',acc)



#karar agaci icin confusion matrix ve metrik degerler

cellTree_dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_dt = cross_val_score(cellTree_dt, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_dt)

print('cv_scores mean:{}'.format(np.mean(cv_scores_dt)))

from sklearn.metrics import classification_report

prec_dt = classification_report(yhat_dt,Y_validation)

print(prec_dt)
#call the models

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 3)

# fit the models

neigh = knn_model.fit(X_train,Y_train)

#predict the mode;

yhatknn=neigh.predict(X_validation)



  #Accuracy evaluation

accknn = metrics.accuracy_score(Y_validation, yhatknn)

print('en yakin komsular icin accuracy',accknn)



#knn=3 icin confusion matrix ve metrik degerler

knn_knn = KNeighborsClassifier(n_neighbors = 3)

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_knn = cross_val_score(knn_knn, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_knn)

print('cv_scores mean:{}'.format(np.mean(cv_scores_knn)))



#knn scores

from sklearn.metrics import classification_report

prec_knn = classification_report(yhatknn,Y_validation)

print(prec_knn)
#lojistik regresyon

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)

LR

#predict

yhatlr = LR.predict(X_validation)

#print('yhat', yhat)

  #Accuracy evaluation

acclr = metrics.accuracy_score(Y_validation, yhatlr)

print('lojistik regresyon icin accuracy',acclr)





#lojistik regresyon icin confusion matrix ve metrik degerler

lr_lr = LogisticRegression(C=0.01, solver='liblinear')

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_lr = cross_val_score(lr_lr, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_lr)

print('cv_scores mean:{}'.format(np.mean(cv_scores_lr)))





from sklearn.metrics import classification_report

prec_lr = classification_report(yhatlr,Y_validation)

print(prec_lr)
#SVM 

from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(X_train, Y_train) 

#predict

yhatsvm = clf.predict(X_validation)

#yhat [0:5]

accsvm = metrics.accuracy_score(Y_validation, yhatsvm)

print('svm icin accuracy',accsvm)







#svm icin confusion matrix ve metrik degerler

clf_svm = svm.SVC(kernel='rbf')

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_svm = cross_val_score(clf_svm, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_svm)

print('cv_scores mean:{}'.format(np.mean(cv_scores_svm)))





from sklearn.metrics import classification_report

prec_svm = classification_report(yhatsvm,Y_validation)

print(prec_svm)
#gaussian NB 

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

#call the models

gnb = GaussianNB()

  #fit the model

gnb.fit(X_train, Y_train) 

  #predict

yhatgnb = gnb.predict(X_validation)

accgnb = metrics.accuracy_score(Y_validation, yhatgnb)

print('gaussian naive bayes icin accuracy',accgnb)





#gaussian naive bayes icin confusion matrix ve metrik degerler

clf_gnb = GaussianNB()

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_gnb = cross_val_score(clf_gnb, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_gnb)

print('cv_scores mean:{}'.format(np.mean(cv_scores_gnb)))



#klasifikasyon tablosu

from sklearn.metrics import classification_report

prec_gnb = classification_report(yhatgnb,Y_validation)

print(prec_gnb)
#linear discriminant analysis 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

#fit the model

lda.fit(X_train, Y_train) 

#predict

yhatlda = lda.predict(X_validation)

acclda = metrics.accuracy_score(Y_validation, yhatlda)

print('linear discriminant analiz icin accuracy',acclda)









#linear discrimant icin confusion matrix ve metrik degerler

clf_ld = LinearDiscriminantAnalysis()

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_ld = cross_val_score(clf_ld, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_ld)

print('cv_scores mean:{}'.format(np.mean(cv_scores_ld)))



#klasifikasyon linear diskrimannt

from sklearn.metrics import classification_report

prec_lda = classification_report(yhatlda,Y_validation)

print(prec_lda)
# RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier(max_depth=5, n_estimators=100, max_features='auto')

rfc.fit(X_train, Y_train) 

#predict

yhat1 = rfc.predict(X_validation)

#yhat [0:5]

#evaluate



#create a new SVM model

rfc_cv = RandomForestClassifier(max_depth=5, n_estimators=100, max_features='auto')

#train model with cv of 10

cv_scores = cross_val_score(rfc_cv, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores)

print('cv_scores mean:{}'.format(np.mean(cv_scores)))











from sklearn.metrics import classification_report, confusion_matrix

import itertools

from sklearn.metrics import f1_score

print('f1_score for Random Forest Classifier:',f1_score(Y_validation, yhat1, average='weighted'))

#print("Train set Accuracy for Random Forest Classifier: ", metrics.accuracy_score(Y_validation, rfc.predict(X_train)))

#print("Test set Accuracy for Random Forest Classifier: ", metrics.accuracy_score(Y_validation, yhat1))

from sklearn.metrics import classification_report

prec_rec = classification_report(yhat1,Y_validation)

print(prec_rec)