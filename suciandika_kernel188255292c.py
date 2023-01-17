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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



dataset=pd.read_csv('../input/mobile-price-classification/train.csv')

dataset.head()
dataset.describe()
dataset.info()
dataset.std()
import seaborn as sns

import matplotlib.pyplot as plt

corr=dataset.corr()

fig = plt.figure(figsize=(15,12))

r = sns.heatmap(corr, cmap='Blues')

r.set_title("KORELASI ANTARA FITUR ")
corr.sort_values(by=["price_range"],ascending=False).iloc[0].sort_values(ascending=False)
import plotly.graph_objects as go



df = pd.read_csv('../input/mobile-price-classification/train.csv')



fig = go.Figure()

fig.add_trace(go.Box(

    y=df['ram'],

    name="RAM",

    jitter=0.3,

    pointpos=-1.8,

    boxpoints='all', # represent all points

    marker_color='rgb(7,40,89)',

    line_color='rgb(7,40,89)'

))



fig.add_trace(go.Box(

    y=df['battery_power'],

    name="Battery Power",

    boxpoints=False, # no data points

    marker_color='rgb(9,56,125)',

    line_color='rgb(9,56,125)'

))



fig.update_layout(title_text="RAM dan Battery Power")

#print(fig)

fig.show()
import pandas as pd

import plotly.express as px



df = pd.read_csv('../input/mobile-price-classification/train.csv')

#df.head()



fig = px.line(df, x = 'ram', y = 'price_range', title='PRICE RANGE - RAM')

fig.show()
import plotly.graph_objects as go

import pandas as pd

dataset = pd.read_csv('../input/mobile-price-classification/train.csv')

fig = go.Figure(data=go.Scatter(

    x=dataset['px_width'],

    name='Phone Width',

    mode='markers',

    marker=dict(

    color='rgb(107,174,214)')

))



fig = go.Figure(data=go.Scatter(

    y=dataset['px_height'],

    name='Phone Heigt',

    mode='markers',

    marker=dict(

    color='rgba(219, 64, 82, 0.6)')

))

#print(fig)

fig.update_layout(title_text="Scatterplot Phone Height Phone Width")

fig.show()
import plotly.graph_objects as go

import pandas as pd

dataset = pd.read_csv('../input/mobile-price-classification/train.csv')

fig = go.Figure(data=go.Scatter(

    x=dataset['price_range'],

    name='Phone Width',

    mode='markers',

    marker=dict(

    color='rgb(107,174,214)')

))



fig = go.Figure(data=go.Scatter(

    y=dataset['ram'],

    name='Phone Heigt',

    mode='markers',

    marker=dict(

    color='rgba(219, 64, 82, 0.6)')

))

fig = go.Figure(data=go.Scatter(

    y=dataset['battery_power'],

    name='Phone Heigt',

    mode='markers',

    marker=dict(

    color='rgba(219, 64, 82, 0.6)')

))

#print(fig)

fig.update_layout(title_text="Scatterplot Phone Height Phone Width")

fig.show()
import plotly.graph_objects as go

import pandas as pd

dataset = pd.read_csv('../input/mobile-price-classification/train.csv')

fig = go.Figure(data=go.Scatter(

    x=dataset['price_range'],

    y=dataset['ram'],

    mode='markers',

    marker=dict()

))

print(fig)

fig.show()
import plotly.io as pio



import pandas as pd



df = pd.read_csv('../input/mobile-price-classification/train.csv')



colors = ['blue', 'orange', 'green', 'red']



opt = []

opts = []

for i in range(0, len(colors)):

    opt = dict(

        target = df['price_range'][[i]].unique(), value = dict(marker = dict(color = colors[i]))

    )

    opts.append(opt)



data = [dict(

  type = 'scatter',

  mode = 'markers',

  x = df['sc_w'],

  y = df['sc_h'],

  text = df['price_range'],

  hoverinfo = 'text',

  opacity = 0.8,

  marker = dict(),

  transforms = [

      dict(

        type = 'groupby',

        groups = df['price_range'],

        styles = opts

      ),

      dict(

        type = 'aggregate',

        groups = df['ram'],

        aggregations = [

            dict(target = 'x', func = 'avg'),

            dict(target = 'y', func = 'avg'),

            dict(target = 'marker.size', func = 'sum')

        ]

      )]

)]



layout = dict(

    title = '<b>Screen Width Screen Heigth - Ram Price Range</b><br>Klasifikasai harga berdasarkan Ukuran Screen, RAM dan Harga',

    yaxis = dict(

        type = 'log'

    )

)



fig_dict = dict(data=data, layout=layout)

pio.show(fig_dict, validate=False)
import plotly.io as pio



import pandas as pd



df = pd.read_csv('../input/mobile-price-classification/train.csv')



colors = ['blue', 'orange', 'green', 'red']



opt = []

opts = []

for i in range(0, len(colors)):

    opt = dict(

        target = df['price_range'][[i]].unique(), value = dict(marker = dict(color = colors[i]))

    )

    opts.append(opt)



data = [dict(

  type = 'scatter',

  mode = 'markers',

  x = df['ram'],

  y = df['wifi'],

  text = df['price_range'],

  hoverinfo = 'text',

  opacity = 0.8,

  marker = dict(),

  transforms = [

      dict(

        type = 'groupby',

        groups = df['price_range'],

        styles = opts

      ),

      dict(

        type = 'aggregate',

        groups = df['price_range'],

        aggregations = [

            dict(target = 'x', func = 'avg'),

            dict(target = 'y', func = 'avg'),

            dict(target = 'marker.size', func = 'sum')

        ]

      )]

)]



layout = dict(

    title = '<b>WIFI RAM - Price Range</b><br>Prediksi harga berdasarkan Wifi dan RAM',

    yaxis = dict(

        type = 'log'

    )

)



fig_dict = dict(data=data, layout=layout)

pio.show(fig_dict, validate=False)
#Internal Memory vs Price Range

import seaborn as sns

import matplotlib.pyplot as plt

sns.pointplot(y="int_memory", x="price_range", data=dataset)
sns.jointplot(x='ram',y='price_range',data=dataset,color='blue',kind='kde');
#3G



labels = '3G-supported','Not supported'

colors = ['lightblue', 'red']

values=dataset['three_g'].value_counts().values

explode=(0, 0.1)



fig1, ax1 = plt.subplots()

ax1.pie(values, explode=explode, colors=colors, autopct='%1.1f%%',shadow=True,startangle=10)

plt.legend(labels, loc='best')

plt.title("Persentase 3G Support\n")

ax1.axis('equal')



#center circle

centre_circle = plt.Circle((0,0),0.4,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.show()



#4G



labels4g = '4G-supported','Not supported'

colors = ['lightgreen', 'red']

values4g = dataset['four_g'].value_counts().values

explode=(0, 0.1)



fig1, ax1 = plt.subplots()

ax1.pie(values4g, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True,startangle=30)

plt.legend(labels4g, loc='best')

plt.title("Persentase 4G Support\n")

ax1.axis('equal')

plt.show()
#Battery power vs Price Range dalam mAh



sns.boxplot(x="price_range", y="battery_power", data=dataset)
plt.figure(figsize=(10,6))

dataset['fc'].hist(alpha=0.5, color='blue',label='Front camera')

dataset['pc'].hist(alpha=0.5,color='red',label='Primary camera')

plt.legend()

plt.xlabel('MegaPixels')
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt; plt.rcdefaults()



dataset=pd.read_csv('../input/mobile-price-classification/train.csv')



price = dataset['price_range']

y_pos = np.arange(len(price))

ram = dataset['ram'] #jumlah nilai sesuai dengan jumlah object



plt.bar(price, ram, align = 'center', alpha=0.5)

plt.xlabel('Price')

plt.ylabel('RAM')

plt.title('Price Range - RAM')

#plt.yticks(y_pos, price, 'ram')



plt.show()
import pandas as pd

import plotly.express as px



fig = px.line(df, x = 'battery_power', y = 'ram', title='Battery Power - RAM')

fig.show()
X=dataset.drop('price_range',axis=1)
y=dataset['price_range']
#Splitting the data

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
# Linear Regression Model



from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
#HASIL LINEAR REGRESI

lm.score(X_test,y_test)
#Creating & Training KNN Model



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)
#Elbow Method For optimum value of K



error_rate = []

for i in range(1,20):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=5)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
#Creating & Training Logistic Regression Model



from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()



logmodel.fit(X_train,y_train)
logmodel.score(X_test,y_test)
#Creating & Training Decision Tree Model



from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()



dtree.fit(X_train,y_train)
dtree.score(X_test,y_test)
#Tree Visualization



feature_names=['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',

       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',

       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',

       'touch_screen', 'wifi']



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)
rfc.score(X_test,y_test)
y_pred=lm.predict(X_test)

plt.scatter(y_test,y_pred)
plt.plot(y_test,y_pred)
# Hasil : KKN



from sklearn.metrics import classification_report,confusion_matrix



pred = knn.predict(X_test)

print(classification_report(y_test,pred))
matrix=confusion_matrix(y_test,pred)

print(matrix)
plt.figure(figsize = (10,7))

sns.heatmap(matrix,annot=True)