# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import cufflinks as cf
cf.go_offline()
%matplotlib inline
dm_hoteles = pd.read_csv('hotel_bookings.csv')
dm_hoteles.head()
dm_hoteles.info()
dm_hoteles.isnull().sum() / len(dm_hoteles) * 100
dm_hoteles['company'].describe()
dm_hoteles = dm_hoteles.drop(['agent', 'company'],axis=1)
dm_hoteles.info()
dm_hoteles.isnull().sum() / len(dm_hoteles) * 100
dm_hoteles = dm_hoteles.dropna(axis=0)
dm_hoteles.isnull().sum() / len(dm_hoteles) * 100
dm_hoteles.describe().transpose()
dm_hoteles = dm_hoteles.drop(dm_hoteles[dm_hoteles['adr']<=0].index,axis=0)

dm_hoteles.describe().transpose()
dm_hoteles['total_guests'] = dm_hoteles['adults'] + dm_hoteles['children'] + dm_hoteles['babies']
dm_hoteles = dm_hoteles.drop(['children',
                              'adults',
                              'babies',
                              'arrival_date_day_of_month',
                              'arrival_date_day_of_month',
                              'customer_type',
                              'meal'],axis=1)
dm_hoteles = dm_hoteles.drop(dm_hoteles[dm_hoteles['total_guests']==0].index,axis=0)
dm_hoteles.columns
dm_hoteles = dm_hoteles.reset_index()
dm_hoteles = dm_hoteles.drop('index',axis=1)
dm_hoteles.info()
dm_hoteles['stay'] = dm_hoteles['stays_in_week_nights'] + dm_hoteles['stays_in_weekend_nights']

dm_hoteles = dm_hoteles.drop(['stays_in_week_nights',
                              'stays_in_weekend_nights'],axis=1)

dm_hoteles.describe().transpose()
dm_hoteles.columns
codigo_paises = pd.read_excel('country codes.xlsx')
dm_hoteles = dm_hoteles.merge(codigo_paises, left_on='country', right_on='code')
dm_hoteles.columns
dm_hoteles.rename(columns = {'arrival_date_month': 'Mes'}, inplace = True)
dm_hoteles.rename(columns = {'adr_pp': 'Precio por Persona'}, inplace = True)

country_data = dm_hoteles.groupby(['country','name'])['total_guests'].sum().reset_index().sort_values(by='total_guests',ascending=False)
country_data['% del total'] = round(country_data['total_guests']/ country_data['total_guests'].sum() * 100, 2)

country_data = dm_hoteles.groupby(['country','name'])['total_guests'].sum().reset_index().sort_values(by='total_guests',ascending=False)
country_data['% del total'] = round(country_data['total_guests']/ country_data['total_guests'].sum() * 100, 2)

guest_map = px.choropleth(country_data,
                          locations=country_data['country'],
                          color=country_data["% del total"],
                          hover_name=country_data['name'],
                          color_continuous_scale=px.colors.sequential.Reds,
                          title="Ciudad de origen de visitantes")
guest_map.show()

country_data_cancelados = dm_hoteles.groupby(['country','name'])['total_guests','is_canceled'].sum().reset_index().sort_values(by='total_guests',ascending=False)
country_data_cancelados['% de cancelados'] = round(country_data_cancelados['is_canceled']/ country_data_cancelados['total_guests'] * 100, 2)


guest_map = px.choropleth(country_data_cancelados,
                    locations=country_data_cancelados['country'],
                    color=country_data_cancelados["% de cancelados"], 
                    hover_name=country_data_cancelados['name'], 
                    color_continuous_scale=px.colors.sequential.Reds,
                    title="Ciudad origen cancelados")
guest_map.show()
meses_ordenados = ['January', "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
dm_hoteles['Mes'] = pd.Categorical(dm_hoteles['Mes'], categories=meses_ordenados, ordered=True)

country_data_cancelados = dm_hoteles.groupby(['country','name','Mes'])['total_guests','is_canceled'].sum().reset_index().sort_values(by='Mes')
country_data_cancelados['% de cancelados'] = round(country_data_cancelados['is_canceled']/ country_data_cancelados['total_guests'] * 100, 2)
country_data_cancelados.dropna(inplace=True)
country_data_cancelados = country_data_cancelados.reset_index()
country_data_cancelados = country_data_cancelados.drop('index',axis=1)

guest_map = px.choropleth(country_data_cancelados,
                    locations=country_data_cancelados['country'],
                    color=country_data_cancelados["total_guests"], 
                    hover_name=country_data_cancelados['name'],
                    animation_frame="Mes",
                    color_continuous_scale=px.colors.sequential.Reds,
                    title="Cambios de Origen por fecha")
guest_map.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

guest_map.show()
country_data_cancelados = dm_hoteles.groupby(['country','name'])['total_guests','is_canceled'].sum().reset_index().sort_values(by='total_guests',ascending=False)
country_data_cancelados['% de cancelados'] = round(country_data_cancelados['is_canceled']/ country_data_cancelados['total_guests'] * 100, 2)

country_data_cancelados.sort_values(by='total_guests',ascending=False).head(10).sort_values(by='% de cancelados',ascending=True).plot(x='name',y='% de cancelados',kind='barh')
plt.suptitle('Top Mayor % de Cancelaciones de los Países de Más Reservas')
plt.show()
tipo_hotel = pd.DataFrame(dm_hoteles['hotel'].value_counts()).reset_index()
tipo_hotel['%'] = round(tipo_hotel['hotel']/tipo_hotel['hotel'].sum() *100,2)
tipo_hotel.rename(columns = {'hotel': 'cantidad de reservaciones'}, inplace = True)
tipo_hotel.rename(columns = {'index': 'hotel'}, inplace = True)
tipo_hotel.plot(x='hotel',y='%', kind='barh')

print(tipo_hotel)

plt.show()

plt.figure(figsize=(12,12))
sb.heatmap(dm_hoteles[dm_hoteles['hotel']=='City Hotel'].corr(),cmap='coolwarm',linecolor='white',linewidths=1,annot=False)

plt.suptitle('Correlación de Variables para Hoteles de Ciudad', fontsize=20)
plt.show()
plt.figure(figsize=(12,12))
sb.heatmap(dm_hoteles[dm_hoteles['hotel']=='Resort Hotel'].corr(),cmap='coolwarm',linecolor='white',linewidths=1,annot=False)

plt.suptitle('Correlación de Variables para Resorts', fontsize=20)
plt.show()
dm_hoteles['adr_pp'] = dm_hoteles['adr']/dm_hoteles['total_guests']
pd.DataFrame(dm_hoteles['adr_pp'].describe()).reset_index()[-7:]
plt.figure(figsize=(12,8))

sb.boxplot(x='reserved_room_type',y='adr_pp', hue='hotel',data=dm_hoteles, showfliers=False,palette='rainbow')
plt.title("Precio por tipo de hotel y por tipo de habitacion", fontsize=16)
plt.xlabel("TIpo de Habitación", fontsize=16)
plt.ylabel("Precio [EUR]", fontsize=16)
plt.legend(loc="upper right")
plt.ylim(0, 160)
plt.show()
dm_hoteles['arrival_date_year'].value_counts().plot(kind='bar', cmap='rainbow')
dm_hoteles.rename(columns = {'arrival_date_month': 'Mes'}, inplace = True)
dm_hoteles.rename(columns = {'adr_pp': 'Precio por Persona'}, inplace = True)

meses_ordenados = ['January', "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
dm_hoteles['Mes'] = pd.Categorical(dm_hoteles['Mes'], categories=meses_ordenados, ordered=True)


mensual_tipohotel = dm_hoteles.groupby(['hotel','Mes'])['Precio por Persona'].mean().sort_values().reset_index()

fig = px.bar(mensual_tipohotel, x="Mes", y="Precio por Persona", color='hotel', barmode='group',
             height=400)

fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':meses_ordenados})
fig.update_layout(title_text='Precio Promedio por Mes y Tipo de Hotel')

fig.show()
sb.set(style="whitegrid")

anio2015 = dm_hoteles[dm_hoteles['arrival_date_year']==2015].groupby(['hotel','Mes'])['Precio por Persona'].mean().sort_values().reset_index()

anio2016 = dm_hoteles[dm_hoteles['arrival_date_year']==2016].groupby(['hotel','Mes'])['Precio por Persona'].mean().sort_values().reset_index()

anio2017 = dm_hoteles[dm_hoteles['arrival_date_year']==2017].groupby(['hotel','Mes'])['Precio por Persona'].mean().sort_values().reset_index()

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)

ax1.tick_params('x', labelrotation=45)
ax2.tick_params('x', labelrotation=45)
ax3.tick_params('x', labelrotation=45)

gp1 = sb.lineplot(x='Mes',y='Precio por Persona',ci='sd',data=anio2015, ax=ax1, hue="hotel" ).set_title("2015", {'fontsize': 16})
gp2 = sb.lineplot(x='Mes',y='Precio por Persona',ci='sd',data=anio2016, ax=ax2, hue="hotel" ).set_title("2016", {'fontsize': 16})
gp3 = sb.lineplot(x='Mes',y='Precio por Persona',ci='sd',data=anio2017, ax=ax3, hue="hotel" ).set_title("2017", {'fontsize': 16})

plt.subplots_adjust(wspace=0.3,hspace=0.5)
plt.show()



fig = plt.figure(figsize=(20,10))

dm_hoteles["Mes"] = pd.Categorical(dm_hoteles["Mes"], categories=meses_ordenados, ordered=True)

sb.lineplot(x='Mes',y='Precio por Persona',data=dm_hoteles,hue="hotel").set_title("Precio promedio por Mes", {'fontsize': 16})

plt.show()
cambio_precio = pd.DataFrame(dm_hoteles.groupby('Mes')['Precio por Persona'].mean()).reset_index()
cambio_precio['% de crecimiento'] = cambio_precio['Precio por Persona'].pct_change() * 100
cambio_precio['% de crecimiento'] = cambio_precio['% de crecimiento'].fillna(0)
cambio_precio = cambio_precio[-11:]

plt.figure(figsize=(20,10))

sb.barplot(x ='Mes', y='% de crecimiento', data=cambio_precio).set_title("Crecimiento Porcentual del Precio Promedio Mensual", {'fontsize': 16})

plt.show()
cambio_precio = pd.DataFrame(dm_hoteles.groupby(['Mes','hotel'])['Precio por Persona','total_guests'].mean()).reset_index()
cambio_precio['% de crecimiento'] = cambio_precio['Precio por Persona'].pct_change() * 100
cambio_precio['% de crecimiento'] = cambio_precio['% de crecimiento'].fillna(0)
cambio_precio.sort_values(by='Mes',ascending=False).dropna(inplace=True)
cambio_precio = cambio_precio.dropna()

fig = px.scatter(cambio_precio, y=cambio_precio["% de crecimiento"],
                    x= cambio_precio["Mes"],
                    range_x = [-1,12],
                    range_y = [cambio_precio["% de crecimiento"].min()-1,cambio_precio["% de crecimiento"].max()+1],
                    animation_frame="Mes",
                    color= "hotel", hover_name="hotel",
                    hover_data=["% de crecimiento"],
                    size='total_guests',
                    title='% de crecimiento mensual por tipo de hotel',
                    height=1000
                    )
fig.update_coloraxes(colorscale="hot")
fig.update(layout_coloraxis_showscale=True)
fig.update_xaxes(title_text="Mes")
fig.update_yaxes(title_text="% de crecimiento")
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
fig.show()
Ocupacion = dm_hoteles.groupby(['Mes','hotel'])['total_guests'].sum().reset_index()

fig = px.bar(Ocupacion, x="Mes", y="total_guests", color='hotel', barmode='group',
             height=400)

fig.update_layout(title_text='Total Guests por Mes y por Hotel')


fig.show()
plt.figure(figsize=(12,8))

sb.distplot(dm_hoteles['stay'],bins=100, kde=False, color='g').set_title('Distribución de Reservacion por Noches',fontsize=16)

plt.show()
duracion = dm_hoteles.groupby(['stay','hotel'])['total_guests'].sum().reset_index().sort_values(by='stay',ascending=False)
plt.figure(figsize=(18,8))
sb.barplot(data=duracion, x="stay", y="total_guests", hue='hotel')
plt.legend(loc='upper right')
plt.suptitle('Duración')
plt.show()
plt.figure(figsize=(20, 20)) #estableciendo el tamaño del grafico
sb.heatmap(pd.pivot_table(dm_hoteles, 
                          values='stay', 
                          index=['Mes'], 
                          columns=['hotel'], 
                          aggfunc='mean').iloc[:, :20], annot=True, fmt='g',cmap='Greens',linecolor='white',linewidths=1) #grafico de heatmap
plt.tick_params('x', labelrotation=45)
plt.tick_params('y', labelrotation=360)
plt.suptitle('Promedio de duración por mes y hotel', fontsize=20)
plt.show()

plt.figure(figsize=(12, 6))

sb.countplot(data=dm_hoteles,x='market_segment', hue='is_canceled')
plt.tick_params('x', labelrotation=45)
plt.suptitle('Reservaciones por Segmento de Mercado', fontsize=20)
plt.show()
plt.figure(figsize=(12, 6))

sb.barplot(data=dm_hoteles,x='market_segment',y='Precio por Persona', hue='reserved_room_type')
plt.tick_params('x', labelrotation=45)
plt.suptitle('Reservaciones por Segmento de Mercado', fontsize=20)
plt.show()
plt.figure(figsize=(12, 6))

canc_mes = dm_hoteles.groupby(['Mes','hotel'])['is_canceled'].sum().reset_index()
sb.lineplot(x='Mes',y='is_canceled',data=canc_mes)

plt.suptitle('Tendencia de Cancelaciones Por Mes', fontsize=20)
plt.show()
plt.figure(figsize=(12, 6))

canc_mes = dm_hoteles.groupby(['Mes','hotel'])['is_canceled'].sum().reset_index()
sb.lineplot(x='Mes',y='is_canceled',data=canc_mes,hue='hotel')

plt.suptitle('Tendencia de Cancelaciones Por Mes', fontsize=20)
plt.show()
plt.figure(figsize=(12,8))

august = dm_hoteles[dm_hoteles['Mes']=='August']

august = august.corr()['is_canceled'].sort_values()

august = august.drop('is_canceled',axis=0)

august.iplot(kind='bar', title = 'Relación de Variables para el Mes (Agosto) de Mayor cantidad de Cancelaciones', fontsize=20)

plt.figure(figsize=(12,8))

corr = dm_hoteles[dm_hoteles['Mes']=='August'].corr()

cmap = sb.diverging_palette(h_neg=10,
                            h_pos=240,
                            as_cmap=True)

mask = np.triu(np.ones_like(corr, dtype=bool))

sb.heatmap(corr, mask = mask, center = 0, cmap = cmap, linewidths=1, annot=True, fmt=".2f")

plt.suptitle('Matriz de Correlación de Variables para el Mes (Agosto) de Mayor cantidad de Cancelaciones', fontsize=20)
plt.show()
dm_hoteles.columns
dm_hoteles.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#dm_hoteles.select_dtypes(include='object').columns

dm_hoteles_modelo = dm_hoteles

dm_hoteles_modelo['hotel'] = le.fit_transform(dm_hoteles_modelo['hotel'])

dm_hoteles_modelo['market_segment'] = le.fit_transform(dm_hoteles_modelo['market_segment'])

dm_hoteles_modelo['Mes'] = le.fit_transform(dm_hoteles_modelo['Mes'])

dm_hoteles_modelo['reserved_room_type'] = le.fit_transform(dm_hoteles_modelo['reserved_room_type'])

dm_hoteles_modelo['assigned_room_type'] = le.fit_transform(dm_hoteles_modelo['assigned_room_type'])

dm_hoteles_modelo['deposit_type'] = le.fit_transform(dm_hoteles_modelo['deposit_type'])

dm_hoteles_modelo.columns
sb.countplot(x='is_canceled',data=dm_hoteles)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()

X = dm_hoteles_modelo.drop(['is_canceled',
                            'arrival_date_year',
                            'arrival_date_week_number',
                            'country',
                            'distribution_channel',
                            'adr',
                            'reservation_status',
                            'reservation_status_date',
                            'name',
                            'code'],axis=1).values

y = dm_hoteles_modelo['is_canceled'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
# CODE HERE
model = Sequential()

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# Choose whatever number of layers/neurons you want.
model.add(Dense(78, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, 
          y=y_train, 
          epochs=50,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )
df_loss = pd.DataFrame(model.history.history)
df_loss.plot()
from sklearn.metrics import confusion_matrix, classification_report

predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.model_selection import KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models=[]
models.append(('LR',LogisticRegression()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('KN',KNeighborsClassifier(10)))
models.append(('NB',GaussianNB()))
models.append(('RF',RandomForestClassifier(n_estimators=100)))
results=[]
names=[]
scoring='accuracy'
for name,model in models:
    kfold=KFold(n_splits=10,random_state=None)
    cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    names.append(name)
    msg=("%s: %f (%f)" % (name,cv_result.mean(),cv_result.std()))
    print(msg)
fig=plt.figure()
fig.suptitle('Algorithms Coparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
RF = RandomForestClassifier(n_estimators=100)

X = dm_hoteles_modelo.drop(['is_canceled',
                            'arrival_date_year',
                            'arrival_date_week_number',
                            'country',
                            'distribution_channel',
                            'adr',
                            'reservation_status',
                            'reservation_status_date',
                            'name',
                            'code'],axis=1)

y = dm_hoteles_modelo['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_train = scaler.fit_transform(X_train)


RF.fit(X_train, y_train)

importancia = pd.DataFrame()
importancia['Variables'] = X.columns
importancia['Score'] = RF.feature_importances_

plt.figure(figsize=(12,8))

importancia.sort_values(by='Score',ascending=True).plot(x='Variables',y='Score',kind='barh')

plt.suptitle('Importancia de variables en las cancelaciones', fontsize=20)
plt.show()
