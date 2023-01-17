import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df1 = pd.read_csv( '/kaggle/input/uncover/UNCOVER/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv' )

df1 = df1.drop(columns=[ 'daterep', 'geoid' ])

df1.head()
df2 = pd.read_csv( '/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv' )

df2 = df2.drop(columns=[ 'recovered', 'delta_recovered', 'active', 'people_tested', 'people_hospitalized', 'province_state', 'uid', 'iso3', 'report_date_string' ])

df2.head()
def fig1(countries_array, population=False):

    plt.figure(figsize=(12,5))

    for country in countries_array:

        if population == True:

            country_population = int((df1[df1['countriesandterritories'] == country])[ 'popdata2018' ].drop_duplicates())

        else:

            country_population = 1

            

        country_data = (df1[df1['countriesandterritories']==country])

        country_data = country_data[ country_data['cases'] > 0 ]

        dates = np.array(country_data[[ 'day', 'month', 'year']])



        date = []

        for d in range(len(np.array(country_data))):

            date.append( str(dates[d,2]) + '-0' + str(dates[d,1]) + '-' + str(dates[d,0]).zfill(2) )



        days = len(country_data)

        day_array = np.arange(0,days,1)





        # Plots



        plt.plot(day_array[::-1], abs(country_data['cases'])/country_population , label = country )

        plt.legend()

        plt.xlabel('Day')

        plt.ylabel('New cases')



    plt.title('New COVID-19 cases', fontsize = 18 )

    



def fig2(countries_array, population=False):

    plt.figure(figsize=(12,5))

    for country in countries_array:

        if population == True:

            country_population = int((df1[df1['countriesandterritories'] == country])[ 'popdata2018' ].drop_duplicates())

        else:

            country_population = 1

            

        country_data = (df1[df1['countriesandterritories']==country])

        country_data = country_data[ country_data['cases'] > 0 ]

        dates = np.array(country_data[[ 'day', 'month', 'year']])



        date = []

        for d in range(len(np.array(country_data))):

            date.append( str(dates[d,2]) + '-0' + str(dates[d,1]) + '-' + str(dates[d,0]).zfill(2) )



        

        days = len(country_data)

        day_array = np.arange(0,days,1)



        # Cumulative

        country_population = int((df1[df1['countriesandterritories'] == country])[ 'popdata2018' ].drop_duplicates())

        country_data2 = df2[df2['country_region']== country]

        country_data2 = country_data2[country_data2['confirmed'] > 0 ]

        ii = []



        for i in range(len(date)):

            ii.append( list(country_data2['last_update'].astype('str')).index(date[i]) )



        country_data2 = country_data2.iloc[ii[::-1]]

        # Plots



        plt.plot(day_array, country_data2['confirmed'], label = country )

        plt.legend()

        plt.xlabel('Day')

        plt.ylabel('Cummulative COVID-19 cases')



    plt.title('Cumulative COVID-19 cases', fontsize = 18 )

fig1(['Colombia'])

fig2(['Colombia'])
sudamerica_countries = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela']

fig1(sudamerica_countries)

fig2(sudamerica_countries)
countries = ['Colombia', 'Spain', 'Italy', 'Germany', 'France', 'Canada', 'Mexico']

fig1(countries)

fig2(countries)
df3 = pd.read_csv( '/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv' )

df3 = df3.drop(columns=[ 'people_tested', 'people_hospitalized', 'uid', 'iso3' ])

df3 = df3.dropna()

print(np.shape(df3))

df3.head()
import sklearn.manifold

import sklearn.datasets

import sklearn.cluster

import umap
X = df3[['lat', 'long', 'incident_rate', 'mortality_rate' ]]

Y = np.array(df3['mortality_rate'])



target = []

for i in range(len(Y)):

    if Y[i]>=0 and Y[i]<1:

        target.append( 0 )

    elif Y[i]>=1 and Y[i]<10:

        target.append( 1 )

    elif Y[i]>=10:

        target.append( 2 )



best_neighbors = 10

min_dist = 0.05

reducer = umap.UMAP(n_neighbors=best_neighbors, min_dist=min_dist, metric='correlation')



# Ahora ejecutamos la fase de aprendizaje

reducer.fit(X)



# Extraemos la representación de los datos en el espacio bidimensional

embedding = reducer.embedding_
# clusters sobre los resultados de tsne

n_clusters = 3

k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)

k_means.fit(embedding) # training

cluster = k_means.predict(embedding) # predice a cual cluster corresponde cada elmento

distance = k_means.transform(embedding) # calcula la distancia de cada elemento al centro de su cluster
plt.figure(figsize=(15,5))

plt.angulos = np.linspace(0,2*np.pi,100)



plt.subplot(1,2,1)

plt.title( 'Neighbors = ' + str(best_neighbors) + ', Min dis = ' + str(min_dist) )

plt.scatter(embedding[:,0], embedding[:,1], c=target, cmap='viridis', s=10.0)

plt.colorbar()





plt.subplot(1,2,2)

plt.title( 'Número de clúster = ' + str(n_clusters) )

plt.scatter(embedding[:,0], embedding[:,1], c=cluster, cmap='viridis', s=10.0)

plt.colorbar()

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import sklearn.metrics

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
X = df3[['country_region', 'lat', 'long', 'incident_rate']]

Y = np.array(df3['mortality_rate'])



# Vamos a hacer un split training test

scaler = StandardScaler()

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.5 )



train_countries = X_train['country_region']

test_countries = X_test['country_region']
model = LinearRegression()

model.fit( X_train[[ 'lat', 'long', 'incident_rate' ]], Y_train )

Y_pred = model.predict( X_test[[ 'lat', 'long', 'incident_rate' ]] )



# The coefficients

print('Coefficients: ', model.coef_)



print('Mean squared error: {:2f}'.format(mean_squared_error(Y_test, Y_pred)))



print('r2: {:2f}'.format( r2_score(Y_test, Y_pred)))
from sklearn.linear_model import LogisticRegression



X = df3[['country_region', 'lat', 'long', 'incident_rate']]

Y = np.array(df3['mortality_rate'])



target = []

for i in range(len(Y)):

    if Y[i]>=0 and Y[i]<1:

        target.append( 0 )

    elif Y[i]>=1 and Y[i]<10:

        target.append( 1 )

    elif Y[i]>=10:

        target.append( 2 )



target = np.array(target)
unique, counts = np.unique(target, return_counts=True)

dict(zip(unique, counts))
X_train, X_test, Y_train, Y_test = train_test_split( X, target, train_size=0.5 )



model = LogisticRegression()

model.fit( X_train[[ 'lat', 'long', 'incident_rate' ]], Y_train )

Y_pred = model.predict( X_test[[ 'lat', 'long', 'incident_rate' ]] )



print('F1', sklearn.metrics.f1_score(Y_test, Y_pred, average='macro') )
Colombia_df3 = (df3[df3['country_region']=='Colombia'])[['lat', 'long', 'incident_rate', 'mortality_rate']]

Colombia_mortality = float(Colombia_df3['mortality_rate'])



if Colombia_mortality>=0 and Colombia_mortality<1:

    Colombia_test = 0 

elif Colombia_mortality>=1 and Colombia_mortality<10:

    Colombia_test = 1

elif Colombia_mortality>=10:

    Colombia_test = 2



print( 'Thrut = ' , Colombia_test , '; Predict = ', int(model.predict(Colombia_df3[['lat', 'long', 'mortality_rate']])) )   