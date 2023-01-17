import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

data = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/craigslistVehicles.csv');
[density, edges] = np.histogram(data['price'], np.logspace(2,9, 8));

fig = plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')

plt.bar(x =range(7), height = np.log(density))

plt.xticks(range(7), (edges[1:]-edges[0:-1])/2, rotation=45)

plt.title('Number of vehicles (logarithmic) vs price range')

plt.xlabel('Price range')

plt.ylabel('# of vehicles (log)');

current_year = 2019;

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='constant', fill_value=-1, verbose=1);

data['year'] = imputer.fit_transform(np.reshape(data['year'].values, (len(data),1)))
imputer = SimpleImputer(missing_values = current_year+1, strategy='constant', fill_value=-1, verbose=1);

data['year'] = imputer.fit_transform(np.reshape(data['year'].values, (len(data),1)))

data['Age'] = current_year - data['year'];
aux = data['Age'].value_counts().to_dict()

x = np.sort(list(aux.keys()));

y = [aux[i] for i in x]

plt.figure(figsize = (16,8));

plt.bar(x[:-1], y[:-1]);

plt.xlabel('Age (years)');

plt.ylabel('# of cars on sale');

plt.title('Volume of cars on sale considering their year of manufacturing');

import scipy.stats as stats    

import random

fit_alpha, fit_loc, fit_beta=stats.gamma.fit(data['Age'][data['Age'] < 140])

y_gamma = stats.gamma.rvs(fit_alpha, loc=fit_loc, scale=1/fit_beta, size=1000000, random_state=1492);

y_gamma_values = plt.hist(y_gamma,bins=121);

plt.close()



fit_alpha, fit_beta, fit_loc, fit_scale = stats.beta.fit(data['Age'][data['Age'] < 140])

y_beta = stats.beta.rvs(fit_alpha, fit_beta, loc=fit_loc, scale=fit_scale, size=1000000, random_state=1492);

y_beta_values = plt.hist(y_beta,bins=121);

plt.close()
y_data = [];

for i in range(121):

    try:

        y_data.append(aux[i]);

    except:

        y_data.append(0);

plt.figure(figsize = (16,8))

est_data =  y_data/np.sum(y_data);

est_gamma = y_gamma_values[0]/np.sum(y_gamma_values[0]);

est_beta = y_beta_values[0]/np.sum(y_beta_values[0]);

plt.bar(range(121), est_data, label = 'Data')

plt.plot(range(121), est_gamma, alpha = 1, label = 'Gamma distribution', color='r',linewidth = 5)

plt.plot(range(121), est_beta, alpha = 1, label = 'Beta distribution', color='g',linewidth = 5)

plt.legend();

plt.xlabel('Age (year)');

plt.ylabel('Price weight');

plt.title('Comparison of two statistical distributions against the distribution of prices of our dataset.');
ages = np.unique(data['Age']);

fig = plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')



for i in range(len(ages)):

    if ages[i]!= (current_year+1):

        y = data['price'][data['Age'] == ages[i]]

        x = data['Age'][data['Age'] == ages[i]]

        plt.scatter(x,(y))



plt.xlabel('Age (years)')

plt.ylabel('Price (\$)')

plt.yscale('symlog')

plt.ylim(0, 1e10);

plt.grid()

plt.title('Raw prices');
z_lim = 3;

def z_score_cleaning(y_vec, z_lim):

    work = True;

    

    while work == True:

        ave   = np.mean(y_vec);

        stdev = np.std(y_vec);

        Z = np.abs(y_vec - ave)/stdev;

        if np.max(Z) > z_lim:

            y_vec = y_vec[Z <= z_lim];

        else:

            work = False;

    return ave, stdev, y_vec
ages = np.unique(data['Age']);

fig = plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')



for i in range(len(ages)):

    if ages[i]!= (current_year+1):

        y = data['price'][data['Age'] == ages[i]]

        dummy, dummy, y_vec = z_score_cleaning(y, z_lim)

        x = np.ones(len(y_vec))*ages[i];

        plt.scatter(x,y_vec)



plt.xlabel('Age (years)')

plt.ylabel('Price (\$)')

plt.grid()

plt.title('Prices after removing outliers using Z-scores');
manufacturers_counts = data['manufacturer'].value_counts().to_dict()

fig = plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')

plt.bar(manufacturers_counts.keys(), manufacturers_counts.values())

plt.xticks(rotation = 45);

plt.xlabel('Manufacturer')

plt.ylabel('# of cars on sale')
fig = plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')

for j in range(9):

    manufacturer = list(manufacturers_counts.keys())[j];

    plt.subplot(3,3,j+1)

    for i in range(len(ages)):

        if ages[i]!= (current_year+1):

            y = data[data['manufacturer']==manufacturer]['price'][data['Age'] == ages[i]]

            x = np.ones(len(y))*ages[i];

            #x = data[data['manufacturer']==manufacturer]['Age'][data['Age'] == ages[i]]

            plt.scatter(x,(y))

    if j > 5:

        plt.xlabel('Age (years)')

    plt.ylabel('Price (\$)')

    plt.yscale('symlog')

    plt.ylim(2, 1e6);

    plt.xlim(0, 120);

    plt.grid();

    plt.title('{0}'.format(manufacturer));
fig = plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')

for j in range(9):

    manufacturer = list(manufacturers_counts.keys())[j];

    plt.subplot(3,3,j+1)

    for i in range(len(ages)):

        if ages[i]!= (current_year+1):

            y = data[data['manufacturer']==manufacturer]['price'][data['Age'] == ages[i]]

            if len(y) > 10:

                ave, stdev, dummy = z_score_cleaning(y, z_lim)

                x = ages[i];

                plt.errorbar(x, ave, yerr=stdev)

    if j > 5:

        plt.xlabel('Age (years)')

    plt.ylabel('Price (\$)')

    plt.xlim(0, 120);

    plt.grid();

    plt.title('{0}'.format(manufacturer));

# pd.isna(data['manufacturer'])

Europe_cars = ['alfa-romeo','aston-martin', 'audi','bmw','ferrari','fiat','jaguar','land rover', 'porche','mercedes-benz',  'morgan','volkswagen', 'volvo', 'rover', 'mini']

Asia_cars = [  'kia', 'infiniti',  'hyundai', 'acura', 'datsun', 'honda', 'lexus', 'mazda', 'mitsubishi','nissan', 'subaru', 'toyota']   

USA_cars = ['lincoln','hennessey','saturn','buick', 'cadillac', 'chevrolet', 'chrysler', 'dodge',  'ford', 'gmc', 'harley-davidson', 'jeep', 'pontiac', 'ram', 'mercury']

manufacturer_region = [];

for i in data['manufacturer']:

    if i in Europe_cars:

        manufacturer_region.append('EU');

    elif i in Asia_cars:

        manufacturer_region.append('AS');

    elif i in USA_cars:

        manufacturer_region.append('USA')

    else:

        manufacturer_region.append('NONE')       



data['Manufacturer_region'] = manufacturer_region;
aux = data['Manufacturer_region'].value_counts().to_dict();

radio_usa = aux['USA']/sum(aux.values());

eps = 0.025;

# radio_sum = radio*2;

circle1 = plt.Circle((radio_usa+eps, radio_usa), 

                     radio_usa,

                     linewidth=5,

                     #facecolor='r',

                     facecolor=((255/255,102/255,102/255,1)) ,

                     edgecolor = ((255/255,0/255,0/255,1))

                    )



radio_asia = aux['AS']/sum(aux.values());

circle2 = plt.Circle((2*radio_usa+radio_asia+eps, radio_asia), 

                     radio_asia,

                     linewidth=5,

                     facecolor= ((178/255,255/255,102/255,1)) ,

                     edgecolor =  ((0/255,255/255,0/255,1))

                    )



radio_europe = aux['EU']/sum(aux.values());

circle3 = plt.Circle((2*(radio_usa+radio_asia)+radio_europe+eps,radio_europe),

                     radio_europe, 

                     linewidth=5,

                     facecolor=((153/255,153/255,255/255,1)) ,

                     edgecolor = ((0/255,0/255,255/255,1)) 

                    )

fig, ax = plt.subplots(figsize = (16,8)) 

ax.add_artist(circle1)

ax.add_artist(circle2)

ax.add_artist(circle3)

plt.text(radio_usa-eps,radio_usa*2+eps,

         'USA\n{0:.2f}%'.format(radio_usa*100),

         fontsize=25)

plt.text(radio_usa*2+radio_asia*3/4,radio_asia*2+eps,

         'Asia\n{0:.2f}%'.format(radio_asia*100),

         fontsize=25)

plt.text(radio_usa*2+radio_asia*2+radio_europe/2,radio_europe*2+eps,

         'EU\n{0:.2f}%'.format(radio_europe*100),

         fontsize=25)



plt.xlim(0,2);

plt.ylim(0, 1.5);
#  z_score_cleaning(y_vec, z_lim)

y_out = {};

age_range = [0, 10, 20,140];



for i_range in range(len(age_range)-1):

    y_aux = [];

    for manufacturer in ['USA','AS','EU']:

        y = data['price'][(data['Manufacturer_region'] == manufacturer) & 

                          (data['Age']>=age_range[i_range]) 

                          & (data['Age']<age_range[i_range+1])]

        [dummy, dummy, y_clean] = z_score_cleaning(y, z_lim);

        y_aux.append(y_clean);

    y_out[i_range] = y_aux;

    



fig = plt.figure(figsize=(16,6))

for i_range in range(len(age_range)-1):

    plt.subplot(1,3,i_range+1)

    plt.boxplot(y_out[i_range],labels=['USA','Asia','Europe']);

    if i_range == 0:

        plt.ylabel('Price ($)');

        plt.title('Age between 0 and 10 years (young)')

    elif i_range == 1:

        plt.title('Age between 10 and 20 years (middle)')

    elif i_range == 2:

        plt.title('Age older than 20 years (old)')

    plt.ylim(0,60000)

    plt.grid()