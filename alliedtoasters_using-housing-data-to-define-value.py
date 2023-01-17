import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.ensemble.partial_dependence import partial_dependence

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.basemap import Basemap, cm



df = pd.read_csv('../input/Melbourne_housing_FULL.csv')

#Shuffle data to remove any grouping

df = df.sample(frac=1, random_state=111)
#Drop rows without pricing

priced = df[df.Price.notnull()]



#Drop rows without "longtitude" (sic)

priced = priced[priced.Longtitude.notnull()].copy()
#Mark whether landsize and building area info was provided

priced['has_landsize'] = np.where(priced.Landsize.notnull(), np.where(priced.Landsize > 0, True, False), False)

priced['has_buildingarea'] = np.where(priced.BuildingArea.notnull(), np.where(priced.BuildingArea > 0, True, False), False)



#Get ratio of building area to land size from rows with this information available

priced['ba_to_lnd'] = np.where(priced.has_buildingarea & priced.has_landsize, priced.BuildingArea/priced.Landsize, False)

#Get median ratio

a2lnd = priced[priced.ba_to_lnd != False].ba_to_lnd.median()

#Use ratio to fill in building area where land size is available

priced.BuildingArea = np.where(~priced.has_buildingarea & priced.has_landsize, priced.Landsize * a2lnd, priced.BuildingArea)

#Use ratio to fill in land size where building area is available

priced.Landsize = np.where(~priced.has_landsize & priced.has_buildingarea, priced.BuildingArea/a2lnd, priced.Landsize)



#Calculate the mean area per room where number of rooms and area is available

priced['rm_area'] = np.where(priced.has_buildingarea, priced.BuildingArea/priced.Rooms, 0)

#Get median ratio

md_rm_area = priced[priced.has_buildingarea].rm_area.median()

#Use median ratio to estimate building area by number of rooms where number of rooms is available

priced.BuildingArea = np.where(~priced.has_buildingarea & (priced.Rooms > 0), priced.Rooms*md_rm_area, priced.BuildingArea)



#Calculate number of rooms to lot size ratio.

priced['rm_lot'] = np.where(priced.has_landsize, priced.Landsize/priced.Rooms, 0)

#Get median ratio

md_lot_rm = priced[priced.has_landsize].rm_lot.median()

#Use median ratio to estimate lot size by number of rooms is number of rooms is given

priced.Landsize = np.where(~priced.has_landsize & (priced.Rooms > 0), priced.Rooms*md_lot_rm, priced.Landsize)



#Fill in ratio information where missing so we can use as features

priced.rm_area = np.where(priced.rm_area==0, md_rm_area, priced.rm_area)

priced.rm_lot = np.where(priced.rm_lot==0, md_lot_rm, priced.rm_lot)
def get_date(date_string):

    """Converts date format string 'dd/mm/yyyy' to

    an integer day: # of days after Jan 1, 2000 (01/01/2000)

    """

    if date_string[1] == '/':

        day = int(date_string[0])

        month = int(date_string[2:4])

        year = int(date_string[5:])

    else:

        day = int(date_string[:2])

        month = int(date_string[3:5])

        year = int(date_string[6:])

    return int((year-2000)*365 + int(month*30.4) + day)



#Apply function to data

priced['date_int'] = priced.Date.apply(get_date)

priced.date_int = priced.date_int - priced.date_int.min()

first_date_readable = priced.loc[priced.date_int.idxmin()].Date

print('first date in set: {}'.format(first_date_readable))



#Droppoing other various null values.

priced = priced[priced.Distance.notnull()] 

priced = priced[priced.Bedroom2.notnull()]

priced = priced[priced.Bathroom.notnull()]

priced = priced[priced.Car.notnull()]

priced.Car = np.where(priced.Car > 5, 5, priced.Car)

priced = priced[priced.BuildingArea.notnull()]

priced = priced[priced.Landsize.notnull()]

print('remaining rows: ', len(priced))
#Correct spelling of latitude and longitude

priced['Latitude'] = priced['Lattitude']

priced['Longitude'] = priced['Longtitude']



#Get our features together



fts = ['Distance', 

       'date_int', 

       'Propertycount', 

       'Longitude', 

       'Latitude', 

       'Bedroom2', 

       'Bathroom', 

       'Car', 

       'Rooms', 

       'BuildingArea',

       'has_buildingarea',

       'rm_lot',

       'rm_area',

       'Landsize',

       'has_landsize'

]



#Build dummies to encode information when necessary

features = pd.concat([pd.get_dummies(priced['Type']), pd.get_dummies(priced['Method']), priced[fts]], axis=1)
#To validate the performance of my model, I'll use an 80-20 training-test split.

cutoff = int(len(priced)*.8)

X = features

Y = priced.Price



X_train = features[:cutoff]

Y_train = priced[:cutoff].Price



X_test = features[cutoff:]

Y_test = priced[cutoff:].Price
#I've settled on these parameters, they work pretty well.

params = {

    'n_estimators': 4000,

    'learning_rate': .015,

    'max_depth': 6,

    'loss': 'huber'

}



#Instantiate model

gbr = GradientBoostingRegressor(**params)

gbr.fit(X_train, Y_train)



#Verify its performance on test set



Y_ = gbr.predict(X_test)



result = pd.DataFrame()

tolerated_error = .1

result['prediction'] = Y_

result['true_value'] = list(Y_test)

result['upper_bound'] = result.true_value + result.true_value*tolerated_error

result['lower_bound'] = result.true_value - result.true_value*tolerated_error

result['within_spec'] = np.where(((result.prediction >= result.lower_bound) & (result.prediction <= result.upper_bound)), True, False)

acc = len(result[result.within_spec])/len(result)

print('TEST SET:\npercent estimations within {}% of actual value: '.format(100*tolerated_error), str(100*acc)[:5], '\n')



result = pd.DataFrame()

tolerated_error = .25

result['prediction'] = Y_

result['true_value'] = list(Y_test)

result['upper_bound'] = result.true_value + result.true_value*tolerated_error

result['lower_bound'] = result.true_value - result.true_value*tolerated_error

result['within_spec'] = np.where(((result.prediction >= result.lower_bound) & (result.prediction <= result.upper_bound)), True, False)

acc = len(result[result.within_spec])/len(result)

print('percent estimations within {}% of actual value: '.format(100*tolerated_error), str(100*acc)[:5], '\n')
important_features = pd.DataFrame()

important_features['feature'] = features.columns

important_features['importance'] = gbr.feature_importances_
important_features.loc[important_features.importance.sort_values(ascending=False).index].head(15)
names = features.columns

fts = [8, 9, (16, 8), 15]

fig, axs = plot_partial_dependence(gbr, features, fts,

                                   feature_names=names,

                                   grid_resolution=50, figsize=(10, 8))

fig.suptitle('Partial dependence of housing value on features,\n'

             'Melbourne housing dataset')

plt.subplots_adjust(wspace=1)  # tight_layout causes overlap with suptitle

fig = plt.figure()



target_feature = (21, 8)

pdp, axes = partial_dependence(gbr, target_feature,

                               X=X_train, grid_resolution=50)

XX, YY = np.meshgrid(axes[0], axes[1])

Z = pdp[0].reshape(list(map(np.size, axes))).T

ax = Axes3D(fig)

surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,

                       cmap=plt.cm.BuPu, edgecolor='k')

ax.set_xlabel(names[target_feature[0]])

ax.set_ylabel(names[target_feature[1]])

ax.set_zlabel('Partial dependence')

#  pretty init view

ax.view_init(elev=22, azim=122)

plt.colorbar(surf)

plt.suptitle('Partial dependence of housing value on distance from\n'

             'central business district and property size')

plt.subplots_adjust(top=0.9)



plt.show()
extent = priced.Longitude.min(), priced.Longitude.max(), priced.Latitude.min(), priced.Latitude.max()



#fig = plt.figure()

map_ = Basemap(projection='merc',

               llcrnrlon=priced.Longitude.min(),

               llcrnrlat=priced.Latitude.min(),

               urcrnrlon=priced.Longitude.max(),

               urcrnrlat=priced.Latitude.max(),

               resolution='h'

)

#fig.canvas.draw()

#img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

#img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
features = [(11, 12)]



plt.set_cmap('Blues')

ax = map_.drawcoastlines(linewidth=3)

fig, axs = plot_partial_dependence(gbr, X_train, features,

                                   feature_names=names,

                                   n_jobs=3, grid_resolution=15,

                                  figsize=(5, 5), **{'ax':ax})

fig.suptitle('Partial dependence of housing price on latitude and longitude')

plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle