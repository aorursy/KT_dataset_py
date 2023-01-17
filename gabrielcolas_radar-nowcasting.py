# IMPORT THE LIBRARIES

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from keras.optimizers import *
# LOCATION OF THE TRAINING DATA

directory = '/kaggle/input/meteonet/'

zone = "NW"
fname_coords = directory + 'Radar_coords/Radar_coords/'+ f'radar_coords_{zone}.npz'

coords = np.load(fname_coords, allow_pickle=True)

#it is about coordinates of the top left corner of pixels -> it is necessary to get the coordinates of the center of pixels

#to perform a correct overlay of data

resolution = 0.01 #spatial resolution of radar data (into degrees)

lat = coords['lats']-resolution/2

lon = coords['lons']+resolution/2
# LOAD ANY RAINFALL RADAR PICKLE

def load_fichier(year,part_month,month):

    directory = '/kaggle/input/meteonet/'

    zone = "NW"

    fname = directory + f'{zone}_rainfall_{str(year)}/{zone}_rainfall_{str(year)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'

    pickle = np.load(fname, allow_pickle=True)

    return pickle
# Convert the given date of the pickle to the missing index.

def indice_miss_date(pickle):

    indice_tab= np.zeros(pickle['miss_dates'].shape[0],dtype=int)

    print("This is the missing date in the data: ",pickle['miss_dates'])

    for miss_date in pickle['miss_dates']:

        print(miss_date)

        compteur=0

        Diff_year = miss_date.year - pickle['dates'][0].year

        Diff_Month = miss_date.month - pickle['dates'][0].month

        Diff_Day = miss_date.day - pickle['dates'][0].day

        Diff_Hour = miss_date.hour - pickle['dates'][0].hour

        Diff_Minute = miss_date.minute - pickle['dates'][0].minute



        indice = Diff_Day*24*12 + Diff_Hour*12 + Diff_Minute/5



        indice_abs= int(indice) / indice

        # We raise error because for now we don't handle that kind of situation

        if Diff_year != 0 or Diff_Month != 0:

            raise NameError('There is a difference in month or year !')

        if indice_abs != 1:

            raise NameError('Indice is not a integer')

        else:

            indice_tab[compteur]= int(indice)

        compteur+=1

    return indice_tab





# cut the input data by the step we have consider

def cut_timestep(pickle,Step_minute):

    Step_data = 5

    Step = Step_minute // Step_data

    print("We make a step every ",Step,"indices")

    Val_selec=np.arange(0,len(pickle['dates']),Step) # Every 15 minits

    return pickle['data'][Val_selec,:,:],pickle['dates'][Val_selec]





# If Missing values.

def cut_timestep_miss(pickle,Echeance_minute,Step_minute) :

    Step_data = 5

    Step = Step_minute // Step_data

    Entrainement = Echeance_minute//15 + 1

    Prediction = Echeance_minute//15

    Tot = Entrainement + Prediction

    Qinit = (Step * Tot) - Step

    Q1= Step * Tot

  #  print('là',Q1)

    indice_tab= indice_miss_date(pickle) # LOAD indice_miss_date function to get the missing indices 

    Tab_index=[]

    

    valeur_init=0

    for indice in indice_tab:

        Quotient= indice // Q1

        Reste = indice % Q1

        if Quotient > 1:

            Index_end = Qinit + (Quotient - 1) * Q1

            New_begin_index = Index_end + Q1 + Step - 1

        elif Quotient == 1:

            Index_end = Qinit * Quotient

            New_begin_index = Index_end + Q1 + Step - 1

        else :

            Index_end = Qinit * Quotient

            New_begin_index = Index_end + Q1 - 1

        Sequence = np.arange(valeur_init , Index_end+1 , 3 , dtype=int)

        Tab_index.append(Sequence[:])

        valeur_init = New_begin_index

     #   print(indice,Quotient,Reste,Index_end,Sequence,New_begin_index,Tab_index,valeur_init)

    Sequence = np.arange(valeur_init ,pickle["dates"].shape[0] , 3 , dtype=int)

    Tab_index.append(Sequence[:])

    return Tab_index

def cut_data_and_coord(data,Pixel,lat,lon,lat_edge=49.5,lon_edge=-2.5):

    lat_only=lat[:,1]

    lon_only=lon[1,:]

    Temp_lat = np.where(lat_only < lat_edge)

    Temp_lon = np.where(lon_only < lon_edge)

    Lat_cut_index = Temp_lat[0][:Pixel]

    Lon_cut_index = Temp_lon[0][:Pixel]

    data_cut = data[:,Lat_cut_index]

    data_cut= data_cut[:,:,Lon_cut_index]

    return data_cut



def cut_timestep_miss2(pickle,Echeance_minute,Step_minute):

    Tab = cut_timestep_miss(pickle,Echeance_minute,Step_minute)

    shape = len(Tab)

    Tab_tot = np.empty(0,dtype=int)

    for i in range(shape):

        Tab_tot=np.concatenate((Tab_tot,Tab[i]))

        

    return pickle['data'][Tab_tot,:,:],pickle['dates'][Tab_tot]
# CODE TO CHANGE DATA BY 1 OR 0

def data_threshold(data_cut,rain_limit):

    data_cut[data_cut > rain_limit ] = 1

    data_cut[data_cut <= rain_limit ] = 0

    return data_cut
def XYTRAIN(data_process):

    X_Train=np.zeros((0,256,256,5))

    Y_Train=np.zeros((0,256,256,4))

    DATA_X=np.zeros((1,256,256,5))

    DATA_Y=np.zeros((1,256,256,4))

    pas_X=0

    pas_Y=4

    print("Valeur max:",int(data_process.shape[0]/9) )

    for globale in range(int(data_process.shape[0]/9)):

        for X in range(1,6):

            DATA_X[0,:,:,X-1]=data_process[pas_X+X-1,:,:]

        for Y in range(1,5):

            DATA_Y[0,:,:,Y-1]=data_process[pas_Y+Y,:,:]

        pas_X+=9

        pas_Y+=9

        X_Train= np.append(X_Train,DATA_X,axis=0)

        Y_Train= np.append(Y_Train,DATA_Y,axis=0)

        DATA_X=np.zeros((1,256,256,5))

        DATA_Y=np.zeros((1,256,256,4))

    return X_Train,Y_Train
def COUNT_RAIN_SITUATION(X_TRAIN,Y_TRAIN):

    Count_X=0

    Count_Y=0

    Tab_delete=[]

    Count=0

    for X in X_TRAIN:

        Temp_x = np.count_nonzero(X == 1)

        if (Temp_x <= 200):

            Count_X += 1

            Tab_delete.append(Count)

        Count +=1

        

    for Y in Y_TRAIN:

        Temp_y = np.count_nonzero(Y == 1)

        if Temp_y <= 200:

            Count_Y += 1

            

    X_TRAIN = np.delete(X_TRAIN,Tab_delete,axis=0)

    Y_TRAIN = np.delete(Y_TRAIN,Tab_delete,axis=0)

        

    return(X_TRAIN,Y_TRAIN,Count_X,Count_Y)
# CONFIGURATION DATA

# CHOOSE THE STEP AND THE ECHEANCE

Echeance_minute = 60

Step_minute = 15

Pixel=256

rain_limit = 0
# GLOBAL CODE FOR GIVEN THE DATASET TO LEARN THE MODEL

month=[1,2,3]

part=[1,2,3]

year=[2016,2017]
## LOOP FOR TRAIN DATA

init=0

for m in month:

    for p in part:

        pickle = load_fichier(2016,p,m)

        print("This is the part " ,p)

    

        if pickle["miss_dates"].shape[0] == 0:

            print("There is no missing data in this chunk")

            data_radar,dates_radar = cut_timestep(pickle,Step_minute)

            print( data_radar.shape)

            data_cut = cut_data_and_coord(data_radar,Pixel,lat,lon)

            print( data_cut.shape)       

            data_process = data_threshold(data_cut,rain_limit)

            print("Data have been threshold")

            X_TRAIN,Y_TRAIN = XYTRAIN(data_process)

            print("Shape of the temporary train data",X_TRAIN.shape,Y_TRAIN.shape)

        

        else:

            print("There is miss dates")

            data_radar,dates_radar = cut_timestep_miss2(pickle,Echeance_minute,Step_minute)

            print(data_radar.shape)

            data_cut = cut_data_and_coord(data_radar,Pixel,lat,lon)

            print( data_cut.shape)

            data_process = data_threshold(data_cut,rain_limit)

            print("Data have been threshold")

            X_TRAIN,Y_TRAIN = XYTRAIN(data_process)

            print("Shape of the temporary train data",X_TRAIN.shape,Y_TRAIN.shape)

        

        if (init == 0):

            X_TEMP=X_TRAIN

            Y_TEMP=Y_TRAIN

            print("STEP N°1: ",X_TEMP.shape,Y_TEMP.shape)

        elif ( init == 1):

            X_train= np.append(X_TEMP,X_TRAIN,axis=0)

            Y_train= np.append(Y_TEMP,Y_TRAIN,axis=0)

            print("STEP N°2: ",X_train.shape,Y_train.shape)

        else:

            X_train= np.append(X_train,X_TRAIN,axis=0)

            Y_train= np.append(Y_train,Y_TRAIN,axis=0)

            print("OTHER STEP : ",X_train.shape,Y_train.shape)

        init += 1

        print("----------------------------------------------------------------------")

    

del(X_TEMP,Y_TEMP,X_TRAIN,Y_TRAIN,data_cut,data_radar,data_process)
# SHOW THE SITUATION WITHOUT METEO DATA

print(X_train.shape,Y_train.shape)

X_train,Y_train,Count_X,Count_Y = COUNT_RAIN_SITUATION(X_train,Y_train)

print(X_train.shape,Y_train.shape,Count_X,Count_Y)
inputs = keras.Input(shape=(256,256,5))



conv1 = layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu", padding="same")(inputs)

conv1 = layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu", padding="same")(conv1)



maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)



conv2 = layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu", padding="same")(maxpool1)

conv2 = layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu", padding="same")(conv2)



maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)



conv3 = layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu", padding="same")(maxpool2)

conv3 = layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu", padding="same")(conv3)



maxpool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)



conv4 = layers.Conv2D(filters=256,kernel_size=(3,3),activation="relu", padding="same")(maxpool3)

conv4 = layers.Conv2D(filters=256,kernel_size=(3,3),activation="relu", padding="same")(conv4)

drop4 = layers.Dropout(0.5)(conv4)



maxpool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)



conv5 = layers.Conv2D(filters=512,kernel_size=(3,3),activation="relu", padding="same")(maxpool4)

conv5 = layers.Conv2D(filters=512,kernel_size=(3,3),activation="relu", padding="same")(conv5)

drop5 = layers.Dropout(0.5)(conv5)





up6 = layers.Conv2D(256, 2, activation = 'relu', padding="same")(layers.UpSampling2D(size = (2,2))(drop5))

merge6 = layers.concatenate([drop4,up6], axis = 3)



conv6 = layers.Conv2D(256, 3, activation = 'relu', padding="same")(merge6)

conv6 = layers.Conv2D(256, 3, activation = 'relu', padding="same")(conv6)







up7 = layers.Conv2D(128, 2, activation = 'relu', padding="same")(layers.UpSampling2D(size = (2,2))(conv6))

merge7 = layers.concatenate([conv3,up7], axis = 3)

conv7 = layers.Conv2D(128, 3, activation = 'relu', padding="same")(merge7)

conv7 = layers.Conv2D(128, 3, activation = 'relu', padding="same")(conv7)







up8 = layers.Conv2D(64, 2, activation = 'relu', padding="same")(layers.UpSampling2D(size = (2,2))(conv7))

merge8 = layers.concatenate([conv2,up8], axis = 3)

conv8 = layers.Conv2D(64, 3, activation = 'relu', padding="same")(merge8)

conv8 = layers.Conv2D(64, 3, activation = 'relu', padding="same")(conv8)



up9 = layers.Conv2D(32, 2, activation = 'relu', padding="same")(layers.UpSampling2D(size = (2,2))(conv8))

merge9 = layers.concatenate([conv1,up9], axis = 3)

conv9 = layers.Conv2D(32, 3, activation = 'relu', padding="same")(merge9)

conv9 = layers.Conv2D(32, 3, activation = 'relu' , padding="same")(conv9)

conv10 = layers.Conv2D(4, 3, activation = 'sigmoid', padding="same")(conv9)





model= keras.Model(inputs,conv10)

model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

model.summary()
history=model.fit(X_train,Y_train,batch_size=32, epochs=10)
# MAKE PLACE IN MEMORY

del(X_train,Y_train)
epoch=[1,2,3,4,5,6,7,8,9,10]



import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(epoch,history.history['loss'],'o-',label="loss")

ax.set_xlabel("Epochs")

ax.set_ylabel("Cross_entropy loss")

ax.legend()

ax.set_title('Loss plot for epoch iteration')

plt.savefig("Loss_Train")
fig, ax = plt.subplots()

ax.plot(epoch,history.history['binary_accuracy'],'o-',label="accuracy")

ax.set_xlabel("Epochs")

ax.set_ylabel("Accuracy metrics")

ax.legend()

ax.set_title('Accuracy plot for epoch iteration')

plt.savefig("Accuracy_Train")
# VARIABLE FOR TEST DATA

year_test= 2018

month_test = [1]

part_test=[1,2]
## LOOP FOR TEST DATA

init=0

for p in part_test:

    pickle = load_fichier(year_test,p,1)

    print("This is the part " ,p)

    

    if pickle["miss_dates"].shape[0] == 0:

        print("There is no missing data in this chunk")

        data_radar,dates_radar = cut_timestep(pickle,Step_minute)

        print( data_radar.shape)

        data_cut = cut_data_and_coord(data_radar,Pixel,lat,lon)

        print( data_cut.shape)       

        data_process = data_threshold(data_cut,rain_limit)

        print("Data have been threshold")

        X_TEST,Y_TEST = XYTRAIN(data_process)

        print("Shape of the temporary train data",X_TEST.shape,Y_TEST.shape)

      #  X_TEST,Y_TEST = PERSISTANCE(data_process)

        

    else:

        print("There is miss dates")

        data_radar,dates_radar = cut_timestep_miss2(pickle,Echeance_minute,Step_minute)

        print(data_radar.shape)

        data_cut = cut_data_and_coord(data_radar,Pixel,lat,lon)

        print( data_cut.shape)

        data_process = data_threshold(data_cut,rain_limit)

        print("Data have been threshold")

        X_TEST,Y_TEST = XYTRAIN(data_process)

        print("Shape of the temporary train data",X_TEST.shape,Y_TEST.shape)

     #   X_TEST,Y_TEST = PERSISTANCE(data_process)

        

    if (init == 0):

        X_TEMP=X_TEST

        Y_TEMP=Y_TEST

        print("STEP N°1: ",X_TEMP.shape,Y_TEMP.shape)

    elif ( init == 1):

        X_test= np.append(X_TEMP,X_TEST,axis=0)

        Y_test= np.append(Y_TEMP,Y_TEST,axis=0)

        print("STEP N°2: ",X_test.shape,Y_test.shape)

    else:

        X_test= np.append(X_test,X_TEST,axis=0)

        Y_test= np.append(Y_test,Y_TEST,axis=0)

        print("OTHER STEP : ",X_test.shape,Y_test.shape)

    init += 1

    print("----------------------------------------------------------------------")

    

del(X_TEST,Y_TEST,data_cut,data_radar,data_process)
# EVALUATE THE MODEL ON TEST DATA.

print("Evaluate on test data")

results = model.evaluate(X_test, Y_test, batch_size=20)

# EVALUATE THE MODEL WITH THAT

print("test loss, test acc:", results)

Y_predict = model.predict(X_test)

#Y_predict.shape
from sklearn.metrics import log_loss

#accuracy = accuracy_score(Y_predict,Y_test)

log_loss= log_loss(np.ravel(Y_test[:,:,:,3]), np.ravel(Y_predict[:,:,:,3]))

print(log_loss)
print(log_loss)

log_loss_tab= [None,0.15,0.2,0.24,0.26]
del(X_test,Y_test)

#del(model)
def LATLON_CUT(lat,lon):

    lat_edge=49.5

    lon_edge=-2.5

    lat_only=lat[:,1]

    lon_only=lon[1,:]

    Temp_lat = np.where(lat_only < lat_edge)

    Temp_lon = np.where(lon_only < lon_edge)

    Lat_cut_index = Temp_lat[0][:Pixel]

    Lon_cut_index = Temp_lon[0][:Pixel]

    lat_format= lat[Lat_cut_index,Lon_cut_index]

    lon_format= lon[Lat_cut_index,Lon_cut_index]

    return lat_format,lon_format
# Lat/Lon formating for plot.

lat_format,lon_format = LATLON_CUT(lat,lon)
import cartopy.crs as ccrs

from cartopy.mpl.geoaxes import GeoAxes

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import AxesGrid

from matplotlib import colors

import cartopy.feature as cfeature

import numpy as np



lllat = lat_format[-1]  #lower left latitude

urlat = lat_format[0]  #upper right latitude

lllon =  lon_format[0]  #lower left longitude

urlon =lon_format[-1]  #upper right longitude

extent = [lllon, urlon, lllat, urlat]

cmap = colors.ListedColormap(['silver','#85F599','blue','#FFFF57','#FFC400','#FF2200'])

cmap = colors.ListedColormap(['white', 'darkslateblue', 

                              'skyblue','cyan','lime','yellow',

                              'orange','brown','red','plum'])

bounds = [0,0.4,0.5,0.55,0.6,0.65,0.7,0.8,0.9,1]

norm = colors.BoundaryNorm(bounds, cmap.N)





lats,lons = np.meshgrid(lat_format,lon_format)



projection = ccrs.PlateCarree()

axes_class = (GeoAxes,dict(map_projection=projection))



fig=plt.figure(figsize=(80,80))

axgr= AxesGrid(fig, 222, axes_class=axes_class,

                    nrows_ncols=(2, 9),

                    axes_pad=0.2,

                    cbar_location='right',

                    cbar_mode= 'single',

                    cbar_size='5%',

                    label_mode='')

                   # shared_all=True)  # note the empty label_mode

Pas= 3

for i, ax in enumerate(axgr):

    ax.coastlines(resolution='50m', linewidth=2)

    ax.gridlines()

    ax.add_feature(cfeature.BORDERS.with_scale('50m'))



    if i < 5:

        p = ax.imshow(X_test[Pas,:,:,i],cmap=cmap,norm=norm, interpolation='none', origin='upper',extent=extent)

    elif i >= 5 and i < 9:

        p = ax.imshow(Y_test[Pas,:,:,i-5],cmap=cmap,norm=norm, interpolation='none', origin='upper',extent=extent)

    elif i >=9 and i < 14:

        p=  ax.imshow(X_test[Pas,:,:,i-9],cmap=cmap, interpolation='none', origin='upper',extent=extent)

    elif i >=14 and i < 19:

        p=  ax.imshow(Y_predict[Pas,:,:,i-14],cmap=cmap, interpolation='none', origin='upper',extent=extent)

axgr.cbar_axes[0].colorbar(p)

plt.show()

fig.savefig("Echeance 3")