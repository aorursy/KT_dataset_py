#Librerías

import numpy as np

import matplotlib.pyplot as plt

from netCDF4 import Dataset

import cartopy.crs as ccrs  # Plot maps

import os

from util_img import *

import fnmatch

Inbox = "../input/goes-l1/GOES_ABIL1nc/" #carpeta principal
# Parámetros de las imágenes

nbandas = 16 #Cantidad de bandas

#nbins =  #?? eran de 16 numero de valores de la imagen 



hora = 's20180961500406' #s20183471515

#img_extent = (-5500000, 5500000, -5500000, 5500000) #dimensiones de la escena en m

img_extent = (-5434894.67527,5434894.67527,-5434894.67527,5434894.67527)

psize = 500 # lado de pixel de referencia en m

N = 5424*4 #numero de pixeles de referencia



filas = 1440 # filas del recorte para la de referencia

columnas = 1440 # filas del recorte para la de referencia



x0=1438000 # Coordenada x del limite superior izquierdo en m

y0=-2441000 # Coordenada y del limite superior izquierdo en m



img_extentr = [x0,x0+columnas*psize,y0-filas*psize,y0]
# combinaciones de canales

total_canales = ['C01','C02','C03', 'C04','C05','C06', 'C07','C08','C09', 'C10','C11','C12','C13','C14','C15','C16']

canal =['C02','C03','C01']

canal2 =['C13','C14','C15']

ch_vapor = ['C08','C09','C10'] #distintos niveles de vapor de agua

ch_nyc = ['C04','C05','C06'] #nieve y cirrus

ch_sur = ['C07','C13','C01'] #nubes convectivas, estratos, superficie
imagRGB = np.zeros([filas, columnas,3])

imagRGB2 = np.zeros([filas, columnas,3])

RGBvapor = np.zeros([filas, columnas,3])

RGBnyc = np.zeros([filas, columnas,3])

RGBsur = np.zeros([filas, columnas,3])
for i in range(3):



    for file_name in os.listdir(Inbox):

        if fnmatch.fnmatch(file_name, '*'+ch_vapor[i]+'*'+hora+'*'):

            img_name=file_name

            

            print ('Importando la imagen: %s' %img_name)

            imagenobj = Dataset(Inbox + img_name, 'r')



            print ('Importando las variables de la imagen: %s' %img_name)

            metadato = imagenobj.variables

            altura=metadato['goes_imager_projection'].perspective_point_height

            semieje_may=metadato['goes_imager_projection'].semi_major_axis

            semieje_men=metadato['goes_imager_projection'].semi_minor_axis

            lon_cen=metadato['goes_imager_projection'].longitude_of_projection_origin

            pol=semieje_may*altura/(semieje_may+altura)

            ecu=semieje_men*altura/(semieje_may+altura)

# img_extent = (-ecu,ecu,-pol,pol)

# img_extent = (-pol,pol,-ecu,ecu)

            icanal = int(metadato['band_id'][:])

            print ('Canal %d' %icanal)



# Recortes crudos

            esc=int(N/metadato['Rad'][:].shape[0])

            Nx=int(columnas/esc) #numero de puntos del recorte en x

            Ny=int(filas/esc) #numero de puntos del recorte en x

            f0=int((-y0/psize+N/2-1.5)/esc) #fila del angulo superior izquierdo

            c0=int((x0/psize+N/2+.5)/esc) #columna del angulo superior izquierdo

            f1=int(f0+Ny) #fila del angulo inferior derecho

            c1=int(c0+Nx) #columna del angulo inferior derecho



            im_rec = metadato['Rad'][:].data[f0:f1,c0:c1]



            plt.figure()

            vmin=im_rec.min()

            vmax=im_rec.max()



            if icanal >= 7 :

                img=plt.imshow(im_rec,vmin=vmin,vmax=vmax,cmap='Greys') #emisivas

            else:

                img=plt.imshow(im_rec,vmin=vmin,vmax=vmax,cmap='Greys_r') #reflectivas

            plt.title(canal[i])

            plt.show()



#%%calibracion 

            print ('Calibrando la imagen')

            pendiente= metadato['Rad'].scale_factor

            ordenada= metadato['Rad'].add_offset    

        #imag_rad =im_rec*pendiente+ordenada #DN ->mW m-2 sr-1 mum-1

            Unit = "Radiancia ["+metadato['Rad'].units+"]"



            if icanal >=7:

        #Parámetros de calibracion

                fk1 = metadato['planck_fk1'][0] # DN -> K

                fk2 = metadato['planck_fk2'][0]

                bc1 = metadato['planck_bc1'][0]

                bc2 = metadato['planck_bc2'][0]

                imag_cal = (fk2 / (np.log((fk1 / im_rec) + 1)) - bc1 ) / bc2-273.15 # DN -> C

                Unit = "Temperatura de Brillo [°C]"

            else:

                k0=imagenobj.variables['kappa0'][0]

                imag_cal = im_rec*k0

                Unit = "Reflectancia"

    

    # print (imagenobj.variables['max_radiance_value_of_valid_pixels'][0])

            print('Interpolando')

            x,y,imag_calm =muestreo(range(f0,f1),range(c0,c1),imag_cal,esc=esc)



            print('Realzando')

            vmin=0

            vmax=imag_calm[1000:,:600].max()

            imag_calm=realce_lineal(vmin,vmax,imag_calm)



            RGBvapor[:imag_calm.shape[0],:imag_calm.shape[1],i]=imag_calm

    

            del imag_cal,im_rec

        

            print("Graficando")

            plt.figure()

            crs=ccrs.Geostationary(central_longitude=lon_cen, satellite_height=altura) #proyeccion geoestacionaria para Goes16

            ax = plt.axes(projection=crs)

            # ax = plt.axes(projection=ccrs.Geostationary(central_longitude=lon_cen)) #proyeccion geoestacionaria para Goes16

            ax.gridlines() #agrega linea de meridianos y paralelos 

            ax.coastlines(resolution='10m',color='blue') #agrega líneas de costa

            img = plt.imshow(RGBvapor,extent=img_extentr)

            plt.show()
RGBvapor.shape
# a)-Distintos nivles de vapor de agua



#Esto me va a servir para probar distintas combinaciones RGB automaticamente:

from itertools import permutations

p = list(permutations(range(3),3)) 

#Me da una lista de 3-tuplas con todas las posibles formas de ordenas los nros 0,1,2



vapor = np.zeros([filas,columnas,3]) # Defino un array nuevo para cargar la imagen ahi



for i in range(len(p)): 

    # Por ej para la combicion(0,2,1) que es la 2da en la lista 'p' de tuplas

    # vapor[:,:,0]=sur8, vapor[:,:,2]=sur9, vapor[:,:,1]=sur10

    

    vapor[:,:,p[i][0]]=RGBvapor[:,:,0] 

    vapor[:,:,p[i][1]]=RGBvapor[:,:,1] 

    vapor[:,:,p[i][2]]=RGBvapor[:,:,2] 

    plt.figure(i+1)

    plt.title('Vapor: ' + str(p[i][0])+'=B8, '+str(p[i][1])+'=B9, '+str(p[i][2])+'=B10')

    plt.imshow(vapor)
for i in range(3):



    for file_name in os.listdir(Inbox):

        if fnmatch.fnmatch(file_name, '*'+ch_nyc[i]+'*'+hora+'*'):

            img_name=file_name

            

            print ('Importando la imagen: %s' %img_name)

            imagenobj = Dataset(Inbox + img_name, 'r')



            print ('Importando las variables de la imagen: %s' %img_name)

            metadato = imagenobj.variables

            altura=metadato['goes_imager_projection'].perspective_point_height

            semieje_may=metadato['goes_imager_projection'].semi_major_axis

            semieje_men=metadato['goes_imager_projection'].semi_minor_axis

            lon_cen=metadato['goes_imager_projection'].longitude_of_projection_origin

            pol=semieje_may*altura/(semieje_may+altura)

            ecu=semieje_men*altura/(semieje_may+altura)

# img_extent = (-ecu,ecu,-pol,pol)

# img_extent = (-pol,pol,-ecu,ecu)

            icanal = int(metadato['band_id'][:])

            print ('Canal %d' %icanal)



# Recortes crudos

            esc=int(N/metadato['Rad'][:].shape[0])

            Nx=int(columnas/esc) #numero de puntos del recorte en x

            Ny=int(filas/esc) #numero de puntos del recorte en x

            f0=int((-y0/psize+N/2-1.5)/esc) #fila del angulo superior izquierdo

            c0=int((x0/psize+N/2+.5)/esc) #columna del angulo superior izquierdo

            f1=int(f0+Ny) #fila del angulo inferior derecho

            c1=int(c0+Nx) #columna del angulo inferior derecho



            im_rec = metadato['Rad'][:].data[f0:f1,c0:c1]



            plt.figure()

            vmin=im_rec.min()

            vmax=im_rec.max()



            if icanal >= 7 :

                img=plt.imshow(im_rec,vmin=vmin,vmax=vmax,cmap='Greys') #emisivas

            else:

                img=plt.imshow(im_rec,vmin=vmin,vmax=vmax,cmap='Greys_r') #reflectivas

            plt.title(icanal)

            plt.show()



#%%calibracion 

            print ('Calibrando la imagen')

            pendiente= metadato['Rad'].scale_factor

            ordenada= metadato['Rad'].add_offset    

        #imag_rad =im_rec*pendiente+ordenada #DN ->mW m-2 sr-1 mum-1

            Unit = "Radiancia ["+metadato['Rad'].units+"]"



            if icanal >=7:

        #Parámetros de calibracion

                fk1 = metadato['planck_fk1'][0] # DN -> K

                fk2 = metadato['planck_fk2'][0]

                bc1 = metadato['planck_bc1'][0]

                bc2 = metadato['planck_bc2'][0]

                imag_cal = (fk2 / (np.log((fk1 / im_rec) + 1)) - bc1 ) / bc2-273.15 # DN -> C

                Unit = "Temperatura de Brillo [°C]"

            else:

                k0=imagenobj.variables['kappa0'][0]

                imag_cal = im_rec*k0

                Unit = "Reflectancia"

    

    # print (imagenobj.variables['max_radiance_value_of_valid_pixels'][0])

            print('Interpolando')

            x,y,imag_calm =muestreo(range(f0,f1),range(c0,c1),imag_cal,esc=esc)



            print('Realzando')

            vmin=0

            vmax=imag_calm[1000:,:600].max()

            imag_calm=realce_lineal(vmin,vmax,imag_calm)



            RGBnyc[:imag_calm.shape[0],:imag_calm.shape[1],i]=imag_calm

    

            del imag_cal,im_rec

        

            print("Graficando")

            plt.figure()

            crs=ccrs.Geostationary(central_longitude=lon_cen, satellite_height=altura) #proyeccion geoestacionaria para Goes16

            ax = plt.axes(projection=crs)

            # ax = plt.axes(projection=ccrs.Geostationary(central_longitude=lon_cen)) #proyeccion geoestacionaria para Goes16

            ax.gridlines() #agrega linea de meridianos y paralelos 

            ax.coastlines(resolution='10m',color='blue') #agrega líneas de costa

            img = plt.imshow(RGBnyc,extent=img_extentr)

            plt.show()
#%% b)- Cirrus y nieve

cyn = np.zeros([filas,columnas,3])



for i in range(len(p)): 

    cyn[:,:,p[i][0]]=RGBnyc[:,:,0]

    cyn[:,:,p[i][1]]=RGBnyc[:,:,1]

    cyn[:,:,p[i][2]]=RGBnyc[:,:,2] 

    plt.figure(i+11)

    plt.title('Cirrus y nieve: ' + str(p[i][0])+'=B4, '+str(p[i][1])+'=B6, '+str(p[i][2])+'=B5')

    plt.imshow(cyn)
for i in range(3):



    for file_name in os.listdir(Inbox):

        if fnmatch.fnmatch(file_name, '*'+ch_sur[i]+'*'+hora+'*'):

            img_name=file_name

            

            print ('Importando la imagen: %s' %img_name)

            imagenobj = Dataset(Inbox + img_name, 'r')



            print ('Importando las variables de la imagen: %s' %img_name)

            metadato = imagenobj.variables

            altura=metadato['goes_imager_projection'].perspective_point_height

            semieje_may=metadato['goes_imager_projection'].semi_major_axis

            semieje_men=metadato['goes_imager_projection'].semi_minor_axis

            lon_cen=metadato['goes_imager_projection'].longitude_of_projection_origin

            pol=semieje_may*altura/(semieje_may+altura)

            ecu=semieje_men*altura/(semieje_may+altura)

# img_extent = (-ecu,ecu,-pol,pol)

# img_extent = (-pol,pol,-ecu,ecu)

            icanal = int(metadato['band_id'][:])

            print ('Canal %d' %icanal)



# Recortes crudos

            esc=int(N/metadato['Rad'][:].shape[0])

            Nx=int(columnas/esc) #numero de puntos del recorte en x

            Ny=int(filas/esc) #numero de puntos del recorte en x

            f0=int((-y0/psize+N/2-1.5)/esc) #fila del angulo superior izquierdo

            c0=int((x0/psize+N/2+.5)/esc) #columna del angulo superior izquierdo

            f1=int(f0+Ny) #fila del angulo inferior derecho

            c1=int(c0+Nx) #columna del angulo inferior derecho



            im_rec = metadato['Rad'][:].data[f0:f1,c0:c1]



            plt.figure()

            vmin=im_rec.min()

            vmax=im_rec.max()



            if icanal >= 7 :

                img=plt.imshow(im_rec,vmin=vmin,vmax=vmax,cmap='Greys') #emisivas

            else:

                img=plt.imshow(im_rec,vmin=vmin,vmax=vmax,cmap='Greys_r') #reflectivas

            plt.title(icanal)

            plt.show()



#%%calibracion 

            print ('Calibrando la imagen')

            pendiente= metadato['Rad'].scale_factor

            ordenada= metadato['Rad'].add_offset    

        #imag_rad =im_rec*pendiente+ordenada #DN ->mW m-2 sr-1 mum-1

            Unit = "Radiancia ["+metadato['Rad'].units+"]"



            if icanal >=7:

        #Parámetros de calibracion

                fk1 = metadato['planck_fk1'][0] # DN -> K

                fk2 = metadato['planck_fk2'][0]

                bc1 = metadato['planck_bc1'][0]

                bc2 = metadato['planck_bc2'][0]

                imag_cal = (fk2 / (np.log((fk1 / im_rec) + 1)) - bc1 ) / bc2-273.15 # DN -> C

                Unit = "Temperatura de Brillo [°C]"

            else:

                k0=imagenobj.variables['kappa0'][0]

                imag_cal = im_rec*k0

                Unit = "Reflectancia"

    

    # print (imagenobj.variables['max_radiance_value_of_valid_pixels'][0])

            print('Interpolando')

            x,y,imag_calm =muestreo(range(f0,f1),range(c0,c1),imag_cal,esc=esc)



            print('Realzando')

            vmin=0

            vmax=imag_calm[1000:,:600].max()

            imag_calm=realce_lineal(vmin,vmax,imag_calm)



            RGBsur[:imag_calm.shape[0],:imag_calm.shape[1],i]=imag_calm

    

            del imag_cal,im_rec

        

            print("Graficando")

            plt.figure()

            crs=ccrs.Geostationary(central_longitude=lon_cen, satellite_height=altura) #proyeccion geoestacionaria para Goes16

            ax = plt.axes(projection=crs)

            # ax = plt.axes(projection=ccrs.Geostationary(central_longitude=lon_cen)) #proyeccion geoestacionaria para Goes16

            ax.gridlines() #agrega linea de meridianos y paralelos 

            ax.coastlines(resolution='10m',color='blue') #agrega líneas de costa

            img = plt.imshow(RGBsur,extent=img_extentr)

            plt.show()
sup = np.zeros([filas,columnas,3])



for i in range(len(p)): 

    sup[:,:,p[i][0]]=RGBsur[:,:,0]

    sup[:,:,p[i][1]]=RGBsur[:,:,1]

    sup[:,:,p[i][2]]=RGBsur[:,:,2]

    

    plt.figure(21+i)

    plt.title(str(p[i][0])+'=B1, '+str(p[i][1])+'=B7, '+str(p[i][2])+'=B13')

    plt.imshow(sup)
ch_sup = ['C07','C13','C06'] #superficie

RGBsup= np.zeros([filas,columnas,3])

for i in range(3):



    for file_name in os.listdir(Inbox):

        if fnmatch.fnmatch(file_name, '*'+ch_sup[i]+'*'+hora+'*'):

            img_name=file_name

            

            print ('Importando la imagen: %s' %img_name)

            imagenobj = Dataset(Inbox + img_name, 'r')



            print ('Importando las variables de la imagen: %s' %img_name)

            metadato = imagenobj.variables

            altura=metadato['goes_imager_projection'].perspective_point_height

            semieje_may=metadato['goes_imager_projection'].semi_major_axis

            semieje_men=metadato['goes_imager_projection'].semi_minor_axis

            lon_cen=metadato['goes_imager_projection'].longitude_of_projection_origin

            pol=semieje_may*altura/(semieje_may+altura)

            ecu=semieje_men*altura/(semieje_may+altura)

# img_extent = (-ecu,ecu,-pol,pol)

# img_extent = (-pol,pol,-ecu,ecu)

            icanal = int(metadato['band_id'][:])

            print ('Canal %d' %icanal)



# Recortes crudos

            esc=int(N/metadato['Rad'][:].shape[0])

            Nx=int(columnas/esc) #numero de puntos del recorte en x

            Ny=int(filas/esc) #numero de puntos del recorte en x

            f0=int((-y0/psize+N/2-1.5)/esc) #fila del angulo superior izquierdo

            c0=int((x0/psize+N/2+.5)/esc) #columna del angulo superior izquierdo

            f1=int(f0+Ny) #fila del angulo inferior derecho

            c1=int(c0+Nx) #columna del angulo inferior derecho



            im_rec = metadato['Rad'][:].data[f0:f1,c0:c1]



            plt.figure()

            vmin=im_rec.min()

            vmax=im_rec.max()



            if icanal >= 7 :

                img=plt.imshow(im_rec,vmin=vmin,vmax=vmax,cmap='Greys') #emisivas

            else:

                img=plt.imshow(im_rec,vmin=vmin,vmax=vmax,cmap='Greys_r') #reflectivas

            plt.title(icanal)

            plt.show()



#%%calibracion 

            print ('Calibrando la imagen')

            pendiente= metadato['Rad'].scale_factor

            ordenada= metadato['Rad'].add_offset    

        #imag_rad =im_rec*pendiente+ordenada #DN ->mW m-2 sr-1 mum-1

            Unit = "Radiancia ["+metadato['Rad'].units+"]"



            if icanal >=7:

        #Parámetros de calibracion

                fk1 = metadato['planck_fk1'][0] # DN -> K

                fk2 = metadato['planck_fk2'][0]

                bc1 = metadato['planck_bc1'][0]

                bc2 = metadato['planck_bc2'][0]

                imag_cal = (fk2 / (np.log((fk1 / im_rec) + 1)) - bc1 ) / bc2-273.15 # DN -> C

                Unit = "Temperatura de Brillo [°C]"

            else:

                k0=imagenobj.variables['kappa0'][0]

                imag_cal = im_rec*k0

                Unit = "Reflectancia"

    

    # print (imagenobj.variables['max_radiance_value_of_valid_pixels'][0])

            print('Interpolando')

            x,y,imag_calm =muestreo(range(f0,f1),range(c0,c1),imag_cal,esc=esc)



            print('Realzando')

            vmin=0

            vmax=imag_calm[1000:,:600].max()

            imag_calm=realce_lineal(vmin,vmax,imag_calm)



            RGBsup[:imag_calm.shape[0],:imag_calm.shape[1],i]=imag_calm

    

            del imag_cal,im_rec

        

            print("Graficando")

            plt.figure()

            crs=ccrs.Geostationary(central_longitude=lon_cen, satellite_height=altura) #proyeccion geoestacionaria para Goes16

            ax = plt.axes(projection=crs)

            # ax = plt.axes(projection=ccrs.Geostationary(central_longitude=lon_cen)) #proyeccion geoestacionaria para Goes16

            ax.gridlines() #agrega linea de meridianos y paralelos 

            ax.coastlines(resolution='10m',color='blue') #agrega líneas de costa

            img = plt.imshow(RGBsup,extent=img_extentr)

            plt.show()
elem = np.zeros([filas,columnas,3])



for i in range(len(p)): 

    elem[:,:,p[i][0]]=RGBsup[:,:,0]

    elem[:,:,p[i][1]]=RGBsup[:,:,1]

    elem[:,:,p[i][2]]=RGBsup[:,:,2]

    

    plt.figure(21+i)

    plt.title(str(p[i][0])+'=B6, '+str(p[i][1])+'=B7, '+str(p[i][2])+'=B13')

    plt.imshow(elem)