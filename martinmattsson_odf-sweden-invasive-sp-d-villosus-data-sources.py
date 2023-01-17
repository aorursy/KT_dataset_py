!apt-get install -y libgdal-dev

!gdal-config --version

!export CPLUS_INCLUDE_PATH=/usr/include/gdal/

!export C_INCLUDE_PATH=/usr/include/gdal/

#!cp /usr/include/gdal/*.h /usr/include/

!cp -n /usr/include/gdal/*.h /usr/include/

#!ls /usr/include/

!pip install GDAL==2.1.0
!pip install rioxarray Pyproj==2.4.1

#lists all installed libraries

#!pip freeze
!ls ../input/cmems-bal-phy-reanalysis-monthlymeans/Winter/
from netCDF4 import Dataset



nc_f = '../input/cmems-bal-phy-reanalysis-monthlymeans/Winter/CMEMS_BAL_PHY_reanalysis_monthlymeans_200402.nc'

#http://marine.copernicus.eu/services-portfolio/access-to-products/?option=com_csw&view=details&product_id=BALTICSEA_REANALYSIS_PHY_003_011

f = Dataset(nc_f, 'r')

#f = netCDF4.Dataset("../input/cmems-bal-phy-reanalysis-monthlymeans/Winter/CMEMS_BAL_PHY_reanalysis_monthlymeans_199403.nc", 'r')



print(f , "\n")

print(f['so'] , "\n")
import xarray as xr

import numpy as np



ncPHY = xr.open_mfdataset("../input/cmems-bal-phy-reanalysis-monthlymeans/Winter/CMEMS_BAL_PHY_reanalysis_monthlymeans_******.nc", combine='by_coords')

#print(ncPHY)

print(ncPHY['so'] , "\n")

print(ncPHY['depth'] , "\n")

so_surf = ncPHY['so'].sel(depth=slice(0.0,2.0))

so_surf_m = so_surf.mean(dim='time')

print(so_surf_m , "\n")



so_surf_m.to_netcdf(path="Sal_surf_m.nc")

print("done")
import rioxarray

import xarray as xr

#https://corteva.github.io/rioxarray/html/modules.html



xso_surf_m = xr.open_dataset("Sal_surf_m.nc")

xso_surf_m.rio.set_crs("EPSG:4326")

print(xso_surf_m['so'] , "\n")



xso_surf_m['so'].rio.to_raster("Sal_surf_m.tif")

print("done")
!rm /kaggle/working/Salinity_today.tif

from osgeo import gdal

from osgeo.gdalconst import GA_Update





#reproject and resample tif

#https://gdal.org/python/index.html

#info EPSG http://www.epsg.org/

warp_opts = gdal.WarpOptions(

    format="GTiff",

    outputBounds = [4030000.000000002, 3400000.000000002, 5440000.000000002, 4830000.000000002],

    outputBoundsSRS="EPSG:3035",

    resampleAlg=gdal.GRIORA_NearestNeighbour,

    xRes=250,

    yRes=250,

    srcSRS="EPSG:4326",

    dstSRS="EPSG:3035",

    srcNodata="nan",

    dstNodata=-9999,

)



src = "Sal_surf_m.tif"

input_raster = gdal.Open(src)

dst_temp = "Salinity_today.tif"

print(warp_opts)

res = gdal.Warp(dst_temp,

                input_raster,

                options=warp_opts)



res = None

input_raster = None

print("done") 
#from osgeo import gdal

#from osgeo.gdalconst import GA_Update



dst_temp = "Salinity_today.tif"

input_raster = gdal.Open(dst_temp, GA_Update)

input_raster_band1 = input_raster.GetRasterBand(1)

gdal.FillNodata(input_raster_band1, maskBand = None, maxSearchDist = 10, smoothingIterations = 0)

input_raster = None

input_raster_band1 = None
from osgeo import gdal

import matplotlib.pyplot as mpl



ds = gdal.Open('Salinity_today.tif').ReadAsArray()

cmap = mpl.cm.jet

cmap.set_under('w')

im = mpl.imshow(ds,vmin=0.0, cmap=cmap)

mpl.colorbar()

mpl.title('Salinity today')
!rm /kaggle/working/Sal_surf_m.nc

!rm /kaggle/working/Sal_surf_m.tif
from netCDF4 import Dataset



nc_f = '../input/cmems-bal-phy-reanalysis-monthlymeans/Winter/CMEMS_BAL_PHY_reanalysis_monthlymeans_200402.nc'

#http://marine.copernicus.eu/services-portfolio/access-to-products/?option=com_csw&view=details&product_id=BALTICSEA_REANALYSIS_PHY_003_011

f = Dataset(nc_f, 'r')

#f = netCDF4.Dataset("../input/cmems-bal-phy-reanalysis-monthlymeans/Winter/CMEMS_BAL_PHY_reanalysis_monthlymeans_199403.nc", 'r')



print(f , "\n")

print(f['thetao'] , "\n")
import xarray as xr

import numpy as np



ncPHY = xr.open_mfdataset("../input/cmems-bal-phy-reanalysis-monthlymeans/Winter/CMEMS_BAL_PHY_reanalysis_monthlymeans_******.nc", combine='by_coords')

#print(ncPHY)

print(ncPHY['thetao'] , "\n")

print(ncPHY['depth'] , "\n")

thetao_surf = ncPHY['thetao'].sel(depth=slice(0.0,2.0))

thetao_surf_m = thetao_surf.mean(dim='time')

print(thetao_surf_m , "\n")



thetao_surf_m.to_netcdf(path="Temperature_surf_m.nc")

print("done")
import rioxarray

import xarray as xr



xso_surf_m = xr.open_dataset("Temperature_surf_m.nc")

xso_surf_m.rio.set_crs("EPSG:4326")

print(xso_surf_m['thetao'] , "\n")



xso_surf_m['thetao'].rio.to_raster("Temperature_surf_m.tif")

print("done")
#from osgeo import gdal

#from osgeo.gdalconst import GA_Update





warp_opts = gdal.WarpOptions(

    format="GTiff",

    outputBounds = [4030000.000000002, 3400000.000000002, 5440000.000000002, 4830000.000000002],

    outputBoundsSRS="EPSG:3035",

    resampleAlg=gdal.GRIORA_NearestNeighbour,

    xRes=250,

    yRes=250,

    srcSRS="EPSG:4326",

    dstSRS="EPSG:3035",

    srcNodata="nan",

    dstNodata=-9999,

)



src = "Temperature_surf_m.tif"

input_raster = gdal.Open(src)

print(src)



dst_temp = "Temperature_today.tif"

print(warp_opts)

res = gdal.Warp(dst_temp,

                input_raster,

                options=warp_opts)



res = None

input_raster = None

print("done")
#from osgeo import gdal

#from osgeo.gdalconst import GA_Update



dst_temp = "Temperature_today.tif"

input_raster = gdal.Open(dst_temp, GA_Update)

input_raster_band1 = input_raster.GetRasterBand(1)

gdal.FillNodata(input_raster_band1, maskBand = None, maxSearchDist = 10, smoothingIterations = 0)



input_raster = None

input_raster_band1 = None
from osgeo import gdal

import matplotlib.pyplot as mpl



ds = gdal.Open('Temperature_today.tif').ReadAsArray()

cmap = mpl.cm.jet

cmap.set_under('w')

im = mpl.imshow(ds,vmin=0.0, cmap=cmap)

mpl.colorbar()

mpl.title('Temperature today')
!rm /kaggle/working/Temperature_surf_m.nc

!rm /kaggle/working/Temperature_surf_m.tif
!ls ../input/emodnetseabedsubstrates

#create temp output folder

!mkdir /kaggle/working/temp

#!ls /kaggle/working
from osgeo import ogr, osr

import os

#Based on https://pcjericks.github.io/py-gdalogr-cookbook/projection.html



shapeList = ['multiscale_50k', 'multiscale_100k', 'multiscale_250k', 'multiscale_1M']

for shape in shapeList:

    driver = ogr.GetDriverByName('ESRI Shapefile')



    # input SpatialReference

    inSpatialRef = osr.SpatialReference()

    inSpatialRef.ImportFromEPSG(4326)



    # output SpatialReference

    outSpatialRef = osr.SpatialReference()

    outSpatialRef.ImportFromEPSG(3035)



    # create the CoordinateTransformation

    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)



    # get the input layer

    finDataSet = "../input/emodnetseabedsubstrates/" + shape + ".shp"

    inDataSet = driver.Open(finDataSet)

    inLayer = inDataSet.GetLayer()



    # create the output layer



    foutputShapefile ="/kaggle/working/temp/" + shape + "_3035.shp"

    if os.path.exists(foutputShapefile):

        driver.DeleteDataSource(foutputShapefile)

    outDataSet = driver.CreateDataSource(foutputShapefile)

    foutlayer = "/kaggle/working/temp/" + shape + "_3035.shp"

    outLayer = outDataSet.CreateLayer(foutlayer, geom_type=ogr.wkbMultiPolygon, options = ['ENCODING=UTF-8'])



    # add fields

    inLayerDefn = inLayer.GetLayerDefn()

    for i in range(0, inLayerDefn.GetFieldCount()):

        fieldDefn = inLayerDefn.GetFieldDefn(i)

        outLayer.CreateField(fieldDefn)



    # get the output layer's feature definition

    outLayerDefn = outLayer.GetLayerDefn()



    # loop through the input features

    inFeature = inLayer.GetNextFeature()

    while inFeature:

        # get the input geometry

        geom = inFeature.GetGeometryRef()

        # reproject the geometry

        geom.Transform(coordTrans)

        # create a new feature

        outFeature = ogr.Feature(outLayerDefn)

        # set the geometry and attribute

        outFeature.SetGeometry(geom)

        for i in range(0, outLayerDefn.GetFieldCount()):

            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))

        # add the feature to the shapefile

        outLayer.CreateFeature(outFeature)

        # dereference the features and get the next input feature

        outFeature = None

        inFeature = inLayer.GetNextFeature()



    # Save and close the shapefiles

    inDataSet = None

    outDataSet = None

    

    #write projection file .prj

    outSpatialRef.MorphToESRI()

    fprj = "/kaggle/working/temp/" + shape + "_3035.prj"

    file = open(fprj, 'w')

    file.write(outSpatialRef.ExportToWkt())

    file.close()

    print("Finished reprojecting "+ shape)



print("all done")
!mkdir /kaggle/working/tempraster

!ls ../input/referens-raster/referens_raster
from osgeo import gdal

from osgeo import ogr

from osgeo import gdalconst

target_ds = None

shapeList = ['multiscale_50k_3035', 'multiscale_100k_3035', 'multiscale_250k_3035', 'multiscale_1M_3035']

refraster = "../input/referens-raster/referens_raster/Standardgrid_Symphony_v1b.tif"



data = gdal.Open(refraster, gdalconst.GA_ReadOnly)

proj = data.GetProjection()

geo_transform = data.GetGeoTransform()

source_layer = data.GetLayer()

x_min = geo_transform[0]

y_max = geo_transform[3]

x_max = x_min + geo_transform[1] * data.RasterXSize

y_min = y_max + geo_transform[5] * data.RasterYSize

x_res = data.RasterXSize

y_res = data.RasterYSize

pixel_width = geo_transform[1]





for shape in shapeList:

    print("Rasterize " + shape)

    finDataSet = "/kaggle/working/temp/" + shape + ".shp"

    foutputRasterfile = "/kaggle/working/tempraster/" + shape + ".tif"

    

    inDataSet = ogr.Open(finDataSet)

    inDataSet1 = inDataSet.GetLayer()

    target_ds = gdal.GetDriverByName('GTiff').Create(foutputRasterfile, x_res, y_res, 1, gdal.GDT_Byte)

    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))

    target_ds.SetProjection (proj)

    band = target_ds.GetRasterBand(1)

    NoData_value = 0

    band.SetNoDataValue(NoData_value)

    band.Fill(0)

    band.FlushCache()

    gdal.RasterizeLayer(target_ds, [1], inDataSet1, options=["ATTRIBUTE=Folk_5cl"])



    inDataSet = None

    target_ds = None



print("done")
from osgeo import gdal

import numpy as np

from osgeo.gdalconst import GDT_Byte





rasterList = ['multiscale_50k_3035', 'multiscale_100k_3035', 'multiscale_250k_3035', 'multiscale_1M_3035']

z_lenght = len(rasterList)

print(z_lenght)



#get numpy shape

finputRasterfile = "/kaggle/working/tempraster/" + rasterList[0] + ".tif"

input_raster = gdal.Open(finputRasterfile)

npArrfirst = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)

print(npArrfirst.shape)

npArr = np.zeros(shape=[z_lenght, npArrfirst.shape[0], npArrfirst.shape[1]], dtype=np.uint8)

npResult = np.zeros(shape=[npArrfirst.shape[0], npArrfirst.shape[1]], dtype=np.uint8)

print(npArr.shape)

first = True

i=1

#read all the tif to numpy 

for fnraster in rasterList:

    if first:

        finputRasterfile = "/kaggle/working/tempraster/" + fnraster + ".tif"

        input_raster = gdal.Open(finputRasterfile)

        npArr[0,:,:] = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)

        print(npArr.shape)

        print(npArr[0,:,:].sum())

        print(npArr.sum())

        first = False

    else:

        

        finputRasterfile = "/kaggle/working/tempraster/" + fnraster + ".tif"

        input_raster = gdal.Open(finputRasterfile)

        npArrtemp = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)

        npArr[i,:,:] = npArrtemp



        print(npArr.shape)

        print(npArr[i,:,:].sum())

        print(npArr.sum())

        i = i+1

        

#loop one cell at the time. If value exist in the best resolution use that. If not test the next and so on.

for E in range(npArrfirst.shape[0]):

    for N in range(npArrfirst.shape[1]):

        for Z in range(z_lenght):

            if npArr[Z,E,N] > 0:

                npResult[E,N] = npArr[Z,E,N]

                break







# create outfile with the same extent as the input raster input_raster.RasterXSize , input_raster.RasterYSize

driver = gdal.GetDriverByName('GTiff')



outfile1 = driver.Create( '/kaggle/working/tempraster/Substrate_all.tif', input_raster.RasterXSize , input_raster.RasterYSize , 1, GDT_Byte)

print(npArrfirst.shape[0])

print(npArrfirst.shape[1])

print(npResult.shape)



#There are inconsistancy in this version of GDAL between ReadAsArray() and WriteArray(npResult). we need to flip y-axis before writing to tif.

npResult = np.flipud(npResult)

outfile1.GetRasterBand(1).WriteArray(npResult)



# setting the spatial ref system same as the input

proj = input_raster.GetProjection()

georef = input_raster.GetGeoTransform()

outfile1.SetProjection(proj)

outfile1.SetGeoTransform(georef)

#Set Nodata

outfile1.GetRasterBand(1).SetNoDataValue(input_raster.GetRasterBand(1).GetNoDataValue())

#write to disk

outfile1 = None

print("done")


from osgeo import gdal

import matplotlib.pyplot as mpl

import matplotlib.colors



ds = gdal.Open('/kaggle/working/tempraster/Substrate_all.tif').ReadAsArray()

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","gold","steelblue"])



im = mpl.imshow(ds,vmin=0.0, cmap=cmap, vmax=5)

mpl.colorbar()

mpl.title('Substrate all')
import numpy as np

from osgeo import gdal

from osgeo.gdalconst import GDT_Byte





finputRaster = '/kaggle/working/tempraster/Substrate_all.tif'

input_raster = gdal.Open(finputRaster)

npArr = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)



print(npArr.max())



npArr[np.where( npArr != 2 )] = 1

npArr[np.where( npArr == 2 )] = 0

print(npArr.max())

print(npArr.min())



#There are inconsistancy in this version of GDAL between ReadAsArray() and WriteArray(npResult). we need to flip y-axis before writing to tif.

npResult = np.flipud(npResult)



# create outfile with the same extent as the input raster

driver = gdal.GetDriverByName('GTiff')

outfile1 = driver.Create( 'Substrate.tif', input_raster.RasterXSize , input_raster.RasterYSize , 1, GDT_Byte)

outfile1.GetRasterBand(1).WriteArray(npArr)



# setting the spatial ref system same as the input

proj = input_raster.GetProjection()

georef = input_raster.GetGeoTransform()

outfile1.SetProjection(proj)

outfile1.SetGeoTransform(georef)

#Set Nodata

outfile1.GetRasterBand(1).SetNoDataValue(-9999)

#write to disk

outfile1 = None



print("done")
from osgeo import gdal

import matplotlib.pyplot as mpl

import numpy as np

import matplotlib.colors



ds = gdal.Open('Substrate.tif').ReadAsArray()



dsref = gdal.Open("../input/referens-raster/referens_raster/Standardgrid_Symphony_v1b.tif").ReadAsArray()



cmapref = mpl.cm.gray_r

cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["gold","white"])



imref = mpl.imshow(dsref, vmin=4.0, cmap=cmapref)

im = mpl.imshow(ds,vmin=0.0,vmax=1.0, cmap=cmap1, alpha=.9)

mpl.colorbar(ticks=np.linspace(0,1,2))

mpl.title('Substrate (sand)')



#deleting temporary files

!rm /kaggle/working/temp/*

!rm /kaggle/working/tempraster/*
from osgeo import gdal

import matplotlib.pyplot as mpl



#https://gdal.org/python/index.html

src = "../input/emodnetbathymetry/Emodnet_bathymetry_C5_2018.asc"

input_raster = gdal.Open(src)



stats = input_raster.GetRasterBand(1).GetStatistics(0,1)



print("Min, Max, Mean, StdDev")

print(stats)

print("Nodata is " + str(input_raster.GetRasterBand(1).GetNoDataValue()))

gt =input_raster.GetGeoTransform()

print("Pixel Size (deg)= ({}, {})".format(gt[1], gt[5]))

input_raster = None

gt =  None

print("done")
from osgeo import gdal



tileList = ['../input/emodnetbathymetry/Emodnet_bathymetry_C5_2018.asc',

              '../input/emodnetbathymetry/Emodnet_bathymetry_C6_2018.asc',

              '../input/emodnetbathymetry/Emodnet_bathymetry_C7_2018.asc',

              '../input/emodnetbathymetry/Emodnet_bathymetry_D5_2018.asc',

              '../input/emodnetbathymetry/Emodnet_bathymetry_D6_2018.asc',

              '../input/emodnetbathymetry/Emodnet_bathymetry_D7_2018.asc']



#BuildVRT(destName, srcDSOrSrcDSTab, **kwargs)

destName = '/kaggle/working/tempraster/C5C7_D5D7_2018.vrt'



gdal.BuildVRT(destName, tileList)



#write to disk

destName = None

print("done")
from osgeo import gdal

from osgeo.gdalconst import GA_Update



#resample tif

#https://gdal.org/python/index.html

#info EPSG http://www.epsg.org/

warp_opts = gdal.WarpOptions(

    format="GTiff",

    outputBounds = [4030000.0, 3400000.0, 5440000.0, 4830000.0],

    srcSRS="EPSG:4326",

    dstSRS="EPSG:3035",

    resampleAlg="near",

    srcNodata=99.0,

    dstNodata=-9999,

)



src = '/kaggle/working/tempraster/C5C7_D5D7_2018.vrt'

input_raster = gdal.Open(src)

dst_temp = "/kaggle/working/tempraster/C5C7_D5D7_2018.tif"

print(warp_opts)

res = gdal.Warp(dst_temp,

                input_raster,

                options=warp_opts)



input_raster = None

res = None

#result statistics

dst_raster = gdal.Open(dst_temp)

stats = dst_raster.GetRasterBand(1).GetStatistics(0,1)

print("Min, Max, Mean, StdDev")

print(stats)

print("Nodata is")

print(dst_raster.GetRasterBand(1).GetNoDataValue())

gt = dst_raster.GetGeoTransform()

print("Pixel Size (m)= ({}, {})".format(gt[1], gt[5]))





print("done")
import numpy as np

from osgeo import gdal

from osgeo.gdalconst import GDT_Float32



src = '/kaggle/working/tempraster/C5C7_D5D7_2018.tif'

input_raster = gdal.Open(src)



npArr = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.float32)



print("Highest land value is "+str(npArr.max()))



npArr[np.where( npArr > -0.5 )] = input_raster.GetRasterBand(1).GetNoDataValue()

print("Lowest water depth" + str(npArr.max()))

print("New land values " + str(npArr.min()))



# create outfile with the same extent as the input raster

driver = gdal.GetDriverByName('GTiff')

outfile1 = driver.Create( '/kaggle/working/tempraster/C5C7_D5D7_2018_reclassify.tif', input_raster.RasterXSize , input_raster.RasterYSize , 1, GDT_Float32)

outfile1.GetRasterBand(1).WriteArray(npArr)



# setting the spatial ref system same as the input

proj = input_raster.GetProjection()

georef = input_raster.GetGeoTransform()

outfile1.SetProjection(proj)

outfile1.SetGeoTransform(georef)

#Set Nodata

outfile1.GetRasterBand(1).SetNoDataValue(input_raster.GetRasterBand(1).GetNoDataValue())

#write to disk

outfile1 = None

print("done")
from osgeo import gdal

from osgeo.gdalconst import GA_Update



#resample tif

#https://gdal.org/python/index.html

#info EPSG http://www.epsg.org/

warp_opts = gdal.WarpOptions(

    format="GTiff",

    outputBounds = [4030000.0, 3400000.0, 5440000.0, 4830000.0],

    xRes=250,

    yRes=250,

    srcSRS="EPSG:3035",

    dstSRS="EPSG:3035",

    resampleAlg="near",

    srcNodata=-9999,

    dstNodata=-9999,

)



src = '/kaggle/working/tempraster/C5C7_D5D7_2018_reclassify.tif'

input_raster5 = gdal.Open(src)

dst_temp = "Depth.tif"

print(warp_opts)

res = gdal.Warp(dst_temp,

                input_raster5,

                options=warp_opts)



res = None

input_raster5 = None



input_raster6 = gdal.Open(dst_temp, GA_Update)

input_raster6_band1 = input_raster6.GetRasterBand(1)

gdal.FillNodata(input_raster6_band1, maskBand = None, maxSearchDist = 10, smoothingIterations = 0)



input_raster6 = None

input_raster6_band1 = None

print("done")
from osgeo import gdal

import matplotlib.pyplot as mpl



ds = gdal.Open('Depth.tif').ReadAsArray()

cmap = mpl.cm.jet

cmap.set_under('w')

im = mpl.imshow(ds,vmin=-2000, cmap=cmap)

mpl.colorbar()

mpl.title('Depth')
#deleting temporary files

!rm /kaggle/working/tempraster/*
from osgeo import gdal

from osgeo.gdalconst import GA_Update



#resample tif

#https://gdal.org/python/index.html

#info EPSG http://www.epsg.org/

warp_opts = gdal.WarpOptions(

    format="GTiff",

    outputBounds = [4030000.0, 3400000.0, 5440000.0, 4830000.0],

    xRes=250,

    yRes=250,

    srcSRS="EPSG:4326",

    dstSRS="EPSG:3035",

    resampleAlg="near",

    srcNodata=-3.40282346639e+38,

    dstNodata=-9999,

)



src = '../input/waveexposure/baltic_wei_swm_with_wh1.tif'

input_raster5 = gdal.Open(src)

dst_temp = "waveexposure.tif"

print(warp_opts)

res = gdal.Warp(dst_temp,

                input_raster5,

                options=warp_opts)



res = None

input_raster5 = None



input_raster6 = gdal.Open(dst_temp, GA_Update)

input_raster6_band1 = input_raster6.GetRasterBand(1)

gdal.FillNodata(input_raster6_band1, maskBand = None, maxSearchDist = 10, smoothingIterations = 0)





input_raster6 = None

input_raster6_band1 = None





print("done")
from osgeo import gdal

import matplotlib.pyplot as mpl



ds = gdal.Open('waveexposure.tif').ReadAsArray()

cmap = mpl.cm.jet

cmap.set_under('w')

im = mpl.imshow(ds,vmin=-2000, cmap=cmap)

mpl.colorbar()

mpl.title('Wave exposure')
from osgeo import gdal

from osgeo.gdalconst import GA_Update





#reproject and resample tif

#https://gdal.org/python/index.html

#info EPSG http://www.epsg.org/

warp_opts = gdal.WarpOptions(

    format="GTiff",

    outputBounds = [4030000.0, 3400000.0, 5440000.0, 4830000.0],

    xRes=250,

    yRes=250,

    dstSRS="EPSG:3035",

    resampleAlg="near",

    dstNodata=-9999.0,)





src = "../input/climatechangeversion20151211/a1b_bau_2099_V_Salt1.tif"

input_raster3 = gdal.Open(src)

dst_temp = "Salinity_ClimateChange.tif"

print(warp_opts)

res = gdal.Warp(dst_temp,

                input_raster3,

                options=warp_opts)



res = None

input_raster3 = None



input_raster4 = gdal.Open(dst_temp, GA_Update)

input_raster4_band1 = input_raster4.GetRasterBand(1)

gdal.FillNodata(input_raster4_band1, maskBand = None, maxSearchDist = 4, smoothingIterations = 0)



input_raster4_band1 = None

input_raster4 = None

print("done")
from osgeo import gdal

import matplotlib.pyplot as mpl



ds = gdal.Open('Salinity_ClimateChange.tif').ReadAsArray()

cmap = mpl.cm.jet

cmap.set_under('w')

im = mpl.imshow(ds,vmin=0.0, cmap=cmap)

mpl.colorbar()

mpl.title('Salinity Climate Change')
from osgeo import gdal

from osgeo.gdalconst import GA_Update



#reproject and resample tif

#https://gdal.org/python/index.html

#info EPSG http://www.epsg.org/

warp_opts = gdal.WarpOptions(

    format="GTiff",

    outputBounds = [4030000.0, 3400000.0, 5440000.0, 4830000.0],

    xRes=250,

    yRes=250,

    dstSRS="EPSG:3035",

    resampleAlg="near",

    dstNodata=-9999,)





src = "../input/climatechangeversion20151211/a1b_bau_2099_V_Temp1.tif"

input_raster = gdal.Open(src)

dst_temp = "Temperature_ClimateChange.tif"

print(warp_opts)

res = gdal.Warp(dst_temp,

                input_raster,

                options=warp_opts)

res = None

input_raster = None



input_raster2 = gdal.Open(dst_temp, GA_Update)

input_raster2_band1 = input_raster2.GetRasterBand(1)

gdal.FillNodata(input_raster2_band1, maskBand = None, maxSearchDist = 4, smoothingIterations = 0)



input_raster2_band1 = None

input_raster2 = None

print("done")
from osgeo import gdal

import matplotlib.pyplot as mpl



ds = gdal.Open('Temperature_ClimateChange.tif').ReadAsArray()

cmap = mpl.cm.jet

cmap.set_under('w')

im = mpl.imshow(ds,vmin=0.0, cmap=cmap)

mpl.colorbar()

mpl.title('Temperature Climate Change')
from osgeo import ogr, osr, gdal

import numpy as np

from pathlib import Path



#Left, Bottom, Right, Top

outputBounds=[4030000.0, 3400000.0, 5440000.0, 4830000.0]

colrows = [5640, 5720]



rasterList = ['Salinity_today.tif',

              'Temperature_today.tif',

              'Substrate.tif',

              'Depth.tif',

              'waveexposure.tif']

             #'..\\1_Sweden\\7_Absence_Shipping traffic\\out\\7_Absence_Shipping traffic.tif']

        

z_lenght = len(rasterList)

print(z_lenght)



#get numpy shape

finputRasterfile = rasterList[0]

input_raster = gdal.Open(finputRasterfile)

npArrfirst = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.float32)

print(npArrfirst.shape)

npArr = np.zeros(shape=[z_lenght, npArrfirst.shape[0], npArrfirst.shape[1]], dtype=np.float32)

npResult = np.zeros(shape=[npArrfirst.shape[0], npArrfirst.shape[1]], dtype=np.float32)

print(npArr.shape)

first = True

i=0

#read all the tif to numpy 

for fnraster in rasterList:



    finputRasterfile = fnraster

    input_raster = gdal.Open(finputRasterfile)

    npArrtemp = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.float32)

    npArr[i,:,:] = npArrtemp



    i = i + 1





finDataSet = '../input/shapeinvasivespecies/Dikerogammarus_villosus_20200110_baltic.shp'

driver = ogr.GetDriverByName('ESRI Shapefile')

inPointShape = driver.Open(finDataSet)

layer = inPointShape.GetLayer()

layerDefinition = layer.GetLayerDefn()



#layer_attr = ['pointid', 'grid_code', 'N_ETRS89', 'E_ETRS89', 'N_WGS84', 'E_WGS84']

layer_attr = ['pointid', 'N_ETRS89', 'E_ETRS89']



f = open('Precence_baltic.csv', 'a')

f.write(layer_attr[0] + ";")

for inraster in rasterList:

    f.write(Path(inraster).stem + ";")

    print(Path(inraster).stem)

f.write("\n")



for feature in layer:

    pointid = feature.GetField(layer_attr[0])

    N = feature.GetField(layer_attr[1])

    E = feature.GetField(layer_attr[2])

    #print(pointid,N,E)

    x = int((E - outputBounds[0])/250)

    y = (colrows[1]-1) - int((N - outputBounds[1])/250)

    #print(pointid,x,y)

    f.write(str(pointid) + ";")

    for z in range(z_lenght):

        f.write(str(npArr[z,y,x]) + ";")

    f.write("\n")

f.close()

print("done")
from osgeo import ogr, osr, gdal

import numpy as np

from pathlib import Path



#Left, Bottom, Right, Top

outputBounds=[4030000.0, 3400000.0, 5440000.0, 4830000.0]

colrows = [5640, 5720]



rasterList = ['Salinity_today.tif',

              'Temperature_today.tif',

              'Substrate.tif',

              'Depth.tif',

              'waveexposure.tif']

             #'..\\1_Sweden\\7_Absence_Shipping traffic\\out\\7_Absence_Shipping traffic.tif']

        

z_lenght = len(rasterList)

print(z_lenght)



#get numpy shape

finputRasterfile = rasterList[0]

input_raster = gdal.Open(finputRasterfile)

npArrfirst = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.float32)

print(npArrfirst.shape)

npArr = np.zeros(shape=[z_lenght, npArrfirst.shape[0], npArrfirst.shape[1]], dtype=np.float32)

npResult = np.zeros(shape=[npArrfirst.shape[0], npArrfirst.shape[1]], dtype=np.float32)

print(npArr.shape)

first = True

i=0

#read all the tif to numpy 

for fnraster in rasterList:



    finputRasterfile = fnraster

    input_raster = gdal.Open(finputRasterfile)

    npArrtemp = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.float32)

    npArr[i,:,:] = npArrtemp



    i = i + 1





finDataSet = '../input/shapeinvasivespecies/Absence_prediction_ETRS89.shp'

driver = ogr.GetDriverByName('ESRI Shapefile')

inPointShape = driver.Open(finDataSet)

layer = inPointShape.GetLayer()

layerDefinition = layer.GetLayerDefn()



#layer_attr = ['pointid', 'grid_code', 'N_ETRS89', 'E_ETRS89', 'N_WGS84', 'E_WGS84']

layer_attr = ['pointid', 'N_ETRS89', 'E_ETRS89']



f = open('Absence_baltic.csv', 'a')

f.write(layer_attr[0] + ";")

for inraster in rasterList:

    f.write(Path(inraster).stem + ";")

    print(Path(inraster).stem)

f.write("\n")



for feature in layer:

    pointid = feature.GetField(layer_attr[0])

    N = feature.GetField(layer_attr[1])

    E = feature.GetField(layer_attr[2])

    #print(pointid,N,E)

    x = int((E - outputBounds[0])/250)

    y = (colrows[1]-1) - int((N - outputBounds[1])/250)

    #print(pointid,x,y)

    f.write(str(pointid) + ";")

    for z in range(z_lenght):

        f.write(str(npArr[z,y,x]) + ";")

    f.write("\n")

f.close()

print("done")
from osgeo import ogr, osr, gdal

import numpy as np

from pathlib import Path



#Left, Bottom, Right, Top

outputBounds=[4030000.0, 3400000.0, 5440000.0, 4830000.0]

colrows = [5640, 5720]



rasterList = ['Salinity_ClimateChange.tif',

              'Temperature_ClimateChange.tif',

              'Substrate.tif',

              'Depth.tif',

              'waveexposure.tif']

             #'..\\1_Sweden\\7_Absence_Shipping traffic\\out\\7_Absence_Shipping traffic.tif']

        

z_lenght = len(rasterList)

print(z_lenght)



#get numpy shape

finputRasterfile = rasterList[0]

input_raster = gdal.Open(finputRasterfile)

npArrfirst = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.float32)

print(npArrfirst.shape)

npArr = np.zeros(shape=[z_lenght, npArrfirst.shape[0], npArrfirst.shape[1]], dtype=np.float32)

npResult = np.zeros(shape=[npArrfirst.shape[0], npArrfirst.shape[1]], dtype=np.float32)

print(npArr.shape)

first = True

i=0

#read all the tif to numpy 

for fnraster in rasterList:



    finputRasterfile = fnraster

    input_raster = gdal.Open(finputRasterfile)

    npArrtemp = np.array(input_raster.GetRasterBand(1).ReadAsArray(), dtype=np.float32)

    npArr[i,:,:] = npArrtemp



    i = i + 1





finDataSet = '../input/shapeinvasivespecies/Absence_prediction_ETRS89.shp'

driver = ogr.GetDriverByName('ESRI Shapefile')

inPointShape = driver.Open(finDataSet)

layer = inPointShape.GetLayer()

layerDefinition = layer.GetLayerDefn()



#layer_attr = ['pointid', 'grid_code', 'N_ETRS89', 'E_ETRS89', 'N_WGS84', 'E_WGS84']

layer_attr = ['pointid', 'N_ETRS89', 'E_ETRS89']



f = open('Prediction_ClimateChange.csv', 'a')

f.write(layer_attr[0] + ";")

for inraster in rasterList:

    f.write(Path(inraster).stem + ";")

    print(Path(inraster).stem)

f.write("\n")



for feature in layer:

    pointid = feature.GetField(layer_attr[0])

    N = feature.GetField(layer_attr[1])

    E = feature.GetField(layer_attr[2])

    #print(pointid,N,E)

    x = int((E - outputBounds[0])/250)

    y = (colrows[1]-1) - int((N - outputBounds[1])/250)

    #print(pointid,x,y)

    f.write(str(pointid) + ";")

    for z in range(z_lenght):

        f.write(str(npArr[z,y,x]) + ";")

    f.write("\n")

f.close()

print("done")
#remove temporary files

!rm -r /kaggle/working/temp

!rm -r /kaggle/working/tempraster



#remove result raster

!rm Salinity_today.tif

!rm Temperature_today.tif

!rm Substrate.tif

!rm Depth.tif

!rm waveexposure.tif

!rm Salinity_ClimateChange.tif

!rm Temperature_ClimateChange.tif



#remove result .csv

#!rm Precence_baltic.csv

#!rm Absence_baltic.csv

#!rm Prediction_ClimateChange.csv