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
!git clone https://github.com/kevinlwebb/Treepedia_Public_SanDiego.git
!pwd
!ls Treepedia_Public_SanDiego/
!pip3 install -r Treepedia_Public_SanDiego/requirements.txt
os.chdir(os.path.join("/kaggle", "working"))
!pip install earthengine-api --upgrade
import os, os.path
import urllib
import fiona
import ee

ee.Authenticate()


ee.Initialize()
image = ee.Image('LANDSAT/LC08/C01/T1/LC08_044034_20140318')

bandNames = image.bandNames()
print('Band names: ', bandNames)
import datetime

# Convert ee.Date to client-side date
ee_date = ee.Date('2020-01-01')
py_date = datetime.datetime.utcfromtimestamp(ee_date.getInfo()['value']/1000.0)
print(py_date)

# Convert client-side date to ee.Date
py_date = datetime.datetime.utcnow()
ee_date = ee.Date(py_date)
print(ee_date)
# Load a Landsat image.
img = ee.Image('LANDSAT/LT05/C01/T1_SR/LT05_034033_20000913')

# Print image object WITHOUT call to getInfo(); prints serialized request instructions.
print(img)

# Print image object WITH call to getInfo(); prints image metadata.
print(img.getInfo())
# Print the elevation of Mount Everest.
dem = ee.Image('USGS/SRTMGL1_003')
xy = ee.Geometry.Point([86.9250, 27.9881])
elev = dem.sample(xy, 30).first().get('elevation').getInfo()
print('Mount Everest elevation (m):', elev)
# Import the Image function from the IPython.display module. 
from IPython.display import Image

# Display a thumbnail of global elevation.
Image(url = dem.updateMask(dem.gt(0))
  .getThumbURL({'min': 0, 'max': 4000, 'dimensions': 512,
                'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}))
# Import the Folium library.
import folium

# Define a method for displaying Earth Engine image tiles to folium map.
def add_ee_layer(self, ee_image_object, vis_params, name):
  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
  folium.raster_layers.TileLayer(
    tiles = map_id_dict['tile_fetcher'].url_format,
    attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    name = name,
    overlay = True,
    control = True
  ).add_to(self)

# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

# Set visualization parameters.
vis_params = {
  'min': 0,
  'max': 4000,
  'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

# Create a folium map object.
my_map = folium.Map(location=[20, 0], zoom_start=3, height=500)

# Add the elevation model to the map object.
my_map.add_ee_layer(dem.updateMask(dem.gt(0)), vis_params, 'DEM')

# Add a layer control panel to the map.
my_map.add_child(folium.LayerControl())

# Display the map.
display(my_map)
# Import the matplotlib.pyplot module.
import matplotlib.pyplot as plt

# Fetch a Landsat image.
img = ee.Image('LANDSAT/LT05/C01/T1_SR/LT05_034033_20000913')

# Select Red and NIR bands, scale them, and sample 500 points.
samp_fc = img.select(['B3','B4']).divide(10000).sample(scale=30, numPixels=500)

# Arrange the sample as a list of lists.
samp_dict = samp_fc.reduceColumns(ee.Reducer.toList().repeat(2), ['B3', 'B4'])
samp_list = ee.List(samp_dict.get('list'))

# Save server-side ee.List as a client-side Python list.
samp_data = samp_list.getInfo()

# Display a scatter plot of Red-NIR sample pairs using matplotlib.
plt.scatter(samp_data[0], samp_data[1], alpha=0.2)
plt.xlabel('Red', fontsize=12)
plt.ylabel('NIR', fontsize=12)
plt.show()
# Get a download URL for an image.
image1 = ee.Image('srtm90_v4')
path = image1.getDownloadUrl({
    'scale': 30,
    'crs': 'EPSG:4326',
    'region': '[[-120, 35], [-119, 35], [-119, 34], [-120, 34]]'
})
print(path)
# San Diego
sd_map = folium.Map(location=[32.729849, -117.144378], zoom_start=12)
sd_map
landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
# setting the Area of Interest (AOI)
sd_aoi = ee.Geometry.Rectangle([-51.84448, -3.92180,
                                   -52.23999, -4.38201])
# Load a Landsat 5 image, select the bands of interest.
image = ee.Image('LANDSAT/LT05/C01/T1_TOA/LT05_044034_20081011').select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'])

# Make an Array Image, with a 1-D Array per pixel.
arrayImage1D = image.toArray()

# Make an Array Image with a 2-D Array per pixel, 6x1.
arrayImage2D = arrayImage1D.toArray(1)
# Display GEE Features or Images using folium.
def Mapdisplay(center, dicc, Tiles="OpensTreetMap",zoom_start=10):
    '''
    :param center: Center of the map (Latitude and Longitude).
    :param dicc: Earth Engine Geometries or Tiles dictionary
    :param Tiles: Mapbox Bright,Mapbox Control Room,Stamen Terrain,Stamen Toner,stamenwatercolor,cartodbpositron.
    :zoom_start: Initial zoom level for the map.
    :return: A folium.Map object.
    '''
    mapViz = folium.Map(location=center,tiles=Tiles, zoom_start=zoom_start)
    for k,v in dicc.items():
        if ee.image.Image in [type(x) for x in v.values()]:
            folium.TileLayer(
                tiles = v["tile_fetcher"].url_format,
                attr  = 'Google Earth Engine',
                overlay =True,
                name  = k
              ).add_to(mapViz)
        else:
            folium.GeoJson(
            data = v,
            name = k
              ).add_to(mapViz)
    mapViz.add_child(folium.LayerControl())
    return mapViz
coefficients = ee.Array([
  [0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863],
  [-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800],
  [0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572],
  [-0.8242, 0.0849, 0.4392, -0.0580, 0.2012, -0.2768],
  [-0.3280, 0.0549, 0.1075, 0.1855, -0.4357, 0.8085],
  [0.1084, -0.9022, 0.4120, 0.0573, -0.0251, 0.0238]
])

# Load a Landsat 5 image, select the bands of interest.
image = ee.Image('LANDSAT/LT05/C01/T1_TOA/LT05_044034_20081011').select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'])

# Make an Array Image, with a 1-D Array per pixel.
arrayImage1D = image.toArray()

# Make an Array Image with a 2-D Array per pixel, 6x1.
arrayImage2D = arrayImage1D.toArray(1)

# Do a matrix multiplication: 6x6 times 6x1.
# Get rid of the extra dimensions.
componentsImage = ee.Image(coefficients).matrixMultiply(arrayImage2D).arrayProject([0]).arrayFlatten([['brightness', 'greenness', 'wetness', 'fourth', 'fifth', 'sixth']])

# Display the first three bands of the result and the input imagery.
vizParams = {
  'bands': ['brightness', 'greenness', 'wetness'],
  'min': -0.1, 'max': [0.5, 0.1, 0.1]
}

# Display the input imagery with the greenness result.
dicc = {
    'image': image.getMapId({'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.5}),
    'components': componentsImage.getMapId(vizParams)
}

coords = [37.562, -122.3]
Mapdisplay(coords, dicc, "Stamen Terrain", 10)
# Load an image.
# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_TOA#description
image = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318')
mapid = image.getMapId({
    'bands': ['B4', 'B3', 'B2'], 
    'min': 0, 
    'max': 0.3})

center=[38., -122.5]

# Display the map with folium!
Mapdisplay(center, {'Median Composite':mapid},zoom_start=10)
# Load an image.
image = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318')

# Create an NDWI image, define visualization parameters and display.
ndwi = image.normalizedDifference(['B3', 'B5'])

# Mask the non-watery parts of the image, where NDWI < 0.4.
ndwiMasked = ndwi.updateMask(ndwi.gte(0.4))
ndwiId = ndwiMasked.getMapId({'min': 0.5, 'max': 1, 'palette': ['00FFFF', '0000FF']})

# Display the map with folium!
center=[38., -122.5]
Mapdisplay(center,{'NDWI masked':ndwiId})

# Load an image.
# [treecover2000, loss, gain, lossyear, first_b30, first_b40, first_b50,
# first_b70, last_b30, last_b40, last_b50, last_b70, datamask]
image = ee.Image('UMD/hansen/global_forest_change_2013')
mapid = image.getMapId({
    'bands': ["loss"], 
    'min': 0, 
    'max': 0.3,
    'palette': ['00FFFF', '0000FF']})

center=[38., -122.5]

# Display the map with folium!
Mapdisplay(center, {'Median Composite':mapid},zoom_start=10)

# Load an image.
# https://sites.google.com/site/earthengineapidocs/tutorials/global-forest-change-tutorial/palettes-and-masking-with-an-introduction-to-javascript-methods-and-variables
# [treecover2000, loss, gain, lossyear, first_b30, first_b40, first_b50,
# first_b70, last_b30, last_b40, last_b50, last_b70, datamask]
image = ee.Image('UMD/hansen/global_forest_change_2013')
mapid = image.getMapId({
    'bands': ["treecover2000"], 
    'min': 0, 
    'max': 100,
    'palette': '000000, 00FF00'})

center=[38., -122.5]

# Display the map with folium!
Mapdisplay(center, {'Median Composite':mapid},zoom_start=10)
# Load an image.
# https://sites.google.com/site/earthengineapidocs/tutorials/global-forest-change-tutorial/palettes-and-masking-with-an-introduction-to-javascript-methods-and-variables
# [treecover2000, loss, gain, lossyear, first_b30, first_b40, first_b50,
# first_b70, last_b30, last_b40, last_b50, last_b70, datamask]
image = ee.Image('UMD/hansen/global_forest_change_2013').mask(ee.Image('UMD/hansen/global_forest_change_2013').select(['treecover2000']))
mapid = image.getMapId({
    'bands': ["treecover2000"], 
    'min': 0, 
    'max': 100,
    'palette': '000000, 00FF00'})

center=[38., -122.5]

# Display the map with folium!
Mapdisplay(center, {'Median Composite':mapid},zoom_start=10)
# Remove clouds

# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR#bands
def maskL8sr(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = 1 << 3
    cloudsBitMask = 1 << 5
    # Get the pixel QA band.
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) and qa.bitwiseAnd(cloudsBitMask).eq(0)
    return image.updateMask(mask)


# Load an image.
image = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate('2016-01-01', '2016-12-31').map(maskL8sr)
mapid = image.getMapId({
    'bands': ['B4', 'B3', 'B2'], 
    'min': 0, 
    'max': 3000,
    "gamma": 1.4})

center=[38., -122.5]

# Display the map with folium!
Mapdisplay(center, {'Median Composite':mapid},zoom_start=10)
# Clouds included
# Load an image.
image = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate('2016-01-01', '2016-12-31')
mapid = image.getMapId({
    'bands': ['B4', 'B3', 'B2'], 
    'min': 0, 
    'max': 3000,
    "gamma": 1.4})

center=[38., -122.5]

# Display the map with folium!
Mapdisplay(center, {'Median Composite':mapid},zoom_start=10)
import numpy as np
import matplotlib.pyplot as plt
# Define an image.
img = ee.Image('LANDSAT/LC08/C01/T1_SR/LC08_038029_20180810').select(['B4', 'B5', 'B6'])

# Define an area of interest.
aoi = ee.Geometry.Polygon(
  [[[-110.8, 44.7],
    [-110.8, 44.6],
    [-110.6, 44.6],
    [-110.6, 44.7]]], None, False)

# Get 2-d pixel array for AOI - returns feature with 2-D pixel array as property per band.
band_arrs = img.sampleRectangle(region=aoi)

# Get individual band arrays.
band_arr_b4 = band_arrs.get('B4')
band_arr_b5 = band_arrs.get('B5')
band_arr_b6 = band_arrs.get('B6')
# Transfer the arrays from server to client and cast as np array.
np_arr_b4 = np.array(band_arr_b4.getInfo())
np_arr_b5 = np.array(band_arr_b5.getInfo())
np_arr_b6 = np.array(band_arr_b6.getInfo())
print(np_arr_b4.shape)
print(np_arr_b5.shape)
print(np_arr_b6.shape)
# Expand the dimensions of the images so they can be concatenated into 3-D.
np_arr_b4 = np.expand_dims(np_arr_b4, 2)
np_arr_b5 = np.expand_dims(np_arr_b5, 2)
np_arr_b6 = np.expand_dims(np_arr_b6, 2)
print(np_arr_b4.shape)
print(np_arr_b5.shape)
print(np_arr_b6.shape)
# Stack the individual bands to make a 3-D array.
rgb_img = np.concatenate((np_arr_b6, np_arr_b5, np_arr_b4), 2)
print(rgb_img.shape)
# Scale the data to [0, 255] to show as an RGB image.
rgb_img_test = (255*((rgb_img - 100)/3500)).astype('uint8')
plt.imshow(rgb_img_test)
plt.show()
!pip3 uninstall -y typing
!pip3 install --upgrade earthpy
import os
from glob import glob
import matplotlib.pyplot as plt
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
# Get data for example
data = et.data.get_data("vignette-landsat")

# Set working directory
os.chdir(os.path.join(et.io.HOME, "earth-analytics"))

# Stack the Landsat 8 bands
# This creates a numpy array with each "layer" representing a single band
# You can use the nodata= parameter to mask nodata values
landsat_path = glob(
    os.path.join(
        "data",
        "vignette-landsat",
        "LC08_L1TP_034032_20160621_20170221_01_T1_sr_band*_crop.tif",
    )
)
landsat_path.sort()
array_stack, meta_data = es.stack(landsat_path, nodata=-9999)
titles = ["Ultra Blue", "Blue", "Green", "Red", "NIR", "SWIR 1", "SWIR 2"]
# sphinx_gallery_thumbnail_number = 1
ep.plot_bands(array_stack, title=titles)
plt.show()
ep.plot_bands(array_stack[4], cbar=True)
plt.show()