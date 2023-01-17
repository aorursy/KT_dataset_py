!pip install rasterio
!pip install geopandas
!pip install sentinelsat
import os, sys
import numpy as np
import rasterio
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import getpass

import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

%matplotlib inline
# connect to the API
user = 'Username' # enter the username
# user = input('Sentinel Hub Username: ')
pw = getpass.getpass('Sentinel Hub Password: ')
api = SentinelAPI(user, pw, 'https://scihub.copernicus.eu/dhus')

# download single scene by known product id
#api.download(<product_id>)
api
gj_geom = {
"type": "FeatureCollection",
"name": "SELECT",
"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
"features": [
{ "type": "Feature", "properties": { }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 78.790282313505003, 14.722851905653091, 0.0 ], [ 78.78944315111346, 14.723002082692179, 0.0 ], [ 78.789509498945648, 14.724354341456941, 0.0 ], [ 78.790479227929708, 14.724338990131089, 0.0 ], [ 78.790282313505003, 14.722851905653091, 0.0 ] ] ] } }
]
}

# search by polygon, time, and Hub query keywords... 
# 7/24/2016 is date of DG strip which was used for impervious surface mapping
# 6/1/2018 - 7/31/2019 are containement dates for durango 416 fire
# 12/4/2017 - 1/12/2018 are containment dates for Thomas fire

footprint = geojson_to_wkt(gj_geom)
products_S2 = api.query(footprint,
                     date = ('20191201', date(2019, 12, 15)),
                     platformname = 'Sentinel-2',
                     cloudcoverpercentage = (0, 30))
 
# products_S1 = api.query(footprint,
#                      date = ('20191201', date(2019, 12, 15)),
#                      platformname = 'Sentinel-1', 
#                      polarizationmode = 'VV')
products_S1 = api.query(footprint,
                         date = ('20191201', date(2019, 12, 15)), 
                        producttype='SLC')
products_S1
s1_items = list(products_S1.items())
s1_items[0]

# should also be able to convert to pandas
s1_res_df = api.to_geodataframe(products_S1)
s2_res_df = api.to_geodataframe(products_S2)
# s1_res_df.plot()
# plt.show()
s1_res_df.shape, s2_res_df.shape

print('Sentinel-1 results')
s1_res_df.head()
print('Sentinel-2 results')
s2_res_df.head()
s2_res_df.describe()['cloudcoverpercentage']
s1_res_df.describe()
## subset Sentinel-1 results by 'producttype'
prod_groups_list = list(s1_res_df.groupby('producttype'))
for i, item in enumerate(prod_groups_list):
    print(i, 'producttype: ', item[0])
    
# GRD_df = prod_groups_list[0][1]
# RAW_df = prod_groups_list[1][1]
SLC_df = prod_groups_list[0][1]
# print('Sentinel-1 GRD products')
# GRD_df.head()
# print('Sentinel-1 RAW products')
# RAW_df.head()
print('Sentinel-1 SLC products')
SLC_df.head()
# download a couple scenes by uuid
# help(api.download)
SLC_uuid = SLC_df['uuid'][1]
SLC_title = SLC_df['title'][1]
if not os.path.exists(SLC_title + '.zip'):
    print('Downloading SLC product')
    SLC_dl = api.download(SLC_uuid)

# same for GRD product
GRD_uuid = GRD_df['uuid'][1]
GRD_title = GRD_df['title'][1]
if not os.path.exists(GRD_title + '.zip'):
    print('Downloading GRD product')
    GRD_dl = api.download(GRD_uuid)
## you can get the path by assigning the download call to a variable 
import zipfile
for zippath in (SLC_dl['path'], GRD_dl['path']):
    zip_ref = zipfile.ZipFile(zippath, 'r')
    zip_ref.extractall(os.path.join('.', 's1_files'))
    zip_ref.close()
SLC_dl
## open the file with rasterio
# SLC_fi = os.path.join('s1_files', SLC_dl['title'] + '.SAFE', 'manifest.safe')
SLC_fi = '/content/s1_files/S1A_IW_SLC__1SDV_20191209T003127_20191209T003155_030264_0375D2_F3F2.SAFE/manifest.safe'
# s1_filepath = r"C:\Projects\sentinel_api\s1_files\S1A_IW_RAW__0SSV_20161216T130058_20161216T130130_014405_0175A1_975B.SAFE\s1a-iw-raw-s-vv-20161216t130058-20161216t130130-014405-0175a1.dat"
# s1_filepath = r"C:/Projects/sentinel_api/s1_files/S1A_IW_GRDH_1SDV_20160830T010152_20160830T010217_012823_01439D_86AC.SAFE/manifest.safe"

for prd in (SLC_title, GRD_title):
    SLC_fi = os.path.join('s1_files', prd + '.SAFE', 'manifest.safe')
    with rasterio.open(SLC_fi, 'r') as src:
        print(src.profile)
        #arr = src.read() # careful... this could kill the kernel
        #print(arr.shape)

    with rasterio.open(SLC_fi, 'r') as src:    
        for ji, window in src.block_windows(1):
            r = src.read(1, window=window)
            print(r.shape)
            break

    with rasterio.open(SLC_fi, 'r') as src:
        r = src.read(1, window=((7000,12500), (7500,12500)))
        print(r.shape)

    if 'SLC' in prd:
        arr = np.abs(r)
        print(arr.min(), arr.max(), arr.mean())
        plt.imshow(arr, cmap='plasma_r', vmin=arr.min(), vmax=arr.mean() + 100)
        plt.colorbar()
        plt.show()
    else:
        print(r.min(), r.max(), r.mean())
        print('clipping to mean + 100')
        plt.imshow(r, cmap='plasma_r', vmin=r.min(), vmax=r.mean() + 100)
        plt.colorbar()
        plt.show()
        
    

# !gdalinfo $SLC_fi

sub_arr = arr[:, 100:200, 100:200]
plt.imshow(sub_arr[0])
plt.show()
plt.imshow(sub_arr[1])
plt.show()
plt.imshow(sub_arr[0] - sub_arr[1])
help(api.query)
