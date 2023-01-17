import geopandas
## get bayfield data
water = geopandas.read_file("https://opendata.arcgis.com/datasets/31f1f67253074ef9afe46cd905bff07a_1.geojson")
# get just price lake
price_lk = bayfield_water[bayfield_water['WATERBODY_NAME'] == 'Price lake']
price_lk
price_lk.plot
# Find area
# transform the data into different crs
price_tr = price_lk.to_crs("EPSG:2287")
price_tr.crs
acre = 43560
price_tr.area / acre
## total acres = 69.975405
price_ext = price_tr.exterior
price_ext.plot()
# create the buffer
price_ext_bud = price_ext.buffer(100)
price_ext_buf.plot()
no_wake_zone = price_tr.intersection(price_ext_buf)
no_wake_zone.plot()
# Get area of no wake zone
no_wake_zone.area

# get area of price lake after subtracting no wake zone
(price_tr.area - no_wake_zone.area) / acre
#convert to acre
##28.904434 acres