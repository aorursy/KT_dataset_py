import geopandas as gpd



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex1 import *
loans_filepath = "../input/geospatial-learn-course-data/kiva_loans/kiva_loans/kiva_loans.shp"



# Your code here: Load the data

world_loans = gpd.read_file(loans_filepath)



world_loans.head(3)
# This dataset is provided in GeoPandas

world_filepath = gpd.datasets.get_path('naturalearth_lowres')

world = gpd.read_file(world_filepath)

world.head()


ax= world.plot(figsize=(10,10), color='none', edgecolor='gray')

world_loans.plot(color='red', ax=ax, markersize=2)



PHL_loans = world_loans.loc[world_loans.country.isin(['Philippines'])].copy()



PHL_loans.head(3)
# Load a KML file containing island boundaries

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

PHL = gpd.read_file("../input/geospatial-learn-course-data/Philippines_AL258.kml", driver='KML')

PHL.head()


ax= PHL.plot(figsize=(10,10), color='none', edgecolor='gray')

PHL_loans.plot(ax=ax, color='red', markersize=2)
