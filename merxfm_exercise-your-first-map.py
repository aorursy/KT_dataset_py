import geopandas as gpd



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex1 import *
loans_filepath = "../input/geospatial-learn-course-data/kiva_loans/kiva_loans/kiva_loans.shp"



# Your code here: Load the data

world_loans = gpd.read_file(loans_filepath)



# Check your answer

q_1.check()



# Uncomment to view the first five rows of the data

world_loans.head()
# This dataset is provided in GeoPandas

world_filepath = gpd.datasets.get_path('naturalearth_lowres')

world = gpd.read_file(world_filepath)

world.head()
# Your code here

ax = world.plot(figsize=(20,30), color='none', edgecolor='black');

world_loans.plot(ax=ax, markersize=5, alpha=0.5, color='green');
# Get credit for your work after you have created a map

q_2.check()
# Your code here

PHL_mask = world_loans['country'] == 'Philippines'

PHL_loans = world_loans[PHL_mask]



# Check your answer

q_3.check()
# Load a KML file containing island boundaries

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

PHL = gpd.read_file("../input/geospatial-learn-course-data/Philippines_AL258.kml", driver='KML')

PHL.head()
# Your code here

ax = PHL.plot(figsize=(10,20), color='none', edgecolor='black');

PHL_loans.plot(ax=ax, alpha=0.4, markersize=10, color='green');
# Get credit for your work after you have created a map

q_4.a.check()
# View the solution (Run this code cell to receive credit!)

q_4.b.solution()