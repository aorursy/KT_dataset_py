from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex1 import *

print("Setup is completed.")



import geopandas as gpd
loans_filepath = "../input/geospatial-learn-course-data/kiva_loans/kiva_loans/kiva_loans.shp"



# load the data

world_loans = gpd.read_file(loans_filepath)



# check your answer

q_1.check()



# uncomment to view the first five rows of the data

world_loans.head()
# lines below will give you a hint or solution code

# q_1.hint()

# q_1.solution()
# This dataset is provided in GeoPandas

world_filepath = gpd.datasets.get_path('naturalearth_lowres')

world = gpd.read_file(world_filepath)

world.head()
# define a base map with county boundaries

ax = world.plot(figsize=(20,20), color='whitesmoke', linestyle=':', edgecolor='lightgray')



# add loans to the base map

world_loans.plot(ax=ax, markersize=2)



# uncomment to see a hint

# q_2.hint()
# Get credit for your work after you have created a map

q_2.check()



# uncomment to see our solution (your code may look different!)

# q_2.solution()
PHL_loans = world_loans[(world_loans['country'] == 'Philippines')]



# check your answer

q_3.check()
# lines below will give you a hint or solution code

# q_3.hint()

# q_3.solution()
# enable fiona driver & load a KML file containing island boundaries

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

PHL = gpd.read_file("../input/geospatial-learn-course-data/Philippines_AL258.kml", driver='KML')

PHL.head()
# define a base map with county boundaries

ax_ph = PHL.plot(figsize=(20,20), color='whitesmoke', linestyle=':', edgecolor='lightgray')



# add loans to the base map

PHL_loans.plot(ax=ax_ph, markersize=2)



# Uncomment to see a hint

#q_4.a.hint()
# Get credit for your work after you have created a map

q_4.a.check()



# Uncomment to see our solution (your code may look different!)

# q_4.a.solution()
# View the solution (Run this code cell to receive credit!)

q_4.b.solution()