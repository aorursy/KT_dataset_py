!pip install mapclassify
import geopandas as gpd
import requests
import zipfile
import io
import mapclassify
import matplotlib.pyplot as plt
# jupyter "magic" to display plots in notebook
%matplotlib inline
shapefile = "https://opendata.arcgis.com/datasets/8697f02bb81c4d2783cdb4bead357490_9.zip?outSR=%7B%22latestWkid%22%3A2264%2C%22wkid%22%3A102719%7D"
local_path = 'tmp/'
print('Downloading shapefile...')
r = requests.get(shapefile)
z = zipfile.ZipFile(io.BytesIO(r.content))
print("Done")
z.extractall(path=local_path) # extract to folder
filenames = [y for y in sorted(z.namelist()) for ending in ['dbf', 'prj', 'shp', 'shx'] if y.endswith(ending)] 
print(filenames)

dbf, prj, shp, shx = [filename for filename in filenames]
charlotte = gpd.read_file(local_path + shp)
print("Shape of the dataframe: {}".format(charlotte.shape))
print("Projection of dataframe: {}".format(charlotte.crs))
charlotte.tail() #last 5 records in dataframe
ax = charlotte.plot()
ax.set_title("Charlotte Map, Default View)");

charlotte["poverty_percentage"]= 100*charlotte["Populati_2"]
ax = charlotte.plot(figsize=(15,15), column='poverty_percentage', scheme='quantiles', cmap="tab20b", legend=True)
ax.set_title("Mecklenburg Count Census Tracts by Percentage in Poverty", fontsize='large')
#add the legend and specify its location
leg = ax.get_legend()
leg.set_bbox_to_anchor((0.95,0.20))
plt.savefig("charlotte_poverty.png", bbox_inches='tight')




          