# imports

import fiona
import matplotlib.pyplot as plt
import pandas as pd
from geopandas.plotting import plot_polygon_collection
from shapely.geometry import shape
# tools

# EEPSG:102003 USA_Contiguous_Albers_Equal_Area_Conic
equal_area_proj = ('+proj=aea'
                   ' +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96'
                   ' +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs ')

def project(geom, p1, p2=equal_area_proj):
    """Convert geom from p1 to p2
    
    Parameters
    ----------
    geom: shapely geometry object
    p1: str or dict
        Parameters for the original projection
    p2: str or dict
        Parameters for the desired projection
        
    Returns
    -------
    shapely geometry object
        An object equivalent to geom, but
        projected into p2 instead
    """
    import pyproj
    from functools import partial    
    from shapely.ops import transform
    
    p1 = pyproj.Proj(p1, preserve_units=True)
    p2 = pyproj.Proj(p2, preserve_units=True)
    project = partial(pyproj.transform, p1, p2)
    transformed = transform(project, geom)
    return transformed
!ls ../input/data-science-for-good/
# retrieve district geometry and properties

loc = '../input/data-science-for-good/cpe-data/Dept_11-00091/11-00091_Shapefiles'
with fiona.open(loc) as c:
    rec = next(iter(c))  # choose first district of the list
    crs = c.crs  # coordinate reference system
rec['properties']
d14_shape = shape(rec['geometry'])
d14_shape = project(d14_shape, crs)  # project into equal-area
d14_shape
# retrieve census tracts shapes and properties

loc = '../input/01-example-workflow/ma_simplified'
with fiona.open(loc) as c:
    records = list(c)  # retrieve all tracts from shapefile
    crs = c.crs  # coordinate reference system
print(f"{len(records)} census tracts available")
# set shapely ``shape`` in each record

for record in records:
    record['shape'] = shape(record['geometry'])
    # project into equal-area
    record['shape'] = project(record['shape'], crs)
# plot census tracts

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

polygons = [r['shape'] for r in records]
plot_polygon_collection(ax, polygons);
percentages = {}

for record in records:
    id = record['properties']['AFFGEOID']
    intersection = record['shape'].intersection(d14_shape)
    percentage = intersection.area / record['shape'].area
    percentages[id] = percentage

list(percentages.items())[:5]
# plot the intercepting census tracts

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

tracts = [r['shape'] for r in records
          if percentages[r['properties']['AFFGEOID']] > 0]
plot_polygon_collection(ax, tracts, edgecolor='white', alpha=0.5)
plot_polygon_collection(ax, [d14_shape], color='red', alpha=0.5)
loc = ('../input/data-science-for-good/cpe-data/Dept_11-00091/11-00091_ACS_data'
       '/11-00091_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv')
df = pd.read_csv(loc, skiprows=[1])  # there are 2 header rows
df = df.set_index('GEO.id')  # prepare for joining
df.head()
###
# Estimate; RACE - Race alone or in combination with one or more other races - Total population - Black or African American
black_varname = 'HC01_VC79'

# Estimate; RACE - Race alone or in combination with one or more other races - Total population
total_varname = 'HC01_VC77'
###

# join percentage values
pct_series = pd.Series(percentages)
pct_series.name = 'percentage'
small = df[[black_varname, total_varname]].join(pct_series, how='left')

# estimate populations inside police district
small['black_pop'] = small[black_varname] * small['percentage']
small['total_pop'] = small[total_varname] * small['percentage']

small.head()
black_pop = small['black_pop'].sum()
total_pop = small['total_pop'].sum()
print(f"Estimated black population: {black_pop:.1f}")
print(f"Estimated total population: {total_pop:.1f}")
print(f"Estimated percentage: {black_pop / total_pop:.1%}")