import pandas as pd
import geopandas as gpd
import geoviews as gv

gv.extension('bokeh')
geometries = gpd.read_file('../input/data-map/TUN_adm1.shp')
geometries.as_matrix

idr=referendum = pd.read_csv('../input/idr-gouv/idr_gouv.csv')
idr.columns
gdf = gpd.GeoDataFrame(pd.merge(geometries, idr))
gdf.head()
plot_opts = dict(tools=['hover'], width=550, height=800, color_index='IDR',
                 colorbar=True, toolbar=None, xaxis=None, yaxis=None)
gv.Polygons(gdf, vdims=['gouvernorat', 'IDR'], label='Regional Development Index in Tunisia (2010)').opts(plot=plot_opts)

renderer = gv.renderer('bokeh')
g_idr=gv.Polygons(gdf, vdims=['gouvernorat', 'IDR'], label='Regional Development Index in Tunisia').opts(plot=plot_opts)
renderer.save(g_idr, 'g_idr')
