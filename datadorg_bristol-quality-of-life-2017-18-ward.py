import os
import json
import pandas as pd
from datashader.utils import lnglat_to_meters
from bokeh.models import ColumnDataSource, LinearColorMapper, HoverTool, Select, Div
from bokeh import palettes
from bokeh.plotting import figure, show
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import row, column
from bokeh.io import output_file
for file in os.listdir("../input"):
    path = os.path.join("../input", file)
    print(f"{os.path.getsize(path):>10} {file}")
df = pd.read_csv('../input/quality-of-life-2017-18-ward-fixed.csv', header=0,
    usecols=[0,1,2,4,5,8,12],
    names=['indicator_ref','indicator','theme','ward_code','ward_name','statistic','geo_shape'])
df = df.dropna(axis=0)
df.info()
ward_df = df[['ward_code','ward_name','geo_shape']].drop_duplicates().sort_values(by=['ward_name']).reset_index(drop=True)
ward_df.head()
ward_codes, xs, ys = [], [], []
for record in ward_df.itertuples():
    data = json.loads(record.geo_shape)
    x, y = [], []
    for coord in data.get('coordinates')[0]:
        easting, northing = lnglat_to_meters(coord[0], coord[1])
        x.append(easting)
        y.append(northing)
    xs.append(json.dumps(x))
    ys.append(json.dumps(y))
    ward_codes.append(record.ward_code)

poly_df = pd.DataFrame({'ward_code': ward_codes, 'xs': xs, 'ys': ys})
poly_df.head()
ward_df = ward_df.drop(columns=['geo_shape'])
ward_df=ward_df.merge(poly_df,how='inner',left_on='ward_code',right_on='ward_code')
ward_df.head()
ind_df = df[['indicator_ref','indicator','theme']].drop_duplicates().sort_values(by=['indicator_ref']).reset_index(drop=True)
ind_df.head()
stats_df = df[['ward_code','indicator_ref','statistic']].sort_values(by=['ward_code','indicator_ref']).reset_index(drop=True)
stats_df.head()
s = stats_df.pivot(index='ward_code',columns='indicator_ref',values='statistic')
s = s.merge(ward_df[['ward_code','ward_name','xs','ys']],how='inner',left_on='ward_code',right_on='ward_code')
data = s.to_dict(orient='list')
data['value'] = data['IQOL17001']
data['xs'] = list(map(json.loads, data['xs']))
data['ys'] = list(map(json.loads, data['ys']))
source = ColumnDataSource(data)
color_mapper = LinearColorMapper(palette=palettes.Inferno256)
tools = "pan,wheel_zoom,box_zoom,reset,hover"
xmin, ymin = lnglat_to_meters(-2.75, 51.4)
xmax, ymax = lnglat_to_meters(-2.5, 51.54)
plot = figure(plot_width=500, plot_height=500, title="Bristol, Quality of Life 2017-18 (ward)",
            x_range=(xmin,xmax), y_range=(ymin,ymax), tools=tools,
            x_axis_type="mercator", y_axis_type="mercator")
plot.add_tile(CARTODBPOSITRON)
plot.patches('xs', 'ys', fill_alpha=0.4,
                fill_color={'field': 'value', 'transform': color_mapper},
                line_color='white', line_width=0.5, source=source)
hover = plot.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [("Ward", "@ward_name"),("Value", "@value")]
ind = ind_df[['indicator_ref','indicator','theme']].sort_values(by=['theme', 'indicator_ref'])
ind = ind.drop_duplicates()
menu = {}
for record in ind.itertuples():
    if not record.theme in menu:
        menu[record.theme] = []
    menu[record.theme].append((record.indicator_ref, record.indicator))
callback = CustomJS(args=dict(source=source), code="""
    var data = source.data;
    var ref = cb_obj.value
    data['value'] = data[ref]
    source.change.emit();
""")

select = Select(value=menu['Community & Living'][0][1], options=menu, title="Theme & Indicator")
select.js_on_change('value', callback)
div = Div(text="""<p>Using data from
<a href="https://opendata.bristol.gov.uk/explore/dataset/quality-of-life-2017-18-ward/">opendata.bristol.gov.uk</a>.</p>
<p>2018-09-25 Robert Smith</p>""", width=400)
output_file("quality.html")
show(row(plot, column(select, div)))
from IPython.display import FileLink
FileLink('quality.html')
