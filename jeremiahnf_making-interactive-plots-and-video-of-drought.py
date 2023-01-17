#plotting
import holoviews as hv
import holoviews.plotting.mpl
hv.extension('matplotlib')
from colorcet import fire, rainbow
from IPython.display import display_png
import bokeh
#video generation
import cv2
import os
%output fig='png'
#widget creation
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import json
with open('../input/bokeh-counties/counties.json') as f:
    counties= json.load(f)
drought=pd.read_csv("../input/united-states-droughts-by-county/us-droughts.csv", encoding='latin1')
drought.head()
counties_int_key={}
for key in counties.keys():
    counties_int_key[int(key)]=counties[key]
counties=counties_int_key

drought["drought_level"]=drought["D0"]/100+drought["D1"]/100+drought["D2"]/100+drought["D3"]/100+drought["D4"]/100
drought.head()
drought_monthly=drought[["FIPS","validEnd","drought_level"]]
drought_monthly.validEnd=pd.to_datetime(drought_monthly.validEnd)
drought_monthly.validEnd=drought_monthly.validEnd.dt.to_period("M")
drought_monthly=drought_monthly.groupby(["FIPS","validEnd"]).drought_level.max()
drought_monthly.head()
counties_drought=pd.DataFrame.from_dict(counties, orient='index')
for year in range(2010,2016):
    for month in range(1,13):
        if month<10:
            month="0"+str(month)
        date=str(year)+"-"+str(month)
        counties_drought[date]=0
        for cid in counties_drought.index:
            if (cid,date) in drought_monthly:
                counties_drought.loc[cid,date]=drought_monthly[(cid,date)]
counties_drought.head()
#1 
counties_drought_continental=counties_drought[~counties_drought["state"].isin(['hi', 'ha','ak','pr','gu','mp','as','vi'])]
#2
counties_drought_dict=counties_drought_continental.to_dict("index")
counties_drought_list = [dict(county)for cid, county in counties_drought_dict.items()]
#3
dates_list=[]
for year in range(2010,2016):
    for month in range(1,13):
        if month<10:
            month="0"+str(month)
        date=str(year)+"-"+str(month)
        dates_list.append(date)
cbar_tick_labels=[(0,'Not in drought'),(1,'Abnormally dry conditions'),(2,'Moderate Drought'),(3,'Severe Drought'),(4,'Extreme Drought'),(5,'Exceptional Drought')]
polygons = hv.Polygons(counties_drought_list, ['lons', 'lats'], vdims=hv.Dimension(dates_list[1], range=(0, 5)), label="Continental US Drought Index")
polygons.options(logz=False, xaxis=None, yaxis=None, cbar_ticks=cbar_tick_labels,
                   show_grid=False, show_frame=False, colorbar=True, color_index=dates_list[1],
                   fig_size=500, edgecolor='black', cmap=list(rainbow))

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
month_input=widgets.IntSlider(
    value=7,
    min=1,
    max=12,
    step=1,
    description='Month:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
year_input=widgets.IntSlider(
    value=2012,
    min=2010,
    max=2015,
    step=1,
    description='Year:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
state_abbrevs=counties_drought[~counties_drought["state"].isin(['ha','pr','gu','mp','as','vi'])].state.unique()
state_input=widgets.Dropdown(
    options=state_abbrevs,
    value='ca',
    description='State:',
    disabled=False,
    continuous_update=False
)
display(month_input)
display(year_input)
display(state_input)
if month_input.value<10:
    month_input_str="0"+str(month_input.value)
    date=str(year_input.value)+"-"+str(month_input_str)
else:
    date=str(year_input.value)+"-"+str(month_input.value)
counties_drought_user_input=counties_drought[counties_drought["state"].isin([state_input.value])]
counties_drought_user_input.head()
counties_drought_user_dict=counties_drought_user_input.to_dict("index")
counties_drought_user_list = [dict(county)for cid, county in counties_drought_user_dict.items()]
cbar_tick_labels=[(0,'Not in drought'),(1,'Abnormally dry conditions'),(2,'Moderate Drought'),(3,'Severe Drought'),(4,'Extreme Drought'),(5,'Exceptional Drought')]
polygons = hv.Polygons(counties_drought_user_list, ['lons', 'lats'], vdims=hv.Dimension(date, range=(0, 5)), label="User Specific Map: "+state_input.value+" "+date)
polygons.options(logz=False, xaxis=None, yaxis=None, cbar_ticks=cbar_tick_labels,
                   show_grid=False, show_frame=False, colorbar=True, color_index=date,
                   fig_size=500, edgecolor='black', cmap=list(rainbow))
plots={}
renderer = hv.plotting.mpl.MPLRenderer.instance(dpi=120)
for date in dates_list:
    print(date)
    polygons.vdims=[hv.Dimension(date, range=(0, 5))]
    plots[date]=polygons.options(logz=False, xaxis=None, yaxis=None, cbar_ticks=cbar_tick_labels,
                   show_grid=False, show_frame=False, colorbar=True, color_index=date,
                   fig_size=500, edgecolor='black', cmap=list(rainbow))
for date in dates_list:
    path=""+date
    renderer.save(plots[date],path ,fmt='png')
dir_path = ""
for f in os.listdir(""):
    if f.endswith("png"):
        images.append(f)
image_path = os.path.join("", images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter("vid.mp4", fourcc, 5.0, (width, height))
dir_path = ""
for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) 

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): 
        break
out.release()
cv2.destroyAllWindows()
import base64
import io
from IPython.display import HTML
#video is hardcoded because within Kaggle reading and writing repeatedly is difficult
video = io.open('../input/drought-video/vid.mp4', 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))
