%matplotlib inline

import pandas as pd

import numpy as np

import gc, os, sys

import matplotlib.pyplot as plt

import cv2

from IPython.display import Image, display, HTML

import calendar



np.seterr(divide='ignore', invalid='ignore')



DTYPE = {

    'building_id': ('int16'),

    'meter': ('int8'),

    'meter_reading': ('float32')

}



INPUT = '../input/ashrae-energy-prediction'
buildings = pd.read_csv(f'{INPUT}/building_metadata.csv', index_col='building_id')

buildings.shape
train = pd.read_csv(f'{INPUT}/train.csv', parse_dates=['timestamp'], dtype=DTYPE)

train.shape
weather = pd.read_csv(f'{INPUT}/weather_train.csv', parse_dates=['timestamp'])

weather.shape
buildings.groupby(['primary_use','site_id']).size().unstack().fillna(0).style.background_gradient(axis=None)
train.timestamp.dt.year.value_counts()
train.timestamp.dt.weekofyear.min(), train.timestamp.dt.weekofyear.max()
def add_xy(df):

    dt = df.timestamp.dt

    df['x'] = ((dt.dayofweek * 24) + dt.hour).astype('int16')

    df['y'] = (dt.weekofyear % 53).astype('int8')
add_xy(train)

add_xy(weather)

train.x.max(), train.y.max()
WIDTH = train.x.max() + 1

HEIGHT = train.y.max() + 1

WIDTH, HEIGHT
PLOTS_GRAY = 'plots_grayscale'

os.makedirs(PLOTS_GRAY, exist_ok=True)
def normalize(c):

    return np.nan_to_num(c / c.max())



def log_normalize(c):

    return normalize(np.log1p(c))



# WARNING! This stretches each color channel independently

#  - it loses relative scale between them

def log_normalize_chan(c):

    return np.dstack([log_normalize(a.T) for a in c.T])



def write_img(fname, img):

    img *= 255

    if len(img.shape) == 3:

        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)

    cv2.imwrite(fname, img)
for (m, b), df in train.groupby(['meter', 'building_id']):

    cc = np.zeros((HEIGHT, WIDTH), dtype=float)

    np.add.at(cc, (df.y, df.x), df.meter_reading)

    write_img(f'{PLOTS_GRAY}/{m}_{b}_log.png', log_normalize(cc))
display(Image(f'{PLOTS_GRAY}/1_161_log.png', width=WIDTH*4))
!ls -1 $PLOTS_GRAY | wc -l
PLOTS_RGB = 'plots_rgb'

os.makedirs(PLOTS_RGB, exist_ok=True)
METERS = ['electricity', 'chilledwater', 'steam', 'hotwater']

M_DICT = {k:i for i, k in enumerate(METERS)}



train.meter.value_counts()
rgb_pngs = {}

for b, df in train.query('meter<3').groupby('building_id'):

    cc = np.zeros((HEIGHT, WIDTH, 3), dtype=float)

    np.add.at(cc, (df.y, df.x, df.meter), df.meter_reading)

    png = f'{PLOTS_RGB}/{b}_log.png'

    rgb_pngs[b] = png

    write_img(png, log_normalize_chan(cc))
!ls -1 $PLOTS_RGB | wc -l
# Find month offsets

month_y_min = train.groupby(train.timestamp.dt.month).y.min()

ylabels = np.asarray(calendar.month_abbr)[month_y_min.index]

yticks = month_y_min.values



# Tick to mark days - each 24 hours

days = list(calendar.day_abbr)

xlabels = days

xticks = np.arange(7) * 24

WEATHER_COLS = ['air_temperature', 'precip_depth_1_hr', 'sea_level_pressure']



def count_gray(df):

    cc = np.zeros((HEIGHT, WIDTH), dtype=float)

    np.add.at(cc, (df.y, df.x), df.meter_reading)

    return cc



def count_rgb(df):

    cc = np.zeros((HEIGHT, WIDTH, 3), dtype=float)

    np.add.at(cc, (df.y, df.x, df.meter), df.meter_reading)

    return cc



# use rank transform of selected columns (fork to try others!)

def weather_rgb(df):

    cc = np.zeros((HEIGHT, WIDTH, 3), dtype=float)

    for i, c in enumerate(WEATHER_COLS):

        cc[df.y, df.x, i] = df[c].rank(pct=True)

    return cc



def detail_list(series):

    return ''.join([f'<li><i>{k}</i>: <b>{v}</b>'

                    for k,v in series.dropna().items()])



def display_plot(plotdata, title):

    fig, ax = plt.subplots(figsize=(14, 6))

    c = ax.imshow(plotdata)

    ax.set_xticks(xticks, False)

    ax.set_xticklabels(xlabels)

    ax.set_yticks(yticks)

    ax.set_yticklabels(ylabels)

    ax.set_title(title)

    plt.tight_layout()

    plt.show()



def show_plot(src_df, building_id, comment):

    df = src_df.query(f'(building_id=={building_id}) and (meter<3)')

    display(HTML(f'<h1 id="b{building_id}">Building {building_id}</h1>'))

    display(HTML(detail_list(buildings.loc[building_id])))

    display(df.groupby('meter').meter_reading.agg(['count', 'mean', 'max']))

    display(HTML(f'<br/>{comment}'))

    p = log_normalize_chan(count_rgb(df))

    display_plot(p, f'Building {building_id}')



COMMENTS = {

    647: "Some have a lot of missing data",

    675: "Nice intricate pattern",

    677: "<a target='_blank' href='https://www.youtube.com/watch?v=ciz_C3xiuN0'>(Thursday) Here's Why I Did Not Go to Work Today</a> (I didn't know this song - Google suggested it.)",

    182: "Single hour spikes <b>much</b> higher than the mean (see stats above) - <a target='_blank' href='https://www.youtube.com/watch?v=qAkZT_4vL_Y'>what's he <i>building</i> in there?</a> (See comments! &darr;)",

    822: "Looks more like an on/off pattern (constant usage for one day, no hourly variation), and reminds me of a <a target='_blank' href='https://www.kaggle.com/jtrotman/eda-talkingdata-temporal-click-count-plots'>ten minute switching pattern in chinese mobile advert click patterns</a> :) ",

    751: "Remember, blue is <i>steam</i>: Some activity at weekends, but weekends have different seasonal pattern",

    1017: "Abrupt drop in electricity, switches to chilledwater (green)?",

    1063: "Monthly pattern in weekends?",

    747: "Who likes <a target='_blank' href='https://www.google.com/search?q=fruit+salad+sweet&source=lnms&tbm=isch'>Fruit Salads?</a>",

    1247: "Interesting interaction between hour of day and position in year - phase shifts over time.",

    1355: "Looks like four or more separate regimes - longer days in the later half (that start one hour later), and missing data around March/April that is common to most site 15 buildings.",

}
# Add this back if you want separate weather plots

# for site, df in weather.groupby('site_id'):

#     display_plot(weather_rgb(df), f'Weather at site {site}')
for building_id, comment in COMMENTS.items():

    show_plot(train, building_id, comment)
import base64



buildings['png'] = pd.Series(rgb_pngs)

buildings['png_size'] = pd.Series(rgb_pngs).map(os.path.getsize)



IMG_PER_ROW = 4



def img_src(fname):

    with open(fname, 'rb') as f:

        return base64.b64encode(f.read()).decode('utf-8')



# Return HTML for a table of plots

def make_table(df, per_row, inline=True):

    src = ""

    for i, (bid, row) in enumerate(df.iterrows()):

        if (i%per_row) == 0:

            if i:

                src += "</tr>"

            src += "<tr>"

        if inline:

            dat = f"data:image/png;base64,{img_src(row.png)}"

        else:

            dat = row.png

        src += (f'<td><img width={WIDTH} height={HEIGHT} src="{dat}">'

                f'<br/>[{bid}] <i>{row.primary_use}</i></td>')

    src += "</tr>"

    return f"<table>{src}</table>"



# Write single HTML table with all plots

with open('pngs_sorted_by_size.html', 'w') as f:

    table_src = make_table(buildings.sort_values('png_size'), per_row=8, inline=False)

    print(f"<html><head><title>ASHRAE Plots</title></head>"

          f"<body>{table_src}</body></html>\n", file=f)



# Generate output section per site

for site, df in buildings.sort_values(['primary_use', 'png_size'], ascending=[True, False]).groupby('site_id'):

    display(HTML(f'<h1 id="s{site}">Site {site}</h1>'))

    

    display(HTML(f'<h2>Weather</h2>'))

    display_plot(weather_rgb(weather.loc[weather.site_id==site]), f'Weather site {site}')

    display(weather.loc[weather.site_id == site].agg({

                WEATHER_COLS[0]: ['min', 'mean', 'max'],

                WEATHER_COLS[1]: ['min', 'mean', 'max'],

                WEATHER_COLS[2]: ['min', 'mean', 'max']

            }).T.round(1)

    )

    

    display(HTML(f'<h2>Stats</h2>'))

    display(

        df.groupby('primary_use').agg({

            'square_feet': ['count', 'mean'],

            'year_built': ['min', 'median', 'max'],

            'floor_count': ['min', 'median', 'max']

        }).T.dropna(how='all').T.fillna('').style.background_gradient(axis=0)

    )

    

    display(HTML(f'<h2>Buildings</h2>'))

    src = make_table(df, IMG_PER_ROW, inline=True)

    src += "<br/><hr/>"

    display(HTML(src))
test = pd.read_csv(f'{INPUT}/test.csv', usecols=['building_id', 'meter'], dtype=DTYPE)

test.shape
2*365*24
test.groupby(['building_id']).size().value_counts().sort_index()
test.groupby(['building_id', 'meter']).size().value_counts()
!7z a -bd -mmt4 -sdel {PLOTS_GRAY}.zip {PLOTS_GRAY} >>compress.log
!7z a -bd -mmt4 -sdel {PLOTS_RGB}.zip {PLOTS_RGB} >>compress.log