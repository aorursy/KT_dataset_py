import io

import base64



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.animation as animation



from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.colors import rgb2hex, Normalize

from matplotlib.colorbar import ColorbarBase



from IPython.display import HTML



import warnings

warnings.filterwarnings('ignore')
# comfirmed case constant

NUM_CASES_MAX = 4000 # upper bound of comfirmed case, in case of some province have very high number



# input, output constants

COVID_19_DATA_PATH = '/kaggle/input/corona-virus-report/covid_19_clean_complete.csv'

MAP_DATA_PATH = '/kaggle/input/cn-map/'

ANIMATION_OUTPUT_PATH = '/kaggle/working/COVID-19_geo_visualization.gif'



# map bounding box constants

LOWER_LEFT_LON = 73.55770111084013

LOWER_LEFT_LAT = 18.159305572509766

UPPER_RIGHT_LON = 134.7739257812502

UPPER_RIGHT_LAT = 53.56085968017586
def num_days_max(month, year):

    # Feb

    if month is 2:

        day = 29 if year % 4 is 0 else 28

    else: # other months

        day = 31 if month in [1, 3, 4, 7, 8, 10, 12] else 30   

    return day



def validDate(date1, date2):

    m1, d1, y1 = date1

    m2, d2, y2 = date2

    

    if y1 < y2: return True

    

    if y1 > y2: return False

    

    if m1 < m2: return True

    

    if m1 > m2: return False

    

    if d1 > d2: return False

    

    return True
# Compute number of days based on latest dataset

data = pd.read_csv(COVID_19_DATA_PATH)



start_month, start_day, start_year = data['Date'][0].split('/')

latest_month, latest_day, latest_year = data['Date'][len(data)-1].split('/')



year = int(start_year); month = int(start_month); day = int(start_day)-1



time_line = {'start': None}

while validDate((month, day, year), (int(latest_month), int(latest_day)-1, int(latest_year))):

    # 12/31

    if month is 12 and day is 31:

        year += 1

        month = 1

        day = 1

    elif day >= num_days_max(month, year): # make sure day is valid after increment

        month +=1

        day = 1

    else:

        day +=1

    

    time_line["{}/{}/{}".format(month,day,year)] = None
data['Country/Region'].replace({"Mainland China": "China", 

                                "Taiwan": "China", 

                                "Hong Kong": "China", 

                                "Macau": "China"}, 

                               inplace=True)

is_China = data['Country/Region'] == "China"

data_China = data[is_China].fillna(0)



group_provinces = data_China.groupby("Province/State")
cmap = plt.get_cmap('Oranges')

vmin = 0; vmax = NUM_CASES_MAX # set range.



# create the dictionary for data and their color value

for date in time_line:

    prov = {}

    for p in group_provinces.groups:

            

        confirmed = 0.0

        

        # get the number of comfired cases for non-start date

        if date is not 'start':

            province_data = group_provinces.get_group(p)

            confirmed = province_data[province_data['Date'] == date].max()['Confirmed']    

            

            # make sure there's no nan

            if np.isnan(confirmed):

                confirmed = 0.0

        

        # get color

        if confirmed >= NUM_CASES_MAX: 

            confirmed = NUM_CASES_MAX

        color = cmap(np.sqrt((confirmed-vmin)/(vmax-vmin)))[:3]

        

        # correct some province names

        if p == 'Macau': 

            p = 'Macao'

        if p == 'Inner Mongolia':

            p = 'Nei Mongol'

        if p == 'Xinjiang':

            p = 'Xinjiang Uygur'

            

        prov[p] = (confirmed, color)

        

    time_line[date] = prov
date_sample = time_line['1/22/20']

print('{Province: (num_cases, (R, G, B))}')

res = [{p: date_sample[p]} for p in list(date_sample)[0:2]]

for e in res:

    print(e)
# Initialize the figure, axis and get the colormap

fig = plt.figure(figsize=(40, 18))

ax = plt.gca()



info_text = plt.text(74, 50, "Start\nComfired: 0",fontsize=30)



# Initialize the basemap with a bounding box drawing around the map.

mp = Basemap(

    llcrnrlon = LOWER_LEFT_LON, 

    llcrnrlat = LOWER_LEFT_LAT, 

    urcrnrlon = UPPER_RIGHT_LON, 

    urcrnrlat = UPPER_RIGHT_LAT

)



# combine maps of mainland China, Taiwan, Hong Kong and Macau

maps = ['CHN', 'TWN', 'HKG', 'MAC']

states_info = []; states = []



for m in maps:

    mp.readshapefile(r'{}{}'.format(MAP_DATA_PATH, m), 'states', drawbounds=True)    

    states_info +=[ d['NAME_1'] for d in mp.states_info] if 'CHN' in m else states_info + [ d['NAME_0'] for d in mp.states_info]

    states += mp.states



def update(frame_num):

    date = list(time_line)[frame_num]

    lis = time_line[date]   



    # add map polygons

    for nshape,seg in enumerate(states): 

        if states_info[nshape] in lis:

            color = rgb2hex(lis[states_info[nshape]][1])  

        else:

            color = rgb2hex(cmap(np.sqrt((0.0-vmin)/(vmax-vmin)))[:3])

        ax.add_patch(Polygon(seg,facecolor=color,edgecolor=color))

    

    # update plot information

    if date is not 'start':

        total_cases = sum([lis[i][0] for i in lis])

        info_text.set_text("Date: {}\nTotal Comfired Cases: {}".format(date.replace('/', '-'), 

                                                                                           int(total_cases)))

ani = animation.FuncAnimation(fig, update, frames=len(time_line))



plt.title('Geographic Map Visualiztion of the Spread of COVID-19 in China', fontsize=30)



# add colorbar

ax_c = fig.add_axes([0.19, 0.07, 0.64, 0.04])

ax_c.tick_params(labelsize=20)



cb = ColorbarBase(ax_c, cmap=cmap, norm=Normalize(0, NUM_CASES_MAX), orientation='horizontal')

cb.set_label(r'Number of Comfirmed Cases from 0 to {}+'.format(NUM_CASES_MAX), size=35)



plt.show()
ani.save(ANIMATION_OUTPUT_PATH, writer='imagemagick', fps=1)
filename = ANIMATION_OUTPUT_PATH

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii'))) 