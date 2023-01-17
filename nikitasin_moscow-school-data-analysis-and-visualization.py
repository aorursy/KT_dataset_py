# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import geopandas as gpd
districts_data = gpd.read_file('/kaggle/input/moscow-districts/mo.csv', encoding='utf-8')

schools_data = pd.read_csv('/kaggle/input/moscow-schools/school_data.csv')
districts_data.head()
schools_data.head()
drop_list = ['Роговское', 'Вороновское', 'Киевский', 'Новофёдоровское',

            'Михайлово-Ярцевское', 'Клёновское', 'Краснопахорское', 'Первомайское', 

             'Троицк', 'Марушкинское', 'Внуковское', 'Внуково', 'Десёновское', 

             'Филимонковское', 'Московский', 'Сосенское', 'Щаповское', 'Рязанское',

             'Кокошкино', 'Воскресенское', 'Рязановское', 'Северное Бутово', 

             'Южное Бутово', 'Щербинка', 'Рязановское', 'Ново-Переделкино', 'Солнцево',

             'Силино', 'Крюково', 'Старое Крюково', 'Савёлки', 'Матушкино', 'Восточный', 'Молжаниновский',

             'Некрасовка', 'Северный']



districts_data = districts_data.loc[~districts_data['NAME'].isin(drop_list), :]
from bokeh.plotting import ColumnDataSource, figure, output_file, show

from bokeh.io import output_notebook

from bokeh.models import GeoJSONDataSource

from bokeh.plotting import figure

import pandas as pd

from bokeh.colors import RGB
lat = schools_data['lattitude'].values

long = schools_data['longitude'].values



source = ColumnDataSource(data=dict(x=long, y=lat, name=schools_data['name'], adres=schools_data['adress']))



districts_data.crs = {'init' :'epsg:32633'}

districts_data = districts_data.to_crs({'init' :'epsg:32633'})



geo_source = GeoJSONDataSource(geojson=districts_data.to_json())



TOOLTIPS = [('name', '@NAME'),]



p = figure(tooltips=TOOLTIPS, y_range=[55.55, 55.95], x_range=[37.2, 38], plot_height=380, plot_width=380)

p.patches('xs', 'ys', alpha=0.9, source=geo_source, color=RGB(16,24,32), line_color = 'gray')

p.circle('x', 'y', size=6, color=RGB(242,170,76), alpha=0.7, source=source)

output_notebook()

show(p)
import numpy as np

from shapely.geometry import Point
# Подсчёт числа школ

num_schools = np.zeros(districts_data.shape[0])



for i in range(0, districts_data.shape[0]):

    district = districts_data['geometry'].iloc[i]

    for j in range(0, schools_data.shape[0]):

        school = Point(tuple([schools_data['longitude'].iloc[j], schools_data['lattitude'].iloc[j]]))

        if district.contains(school):

            num_schools[i] += 1



districts_data['num_schools'] = num_schools
districts_data.head()
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="whitegrid")



f, axis = plt.subplots(2, figsize=(10, 7))

sns.boxplot(x=districts_data['num_schools'], linewidth=2.5, ax=axis[0]).set_xticks(range(0, 16, 1))

sns.distplot(districts_data['num_schools'], ax=axis[1], kde=False).set_xticks(range(0, 16, 1))



mean = districts_data['num_schools'].mean()

median = districts_data['num_schools'].median()

mode = districts_data['num_schools'].mode().values[0]





for ax in axis:

    ax.axvline(mean, color='r',)

    ax.axvline(median, color='orange')

    ax.axvline(mode, color='green', linestyle='--')

    ax.xaxis.label.set_visible(False)



axis[1].legend({'Mean': mean, 'Median': median, 'Mode': mode}, fontsize=17)

axis[0].set_title('Распределение количества школ', fontsize=20, pad=10)

plt.show()
from bokeh.io import output_notebook, show

from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar

from bokeh.plotting import figure

from bokeh.palettes import inferno, linear_palette, Viridis256, brewer

from bokeh.transform import factor_cmap

from bokeh.layouts import gridplot

from bokeh.layouts import column
# Удалим районы в которых нет школ

districts_data = districts_data.loc[districts_data['num_schools']!=0, :]



districts_data = pd.DataFrame(districts_data)

districts_data = districts_data.sort_values(by=['num_schools'])

districts_data = districts_data.dropna()
source = ColumnDataSource(data=dict(district=districts_data['NAME'], num_schools=districts_data['num_schools']))

TOOLTIPS = [('Район:', '@district'),

            ('Число школ:', '@num_schools'),]

p1 = figure(x_range=districts_data['NAME'], plot_height=300, plot_width=700,tooltips=TOOLTIPS)

p1.vbar(top='num_schools', x='district', source=source, width=0.9,

       fill_color=factor_cmap('district', palette=linear_palette(Viridis256, 120), factors=districts_data['NAME']))

#p1.yaxis.major_label_text_font_size = "9px"

p1.xgrid.grid_line_color = None

p1.xaxis.visible = False





districts_data=gpd.GeoDataFrame(districts_data)

geo_source = GeoJSONDataSource(geojson=districts_data.to_json())

color_mapper = LinearColorMapper(palette=linear_palette(Viridis256, 120))





TOOLTIPS = [('Район', '@NAME'), 

            ('Количество школ:', '@num_schools'),]

p2 = figure(plot_height=700, plot_width=700,tooltips=TOOLTIPS, y_range=[55.55, 55.95], x_range=[37.2, 38])

p2.patches('xs', 'ys', alpha=0.9, source=geo_source, fill_color={'field': 'num_schools', 'transform': color_mapper}, 

          line_color = 'black', line_width = 0.6, fill_alpha = 1)



color_bar = ColorBar(color_mapper=color_mapper, orientation='horizontal', location=(0,0), padding=0)

p2.add_layout(color_bar, 'below')



show(column(p2, p1))
p = sns.catplot(x='ABBREV_AO', y='num_schools', data=districts_data, kind='box')

p.fig.set_size_inches(10, 5)
# Вычислим площадь районов в кв км

districts_data['area']=gpd.GeoDataFrame(districts_data)['geometry'].area/10**6

districts_data = districts_data.sort_values(by=['area'])
f, axis = plt.subplots(2, figsize=(15, 9))



sns.boxplot(x=districts_data['area'], ax=axis[0], linewidth=2.5)

sns.distplot(districts_data['area'], ax=axis[1], kde=False)



mean = districts_data['area'].mean()

median = districts_data['area'].median()

mode = districts_data['area'].mode().values[0]



for ax in axis:

    ax.axvline(mean, color='r',)

    ax.axvline(median, color='orange')

    ax.axvline(mode, color='green', linestyle='--')

    ax.xaxis.label.set_visible(False)



axis[1].legend({'Mean': mean, 'Median': median, 'Mode': mode}, fontsize=17)

axis[0].set_title('Распределение площади районов', fontsize=20, pad=10)

plt.show()
source = ColumnDataSource(data=dict(district=districts_data['NAME'], area=districts_data['area']))



p = figure(y_range=districts_data['NAME'], x_axis_location='above', plot_height=1000, plot_width=550)



p.hbar(y='district', right='area', source=source, height=0.9, 

       fill_color=factor_cmap('district', palette=linear_palette(Viridis256, 120), factors=districts_data['NAME']))



p.xaxis.axis_label = 'Площадь'

p.yaxis.major_label_text_font_size='12px'

p.ygrid.grid_line_color = None

show(p)
# Найдём относительное количество школ к площади района

districts_data['schools_on_area'] = districts_data['num_schools']/districts_data['area']

districts_data = districts_data.sort_values(by='schools_on_area', ascending=False)
#sns.scatterplot(x=districts_data['area']*10**9, y=districts_data['num_schools'])

p = sns.jointplot('area', 'num_schools', data=districts_data, height=7)

p.set_axis_labels('area', 'num_schools', fontsize=17, fontweight='bold')

p.annotate(stats.pearsonr, fontsize=14)
from scipy import stats



filtered_data = districts_data.loc[~pd.Series(districts_data['area']>6*10**(-9)), :]

filtered_data['area'] = np.log(filtered_data['area'])

p = sns.jointplot('area', 'num_schools', data=filtered_data, color='b', height=7)

p.set_axis_labels('log(area)', 'num_schools', fontsize=15, fontweight='bold')

p.annotate(stats.pearsonr, fontsize=14)
source = ColumnDataSource(data=dict(district=districts_data['NAME'], schools_on_area=districts_data['schools_on_area']))

p = figure(x_range=districts_data['NAME'], plot_height=500, plot_width=1100)

p.vbar(x='district', top='schools_on_area', width=0.8, source=source,

       fill_color=factor_cmap('district', palette=linear_palette(Viridis256, 120), factors=districts_data['NAME']))





p.xaxis.major_label_orientation = 1

p.xaxis.major_label_text_font_size = '12px'

p.xgrid.grid_line_color = None

show(p)