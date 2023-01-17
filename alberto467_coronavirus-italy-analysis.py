import pandas as pd

df = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')
df['Date'] = pd.to_datetime(df['Date'])

regioni = df.drop(columns=['SNo','Latitude','RegionCode','Longitude'])
regioni = regioni.groupby(by=['RegionName','Date']).sum()

italia = regioni.groupby(by=['Date']).sum()
regioni.index.get_level_values(0).unique()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
from scipy.signal import savgol_filter
import numpy as np

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

munits.registry[np.datetime64] = mdates.ConciseDateConverter()

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(8, 10),
    constrained_layout=True
)

ax1.plot(italia.index, savgol_filter(italia['TotalPositiveCases'].values, 15, 1))
ax1.plot(italia['TotalPositiveCases'], color='black', alpha=0.25)

ax1_2 = ax1.twinx()

ax1_2.plot(italia.index, savgol_filter(italia['TestsPerformed'].values, 3, 1), color='red')
ax1_2.tick_params(axis='y', labelcolor='red')

ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid(which='major',alpha=0.7)
ax1.grid(which='minor',alpha=0.35)
ax1.set_title('Total Cases')

ax2.plot(italia.index, savgol_filter(italia['NewPositiveCases'].values, 13, 1))
ax2.plot(italia['NewPositiveCases'], color='black', alpha=0.25)

ax2_2 = ax2.twinx()

ax2_2.plot(italia.index, savgol_filter(np.concatenate((italia['TestsPerformed'].values[:1], np.subtract(italia['TestsPerformed'].values[1:],italia['TestsPerformed'].values[:-1]))), 13, 1), color='red')
ax2_2.tick_params(axis='y', labelcolor='red')

ax2.grid(which='major',alpha=0.7)
ax2.grid(which='minor',alpha=0.35)
ax2.set_title('New Cases')

plt.show()
new_cases_by_week = pd.DataFrame(dict(DayOfWeek=italia.index.weekday,NewPositiveCases=italia['NewPositiveCases'])).groupby('DayOfWeek').sum().reset_index()
min_value = np.min(new_cases_by_week['NewPositiveCases']) * 0.95
plt.bar(new_cases_by_week['DayOfWeek'], new_cases_by_week['NewPositiveCases']-min_value, bottom=min_value)
plt.show()
print(regioni.index.get_level_values(1).max())
regioni_ora = regioni.groupby(level=0).last()
import json

regioni_geo = json.load(open('/kaggle/input/covid19-italy-regional-data/regioni-con-trento-bolzano.geojson'))
code_poly_map = [ 7, 9, 4, 8, 15, 20, 6, 19, 17, 3, 13, 1, 2, 18, 10, 11, 12, 14, 5, 0, 16 ]

list(zip([ regioni_geo['features'][code_poly_map[i]]['properties']['Regione'] for i in range(21)], regioni.index.get_level_values(0).unique()))
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as colorbar

fig, axs = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [60, 1]})

jet = cm = plt.get_cmap('OrRd') 
cNorm  = colors.Normalize(vmin=0, vmax=regioni_ora['TotalPositiveCases'].max())
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

for i, value in enumerate(regioni_ora['TotalPositiveCases'].values):
    axs[0].add_patch(PolygonPatch(regioni_geo['features'][code_poly_map[i]]['geometry'], fc=scalarMap.to_rgba(value), ec='black'))
    
axs[0].axis('scaled')
axs[0].axis('off')
axs[0].set_aspect(1.35)

colorbar.ColorbarBase(
    axs[1],
    cmap=jet,
    norm=cNorm
)

plt.show()
province = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_province.csv')

province['Latitude'] = province['Latitude'].replace(0, np.nan)
province['Longitude'] = province['Longitude'].replace(0, np.nan)

province = province.dropna(subset=['RegionCode','ProvinceName','Latitude','Longitude'])

province = province.drop(columns=['SNo','RegionCode','Latitude','Longitude']).groupby(by=['ProvinceCode','Date']).sum()
province
province_ora = province.groupby(level=0).last()
province_ora['TotalPositiveCases'].loc[1]
import json

province_geo = json.load(open('/kaggle/input/province-italiane/limits_IT_provinces.geojson'))
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from dateutil.parser import parse

jet = plt.get_cmap('OrRd')
cNorm  = colors.Normalize(vmin=0, vmax=province_ora['TotalPositiveCases'].max())
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
fig, ax = plt.subplots(1, 1, figsize=(9, 12), facecolor='#34b8e9')
cbaxes = fig.add_axes([0.78, 0.59, 0.0125, 0.225])
colorbar = plt.colorbar(scalarMap, cax=cbaxes)
colorbar.ax.set_facecolor('#FFF')

for feat in regioni_geo['features']:
    ax.add_patch(PolygonPatch(feat['geometry'], fc='#0000', ec='#000F', linewidth=2))

poly_map = {}
    
for feat in province_geo['features']:
    poly = PolygonPatch(feat['geometry'], ec='#000B', linewidth=0.65)
    poly_map[feat['properties']['prov_istat_code_num']] = poly
    ax.add_patch(poly)

for feat in regioni_geo['features']:
    ax.add_patch(PolygonPatch(feat['geometry'], fc='#0000', ec='#000F', linewidth=1))

ax.axis('off')
ax.set_aspect(1.35)
ax.set_xlim(6, 19)
ax.set_ylim(36.15, 47.65)
title = ax.set_title('Ora', fontsize='xx-large', fontweight='demibold', color='#FFF')

ax.text(0.03, 0.02, 'u/alberto_467',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
       color='#FFF')

def update(date, data):
    title.set_text(parse(date).strftime('%d/%m/%Y'))
    for ids, values in data.iterrows():
        provincia_code = ids[0]
        value = values[0]
        poly_map[provincia_code].set_facecolor(scalarMap.to_rgba(value))

for i, (date, data) in enumerate(province.groupby(level=1)):
    update(date, data)
    print('Generating %d-th image' % i)
    fig.savefig('%03d-frame.png' % i, dpi=120, facecolor='#34b8e9', bbox_inches='tight')
!convert -delay 1x5 *-frame.png -delay 3x1 056-frame.png -loop 0 province.gif
!gifsicle -O3 province.gif > province2.gif