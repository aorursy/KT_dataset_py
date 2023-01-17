import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import geopandas as gpd

import os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

print(os.listdir("../input"))

map_df=gpd.read_file('../input/IND_adm1.shp')



map_df["rep"] = map_df["geometry"].centroid

map_points = map_df.copy()

map_points.set_geometry("rep", inplace = True)
map_df.plot()

map_df[map_df['NAME_1']=='Jammu and Kashmir']
map_df1=gpd.read_file('../input/Admin2.shp')

map_df.loc[14,'geometry']=map_df1.loc[12,'geometry']

map_df.plot()

#map_df1.iloc[12]=map_df.iloc[14]

df=pd.read_excel('../input/IN_HealthData.xls')

df.rename(columns={"Population and Household Profile - Sex ratio of the total population (females per 1000 males)":"Female_Ratio"}, inplace=True)

df_femaleratio = df[['India/States/UTs','Female_Ratio', 'Area']][(df['Survey']=='NFHS-4')]

ind_total = df_femaleratio[(df_femaleratio['Area']=='Total') & (df_femaleratio['India/States/UTs']=='India')]

ind_rural =df_femaleratio[(df_femaleratio['Area']=='Rural') & (df_femaleratio['India/States/UTs']=='India')]

ind_urban = df_femaleratio[(df_femaleratio['Area']=='Urban') & (df_femaleratio['India/States/UTs']=='India')]

df_femaleratio.head()

print(ind_total['Female_Ratio'])

pd.set_option('display.max_colwidth', 50)

df_femaleratio_t=df_femaleratio[df_femaleratio['Area']=='Total']

merged = map_df.set_index('NAME_1').join(df_femaleratio_t.set_index('India/States/UTs'))

merged.index.rename('STATE', inplace=True)



df_femaleratio_r=df_femaleratio[df_femaleratio['Area']=='Rural']



merged_r = map_df.set_index('NAME_1').join(df_femaleratio_r.set_index('India/States/UTs'))

merged_r.index.rename('STATE', inplace=True)

merged_r.fillna(0, inplace=True)



df_femaleratio_u=df_femaleratio[df_femaleratio['Area']=='Urban']



merged_u = map_df.set_index('NAME_1').join(df_femaleratio_u.set_index('India/States/UTs'))

merged_u.index.rename('STATE', inplace=True)

merged_u.fillna(0, inplace=True)

variable = 'Female_Ratio'

vmin, vmax = 850, 1100

fig, ax = plt.subplots(1, figsize=(15, 10))

merged.plot(column=variable, cmap='cividis', linewidth=1, ax=ax, edgecolor='0.8', vmin=vmin, vmax=vmax)

print(merged.size)

#ax = merged.plot(figsize = (15, 12), color = "whitesmoke", edgecolor = "lightgrey", linewidth = 0.5)

texts = []

style1 = dict(size=9.5, color='blue')

style2 = dict(size=10.5, color='black', weight='bold')



for x, y, label, state in zip(map_points.rep.x, map_points.rep.y, merged['Female_Ratio'].astype(np.int64), merged['TYPE_1']):

    if state=='State':

        texts.append(plt.text(x, y, label, fontsize = 10, **style2))

    else:

         texts.append(plt.text(x, y, label, fontsize = 10, **style1))



axins = inset_axes(ax,

                   width="5%",  # width = 5% of parent_bbox width

                   height="50%",  # height : 50%

                   loc='center left',

                   bbox_to_anchor=(1.05, 0., 1, 1),

                   bbox_transform=ax.transAxes,

                   borderpad=0,

                   )

sm = plt.cm.ScalarMappable(cmap='cividis', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm, cax=axins)



ax.axis('off')

ax.set_title('Sex ratio of the total population (females per 1000 males)', fontdict={'fontsize': '25', 'fontweight' : '3'})

#ax.set_title(ind_total['Female_Ratio'], fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate('Source for Data: National Family Health Survey-4, published by data.gov.in',xy=(0.1, .05),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

ax.annotate('Source for Map and Spatial Data: http://projects.datameet.org/maps/',xy=(0.1,0.08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

fig.savefig("map1_export.png", dpi=800)



variable = 'Female_Ratio'

vmin, vmax = 850, 1100

fig, axs = plt.subplots(1,2 , figsize=(15,20), squeeze=False)

rect = fig.patch

#rect.set_facecolor('grey')



texts = []

texts1= []

texts2=[]

style1 = dict(size=9, color='blue')

style2 = dict(size=10.5, color='black')





merged_r.plot(column='Female_Ratio', cmap='cividis', linewidth=1, ax=axs[0][0], edgecolor='0.8', vmin=vmin, vmax=vmax)

for x, y, label, state in zip(map_points.rep.x, map_points.rep.y, merged_r['Female_Ratio'].astype(np.int64), merged_r['TYPE_1']):

    if state=='State':

        texts.append(axs[0][0].text(x, y, label, fontsize = 10, **style2))

    else:

         texts.append(axs[0][0].text(x, y, label, fontsize = 10, **style1))

axs[0][0].axis('off')

axs[0][0].set_title('Rural - Sex ratio of the total population (females per 1000 males)', fontdict={'fontsize': '10', 'fontweight' : '3'})





merged_u.plot(column='Female_Ratio', cmap='cividis', linewidth=1, ax=axs[0][1], edgecolor='0.8', vmin=vmin, vmax=vmax)

for x, y, label, state in zip(map_points.rep.x, map_points.rep.y, merged_u['Female_Ratio'].astype(np.int64), merged_u['TYPE_1']):

    if state=='State':

        texts2.append(axs[0][1].text(x, y, label, fontsize = 10, **style2))

    else:

         texts2.append(axs[0][1].text(x, y, label, fontsize = 10, **style1))

axs[0][1].axis('off')

axs[0][1].set_title('Urban - Sex ratio of the total population (females per 1000 males)', fontdict={'fontsize': '10', 'fontweight' : '3'})



#fig.delaxes(axs[0][1])



axins = inset_axes(axs[0][1],

                   width="5%",  # width = 5% of parent_bbox width

                   height="50%",  # height : 50%

                   loc='center left',

                   bbox_to_anchor=(1.05, 0., 1, 1),

                   bbox_transform=axs[0][1].transAxes,

                   borderpad=0,

                   )

sm = plt.cm.ScalarMappable(cmap='cividis', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm, cax=axins)



fig.savefig("map_export.png", dpi=800)

merged.plot(column='Female_Ratio', cmap='cividis', linewidth=1, ax=axs[0,0], edgecolor='0.8', vmin=vmin, vmax=vmax)

for x, y, label, state in zip(map_points.rep.x, map_points.rep.y, merged['Female_Ratio'].astype(np.int64), merged['TYPE_1']):

    if state=='State':

        texts1.append(axs[0,0].text(x, y, label, fontsize = 10, **style2))

    else:

         texts1.append(axs[0,0].text(x, y, label, fontsize = 10, **style1))

#ax = merged.plot(figsize = (15, 12), color = "whitesmoke", edgecolor = "lightgrey", linewidth = 0.5)



axs[0,0].axis('off')

axs[0,0].set_title('Total - Sex ratio of the total population (females per 1000 males)', fontdict={'fontsize': '10', 'fontweight' : '3'})

#axs[0,1].annotate('Source for Data: National Family Health Survey-4, published by data.gov.in',xy=(0.1, .05),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

#axs[0,1].annotate('Source for Map and Spatial Data: http://projects.datameet.org/maps/',xy=(0.1,0.08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
