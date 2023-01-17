import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

input_data = "../input/forcedataset/force-ai-well-logs/train.csv"

TARGET_1 = "FORCE_2020_LITHOFACIES_LITHOLOGY"

TARGET_2 = "FORCE_2020_LITHOFACIES_CONFIDENCE"

WELL_NAME = 'WELL'
df = pd.read_csv(input_data, sep=';')
df.head()
df.columns
df.dtypes
lithology_keys = {30000: 'Sandstone',

                     65030: 'Sandstone/Shale',

                     65000: 'Shale',

                     80000: 'Marl',

                     74000: 'Dolomite',

                     70000: 'Limestone',

                     70032: 'Chalk',

                     88000: 'Halite',

                     86000: 'Anhydrite',

                     99000: 'Tuff',

                     90000: 'Coal',

                     93000: 'Basement'}
lithology_numbers = {30000: 0,

             65030: 1,

             65000: 2,

             80000: 3,

             74000: 4,

             70000: 5,

             70032: 6,

             88000: 7,

             86000: 8,

             99000: 9,

             90000: 10,

             93000: 11}
list_rock_type = list(df['FORCE_2020_LITHOFACIES_LITHOLOGY'].unique())
list_rock_type
data_summary = df.describe()
not_for_summary = ['FORCE_2020_LITHOFACIES_LITHOLOGY','FORCE_2020_LITHOFACIES_CONFIDENCE']

for_summary = [c for c in data_summary.columns if c not in not_for_summary]
data_summary = data_summary[for_summary]
data_summary
# Read the Well's name and the number of the records for each Well



wells = np.unique(df['WELL'])

wells
plt.figure(figsize=(20,10))

df['WELL'].value_counts().plot(kind = 'bar')

plt.show()
rock_types_count = df[TARGET_1].value_counts()
rock_types_count
df[TARGET_1].value_counts().plot(kind = 'barh')
df_30000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 30000]

df_65030 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 65030]

df_65000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 65000]

df_80000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 80000]

df_74000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 74000]

df_70000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 70000]

df_70032 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 70032]

df_88000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 88000]

df_86000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 86000]

df_99000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 99000]

df_90000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 90000]

df_93000 = df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == 93000]
na_count = df.isna().sum()

N = df.shape[0]

na_count = na_count / N

na_count
na_count.plot(kind = 'bar')
# Feature summary for different types of Rock

def report_col(column):

    merge_df = []

    for i in range(len(list_rock_type)):

        merge_df.append(df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == list_rock_type[i]][column].describe().rename(list_rock_type[i]))

    

    merge_df = pd.concat(merge_df, axis = 1)

    return merge_df
def draw_box_rockType(col):

    merge_dat = []

    for rock in list_rock_type:

        merge_dat.append(df[df["FORCE_2020_LITHOFACIES_LITHOLOGY"] == rock][col].rename(rock))

    merge_dat = pd.concat(merge_dat, axis = 1)

    return merge_dat

    
df['CALI'].describe()
df_for_plot = df.copy()

df_for_plot['FORCE_2020_LITHOFACIES_LITHOLOGY'] = df_for_plot['FORCE_2020_LITHOFACIES_LITHOLOGY'].astype(str)
xticks = ["65000", "30000", "65030", "70000", "80000", "99000", "70032", "88000", "90000", "74000", "86000", "93000"]
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'CALI', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('CALI')
draw_box_rockType('CALI').plot(kind='box', figsize = (20,10))
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'Z_LOC', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('Z_LOC')
draw_box_rockType('Z_LOC').plot(kind='box', figsize = (20,10))
df['RMED'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'RMED', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('RMED')
draw_box_rockType('RMED').plot(kind='box', figsize = (20,10))
df['RDEP'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'RDEP', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('RDEP')
draw_box_rockType('RDEP').plot(kind='box', figsize = (20,10))
df['RHOB'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'RHOB', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('RHOB')
draw_box_rockType('RHOB').plot(kind='box', figsize = (20,10))
df['GR'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'GR', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('GR')
draw_box_rockType('GR').plot(kind='box', figsize = (20,10))
df['PEF'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'PEF', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('PEF')
draw_box_rockType('PEF').plot(kind='box', figsize = (20,10))
df['DTC'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'DTC', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('DTC')
draw_box_rockType('DTC').plot(kind='box', figsize = (20,10))
df['SP'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'SP', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('SP')
draw_box_rockType('SP').plot(kind='box', figsize = (20,10))
df['ROP'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'ROP', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('ROP')
draw_box_rockType('ROP').plot(kind='box', figsize = (20,10))
df['DRHO'].describe()
df_for_plot.plot(kind = 'scatter', x = 'FORCE_2020_LITHOFACIES_LITHOLOGY', y = 'DRHO', subplots = True, figsize = (20, 10), xticks = xticks)
report_col('DRHO')
draw_box_rockType('DRHO').plot(kind='box', figsize = (20,10))
heatmap_col = [col for col in df.columns if df[col].dtypes == 'float64']
plt.figure(figsize=(20,10))

sns.heatmap(df[heatmap_col].corr(), annot = True)

plt.show()
#color coding the rocks.



facies_color_map = { 'Sandstone': '#F4D03F',

                     'Sandstone/Shale': '#7ccc19',

                     'Shale': '#196F3D',

                     'Marl': '#160599',

                     'Dolomite': '#2756c4',

                     'Limestone': '#3891f0',

                     'Chalk': '#80d4ff',

                     'Halite': '#87039e',

                     'Anhydrite': '#ec90fc',

                     'Tuff': '#FF4500',

                     'Coal': '#000000',

                     'Basement': '#DC7633'}



# get a list of the color codes.

facies_colors = [facies_color_map[mykey] for mykey in facies_color_map.keys()]
confidence_color_map = {0:'#FF0000', 1: '#00FF00', 2: '#0000FF'}

confidence_color = [confidence_color_map[mykey] for mykey in confidence_color_map.keys()]
wells_df = df.copy();

def map_lith_key(lith_map, row):

    

    lith_key = row['FORCE_2020_LITHOFACIES_LITHOLOGY']

    

    if lith_key in lith_map:

        return lith_map[lith_key]

    else:

        print('Warning: Key {} not found in map'.format(lith_key))

        return np.nan



wells_df['LITHOLOGY'] = wells_df.apply (lambda row: map_lith_key(lithology_keys, row), axis=1)

wells_df['LITH_LABEL'] = wells_df.apply (lambda row: map_lith_key(lithology_numbers, row), axis=1)
#function to plot data in relation to depth and Rock Coloring

def make_facies_log_plot(log_df, curves, facies_colors, confidence_color):

    

    #make sure logs are sorted by depth

    logs = log_df.sort_values(by='DEPTH_MD')

    cmap_facies = colors.ListedColormap(

            facies_colors[0:len(facies_colors)], 'indexed')

    

    ztop=logs.DEPTH_MD.min(); zbot=logs.DEPTH_MD.max()

    

    cluster=np.repeat(np.expand_dims(logs['LITH_LABEL'].values,1), 100, 1)

    

    num_curves = len(curves)

    f, ax = plt.subplots(nrows=1, ncols=num_curves+2, figsize=(num_curves*3, 12))

    

    for ic, col in enumerate(curves):

        

        

     

        curve = logs[col]

            

        ax[ic].plot(curve, logs['DEPTH_MD'])

        ax[ic].set_xlabel(col)

        ax[ic].set_yticklabels([]);



    # make the lithfacies column

    im=ax[num_curves].imshow(cluster, interpolation='none', aspect='auto',

                    cmap=cmap_facies,vmin=0,vmax=11)

    

    divider = make_axes_locatable(ax[num_curves])

    cax = divider.append_axes("right", size="20%", pad=0.05)

    cbar=plt.colorbar(im, cax=cax)

    cbar.set_label((12*' ').join(['  SS', 'SS-Sh', 'Sh', 

                                ' Ml', 'Dm', 'LS', 'Chk ', 

                                '  Hl', 'Ann', 'Tuf', 'Coal', 'Bsmt']))

    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    

    



    ax[num_curves].set_xlabel('Facies')

    ax[num_curves].set_yticklabels([])

    ax[num_curves].set_xticklabels([])

    

    f.suptitle('Well: %s'%logs.iloc[0]['WELL'], fontsize=14,y=0.94)

    

    # make the confidence column

    cluster_1 = np.repeat(np.expand_dims(logs['FORCE_2020_LITHOFACIES_CONFIDENCE'].values - 1,1), 100, 1)

    cmap_confidence = colors.ListedColormap(

            confidence_color[0:len(confidence_color)], 'indexed')

    im_1 = ax[num_curves+1].imshow(cluster_1, interpolation = 'none', aspect = 'auto', cmap = cmap_confidence, vmin = 0, vmax = 2)

    

    divider_1 = make_axes_locatable(ax[num_curves + 1])

    cax_1 = divider_1.append_axes("right", size = "20%", pad = 0.05)

    cbar_1 = plt.colorbar(im_1, cax = cax_1)

    cbar_1.set_label((72*' ').join([' 1','2','3']))

    cbar_1.set_ticks(range(0,1)); cbar_1.set_ticklabels('')

    

    ax[num_curves + 1].set_xlabel('Confidence')

    ax[num_curves + 1].set_yticklabels([])

    ax[num_curves + 1].set_xticklabels([])

    

    for i in range(len(ax)-1):

        ax[i].set_ylim(ztop,zbot)

        ax[i].invert_yaxis()

        ax[i].grid()

        ax[i].locator_params(axis='x', nbins=3)

    

    

    plt.show()
leftout_col = ['DEPTH_MD', 'FORCE_2020_LITHOFACIES_LITHOLOGY', 'FORCE_2020_LITHOFACIES_CONFIDENCE', 'WELL', 'GROUP', 'FORMATION','X_LOC', 'Y_LOC', 'Z_LOC','LITHOLOGY', 'LITH_LABEL', 'RSHA', 'SGR', 'NPHI', 'ROP', 'DTS', 'DCAL', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']
make_facies_log_plot(wells_df[wells_df['WELL'] == wells[0]] ,set(wells_df.columns) - set(leftout_col), facies_colors, confidence_color)
make_facies_log_plot(wells_df[wells_df['WELL'] == wells[1]] ,set(wells_df.columns) - set(leftout_col), facies_colors, confidence_color)