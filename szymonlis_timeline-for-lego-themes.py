import pandas as pd
import numpy as np

import matplotlib
import matplotlib.mlab as mlab
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns

%matplotlib inline
matplotlib.style.use('seaborn')
# LEGO SETS
DF_Sets = pd.read_csv('../input/sets.csv')

# removing not relevant series
del DF_Sets['set_num']
del DF_Sets['name']
del DF_Sets['num_parts']

DF_Sets.head()
# THEMES SETS
DF_Themes = pd.read_csv('../input/themes.csv')

# renaming columns
DF_Themes = ( DF_Themes.rename(columns = {'id':'theme_id', 'name':'theme_name'}) )

DF_Themes.head()
# merge sets with themes
DF_Sets_Themes = pd.merge(DF_Sets, DF_Themes, how="inner", on = ['theme_id'])

del DF_Sets_Themes['theme_id']

DF_Sets_Themes.head()
# Decoding parent_id using themes dataset

# --- MID LEVEL --- #
DF_Mid_Themes = pd.merge(DF_Sets_Themes, DF_Themes, how="left", left_on = ['parent_id'], right_on = ['theme_id'])

del DF_Mid_Themes['parent_id_x']
del DF_Mid_Themes['theme_id']

DF_Mid_Themes = ( DF_Mid_Themes.rename(columns=
                                {'theme_name_x':'low_theme', 'theme_name_y':'mid_theme', 'parent_id_y':'parent_id'})
                  )

# check if there is not NaN value in low_theme series
assert ( len(DF_Mid_Themes[DF_Mid_Themes['low_theme'].isnull()]) == 0), "NaN values in Low Theme Series"


# --- TOP LEVEL --- #
DF_Chain_Themes = pd.merge(DF_Mid_Themes, DF_Themes, how="left", left_on = ['parent_id'], right_on = ['theme_id'])

del DF_Chain_Themes['parent_id_x']
del DF_Chain_Themes['theme_id']

# check if there all themes in top_level do not have parent
assert ( len(DF_Chain_Themes[DF_Chain_Themes['parent_id_y'].isnull()]) == DF_Sets_Themes.shape[0]) #Assert 11673 - full size

del DF_Chain_Themes['parent_id_y']

DF_Chain_Themes = ( DF_Chain_Themes.rename(columns = {'theme_name' : 'top_theme'}))

# --- ARRANGE THE THEMES IN LEVELS --- #
# some of the top level themes are now in the 'mid_theme' or 'low_theme' column
# to fix it this top-themes has beem shifted to higher levels themes columns and
# the 'null' value is entered to previously occupied cell by theme

# top_theme in 'low_theme'
_NaN_mask = ((DF_Chain_Themes['mid_theme'].isnull()) & (DF_Chain_Themes['top_theme'].isnull()))
DF_Chain_Themes.loc[_NaN_mask, 'top_theme'] = DF_Chain_Themes['low_theme']
DF_Chain_Themes.loc[_NaN_mask, 'low_theme'] = 'null'
DF_Chain_Themes.loc[_NaN_mask, 'mid_theme'] = 'null'

# check if there is not NaN value in mid_theme series
assert ( len(DF_Chain_Themes[DF_Chain_Themes['mid_theme'].isnull()]) == 0), "NaN values in Mid Theme Series"

#top_theme in 'mid_theme'
_NaN_mask = ( DF_Chain_Themes['top_theme'].isnull() )
DF_Chain_Themes.loc[_NaN_mask, 'top_theme'] = DF_Chain_Themes['mid_theme']
DF_Chain_Themes.loc[_NaN_mask, 'mid_theme'] = DF_Chain_Themes['low_theme']
DF_Chain_Themes.loc[_NaN_mask, 'low_theme'] = 'null'

# check if there is not NaN value in mid_theme series
assert ( len(DF_Chain_Themes[DF_Chain_Themes['top_theme'].isnull()]) == 0), "NaN values in Top Theme Series"

DF_Chain_Themes.head()
# ROOT AND TOP THEMES NODES 

_unique_topThemes = len(DF_Chain_Themes['top_theme'].unique())
_startYear = DF_Chain_Themes['year'].min() - 10 # -10 to offset a starting point

# group by top theme and get the year of release
group_themes = DF_Chain_Themes.groupby(['top_theme']).min().year

assert (len(group_themes)), _unique_topThemes

TEMP_DF = pd.DataFrame(data={'year':group_themes.tolist(),
                            'theme' : group_themes.index.get_level_values(0)})

# create column for list of sub themes of the theme
TEMP_DF['sub_themes'] = [list()]*len(TEMP_DF)

# create a dictonary
Themes_Dict = {'theme':'Root', 'year': _startYear, 'sub_themes': TEMP_DF.to_dict('records')}

assert (len(Themes_Dict['sub_themes'])), _unique_topThemes

TEMP_DF.head()
# MID THEMES NODES

# Function for adding mid theme to themes dictonary
def AddMidThemesDF2Dict(in_data, out_dict):
    
    # iterate over each unique top theme
    for index, row in in_data.iterrows():

        # create mid theme dict
        _dict = {}
        _dict = {'theme':row['mid_theme'], 'year': row['year'], 'sub_themes': list()}.copy() 

        # add to dict
        _top_dict = list(filter(lambda x: x['theme'] == row['top_theme'], out_dict['sub_themes']))[0]
    
        # find index of the neasted top dict
        _top_index = out_dict['sub_themes'].index(_top_dict)
     
        _tempList = out_dict['sub_themes'][_top_index]['sub_themes'].copy()
        _tempList.append(_dict)
    
        out_dict['sub_themes'][_top_index]['sub_themes'] = _tempList

        
    return out_dict
# this will create the mid_theme item in dictonary, but only for mid_theme which are the parent for others themes
# mid themes which do not split to sub-themes are picked in the next step

_unique_midThemes = len(DF_Chain_Themes['mid_theme'].unique())

# group by mid themes a and get the year of release
group_themes = DF_Chain_Themes.groupby(['top_theme', 'mid_theme']).min().year

# --- creat a data frame --- 
TEMP_DF = pd.DataFrame(data={'year':group_themes.tolist(),
                            'top_theme' : group_themes.index.get_level_values(0),
                            'mid_theme' : group_themes.index.get_level_values(1)})


# drop row with 'null' mid theme
TEMP_DF = TEMP_DF.drop(TEMP_DF[TEMP_DF['mid_theme']=='null'].index)

# --- put data frame to dictonary ---
Themes_Dict = AddMidThemesDF2Dict(TEMP_DF, Themes_Dict)

TEMP_DF.head()
# Function for adding low theme to themes dictonary
def AddLowThemesDF2Dict(in_data, out_dict):
    
    # iterate over each unique top theme
    for index, row in in_data.iterrows():
             
        # create theme dict - item
        _dict = {}
        _dict = {'theme':row['low_theme'], 'year': row['year'], 'sub_themes': list()}.copy()   

    
        # find index of for this row(top theme) in input/output dict
        _top_dict = list(filter(lambda x: x['theme'] == row['top_theme'], out_dict['sub_themes']))[0]    
        _top_index = out_dict['sub_themes'].index(_top_dict)
    
        # find index of for this row(mid theme) in input/output dict
        _mid_dict = list(filter(lambda x: x['theme'] == row['mid_theme'], out_dict['sub_themes'][_top_index]['sub_themes']))[0]
        _mid_index = out_dict['sub_themes'][_top_index]['sub_themes'].index(_mid_dict)
           
        #add the new item
        _tempList = out_dict['sub_themes'][_top_index]['sub_themes'][_mid_index]['sub_themes'].copy()
        _tempList.append(_dict)
    
    
        out_dict['sub_themes'][_top_index]['sub_themes'][_mid_index]['sub_themes'] = _tempList
    
    
    return out_dict
# --- creat a data frame --- 
_unique_lowThemes = len(DF_Chain_Themes['low_theme'].unique())

# group by top,mid adn low themes a and get the year of release
group_themes = DF_Chain_Themes.groupby(['top_theme', 'mid_theme', 'low_theme']).min().year

TEMP_DF = pd.DataFrame(data={'year':group_themes.tolist(),
                            'top_theme' : group_themes.index.get_level_values(0),
                            'mid_theme' : group_themes.index.get_level_values(1),
                            'low_theme' : group_themes.index.get_level_values(2)})


# drop row with 'null' low theme
TEMP_DF = TEMP_DF.drop( TEMP_DF[TEMP_DF['low_theme'] == 'null'].index )

# --- put data frame to dictonary ---
Themes_Dict = AddLowThemesDF2Dict(TEMP_DF, Themes_Dict)

TEMP_DF.head()
# sort the dictonary according to year

# top theme
Themes_Dict['sub_themes'].sort(key= lambda x: x['year'])

for mid_item in Themes_Dict['sub_themes']:
    
    #mid theme
    mid_item['sub_themes'].sort(key = lambda x: x['year'])
    
    for low_item in mid_item['sub_themes']:
        
        #low theme
        low_item['sub_themes'].sort(key = lambda x: x['year'])
# TOP-THEMES
group_themes = DF_Chain_Themes.groupby(['top_theme']).min().year

TOP_DF = pd.DataFrame(data={'year':group_themes.tolist(),
                            'top_theme' : group_themes.index.get_level_values(0)})

# group by year
group_top_by_year = TOP_DF.groupby(['year']).size()

TOP_THEMES_DATA = pd.DataFrame(data={'count': group_top_by_year.tolist(),
                                  'year': group_top_by_year.index.get_level_values(0)})
# ALL THEMES
group_themes = DF_Chain_Themes.groupby(['top_theme', 'mid_theme', 'low_theme']).min().year

ALL_DF = pd.DataFrame(data={'year':group_themes.tolist(),
                            'top_theme' : group_themes.index.get_level_values(0),
                            'mid_theme' : group_themes.index.get_level_values(1),
                            'low_theme' : group_themes.index.get_level_values(2)})

# group by year
group_all_by_year = ALL_DF.groupby(['year']).size()

ALL_THEMES_DATA = pd.DataFrame(data={'count': group_all_by_year.tolist(),
                                  'year': group_all_by_year.index.get_level_values(0)})
title_font = {'fontname':'Arial', 'size':'24'}
axis_font = {'fontname':'Arial', 'size':'18'}

fig, ax = plt.subplots(figsize=(15, 10))

ax.bar(ALL_THEMES_DATA['year'], ALL_THEMES_DATA['count'], align='center', color='#63B1BCAA', label='All Themes')
ax.bar(TOP_THEMES_DATA['year'], TOP_THEMES_DATA['count'], align='center', color='#8D6E97FF', label='Main Themes')

ax.set_xlabel('Year', **axis_font)
ax.set_ylabel('Count', **axis_font)
ax.set_title('Number of The New LEGO Theme Releases per Year (since 1950)', **title_font)

plt.tick_params(axis='both', which='major', labelsize=14)

ax.legend(loc='upper left', fontsize=16)
plt.show()
# preparing data for Pie Chart plot of the themes number per year

DF_Top_Themes_Sorted = DF_Chain_Themes.sort_values(by=['year'], ascending=True)
DF_Top_Themes_Sorted = DF_Top_Themes_Sorted.reset_index()


# cumulative value
# create data frame where: i) columns are top themes, 
# ii) rows are years, iii) values are number of themes and subthemes released that year
group_full_top_by_year = DF_Top_Themes_Sorted.groupby(['year'])

year_keys = (DF_Chain_Themes['year'].unique()).tolist()
year_keys = sorted(year_keys)

rows_list = []

cumulative_dict = {}

for row in year_keys:

        curr_dict = {}
                
        small_df = group_full_top_by_year.get_group(row)
        small_df_grouped = small_df.groupby('top_theme').size()
        
        curr_dict.update({'year':row})
        curr_dict.update(small_df_grouped) 
        
        #add to the previous row - cumulation of values
        cumulative_dict = { k: curr_dict.get(k, 0) + cumulative_dict.get(k, 0) for k in set(curr_dict) | set(cumulative_dict) }
               
        cumulative_dict.update({'year':row})
        
        rows_list.append(cumulative_dict)


# convert list of dictonaries to panda data frame
df_themes_cml_count = pd.DataFrame(rows_list)

df_themes_cml_count = df_themes_cml_count.fillna(0)

#put year column on front
cols = df_themes_cml_count.columns.tolist()
cols = cols[-1:] + cols[:-1]

df_themes_cml_count = df_themes_cml_count[cols]
df_themes_cml_count.head()
def GetPieData(cml_dataframe, year_index):
    
    r = year_index;

    # get year
    pie_year = cml_dataframe[r:r+1]['year'].values[0]

    # mask - get the columnes with value
    pie_mask = (cml_dataframe.iloc[r:r+1, 1:] > 0.0).values[0]

    # select column
    cols = cml_dataframe.columns.tolist()
    labels = np.array([cols[1:]])[:, pie_mask][0]

    # select data
    pie_data = (cml_dataframe.iloc[r:r+1, 1:].values)[:, pie_mask][0]

    #and sort in revers order
    p = pie_data.argsort()[::-1]

    pie_data = pie_data[p]
    labels = labels[p]
    
    return pie_data, labels, pie_year
def PlotPieGraph(year_index, graph_ax, anchor):
    
    pie_data, labels, pie_year = GetPieData(df_themes_cml_count, year_index)
    
    patches, texts = graph_ax.pie( pie_data, colors=cs, counterclock=False, startangle=90 )
    graph_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


    # put year in the middle
    centre_circle = plt.Circle((0,0),0.25,color='#FFFFFF', fc='white',linewidth=1.25)
    graph_ax.add_artist(centre_circle)
    graph_ax.text(0, 0, str(pie_year), horizontalalignment='center', verticalalignment='center', color='#000000', fontsize=24)
          
    max_lables = 25
    if ( len(patches) > max_lables):
        graph_ax.legend(patches[0:max_lables-1], labels[0:max_lables-1], loc='center', bbox_to_anchor=anchor, fontsize=12)
    else:
        graph_ax.legend(patches, labels, loc='center', bbox_to_anchor=anchor, fontsize=12)
# plot
title_font = {'fontname':'Arial', 'size':'24'}
axis_font = {'fontname':'Arial', 'size':'18'}

cs=sns.color_palette("Set2", 10)

fig = plt.figure(figsize=(15, 12))

# --- first graph
ax1 = fig.add_subplot(221)
PlotPieGraph(8, ax1, (1.1,0.5))

# --- second graph
ax2 = fig.add_subplot(222)
PlotPieGraph(18, ax2, (1.1,0.5))

# --- third graph
ax3 = fig.add_subplot(223)
PlotPieGraph(28, ax3, (1.1,0.5))

# --- forth graph
ax4 = fig.add_subplot(224)
PlotPieGraph(38, ax4, (1.1,0.5))


fig.suptitle("Size of Main Themes (size defined as number of sub-themes included in this main theme)", **title_font)
fig.text(0.5, 0.9,'Legend includes only top 25 themes', ha='center', va='center', **axis_font)

plt.show()
# plot
title_font = {'fontname':'Arial', 'size':'24'}
axis_font = {'fontname':'Arial', 'size':'18'}

cs=sns.color_palette("Set2", 10)

fig = plt.figure(figsize=(20, 12))

# --- first graph
ax1 = fig.add_subplot(121)
PlotPieGraph(48, ax1, (1.15,0.5))

# --- second graph
ax2 = fig.add_subplot(122)
PlotPieGraph(58, ax2, (1.15,0.5))


fig.suptitle("Size of Main Themes (size defined as number of sub-themes included in this main theme)", **title_font)
fig.text(0.5, 0.9,'Legend includes only top 25 themes', ha='center', va='center', **axis_font)

plt.show()
# plot
title_font = {'fontname':'Arial', 'size':'24'}
axis_font = {'fontname':'Arial', 'size':'18'}

cs=sns.color_palette("Set2", 10)

fig = plt.figure(figsize=(20, 12))

# --- first graph
ax1 = fig.add_subplot(111)
PlotPieGraph(65, ax1, (1.0,0.5))


fig.suptitle("Size of Main Themes (size defined as number of sub-themes included in this main theme)", **title_font)
fig.text(0.5, 0.9,'Legend includes only top 25 themes', ha='center', va='center', **axis_font)

plt.show()
# Helpers functions

# get number of sub-themes in top-theme
def GetTopThemeSize(theme):
    
    # 1 for itslef + count of mid_themes
    k = 1 + len(theme['sub_themes'])
    
    # + count of low_themes            
    for sub_item in theme['sub_themes']:
        
        k = k + (len(sub_item['sub_themes'] ))
    
    return k


# get number of sub-themes in mid-theme
def GetMidThemeSize(theme):
    
    # 1 for itslef + count of mid_themes
    k = 1 + len(theme['sub_themes'])
    
    return k


# get number of all themes
def GetAllThemeSize(themes):
    
    k = 1
    
    for item in themes:
    
        k = k + GetTopThemeSize(item) + 1
        
    return k


# get index of selected themes
def SelectPlotData(themes_list, themes_dict):
    
    themes_indexes = []
    
    for theme in themes_list:
        
        # find index of for this theme in input/output dict
        _dict = list(filter(lambda x: x['theme'] == theme, themes_dict['sub_themes']))[0]
        _index = themes_dict['sub_themes'].index(_dict)
        
        themes_indexes.append(_index)
        
    return themes_indexes
def PlotTopThemes(themes, plot_ax, _color_rect='C0', _color_font='C0', _alpha_rect=0.5):
        
    patches = []
    
    k = 0
    
    for item in themes:
        
        k = k + 1
        x = item['year']       
        w = 2050 - x
        
        h = GetTopThemeSize(item)
        
        rect = Rectangle( (x, k-0.05), w, h+0.1, color=_color_rect)
        patches.append(rect) 
        
        plt.text(x - 0.5, k, item['theme'].upper(), horizontalalignment='right', verticalalignment='bottom', color=_color_font, fontsize=14)
           
        k = k + h

    p = PatchCollection(patches, match_original=True, alpha=_alpha_rect)

    plot_ax.add_collection(p)
def PlotMidThemes(themes, plot_ax, _color_rect='C1', _color_font='C1', _alpha_rect=0.5):
        
    patches = []

    k = 0
    for top_item in themes:
        
        k = k + 1   
        top_h = GetTopThemeSize(top_item)

        m = k + 1
        for mid_item in top_item['sub_themes']:
            
            mid_h = GetMidThemeSize(mid_item)        
            mid_x = mid_item['year']
            mid_w = 2050 - mid_x
            
            rect = Rectangle( (mid_x, m), mid_w, mid_h, color=_color_rect)
            patches.append(rect)
        
            plt.text(mid_x + 0.5, m, mid_item['theme'].upper(), horizontalalignment='left', verticalalignment='bottom', color=_color_font, fontsize=13)
             
            m = m + mid_h
            
        k = k + top_h 
        
    p = PatchCollection(patches, match_original=True, alpha=_alpha_rect)
    plot_ax.add_collection(p)
def PlotLowThemes(themes, plot_ax, _color_rect='C2', _color_font='C2', _alpha_rect=0.5):
    
    patches = []
    
    k = 0
    for top_item in themes:
    
        k = k + 1   
        top_h = GetTopThemeSize(top_item)
    
        m = k + 1
        for mid_item in top_item['sub_themes']:
            
            l = m + 1
            mid_h = GetMidThemeSize(mid_item) 
    
            for low_item in mid_item['sub_themes']:
            
                low_h = 1
                low_x = low_item['year']
                low_w = 2050 - low_x
                
                rect = Rectangle( (low_x, l), low_w, low_h, color=_color_rect)
                patches.append(rect)
                    
                plt.text(low_x + 0.5, l, low_item['theme'].upper(), horizontalalignment='left', verticalalignment='bottom', color=_color_font, fontsize=12)
            
                l = l + low_h
                
            m = m + mid_h
            
        k = k + top_h 
        
    p = PatchCollection(patches, match_original=True, alpha=_alpha_rect)
    plot_ax.add_collection(p)
#plot graph

axis_font = {'fontname':'Arial', 'size':'24'}

fig, ax = plt.subplots(figsize=(20, 200))

ax.set_facecolor((1, 1, 1))

ax.grid(b=True, which='major', color='k', linestyle='--')

#_select_themes = [100, 101, 102, 103, 104, 105, 106, 107, 108]

#_plot_data = [Themes_Dict['sub_themes'][i] for i in _select_themes]


#1950 -1980 [0:23]
#1980 - 2000 [23:42]
#2000 - 2010 [42:72]
#2010 - 2018 [72:108]
_plot_data = Themes_Dict['sub_themes'][0:108]

# --- TOP THEMES --- #
PlotTopThemes(_plot_data, ax, _color_rect='#8D6E97', _color_font='#222222', _alpha_rect=0.6) ## C09C83  7B6469 6D3332


plt.xlim((1945, 2018))

h = GetAllThemeSize(_plot_data)

plt.ylim((0, h))

plt.title('LEGO Main Themes Over the Time 1950-2018', **axis_font)
plt.xlabel('Year', **axis_font)
plt.yticks([])

plt.tick_params(axis='both', which='major', labelsize=18)

top_patch = matplotlib.patches.Patch(color='#8D6E97AA', label='TOP THEME')
plt.legend(handles=[top_patch], bbox_to_anchor=(0.15, 1.0), borderpad=1, fontsize=16, frameon = 1)


plt.show()
#plot graph

axis_font = {'fontname':'Arial', 'size':'24'}

fig, ax = plt.subplots(figsize=(20, 40))

ax.set_facecolor((1, 1, 1))

ax.grid(b=True, which='major', color='k', linestyle='--')

_select_themes = SelectPlotData(['Classic', 'Train', 'Technic'], Themes_Dict)
_plot_data = [Themes_Dict['sub_themes'][i] for i in _select_themes]

# --- TOP THEMES --- #
PlotTopThemes(_plot_data, ax, _color_rect='#8D6E97', _color_font='#222222', _alpha_rect=0.6) ## C09C83  7B6469 6D3332

# --- MID THEMES --- #
PlotMidThemes(_plot_data, ax, _color_rect='#6787B7', _color_font='#555555', _alpha_rect=0.6) ##ECDCC8  672E45 00797C

# --- LOW THEMES --- #
PlotLowThemes(_plot_data, ax, _color_rect='#63B1BC', _color_font='#555555', _alpha_rect=0.7) ##A5B99C B1C9E8 FF9E1B

    
plt.xlim((1945, 2018))

h = GetAllThemeSize(_plot_data)

plt.ylim((0, h))

plt.title('LEGO Themes Time Line', **axis_font)
plt.xlabel('Year', **axis_font)
plt.yticks([])

plt.tick_params(axis='both', which='major', labelsize=18)

top_patch = matplotlib.patches.Patch(color='#8D6E97AA', label='TOP THEME')
mid_patch = matplotlib.patches.Patch(color='#6787B7AA', label='MID THEME')
low_patch = matplotlib.patches.Patch(color='#63B1BCAA', label='LOW THEME')
plt.legend(handles=[top_patch, mid_patch, low_patch ], bbox_to_anchor=(0.15, 1.0), borderpad=1, fontsize=16, frameon = 1)


plt.show()
#plot graph

axis_font = {'fontname':'Arial', 'size':'24'}

fig, ax = plt.subplots(figsize=(20, 40))

ax.set_facecolor((1, 1, 1))

ax.grid(b=True, which='major', color='k', linestyle='--')

_select_themes = SelectPlotData(['Town'], Themes_Dict)
_plot_data = [Themes_Dict['sub_themes'][i] for i in _select_themes]

# --- TOP THEMES --- #
PlotTopThemes(_plot_data, ax, _color_rect='#8D6E97', _color_font='#222222', _alpha_rect=0.6) ## C09C83  7B6469 6D3332

# --- MID THEMES --- #
PlotMidThemes(_plot_data, ax, _color_rect='#6787B7', _color_font='#555555', _alpha_rect=0.6) ##ECDCC8  672E45 00797C

# --- LOW THEMES --- #
PlotLowThemes(_plot_data, ax, _color_rect='#63B1BC', _color_font='#555555', _alpha_rect=0.7) ##A5B99C B1C9E8 FF9E1B

    
plt.xlim((1945, 2018))

h = GetAllThemeSize(_plot_data)

plt.ylim((0, h))

plt.title('LEGO Themes Time Line', **axis_font)
plt.xlabel('Year', **axis_font)
plt.yticks([])

plt.tick_params(axis='both', which='major', labelsize=18)

top_patch = matplotlib.patches.Patch(color='#8D6E97AA', label='TOP THEME')
mid_patch = matplotlib.patches.Patch(color='#6787B7AA', label='MID THEME')
low_patch = matplotlib.patches.Patch(color='#63B1BCAA', label='LOW THEME')
plt.legend(handles=[top_patch, mid_patch, low_patch ], bbox_to_anchor=(0.15, 1.0), borderpad=1, fontsize=16, frameon = 1)


plt.show()
#plot graph

axis_font = {'fontname':'Arial', 'size':'24'}

fig, ax = plt.subplots(figsize=(20, 40))

ax.set_facecolor((1, 1, 1))

ax.grid(b=True, which='major', color='k', linestyle='--')

_select_themes = SelectPlotData(['Star Wars', 'Creator', 'Collectible Minifigures'], Themes_Dict)
_plot_data = [Themes_Dict['sub_themes'][i] for i in _select_themes]

# --- TOP THEMES --- #
PlotTopThemes(_plot_data, ax, _color_rect='#8D6E97', _color_font='#222222', _alpha_rect=0.6) ## C09C83  7B6469 6D3332

# --- MID THEMES --- #
PlotMidThemes(_plot_data, ax, _color_rect='#6787B7', _color_font='#555555', _alpha_rect=0.6) ##ECDCC8  672E45 00797C

# --- LOW THEMES --- #
PlotLowThemes(_plot_data, ax, _color_rect='#63B1BC', _color_font='#555555', _alpha_rect=0.7) ##A5B99C B1C9E8 FF9E1B

    
plt.xlim((1945, 2018))

h = GetAllThemeSize(_plot_data)

plt.ylim((0, h))

plt.title('LEGO Themes Time Line', **axis_font)
plt.xlabel('Year', **axis_font)
plt.yticks([])

plt.tick_params(axis='both', which='major', labelsize=18)

top_patch = matplotlib.patches.Patch(color='#8D6E97AA', label='TOP THEME')
mid_patch = matplotlib.patches.Patch(color='#6787B7AA', label='MID THEME')
low_patch = matplotlib.patches.Patch(color='#63B1BCAA', label='LOW THEME')
plt.legend(handles=[top_patch, mid_patch, low_patch ], bbox_to_anchor=(0.15, 1.0), borderpad=1, fontsize=16, frameon = 1)


plt.show()