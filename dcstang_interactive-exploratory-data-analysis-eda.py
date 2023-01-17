"""

1 Cathedral family - Rock outcrop complex, extremely stony.

2 Vanet - Ratake families complex, very stony.

3 Haploborolis - Rock outcrop complex, rubbly.

4 Ratake family - Rock outcrop complex, rubbly.

5 Vanet family - Rock outcrop complex complex, rubbly.

6 Vanet - Wetmore families - Rock outcrop complex, stony.

7 Gothic family.

8 Supervisor - Limber families complex.

9 Troutville family, very stony.

10 Bullwark - Catamount families - Rock outcrop complex, rubbly.

11 Bullwark - Catamount families - Rock land complex, rubbly.

12 Legault family - Rock land complex, stony.

13 Catamount family - Rock land - Bullwark family complex, rubbly.

14 Pachic Argiborolis - Aquolis complex.

15 unspecified in the USFS Soil and ELU Survey.

16 Cryaquolis - Cryoborolis complex.

17 Gateview family - Cryaquolis complex.

18 Rogert family, very stony.

19 Typic Cryaquolis - Borohemists complex.

20 Typic Cryaquepts - Typic Cryaquolls complex.

21 Typic Cryaquolls - Leighcan family, till substratum complex.

22 Leighcan family, till substratum, extremely bouldery.

23 Leighcan family, till substratum - Typic Cryaquolls complex.

24 Leighcan family, extremely stony.

25 Leighcan family, warm, extremely stony.

26 Granile - Catamount families complex, very stony.

27 Leighcan family, warm - Rock outcrop complex, extremely stony.

28 Leighcan family - Rock outcrop complex, extremely stony.

29 Como - Legault families complex, extremely stony.

30 Como family - Rock land - Legault family complex, extremely stony.

31 Leighcan - Catamount families complex, extremely stony.

32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.

33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.

34 Cryorthents - Rock land complex, extremely stony.

35 Cryumbrepts - Rock outcrop - Cryaquepts complex.

36 Bross family - Rock land - Cryumbrepts complex, extremely stony.

37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.

38 Leighcan - Moran families - Cryaquolls complex, extremely stony.

39 Moran family - Cryorthents - Leighcan family complex, extremely stony.

40 Moran family - Cryorthents - Rock land complex, extremely stony.

"""
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#data manipulation / dataframe libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



from bokeh.plotting import output_notebook, figure, show

from bokeh.models import ColumnDataSource, Div, Select, Button, ColorBar, CustomJS, Range1d

from bokeh.models.tools import HoverTool

from bokeh.layouts import row, column, layout

from bokeh.transform import cumsum, linear_cmap

from bokeh.palettes import Blues8

from bokeh.models.widgets import Tabs,Panel

from bokeh.io import show

from bokeh.transform import stack, factor_cmap

output_notebook() #set bokeh output to be inline with notebook



import holoviews as hv

from holoviews import opts

hv.extension('bokeh')



from ipywidgets import widgets

import missingno



print("Libraries loaded: NumPy, Pandas, Matplotlib as plt, Seaborn as sns, Missingno, Bokeh")
train_data = pd.read_csv("../input/learn-together/train.csv")



test_data = pd.read_csv("../input/learn-together/test.csv")



sample_subs = pd.read_csv("../input/learn-together/sample_submission.csv")



print("Loaded: train_data, Filesize: {:0.2f}MB".format((train_data.memory_usage().sum()/1000000)))

print("Loaded: test_data, Filesize:  {:0.2f}MB".format((test_data.memory_usage().sum()/1000000)))

print("Loaded: sample_subs, Filesize:  {:0.2f}MB".format((sample_subs.memory_usage().sum()/1000000)))
train_data.info()
train_data.columns
missingno.matrix(train_data, figsize=(20,5), labels=True, fontsize=10)
#initialize labels

labels = ['Spruce', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
#plot kde distribution for 7 cover types and their elevation

df2 = train_data[['Elevation', 'Cover_Type']].copy()



#create empty dictionary for loop

distElevationDict = {}

plotElevationDict = {}



#create 7 distribution plots 

for i in range(1,8):

    distElevationDict[i-1] = df2.loc[df2['Cover_Type'] == i]

    plotElevationDict[i-1] = hv.Distribution(distElevationDict[i-1], label=labels[i-1])

    

#overlay all 7 cover types

elevDistPlot = (plotElevationDict[0] *

                plotElevationDict[1] * 

                plotElevationDict[2] * 

                plotElevationDict[3] * 

                plotElevationDict[4] * 

                plotElevationDict[5] *

                plotElevationDict[6])
#display plot

#elevDistPlot.opts.info()

elevDistPlot.opts(

    opts.Distribution(height=500, width=800, title="Elevation Distribution by Type", xlim=(1800,4200), muted_alpha=0.2)

)
#plot stacked bar for wilderness type

wilderness_list = ['Rawah', 'Neota', 'Comanche Peak', 'Cache la Poudre']



#get dataframe ready

df3 = train_data[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

       'Wilderness_Area4', 'Cover_Type']].copy()



#reshape to enter into Holoviews

wildernessTypeDF = pd.DataFrame(data=[

    (wilderness_list[0],labels[0],df3[(df3['Wilderness_Area1']==1) & (df3['Cover_Type']==1)].sum()[0]),

    (wilderness_list[0],labels[1],df3[(df3['Wilderness_Area1']==1) & (df3['Cover_Type']==2)].sum()[0]),

    (wilderness_list[0],labels[2],df3[(df3['Wilderness_Area1']==1) & (df3['Cover_Type']==3)].sum()[0]),

    (wilderness_list[0],labels[3],df3[(df3['Wilderness_Area1']==1) & (df3['Cover_Type']==4)].sum()[0]),

    (wilderness_list[0],labels[4],df3[(df3['Wilderness_Area1']==1) & (df3['Cover_Type']==5)].sum()[0]),

    (wilderness_list[0],labels[5],df3[(df3['Wilderness_Area1']==1) & (df3['Cover_Type']==6)].sum()[0]),

    (wilderness_list[0],labels[6],df3[(df3['Wilderness_Area1']==1) & (df3['Cover_Type']==7)].sum()[0]),

    (wilderness_list[1],labels[0],df3[(df3['Wilderness_Area2']==1) & (df3['Cover_Type']==1)].sum()[1]),

    (wilderness_list[1],labels[1],df3[(df3['Wilderness_Area2']==1) & (df3['Cover_Type']==2)].sum()[1]),

    (wilderness_list[1],labels[2],df3[(df3['Wilderness_Area2']==1) & (df3['Cover_Type']==3)].sum()[1]),

    (wilderness_list[1],labels[3],df3[(df3['Wilderness_Area2']==1) & (df3['Cover_Type']==4)].sum()[1]),

    (wilderness_list[1],labels[4],df3[(df3['Wilderness_Area2']==1) & (df3['Cover_Type']==5)].sum()[1]),

    (wilderness_list[1],labels[5],df3[(df3['Wilderness_Area2']==1) & (df3['Cover_Type']==6)].sum()[1]),

    (wilderness_list[1],labels[6],df3[(df3['Wilderness_Area2']==1) & (df3['Cover_Type']==7)].sum()[1]),

    (wilderness_list[2],labels[0],df3[(df3['Wilderness_Area3']==1) & (df3['Cover_Type']==1)].sum()[2]),

    (wilderness_list[2],labels[1],df3[(df3['Wilderness_Area3']==1) & (df3['Cover_Type']==2)].sum()[2]),

    (wilderness_list[2],labels[2],df3[(df3['Wilderness_Area3']==1) & (df3['Cover_Type']==3)].sum()[2]),

    (wilderness_list[2],labels[3],df3[(df3['Wilderness_Area3']==1) & (df3['Cover_Type']==4)].sum()[2]),

    (wilderness_list[2],labels[4],df3[(df3['Wilderness_Area3']==1) & (df3['Cover_Type']==5)].sum()[2]),

    (wilderness_list[2],labels[5],df3[(df3['Wilderness_Area3']==1) & (df3['Cover_Type']==6)].sum()[2]),

    (wilderness_list[2],labels[6],df3[(df3['Wilderness_Area3']==1) & (df3['Cover_Type']==7)].sum()[2]),

    (wilderness_list[3],labels[0],df3[(df3['Wilderness_Area4']==1) & (df3['Cover_Type']==1)].sum()[3]),

    (wilderness_list[3],labels[1],df3[(df3['Wilderness_Area4']==1) & (df3['Cover_Type']==2)].sum()[3]),

    (wilderness_list[3],labels[2],df3[(df3['Wilderness_Area4']==1) & (df3['Cover_Type']==3)].sum()[3]),

    (wilderness_list[3],labels[3],df3[(df3['Wilderness_Area4']==1) & (df3['Cover_Type']==4)].sum()[3]),

    (wilderness_list[3],labels[4],df3[(df3['Wilderness_Area4']==1) & (df3['Cover_Type']==5)].sum()[3]),

    (wilderness_list[3],labels[5],df3[(df3['Wilderness_Area4']==1) & (df3['Cover_Type']==6)].sum()[3]),

    (wilderness_list[3],labels[6],df3[(df3['Wilderness_Area4']==1) & (df3['Cover_Type']==7)].sum()[3]),

                                     ],

                                columns =['Wilderness_Type','Cover_Type','Count'])



bars = hv.Bars(wildernessTypeDF, ['Wilderness_Type', 'Cover_Type'], 'Count')

bars.opts(width=800, height=800, title="Count by Wilderness Type")

stacked = bars.opts(stacked=True, clone=True, legend_position="bottom",

                    tools=['hover'], bar_width=0.5, fill_alpha=0.6, hover_fill_alpha=1)

stacked
#soil type plotting

soil_list = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3','Soil_Type4', 

             'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

             'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

             'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

             'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

             'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

             'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

             'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

             'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

             'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']



#get dataframe ready

df4 = train_data[['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',

                  'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

                  'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

                  'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

                  'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

                  'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

                  'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

                  'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

                  'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

                  'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']].copy()
#prepare dataframe

soilTypeDF = pd.DataFrame(columns=['Soil_Type','Cover_Type','Count'])



for n in range(0,7):

    for i in range(0,40):

        soilTypeDF = soilTypeDF.append(

        {

            'Soil_Type' : soil_list[i],

            'Cover_Type' : labels[n],

            'Count' : df4[(df4[soil_list[i]]==1) & (df4['Cover_Type']==n+1)].sum()[i]

        }, ignore_index=True)



"""

for i in range(0,40):

    soilTypeDF = soilTypeDF.append(

    {

        'Soil_Type' : soil_list[i],

        'Cover_Type' : labels[1],

        'Count' : df4[(df4[soil_list[i]]==1) & (df4['Cover_Type']==2)].sum()[i]

    }, ignore_index=True)    



for i in range(0,40):

    soilTypeDF = soilTypeDF.append(

    {

        'Soil_Type' : soil_list[i],

        'Cover_Type' : labels[1],

        'Count' : df4[(df4[soil_list[i]]==1) & (df4['Cover_Type']==3)].sum()[i]

    }, ignore_index=True)

"""





bars2 = hv.Bars(soilTypeDF, ['Soil_Type', 'Cover_Type'], 'Count')
bars2.opts(width=800, height=1100, title="Count by Soil Type", invert_axes=True)

stacked2 = bars2.opts(stacked=True, clone=True, legend_position="bottom", tools=['hover'],

                    bar_width=0.5, fill_alpha=0.6, hover_fill_alpha=1)

stacked2
#hillshade plotting

hillshade_list = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']



#get dataframe ready

df5 = train_data[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Cover_Type']].copy()

#hillshade is ordinal integer with range 0 - 255



df5['Cover_Type'].replace([1,2,3,4,5,6,7],['Spruce', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'], inplace=True)



#create boxplot chart

boxwhisker = hv.BoxWhisker(

    (['Hillshade_9am']*15120 + ['Hillshade_Noon']*15120 + ['Hillshade_3pm']*15120, 

    list(df5['Cover_Type'].values)*3, 

    list(df5['Hillshade_9am'].values) + list(df5['Hillshade_Noon'].values) + list(df5['Hillshade_3pm'].values)),

    ['Group', 'Cover_Type'], vdims='Hillshade'

).opts(box_color='Hillshade')

boxwhisker.opts(width=800, height=650, 

                xrotation=90, title="Visualization of Cover_Type by Hillshade Index",

               )



grouped2 = boxwhisker.opts(

    clone=True, ylim=(0,280), legend_position="bottom", tools=['hover'],

    box_width=0.5, box_fill_alpha=0.6, box_hover_fill_alpha=1, outlier_fill_alpha=0.1,

    outlier_line_width=0, whisker_alpha=0.2

)

grouped2

#distance plotting

hillshade_list = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']



#get dataframe ready

df6 = train_data[['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Cover_Type']].copy()



df6['Cover_Type'].replace([1,2,3,4,5,6,7],['Spruce', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'], inplace=True)



#create boxplot2

boxwhisker2 = hv.BoxWhisker(

    (['Horizontal_Distance_To_Hydrology']*15120 + 

    ['Vertical_Distance_To_Hydrology']*15120 + 

    ['Horizontal_Distance_To_Roadways']*15120 +

    ['Horizontal_Distance_To_Fire_Points']*15120, 

    list(df6['Cover_Type'].values)*4, 

    list(df6['Horizontal_Distance_To_Hydrology'].values) + 

    list(df6['Vertical_Distance_To_Hydrology'].values) + 

    list(df6['Horizontal_Distance_To_Roadways'].values) +

    list(df6['Horizontal_Distance_To_Fire_Points'].values)),

    ['Group', 'Cover_Type'], vdims='Distance'

).opts(box_color='Distance')



boxwhisker2.opts(width=800, height=650, 

                xrotation=90, title="Visualization of Cover_Type by Distances to Features",

               )



grouped3 = boxwhisker2.opts(

    clone=True, legend_position="bottom", tools=['hover'],

    box_width=0.5, box_fill_alpha=0.6, box_hover_fill_alpha=1, outlier_fill_alpha=0.1,

    outlier_line_width=0, whisker_alpha=0.2

)



grouped3