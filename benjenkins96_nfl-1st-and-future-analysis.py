import matplotlib.pylab as plt

import seaborn as sns

import numpy as np

import pandas as pd

import pandas_profiling



import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot

init_notebook_mode(connected=True)

from PIL import Image

from scipy import stats



pd.options.mode.chained_assignment = None
import warnings

warnings.filterwarnings("ignore")



import seaborn as sns



from functools import reduce





import os 

import gc

import psutil



%matplotlib inline
print(os.listdir("../input/nfl-playing-surface-analytics/"))
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
InjuryRecord = import_data("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

PlayList = import_data("../input/nfl-playing-surface-analytics/PlayList.csv")

PlayerTrackData = import_data("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
print(PlayerTrackData.shape)

print(PlayList.shape)

print(InjuryRecord.shape)
PlayList.head()
PlayList['StadiumType'].unique()
array_outdoors = ['Outdoor', 'Oudoor', 'Outdoors',

       'Ourdoor', 'Outddors', 'Heinz Field', 'Outdor', 'Outside', 'Cloudy']

array_indoors = ['Indoors', 'Indoor', 'Indoor', 'Retractable Roof']

array_open = ['Open','Outdoor Retr Roof-Open', 'Retr. Roof-Open', 'Indoor, Open Roof',

       'Domed, Open', 'Domed, open', 'Retr. Roof - Open']

array_closed = ['Closed Dome', 'Domed, closed', 'Dome', 'Domed',

       'Retr. Roof-Closed', 'Outdoor Retr Roof-Open', 'Retractable Roof', 'Indoor, Roof Closed', 'Retr. Roof - Closed', 'Bowl', 'Dome, closed',

       'Retr. Roof Closed']



PlayList['StadiumType'] = PlayList['StadiumType'].replace(array_outdoors, 'Outdoors')

PlayList['StadiumType'] = PlayList['StadiumType'].replace(array_indoors, 'Indoors')

PlayList['StadiumType'] = PlayList['StadiumType'].replace(array_open, 'Open')

PlayList['StadiumType'] = PlayList['StadiumType'].replace(array_closed, 'Closed')
PlayList['Weather'].unique()
array_clear = ['Clear and warm', 'Sunny', 'Clear',

       'Sunny and warm', 'Clear and Cool',

       'Clear and cold', 'Sunny and cold', 'Partly Sunny',

       'Mostly Sunny', 'Clear Skies', 'Partly sunny', 

       'Sunny and clear', 'Clear skies',

       'Sunny Skies', 'Fair', 'Partly clear', 

       'Heat Index 95', 'Sunny, highs to upper 80s', 

       'Mostly sunny', 'Sunny, Windy', 'Mostly Sunny Skies', 

       'Clear and Sunny', 'Clear and sunny',

       'Clear to Partly Cloudy', 'Cold']



array_cloudy = ['Mostly Cloudy', 'Cloudy',

       'Cloudy, fog started developing in 2nd quarter',

       'Partly Cloudy', 'Mostly cloudy', 'Cloudy and cold',

       'Cloudy and Cool', 'Partly cloudy', 

       'Party Cloudy', 'Hazy', 'Partly Clouidy',

       'Overcast', 'Cloudy, 50% change of rain',

       'Mostly Coudy', 'Cloudy, chance of rain',

       'Sun & clouds', 'Cloudy, Rain',

       'cloudy', 'Coudy']



array_indoors = ['Controlled Climate','Indoor',

       'N/A (Indoors)', 'Indoors', 'N/A Indoor']



array_precip = ['Rain',

       'Snow',

       'Scattered Showers',

       'Light Rain',

       'Heavy lake effect snow', 'Cloudy, Rain',

       'Rainy',

       'Cloudy, light snow accumulating 1-3"',

       'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

       'Rain shower', 'Rain likely, temps in low 40s.', 'Rain Chance 40%', 'Rain likely, temps in low 40s.',

       'Cloudy, 50% change of rain', '10% Chance of Rain', 'Showers', '30% Chance of Rain']



PlayList['Weather'] = PlayList['Weather'].replace(array_clear, 'Clear')

PlayList['Weather'] = PlayList['Weather'].replace(array_cloudy, 'Cloudy')

PlayList['Weather'] = PlayList['Weather'].replace(array_indoors, 'Indoors')

PlayList['Weather'] = PlayList['Weather'].replace(array_precip, 'Precipitation')
total = pd.merge(PlayList, InjuryRecord, on='PlayKey',how='left')
final = pd.merge(total, PlayerTrackData, on='PlayKey',how='left')
final['DM_M1'] = final['DM_M1'].fillna(0).astype(int)

final['DM_M7'] = final['DM_M7'].fillna(0).astype(int)

final['DM_M28'] = final['DM_M28'].fillna(0).astype(int)

final['DM_M42'] = final['DM_M42'].fillna(0).astype(int)
injury = pd.merge(InjuryRecord, PlayList, on='PlayKey',how='inner')
injured = pd.merge(injury, PlayerTrackData, on='PlayKey',how='inner')
injury_rate = injured['FieldType'].value_counts()/final['FieldType'].value_counts() 
ax = injury_rate.plot(title='Injuries Occur Over 1.8x More Often On Synthetic Fields', 

                      kind='barh', figsize=(12,8), color='#2678B2', fontsize=12)

vals = ax.get_xticks()

ax.xaxis.label.set_size(14)

ax = plt.xlabel('Injury Rate (%)')

ax = plt.ylabel('Field Type')
injured_syn = injured[injured['Surface']=='Synthetic']

injured_nat = injured[injured['Surface']=='Natural']
inj_s_syn = injured_syn.groupby('PlayKey', as_index=False)['s'].max()

inj_s_nat = injured_nat.groupby('PlayKey', as_index=False)['s'].max()
ax = sns.kdeplot(data=inj_s_syn ['s'], label='Synthetic', shade=True)

ax = sns.kdeplot(data=inj_s_nat['s'], label='Natural', shade=True)

ax = plt.title("Distribution of Max Speed of Synthetic and Natural Turf")

ax = plt.xlabel('Yards Per Second')

ax = plt.ylabel('Density')
stats.ttest_ind(inj_s_syn['s'], inj_s_nat['s'], equal_var = False)
inj_dis_syn = injured_syn.groupby('PlayKey', as_index=False)['dis'].max()

inj_dis_nat = injured_nat.groupby('PlayKey', as_index=False)['dis'].max()
ax = sns.kdeplot(data=inj_dis_syn ['dis'], label='Synthetic', shade=True)

ax = sns.kdeplot(data=inj_dis_nat['dis'], label='Natural', shade=True)

ax = plt.title("Distribution of Max Distance of Synthetic and Natural Turf")

ax = plt.xlabel('Yards Per Second')

ax = plt.ylabel('Density')
injured_syn['a'] = (injured_syn.s - injured_syn.s.shift(1))/ (injured_syn.time - injured_syn.time.shift(1))

injured_syn.a.iloc[0] = 0
injured_nat['a'] = (injured_nat.s - injured_nat.s.shift(1))/ (injured_nat.time - injured_nat.time.shift(1))

injured_nat.a.iloc[0] = 0 
inj_a_syn = injured_syn.groupby('PlayKey', as_index=False)['a'].max()

inj_a_nat = injured_nat.groupby('PlayKey', as_index=False)['a'].max()
ax = sns.kdeplot(data=inj_a_syn ['a'], label='Synthetic', shade=True)

ax = sns.kdeplot(data=inj_a_nat['a'], label='Natural', shade=True)

ax = plt.title("Distribution of Max Acceleration on Synthetic and Natural Turf")

ax = plt.xlabel('Yards Per Second Per Second')

ax = plt.ylabel('Density')
stats.ttest_ind(inj_a_nat['a'], inj_a_syn['a'], equal_var = False)
inj_a_syn = injured_syn.groupby('PlayKey', as_index=False)['a'].min()

inj_a_nat = injured_nat.groupby('PlayKey', as_index=False)['a'].min()
ax = sns.kdeplot(data=inj_a_syn ['a'], label='Synthetic', shade=True)

ax = sns.kdeplot(data=inj_a_nat['a'], label='Natural', shade=True)

ax = plt.title("Distribution of Max Deceleration of Synthetic and Natural Turf")

ax = plt.xlabel('Yards Per Second Per Second')

ax = plt.ylabel('Density')
# Subset the injured dataset by surface

synthetic = injured.query("Surface == 'Synthetic'")

natural = injured.query("Surface == 'Natural'")



# Set up the figure

f, ax = plt.subplots(figsize=(12, 10))

ax.set_aspect("equal")



# Draw the two density plots

ax = sns.kdeplot(synthetic.x, synthetic.y,

                 cmap="Reds", shade=True, shade_lowest=False)

ax = sns.kdeplot(natural.x, natural.y,

                 cmap="Blues", shade=True, shade_lowest=False)



# Add labels to the plot

red = sns.color_palette("Reds")[-2]

blue = sns.color_palette("Blues")[-2]

ax.text(2.5, 8.2, "natural", size=16, color=blue)

ax.text(2.5, 37, "synthetic", size=16, color=red)

ax = plt.title("Location of Injuries on Synthetic and Natural Turf")
ax = sns.kdeplot(data=injured_syn ['PlayerGamePlay'], label='Synthetic', shade=True)

ax = sns.kdeplot(data=injured_nat['PlayerGamePlay'], label='Natural', shade=True)

ax = plt.title("Distribution of the Number of Plays Until Injury on Synthetic and Natural Turf")

ax = plt.xlabel('Number of Plays')

ax = plt.ylabel('Density')

ttest = stats.ttest_ind(injured_syn['PlayerGamePlay'], injured_nat['PlayerGamePlay'], equal_var = False)

if ttest.pvalue < .05:

    print('The difference in the number of plays until injury on synthetic and natural turf is statistically significant.')

else:

    print('The difference in the number of plays until injury on synthetic and natural turf is not statistically significant.')

print('p-value:', '%f' % ttest.pvalue)
syn = injured[injured['Surface']=='Synthetic']

nat = injured[injured['Surface']=='Natural']
playkey = np.random.choice(injured.PlayKey[~injured.PlayKey.isna()])
import plotly.express as px
injuries = pd.get_dummies(injured, columns = ['FieldType'], dummy_na = True, drop_first = True)
injuries.columns
fig = px.parallel_coordinates(injuries, color="FieldType_Synthetic",

                              dimensions=['x', 'y', 's',

                                          'o', 'dis', 'dir'],

                              title="Parallel Coordinates of Tracking Data Amongst Injured Population Across Playing Surfaces",

                              color_continuous_scale=px.colors.diverging.Tealrose)

fig.show()
fig = px.density_contour(injured, x="x", y="y", color="BodyPart", marginal_x="box", marginal_y="box")

fig.show()
fig = px.density_heatmap(injured_syn, x="x", y="y", marginal_x="histogram", marginal_y="histogram")

fig.show()
fig = px.density_heatmap(injured_nat, x="x", y="y", marginal_x="histogram", marginal_y="histogram")

fig.show()
fig = px.line_3d(injured, x="x", y="y", z="s", color="BodyPart", line_dash="BodyPart")

fig.show()
fig = px.line(injured, x="x", y="y", color='PlayKey')

fig.show()
specific_player = injured[injured['PlayKey']=='43505-2-49']
fig = px.scatter_polar(specific_player, r="o", theta="dir", color='time')

fig.show()
import pandas as pd

straight = pd.read_csv("../input/straight/straight.csv")
fig = px.scatter_polar(straight, r="o", theta="dir", color='time',

                      title="Example of Player Running and Facing the Same Direction")

fig.show()
import pandas as pd

directions = pd.read_csv("../input/dummies/directions.csv")
fig = px.scatter_polar(directions, r="o", theta="dir", color='time',

                      title='Comparison of Player Running and Looking Straight, Running and Looking Left and Right')

fig.show()