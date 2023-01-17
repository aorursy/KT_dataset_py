import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
games=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

injrec=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')
games.fillna({'StadiumType': 'NoData',

                     'Weather': 'NoData',

                     }, inplace=True)



games.fillna(method='ffill', inplace=True)



games.replace('Missing Data', np.nan, inplace=True)



games['Temperature'].replace(-999, np.nan, inplace=True)



games.PlayType=games.PlayType.str.replace('Not Returned', '').str.replace('Returned', '').str.strip()

games.PlayType.replace('0', np.nan, inplace=True)
games.Weather=games.Weather.replace({'Indoors': 'Controlled Climate', 'Rain': 'Rain and Bad', 

                                     'N/A (Indoors)': 'Controlled Climate', 'Snow': 'Rain and Bad',

                                     'Indoor': 'Controlled Climate', 'Overcast': 'Rain and Bad',

                                     'N/A Indoor': 'Controlled Climate', 'Clear and cold': 'Clear',

                                     'Mostly Cloudy': 'Cloudy', 'Sunny and clear': 'Sunny',

                                     'Mostly cloudy': 'Cloudy', 'Rain Chance 40%': 'Rain and Bad',

                                     'Cloudy and cold': 'Cloudy', 'Sunny, highs to upper 80s': 'Sunny',

                                     'Cloudy and Cool': 'Cloudy', 'Cloudy, light snow accumulating 1-3"': 'Rain and Bad',

                                     'Partly cloudy': 'Partly Cloudy', 'Scattered Showers': 'Rain and Bad',

                                     'Party Cloudy': 'Partly Cloudy', 'Cold': 'Rain and Bad',

                                     'Partly Clouidy': 'Partly Cloudy', 'Sunny and cold': 'Sunny', 

                                     'Mostly Coudy': 'Partly Cloudy', 'Partly sunny': 'Sunny',

                                     'cloudy': 'Cloudy', 'Cloudy, fog started developing in 2nd quarter ': 'Rain and Bad',

                                     'Coudy': 'Cloudy', 'Showers': 'Rain and Bad',

                                     'Mostly Sunny': 'Sunny', 'Rainy': 'Rain and Bad',

                                     'Partly Sunny': 'Sunny', 'Clear to Partly Cloudy': 'Partly Cloudy',

                                     'Mostly sunny': 'Sunny', 'Rain shower': 'Rain and Bad',

                                     'Sunny Skies': 'Sunny', 'Heat Index 95': 'Rain and Bad',

                                     'Sunny and warm': 'Sunny', 'Cloudy, Rain': 'Rain and Bad',

                                     'Clear and Sunny': 'Clear', 'Heavy lake effect snow': 'Rain and Bad',

                                     'Clear and sunny': 'Clear', '30% Chance of Rain': 'Rain and Bad', 

                                     'Mostly Sunny Skies': 'Sunny', 'Cloudy, chance of rain ': 'Rain and Bad',

                                     'Clear Skies': 'Clear', 'Cloudy, 50% change of rain': 'Rain and Bad',

                                     'Clear skies': 'Clear', 'Rain likely, temps in low 40s.': 'Rain and Bad',

                                     'Clear and Cool': 'Clear', 'Sunny, Windy': 'Sunny',

                                     'Fair': 'Clear', 'Cloudy, fog started developing in 2nd quarter': 'Rain and Bad',

                                     'Light Rain': 'Rain and Bad', '10% Chance of Rain': 'Rain and Bad',

                                     'Clear and warm': 'Clear', 'Cloudy, chance of rain': 'Cloudy',

                                     'Hazy': 'Cloudy', 'Partly clear': 'Clear', 'Sun & clouds': 'Sunny',

'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.': 'Rain and Bad'})     
games.Weather=games.Weather.replace({'Controlled Climate': 'CC',

                                    'Rain and Bad': 'RaB',

                                    'Clear': 'Cr',

                                    'Sunny': 'Sun',

                                    'Partly Cloudy': 'PC',

                                    'Cloudy': 'Cl',

                                    'NoData': 'ND'})
injrec_full=pd.merge(games, injrec)

injrec_full.shape
injrec_full=pd.merge(games, injrec, how='right')

injrec_full['Injuries']=injrec_full[['DM_M1', 'DM_M7', 'DM_M28', 'DM_M42']].sum(axis=1)

inj_dict={1: 'Fine', 2: 'Little problem', 3: 'Serious Trouble', 4: 'Catastrophic Event'}

injrec_full['Injuries']=injrec_full['Injuries'].map(inj_dict)

injrec_full.drop(['DM_M1','DM_M7', 'DM_M28', 'DM_M42'], axis=1, inplace=True)

injrec_full.tail(3)
miss_data=games.loc[games['GameID']=='33337-2'].tail(1)

for i in range(78, len(injrec_full)):

    miss_data=miss_data.append(games.loc[games['GameID']==injrec_full.iloc[i][1]].tail(1))
miss_data_new=pd.merge(miss_data, injrec_full)

miss_data_new.shape
injrec_full=injrec_full.append(miss_data, sort=False)



def get_column_array(injrec_full, column):

    expected_length = len(injrec_full)

    current_array = injrec_full[column].dropna().values

    if len(current_array) < expected_length:

        current_array = np.append(current_array, [''] * (expected_length - len(current_array)))

    return current_array



injrec_full=pd.DataFrame({column: get_column_array(injrec_full, column) for column in injrec_full.columns})



injrec_full.replace('', np.nan, inplace=True)

injrec_full=injrec_full.dropna(thresh=10)

injrec_full.tail(3)
injrec_full['Temperature']=injrec_full['Temperature'].astype(float)

injrec_full['PlayerDay']=injrec_full['PlayerDay'].astype(float)

injrec_full['PlayerGame']=injrec_full['PlayerGame'].astype(float)

injrec_full['PlayerGamePlay']=injrec_full['PlayerGamePlay'].astype(float)
sns.set(rc={'axes.labelsize':30,

            'figure.figsize':(13.7,10.27),

            'xtick.labelsize':25,

            'ytick.labelsize':20})
fig, axs = plt.subplots(1, 2)



axs[0].pie(injrec['Surface'].value_counts(),

           autopct='%1.0f%%',

          colors=['#CDC70A', '#2A741B'],

          textprops={'fontsize': 18},

          shadow=True)



axs[1].pie(games['FieldType'].value_counts(),

           autopct='%1.0f%%',

           startangle=180,

          colors=['#2A741B', '#CDC70A'],

           textprops={'fontsize': 18},

          shadow=True)



axs[0].set_title('Distribution of Injuries by Surface',

                    fontsize= 24)



axs[1].set_title('Distribution of games by Surface',

                     fontsize= 24)



axs[1].legend(labels=games['FieldType'].value_counts().index,

    loc='upper right',

    prop={'size': 18},

     bbox_to_anchor=(1.4, 0.9),

             title='Surface',

             title_fontsize='28')



plt.show()
chart=sns.countplot(x='Surface',hue='Injuries',data=injrec_full, 

              palette=['#820108', '#FF4C56', '#EEF10B', '#0BF1CE'],

             edgecolor=(0,0,0),

                  linewidth=2)



chart.legend(prop={'size': 22},

     bbox_to_anchor=(1, 0.9),

            title='Injuries',

            title_fontsize='35')



chart.set_title('Surface Injury Distribution',

               fontsize=40)



chart.set(yticks=range(0, 21,2))



plt.show()
chart=sns.catplot(x='BodyPart', hue='Injuries', data=injrec_full, col='Surface', kind='count',

                  palette=['#820108', '#FF4C56', '#EEF10B', '#0BF1CE'],edgecolor=(0,0,0),

                  linewidth=2, legend=False)



chart.set_xticklabels(size=20)



chart.set_xlabels(size=24)



chart.set(yticks=range(0, 13,2))



chart.add_legend(prop={'size': 18},

            title='Injuries',

            title_fontsize='30')



plt.show()
chart1=sns.countplot(x='PlayType', hue='Injuries', data=injrec_full, 

              palette=['#820108', '#FF4C56', '#EEF10B', '#0BF1CE'],

             edgecolor=(0,0,0),

                  linewidth=2)



chart1.set_title('Distribution of Injuries by Play Type',

                     fontsize= 38)



chart1.legend(prop={'size': 22},

     bbox_to_anchor=(1, 0.9), title='Injuries',

            title_fontsize='35')



plt.show()
fig, axs = plt.subplots(1, 2)



axs[0].pie(injrec_full['PlayType'].value_counts(),

           radius=1.25,

           autopct='%1.0f%%',

           explode=(0, 0, 0.1, 0.2, 0),

           colors=['#094275', '#FA7D0F', '#0BA006', '#FF4513', '#DDE30F'],

           textprops={'fontsize': 18},

           shadow=True)



axs[1].pie(games['PlayType'].value_counts(),

           radius=1.25,

           autopct='%1.0f%%',

           explode=(0, 0, 0.2, 0.3, 0, 0),

           #startangle=180,

           colors=['#094275', '#FA7D0F', '#FF4513','#0BA006', '#7536A7', '#DDE30F'],

           textprops={'fontsize': 18},

           shadow=True)



axs[0].set_title('Distribution of Injuries by Play Type',

                     fontsize= 22)



axs[1].set_title('Distribution of games by Play Type',

                     fontsize= 22)



axs[1].legend(labels=games['PlayType'].value_counts().index,

    loc='upper right',

    prop={'size': 18},

     bbox_to_anchor=(1.6, 0.9),

             title='Play Type',

             title_fontsize='28')



plt.show()
chart=sns.catplot(x='PlayType', hue='Injuries', data=injrec_full, col='Surface', kind='count',

                  palette=['#820108', '#FF4C56', '#EEF10B', '#0BF1CE'],edgecolor=(0,0,0),

                  linewidth=2, legend=False)



chart.set_xticklabels(size=15)



chart.set_xlabels(size=24)



chart.set(yticks=range(0, 13,2))



chart.add_legend(prop={'size': 18},

            title='Injuries',

            title_fontsize='30')



plt.show()
chart=sns.catplot(x='PlayType', hue='BodyPart', data=injrec_full, col='Surface', kind='count',

                  palette=['#272A7C', '#1696C6', '#1E7928', '#CCE914', '#FF0000'],edgecolor=(0,0,0),

                  linewidth=2, legend=False)



chart.set_xticklabels(size=15)



chart.set_xlabels(size=24)



chart.add_legend(prop={'size': 18},

            title='Injuries',

            title_fontsize='30')



chart.set(yticks=range(0, 15,2))



plt.show()
fig, axs = plt.subplots(1, 2)



axs[0].pie(injrec_full['PositionGroup'].value_counts(),

           radius=1.2,

           autopct='%1.0f%%',

           explode=(0, 0.1, 0, 0.2, 0, 0.2, 0.1),

          colors=['#8E08FE' ,'#258187', '#A525B9', '#D6D676', '#D11C4D', '#2B8138', '#C49144'],

           textprops={'fontsize': 18},

          shadow=True)



axs[1].pie(games['PositionGroup'].value_counts(),

           radius=1.2,

           autopct='%1.0f%%',

            explode=(0, 0, 0.1, 0.1, 0, 0.25, 0.1, 0, 0),

          colors=['#8E08FE', '#A525B9', '#2B8138','#258187', '#D11C4D', '#D6D676', '#C49144', '#F96905', '#D1BFB2'],

           textprops={'fontsize': 18},

          shadow=True)



axs[0].set_title('Position Group Injury Distribution',

                     fontsize= 24)



axs[1].set_title('Position Group Games Distribution',

                     fontsize= 24)



axs[1].legend(labels=games['PositionGroup'].value_counts().index,

    loc='upper right',

    prop={'size': 16},

     bbox_to_anchor=(1.7, 0.9),

             title='Position Group',

             title_fontsize='28')



plt.show()
chart1=sns.countplot(x='PositionGroup', hue='Injuries', data=injrec_full, 

              palette=['#820108', '#FF4C56', '#EEF10B', '#0BF1CE'],

             edgecolor=(0,0,0),

                  linewidth=2)



chart1.set_title('Distribution of Injuries by Position Group',

                     fontsize= 38)



chart1.legend(prop={'size': 22},

     bbox_to_anchor=(1, 0.9), title='Injuries',

            title_fontsize='35')



plt.show()
chart=sns.catplot(x='PositionGroup', hue='Injuries', data=injrec_full, col='Surface', kind='count',

                  palette=['#820108', '#FF4C56', '#EEF10B', '#0BF1CE'],edgecolor=(0,0,0),

                  linewidth=2, legend=False)



chart.set_xticklabels(size=15)



chart.set_xlabels(size=24)



chart.set(yticks=range(0, 13,2))



chart.add_legend(prop={'size': 18},

            title='Injuries',

            title_fontsize='30')



plt.show()
chart=sns.catplot(x='PositionGroup', hue='BodyPart', data=injrec_full, col='Surface', kind='count',

                  palette=['#272A7C', '#1696C6', '#1E7928', '#CCE914', '#FF0000'],edgecolor=(0,0,0),

                  linewidth=2, legend=False)

                  #height=15, aspect=1)

    

chart.set_xticklabels(size=15)



chart.set_xlabels(size=24)



chart.add_legend(prop={'size': 18},

            title='Injuries',

            title_fontsize='30')



chart.set(yticks=range(0, 13,2))



plt.show()
wth=['Cloudy', 'Sunny', 'Clear', 'Partly Cloudy','Rain and Bad', 'Controlled Climate', 'NoData']



fig, axs = plt.subplots(1, 2)



axs[0].pie(injrec_full['Weather'].value_counts(),

           radius=1.2,

           autopct='%1.0f%%',

         explode=(0.1, 0, 0.1, 0, 0.2, 0.2, 0),

          colors=['#F7FE00', '#498484', '#1550A0', '#B8CCF2', '#DE00FF', '#D2355D', '#453547' ],

           textprops={'fontsize': 18},

          shadow=True)



axs[1].pie(games['Weather'].value_counts(),

           radius=1.2,

           autopct='%1.0f%%',

            explode=(0.1, 0, 0.1, 0, 0.2, 0.2, 0),

         colors=['#498484', '#F7FE00', '#B8CCF2','#1550A0', '#D2355D', '#DE00FF', '#453547'],

           textprops={'fontsize': 18},

          shadow=True)



axs[0].set_title('Weather Injury Distribution',

                     fontsize=24)

axs[1].set_title('Weather Games Distribution',

                     fontsize=24)



axs[1].legend(labels=wth,

    loc='upper right',

    prop={'size': 20},

     bbox_to_anchor=(1.8, 0.9),

             title='Weather',

             title_fontsize='28')



plt.show()
chart1=sns.countplot(x='Weather', hue='Injuries', data=injrec_full, 

              palette=['#820108', '#FF4C56', '#EEF10B', '#0BF1CE'],

             edgecolor=(0,0,0),

                  linewidth=2)



chart1.set_title('Distribution of Injuries by Position Group',

                     fontsize= 38)



chart1.legend(prop={'size': 25},

     bbox_to_anchor=(1, 0.9), title='Injuries',

            title_fontsize='45')



plt.show()
chart=sns.catplot(x='Weather', hue='Injuries', data=injrec_full, col='Surface', kind='count', 

                  palette=['#820108', '#FF4C56', '#EEF10B', '#0BF1CE'],edgecolor=(0,0,0),

                  linewidth=2, legend=False)



chart.set_xticklabels(size=20)



chart.set_xlabels(size=24)



chart.set(yticks=range(0, 13,2))



chart.add_legend(prop={'size': 18},

            title='Injuries',

            title_fontsize='80')



plt.show()
chart=sns.catplot(x='Weather', hue='BodyPart', data=injrec_full, col='Surface', kind='count',

                  palette=['#272A7C', '#1696C6', '#1E7928', '#CCE914', '#FF0000'],edgecolor=(0,0,0),

                  linewidth=2, legend=False)



chart.set_xticklabels(size=20)



chart.set_xlabels(size=24)



chart.set(yticks=range(0, 13,2))



chart.add_legend(prop={'size': 18},

            title='Injuries',

            title_fontsize='30')



plt.show()
labels=['Temperature', 'PlayerDay', 'PlayerGame', 'PlayerGamePlay']

x=injrec_full[['Temperature', 'PlayerDay', 'PlayerGame', 'PlayerGamePlay']].mean().astype(int)

y=games[['Temperature', 'PlayerDay', 'PlayerGame', 'PlayerGamePlay']].mean().astype(int)

z = np.arange(len(x))  

width = 0.35  



fig, ax = plt.subplots()



rects1 = ax.bar(z - width/2, x, width, label='Injuries games', edgecolor=(0,0,0),

                  linewidth=2)

rects2 = ax.bar(z + width/2, y, width, label='All Games', edgecolor=(0,0,0),

                  linewidth=2)



ax.set_ylabel('Count')

ax.set_title('Average Comparison', fontsize=26)

ax.set_xticks(z)

ax.set_xticklabels(labels)

ax.legend( loc='upper right',

    prop={'size': 20})



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 4), 

                    textcoords="offset points",

                    ha='center', va='bottom', size=20)

autolabel(rects1)

autolabel(rects2)



fig.tight_layout()

fig.set_size_inches(12, 8)



plt.show()
labels=['Temperature', 'PlayerDay', 'PlayerGame', 'PlayerGamePlay']

x=injrec_full[['Temperature', 'PlayerDay', 'PlayerGame', 'PlayerGamePlay']].median().astype(int)

y=games[['Temperature', 'PlayerDay', 'PlayerGame', 'PlayerGamePlay']].median().astype(int)

z = np.arange(len(x))  

width = 0.35 



fig, ax = plt.subplots()



rects1 = ax.bar(z - width/2, x, width, label='Injuries games', edgecolor=(0,0,0),

                  linewidth=2)

rects2 = ax.bar(z + width/2, y, width, label='All games', edgecolor=(0,0,0),

                  linewidth=2)



ax.set_ylabel('Count')

ax.set_title('Median Comparison',  fontsize=26)

ax.set_xticks(z)

ax.set_xticklabels(labels)

ax.legend(loc='upper right',

    prop={'size': 20})



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3), 

                    textcoords="offset points",

                    ha='center', va='bottom',

                    size=20)

autolabel(rects1)

autolabel(rects2)



fig.tight_layout()

fig.set_size_inches(12, 8)



plt.show()
fig = plt.figure()

ax = fig.suptitle('Distribution of Injured games and all games by', fontsize=36)



ax1=plt.subplot(221)

sns.kdeplot(injrec_full['PlayerDay'], shade=True, color='#FF0000', legend=False)

sns.kdeplot(games['PlayerDay'], shade=True, color='#002CF2', legend=False)

ax1.set_title('Player Day', fontsize=24)



ax2=plt.subplot(2, 2, 2)

sns.kdeplot(injrec_full['PlayerGame'], shade=True, color='#FF0000', label='With Injuries')

sns.kdeplot(games['PlayerGame'], shade=True, color='#002CF2', label='All')

plt.legend(loc='upper right',

           bbox_to_anchor=(1.75, 0.9),

    prop={'size': 24},

          title='Games',

             title_fontsize='28')

ax2.set_title('Player Game', fontsize=24)



ax3=plt.subplot(2, 2, 3)

sns.kdeplot(injrec_full['PlayerGamePlay'], shade=True, color='#FF0000', legend=False)

sns.kdeplot(games['PlayerGamePlay'], shade=True, color='#002CF2', legend=False)

ax3.set_title('Player Game Play', fontsize=24)



ax4=plt.subplot(2, 2, 4)

sns.kdeplot(injrec_full['Temperature'], shade=True, color='#FF0000', legend=False)

sns.kdeplot(games['Temperature'], shade=True, color='#002CF2', legend=False)

ax4.set_title('Temperature', fontsize=24)



plt.show()