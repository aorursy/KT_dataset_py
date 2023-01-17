import pandas as pd, seaborn as sns, numpy as np, matplotlib.pyplot as plt, plotly.graph_objects as go
DATA = {'Case': pd.read_csv('../input/coronavirusdataset/Case.csv'),

        'PatientRoute': pd.read_csv('../input/coronavirusdataset/PatientRoute.csv'),

        'PatientInfo': pd.read_csv('../input/coronavirusdataset/PatientInfo.csv'),

        'TimeAge': pd.read_csv('../input/coronavirusdataset/TimeAge.csv')

        } #These are the datasets that we are going to analyse one by one 
for key in list(DATA.keys()):

    print(key+'\'s dataset shape == '+str(DATA[key].shape))
#What are the most 10 places (infection_case) occured that contributed to a group infection and how many people got infected in total

fig, ax = plt.subplots(figsize=(10, 7))

clusters = DATA['Case'][DATA['Case']['group'] == True]



count_infected_ppl_by_infection_case = clusters.groupby('infection_case').confirmed.sum()[clusters['infection_case'].value_counts().keys()]

count_infected_ppl_by_infection_case = zip(count_infected_ppl_by_infection_case.keys(), count_infected_ppl_by_infection_case.values.astype(str))

count_infected_ppl_by_infection_case = [a+'('+b+')' for a, b in count_infected_ppl_by_infection_case]



sns.barplot(ax = ax, x= clusters['infection_case'].value_counts().values[:10],\

            y = count_infected_ppl_by_infection_case[:10])

#looks like most of groups got infected in Shincheonji Church with 5012 people got infected in total
#Some general information about the DATA['Case']

DATA['Case'].drop(['case_id', 'latitude', 'longitude'], axis=1).describe(include='all')

#NaN values just mean a column with type object cannot have any value for that row for instance a city column can't have Q1(25%) or a mean so it've put NaN and that logic to can be applied to integer or float columns...
#Let's move to the PatientRoute dataset and answer the question what's the most infected Routes

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

m15freq_province = DATA['PatientRoute']['province'].value_counts().keys() #most 15 frequent province



sns.scatterplot(ax = axes[0], x=DATA['PatientRoute']['longitude'], y=DATA['PatientRoute']['latitude'],\

                hue=DATA['PatientRoute']['province'].values, s=40)

plt.title('high density clusters')

sns.scatterplot(ax=axes[1], x=DATA['PatientRoute']['longitude'], y=DATA['PatientRoute']['latitude'], alpha= .2, s=40)

plt.show()

#{The Most infected Routes are in Seoul and it's surronding that makes sense 

#since Seoul is a dense city so if some people got infected in Seoul the virus will spread quickly.}

#{The bottum right cluster Daegu is where Shincheonji Church is located, a church

#that was the reason for about 5000 people getting infected.}





regions_colors={'Incheon':'blue', 'Gyeonggi-do':'orange', 'Seoul':'green', 'Jeollabuk-do':'red', 'Gangwon-do':'purple',

       'Jeollanam-do':'brown', 'Gwangju':'pink', 'Daegu':'grey'}

fig = go.Figure()

for province in DATA['PatientRoute']['province'].unique():

    fig.add_trace(go.Scattergeo(

            lon = DATA['PatientRoute'][DATA['PatientRoute']['province'] == province]['longitude'],

            lat = DATA['PatientRoute'][DATA['PatientRoute']['province'] == province]['latitude'],

            marker_color=[regions_colors[province] for _ in range(len(DATA['PatientRoute'][DATA['PatientRoute']['province'] == province]))],

            name=province



            ))



fig.update_layout(

            title = 'Most Infected Routes with COVID-19 in S. Korea',

            geo_scope='asia',

            )

fig.show()
#Let's move to the next dataset

DATA['PatientInfo'].head()

# Looks like we have some NaN values let's see how many per column
nans_perc = pd.DataFrame(DATA['PatientInfo'].isna().sum() / DATA['PatientInfo'].shape[0])

nans_perc[0] = nans_perc[0].apply(lambda x: str(x)[:4]+'%')

nans_perc.columns = ['NaNs %']

nans_perc.T #Quiet a miss,  let's focus on columns with low percentage of NaN values
sns.countplot(x=DATA['PatientInfo']['sex'], order=['female', 'male'], \

              hue= DATA['PatientInfo']['state'], palette='winter_r', )

#{Eventhought the number of isolated males less than isolated females the number

#of deceased males slightly more than the deceased females}
fig, ax = plt.subplots(figsize=(10, 7))

DATA['PatientInfo']['age'] = DATA['PatientInfo']['age'].apply(lambda x: int(str(x)[:str(x).index('s')]) \

                                                              if pd.isna(x) == False else x)

sns.countplot(DATA['PatientInfo']['age'],hue=DATA['PatientInfo']['state'], color='red',ax=ax)

plt.yticks(ticks=[i for i in range(0, 450, 25)])

plt.legend(loc='upper right')

plt.show()

for i in range(0, 110, 10):

    state_per_age = DATA['PatientInfo']['state'][DATA['PatientInfo']['age'] == i]

    print(str(i)+'s: isolated: {}, released: {}, deceased: {}'.\

          format((state_per_age == 'isolated').sum()/len(state_per_age),\

          (state_per_age == 'released').sum()/len(state_per_age), (state_per_age == 'deceased').sum()/len(state_per_age)))

#it seems like the less the age the more likely of released state and less likely of the deceased state 

    
# the percentage of releasing, isolating, deceasing,

DATA['PatientInfo']['state'].value_counts()/ len(DATA['PatientInfo']['state'].dropna())*100
fig, ax = plt.subplots(figsize=(11, 5))

sns.violinplot(data=DATA['PatientInfo'], x='state', y='age', hue='sex', ax=ax)

plt.legend(loc='lower right')

#isolated and released distributions look similar that make sense since most isolated cases will be released

#deceased cases have realtivly high age median
fig, ax = plt.subplots(figsize=(12, 5))

plt.xticks(rotation=20)

plt.title('the accumulated number of infected people at each day')

plt.ylabel('number of cases')

infected_ppl = DATA['TimeAge'].groupby('date').confirmed.sum()

sns.scatterplot(x=infected_ppl.keys(), y=infected_ppl.values, marker='x', s=100)

#{it looks like the begining of a normal-like curve with low kurtosis (low peak) and that's what affected countries are aiming for till they discover a vaccine,

#and this chart is showing the progress of South Korea at reducing the number of infected poeple per day}
fig, ax = plt.subplots(figsize=(12, 5))

plt.xticks(rotation=20)

plt.title('the accumulated number of deaths')

plt.ylabel('number of deaths')

deceased = DATA['TimeAge'].groupby('date').deceased.sum()

sns.scatterplot(x=deceased.keys(), y=deceased.values, marker='x', s=100, ax=ax)

#it looks like a linear trend, let's try to fit an ANN model on this data and predict today's number of deceased
import keras 

model = keras.models.Sequential([keras.layers.Dense(50, activation='relu'),

                                 keras.layers.Dense(50, activation='relu'),

                                 keras.layers.Dense(50, activation='relu'),

                                 keras.layers.Dense(1),

                                ])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(np.c_[range(len(deceased))], np.c_[deceased.values], epochs=200)
#last input value was 20 which correspond to the date 2020-03-22 today's date is 2020-03-25 so we just add 3 to 20

x = 23

print('total deaths =', np.round(model.predict(np.c_[x])[0][0]))

#the model outputed 128 deaths not a bad prediction since today's number of deaths in S. Korea is 126