import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
#dependencies

import datetime as datetime

from sklearn import preprocessing

import brewer2mpl
train = pd.read_csv('../input/act_train.csv', parse_dates=['date'])

test = pd.read_csv('../input/act_test.csv', parse_dates=['date'])

ppl = pd.read_csv('../input/people.csv', parse_dates=['date'])



df_train = pd.merge(train, ppl, on='people_id')

df_test = pd.merge(test, ppl, on='people_id')

del train, test, ppl
for d in ['date_x', 'date_y']:

    print('Start of ' + d + ': ' + str(df_train[d].min().date()))

    print('  End of ' + d + ': ' + str(df_train[d].max().date()))

    print('Range of ' + d + ': ' + str(df_train[d].max() - df_train[d].min()) + '\n')
plus = sum(df_train.loc[:, 'outcome'] == 0)

minus = sum(df_train.loc[:, 'outcome'] == 1)



print (plus, minus)

print (df_train['outcome'].unique())
set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors



font = {'family' : 'sans-serif',

        'color'  : 'teal',

        'weight' : 'bold',

        'size'   : 18,

        }

plt.rc('font',family='serif')

plt.rc('font', size=16)

plt.rc('font', weight='bold')

#plt.style.use('seaborn-poster')

#plt.style.use('bmh')

#plt.style.use('ggplot')

plt.style.use('seaborn-dark-palette')

#plt.style.use('presentation')

print (plt.style.available)



# Get current size

fig_size = plt.rcParams["figure.figsize"]

 

# Set figure width to 6 and height to 6

fig_size[0] = 6

fig_size[1] = 6

plt.rcParams["figure.figsize"] = fig_size
from matplotlib import rcParams

rcParams['font.size'] = 12

#print (rcParams.keys())

rcParams['text.color'] = 'black'



piechart = plt.pie(

    (minus, plus),

    labels=('plus', 'minus'),

    shadow=False,

    colors=('teal', 'crimson'),

    explode=(0.08,0.08), # space between slices 

    startangle=90,    # rotate conter-clockwise by 90 degrees

    autopct='%1.1f%%',# display fraction as percentages

)



plt.axis('equal')   

plt.title("Outcome Ratio - Train Data", y=1.08,fontdict=font)

plt.tight_layout()

plt.savefig('Outcome-train.png', bbox_inches='tight')
tdates = df_train['date_x'][:2]

print(int(tdates[:1].to_string(index=False).split('-')[0]))

print (int(tdates[:1].to_string(index=False).split('-')[1]))

print (int(tdates[:1].to_string().split('-')[2]))

#print (tdates, str(tdates[:4]).astype(int))
def data_cleanser(data, is_train):

    



    def adjust_dates(dates, diff):

        return dates - diff

    

    if(is_train):

        pass         



    #Slide dates to past dates

    df_dates = data['date_x']

    diff = df_dates.max() - df_dates.min()

    diff2 = df_dates.max() - pd.Timestamp(pd.datetime.now().date())

    diffdays = diff + diff2

    data['adj_date'] = pd.to_datetime(adjust_dates(data['date_x'], diffdays))



    #print (data['adj_date'][:2])

    #covert dates to Yer, Month, Day

    #Break date time components into Y,M,D,H,M components

    #darr = df_dates.to_string().split('-')

    #data['Year'] = int(data['date_x'].to_string(index=False).split('-')[0])

    #data['Month'] = int(data['date_x'].to_string().split('-')[1])

    #data['Day'] = int(data['date_x'].to_string(header=False).split('-')[2])

    

    

    #data.drop(['AnimalID','OutcomeSubtype'],axis=1, inplace=True)

    #data['OutcomeType'] = data['OutcomeType'].map({'Return_to_owner':4, 'Euthanasia':3, 'Adoption':0, 'Transfer':5, 'Died':2})





    # Convert Color to numeric classes

    #breed = preprocessing.LabelEncoder()

    #to convert into numbers

    #data.Breed = breed.fit_transform(data.Breed)

    

            

    return data.drop(['date_x'],axis=1)
#tdates = data['adj_dates'][:2]

#print(tdates[:1].to_string(index=False).split('-')[2])





data = data_cleanser(df_train, True)



data.head()
data.head()



feature_cols = ['people_id', 'outcome']

data[feature_cols].head()

data['cnt'] = 1

print (data.groupby(['people_id'])['cnt'].sum())