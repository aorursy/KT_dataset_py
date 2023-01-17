import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from math import radians



# The following option is form here ...

# Not sure what this does ..

# https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/simple-exploration-notebook-2-connect

pd.options.mode.chained_assignment = None  # default='warn'



train = pd.read_json("../input/train.json")

test  = pd.read_json("../input/test.json")

print("Train Rows : ", train.shape[0])

print("Test Rows : ", test.shape[0])
train.head()
predictor = 'interest_level'

numericCols = ['bathrooms', 'bedrooms', 'price']

pictureCols = ['photos']

mapColumns  = ['latitude', 'longitude']

descriptions = ['features', 'description'] # basically we need to combine these
sns.barplot(x = 'index', y='interest_level',

    data=train['interest_level'].value_counts().reset_index())

plt.xlabel('interest'); plt.ylabel('counts')
# The building_id is a dummy variable

temp = train.ix[ :, numericCols + [predictor, 'building_id']]

counts = train['interest_level'].value_counts()

counts
temp.head()
col = numericCols[0]

for col in numericCols:

    if len(temp[col].unique()) > 20:

        # If there are too many unique values, bin them ...

        # https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/simple-exploration-notebook-2-connect

        # This part is a mess. I need to think about a better wau of coding this

        # -------------------------------

        temp1   = temp.ix[ temp[col] <= np.percentile(train[col].values, 99) ,[col, 'interest_level', 'building_id']]

        _, bins = np.histogram(temp1.price)

        means  = 0.5*(bins[1:] + bins[:-1])

        splits = list(zip(bins[:-1],bins[1:]))

        function1 = lambda m: list(filter( lambda n: n is not None , [ ( np.mean(s) if ((m >= s[0]) and (m < s[1])) else None) for s in splits ]))

        function2 = lambda m: m[0] if (len(m) == 1) else means[-1] + 100

        

        temp1[col] = temp1[col].apply( function1 ).apply( function2 )

        

    else:

        temp1 = temp[[col, 'interest_level', 'building_id']]    

    

    temp1 = temp1.groupby([col, 'interest_level']).agg(len).reset_index()

    temp1['normalizer'] = temp1.interest_level.map(counts)

    temp1[col + '_interest'] = temp1['building_id']/temp1['normalizer']

    plt.figure()

    sns.barplot(data = temp1, x = col, y=col+'_interest', hue='interest_level')

    plt.legend(loc='upper right')

    plt.ylabel('normalized interest level for ' + col)
plt.subplot(121); sns.distplot( train.latitude )

plt.subplot(122); sns.distplot( train.longitude )
def limits(df, col):

    llimit = np.percentile(df[col].values, 1)

    ulimit = np.percentile(df[col].values, 99)

    

    return llimit, ulimit



lonL, lonU = limits(train, 'longitude' )

latL, latU = limits(train, 'latitude' )



mask  = train.latitude  > latL

mask &= train.latitude  < latU

mask &= train.longitude > lonL

mask &= train.longitude < lonU



temp = train.ix[ mask, ['latitude', 'longitude', 'interest_level'] ]



west, south, east, north = ( temp.longitude.min(), 

                             temp.latitude.min(), 

                             temp.longitude.max(), 

                             temp.latitude.max())
interest = 'low'

X, Y = np.meshgrid( np.linspace(west, east, 50), np.linspace(south, north, 50) )



interests = ['low', 'medium', 'high']



for interest in interests:

    longs = np.radians(temp[temp.interest_level == interest]['longitude'].values)

    lats  = np.radians(temp[temp.interest_level == interest]['latitude'].values)



    # We should vectorize the calculations

    def valueFunc(pt):

        '''

        Given a point, return the kde value

        '''

        lon, lat = map(radians, pt)

    

        dLongs = longs - lon

        dLats  = lats  - lat

    

        a = np.sin( dLats / 2 )**2 + np.cos(lat)*np.cos(lats)*np.sin(dLongs/2)**2

        c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )

    

        return c

    

    vals = np.array(list(map( valueFunc,  list(zip(X.reshape(1, -1)[0], Y.reshape(1, -1)[0])))))

    vals = -1 *  (vals ** 2)/( 2 * 5e-5**2 )  

    vals = np.exp( vals  )

    vals = vals.mean(axis=1).reshape( X.shape )

    plt.figure()

    plt.contourf(X, Y, vals, aspect='equal', cmap='viridis')

    plt.title(interest)