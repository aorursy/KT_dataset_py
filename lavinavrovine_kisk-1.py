fruits = ['orange', 'apple', 'pear', 'banana', 'apple', 'banana']
# Calculate the length of fruits

# Insert 'kiwi' into fruits and show the result without using print()

# Create a new list with the first 3 elements of fruits

# Create a new list with the last 2 elements of fruits

# %load solutions/1-1.py
czech_ryanair = {
    'BRQ' : ['LTN'],
    'PRG' : ['CRL', 'CIA', 'BGY', 'TPS','DUB','STN','LPL'],
    'OSR' : ['BGY', 'STN'],
}
# Iterate through czech_ryanair printing the keys and values
for key, value in czech_ryanair.items():
    print('Key: {}, Value: {}'.format(key, value))
# Which destinations are accessible from Prague (PRG)?

# Construct a list of all the destinations ryanair flies to from czech republic

# %load solutions/1-2.py
john_classes = {'monday', 'tuesday', 'wednesday'}
eric_classes = set(['wednesday', 'thursday'])
john_classes
eric_classes
union = john_classes | eric_classes
union
# Print out a list of all the unique destinations ryanair flies to from czech republic

# %load solutions/1-3.py
import pandas as pd
pd.options.display.max_rows = 8
%matplotlib inline
import matplotlib.pyplot as plt
# Need to add data via Add Data->COmpetitions-->Titanic
df = pd.read_csv('../input/train.csv')
df
df.Age.hist()

# df.

df['Name']
df.Name

# Try to select multiple columns
df.iloc[0]
df.loc[0, :]

import numpy as np

# Date range creation function
i = pd.date_range(pd.Timestamp('2016-01-01'), pd.Timestamp('today'))

# Create a dataframe with a series of random numbers
date_df = pd.DataFrame(np.random.randn(len(i)), columns=['random'], index=i)

date_df.head()
date_df.loc['2016-01-04', :]

df.loc[5:8,'Fare']
# df.loc[0:5, 'PassengerId', 'Name']
df.loc[0:5, ['PassengerId', 'Name']]






df.Sex == 'male'
type(df.Sex == 'male')
df[df.Sex == 'male']




df.Sex.unique()
df.isnull()
#df[df.isnull().any(axis=1)]
df.Sex.value_counts()
df.count()


# %load solutions/1-6.py
pd.options.display.max_rows = 20
df.groupby('Sex').count()
# %load solutions/1-7.py



import pandas as pd
pd.options.display.max_rows = 8
%matplotlib inline
import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv')
df.info()
df.values
df.index
df.columns
df.groupby(['Sex','Pclass'])['Fare'].mean()
df.groupby(['Sex','Pclass'])['Fare'].mean().unstack()
df.groupby(['Sex','Pclass'])['Fare'].mean().unstack().stack()
avg_fare_groupedby_sex_v_class = df.groupby(['Sex','Pclass'])['Fare'].mean().unstack()
avg_fare_groupedby_sex_v_class
abs(avg_fare_groupedby_sex_v_class.loc[:,3] - avg_fare_groupedby_sex_v_class.loc[:,1])



avg_fare_groupedby_sex_v_class.loc[:,'abs_diff_1_3'] = abs(avg_fare_groupedby_sex_v_class.loc[:,3] - avg_fare_groupedby_sex_v_class.loc[:,1])
avg_fare_groupedby_sex_v_class



df.sort_index(axis=0, ascending=False)
df.sort_values(by='Pclass')

df.info()
df.dropna().info()
len(df.dropna()) / len(df)
df.Cabin
df.Cabin.fillna(value='Dorm')



import numpy as np
df_age_fare = df[['Age','Fare']]
df_age_fare.describe()
df_age_fare = df[['Age','Fare']]
df_norm_1 = df_age_fare.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
df_norm_1.describe()
df_age_fare = df[['Age','Fare']]

def my_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

df_norm_2 = df_age_fare.apply(my_norm)
df_norm_2.describe()
df_norm_3 = (df_age_fare - df_age_fare.min()) / (df_age_fare.max() - df_age_fare.min())
df_norm_3.describe()
df_age_fare.hist();
df_age_norm_3.hist();



# http://stackoverflow.com/questions/16734621/random-walk-pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def geometric_brownian_motion(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):        
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) ### geometric brownian motion ###
    return S

dates = pd.date_range('2012-01-01', '2016-02-22')
T = (dates.max()-dates.min()).days / 365
N = dates.size
start_price = 100
y = pd.Series(geometric_brownian_motion(T, N, sigma=0.1, S0=start_price), index=dates)
y.plot()
# plt.show()
type(y.index)
y
y.loc['2014-10']
y.loc['2014']
y.rolling(window=30).mean().plot()
y.plot()
y.rolling(window=30).mean().plot()
moving_averages = [30, 60, 200]

def calculate_moving_averages(df, moving_average_list):

    # Fill me in
    
    return df

# Uncomment the next line
# calculate_moving_averages(y, moving_averages)
