# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import parallel_coordinates



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn.model_selection import KFold, train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('whitegrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')



        

pd.options.display.max_rows = 999

pd.options.display.max_columns = 999

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')



df = pd.concat([train, test])
display(df.head())
df.info()
target_dist = df['Cover_Type'].value_counts()/len(train)
plt.figure(figsize=(10, 6))

plt.title('Target Variable Distribution')

sns.barplot(x=target_dist.index, y=target_dist.values, alpha=.85)

plt.show()
temp = df[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Cover_Type']]

temp = temp.groupby('Cover_Type').mean().reset_index()
trace1 = go.Parcoords(

        line=dict(color=temp['Cover_Type'],

                 colorscale = 'Electric'),

        dimensions = list([

            dict(range = [temp['Hillshade_3pm'].min(), temp['Hillshade_9am'].max()],

                 label = 'Hillshade_9am', values = temp['Hillshade_9am'].values),

            dict(range = [temp['Hillshade_3pm'].min(), temp['Hillshade_9am'].max()],

                 label = 'Hillshade_Noon', values = temp['Hillshade_Noon'].values),

            dict(range = [temp['Hillshade_3pm'].min(), temp['Hillshade_9am'].max()],

                 label = 'Hillshade_3pm', values = temp['Hillshade_3pm'].values)

        ])

)



data = [trace1]



fig = go.Figure(data=data)



iplot(fig)
f, ax = plt.subplots(2, 2, figsize=(25, 12))



ax[0, 0].set_title('Elevation Distribution by Cover Type')

sns.violinplot(df['Cover_Type'], df['Elevation'], ax=ax[0,0])

ax[0, 1].set_title('Horizontal Distance to Fire Distribution by Cover Type')

sns.violinplot(df['Cover_Type'], df['Horizontal_Distance_To_Fire_Points'], ax=ax[0,1])

ax[1, 0].set_title('Horizontal Distance to Hydrology Distribution by Cover Type')

sns.violinplot(df['Cover_Type'], df['Horizontal_Distance_To_Hydrology'], ax=ax[1,0])

ax[1, 1].set_title('Horizontal Distance to Roadways Distribution by Cover Type')

sns.violinplot(df['Cover_Type'], df['Horizontal_Distance_To_Roadways'], ax=ax[1,1])

plt.show()

plt.figure(figsize=(25, 6))

plt.title('Vertical Distance To Hydrology Distribution by Cover Type')

sns.violinplot(x='Cover_Type', y='Vertical_Distance_To_Hydrology', data=df)

plt.show()
aspect = df.groupby('Cover_Type')['Aspect'].mean()

slope = df.groupby('Cover_Type')['Slope'].mean()
f, ax = plt.subplots(1, 2, figsize=(25, 6))



ax[0].set_title('Average Aspect by Cover Type')

sns.barplot(x=aspect.index, y=aspect.values, alpha=.85, ax=ax[0])

ax[1].set_title('Average Slope by Cover Type')

sns.barplot(x=slope.index, y=slope.values, alpha=.85, ax=ax[1])



plt.show()
df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']

df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])

df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])

df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])

df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])

df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])



df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5



df.slope_hyd=df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any



df['Mean_Amenities'] = (df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 

df['Mean_Fire_Hyd'] = (df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2 
soils = ['Soil_Type1', 'Soil_Type10', 'Soil_Type11',

       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',

       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',

       'Soil_Type2', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',

       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',

       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type3',

       'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',

       'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',

       'Soil_Type38', 'Soil_Type39', 'Soil_Type4', 'Soil_Type40', 'Soil_Type5',

       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9']
bins_columns = ['Aspect',

                'Elevation',

                'Hillshade_3pm',

                'Hillshade_9am',

                'Hillshade_Noon',

                'Horizontal_Distance_To_Hydrology',

                'Horizontal_Distance_To_Fire_Points',

                'Horizontal_Distance_To_Roadways',

                'Vertical_Distance_To_Hydrology',

                'Slope', 'HF1', 'HF2', 'HR1', 'HR2', 

                'FR1', 'FR2', 'slope_hyd', 'Mean_Amenities',

                'Mean_Fire_Hyd'

               ]
# for col in bins_columns:

#     df[col] = pd.qcut(df[col], 4)

#     feat_dummies = pd.get_dummies(df[col], prefix=col)

#     df = pd.concat([df, feat_dummies], axis=1)

    

# df.drop(bins_columns, 1, inplace=True)
print(f'The final dataset has {df.shape[1]} features') 
train = df[df['Cover_Type'].notnull()]

test = df[df['Cover_Type'].isnull()].drop('Cover_Type', 1)



train['Cover_Type'] = train['Cover_Type'].astype('int')
from sklearn.model_selection import KFold, train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
X = train.drop(['Cover_Type', 'Id'], 1)

y = train['Cover_Type']
lr = LogisticRegression()

rf = RandomForestClassifier(random_state=1)

dt = DecisionTreeClassifier(random_state=1)



kfold = KFold(11, random_state=1)
def train_model(model, X, y, cv, scoring_metric='accuracy'):

    accuracy = cross_val_score(model, X, y, cv=kfold, scoring=scoring_metric)

    return accuracy.mean()
%%time

print('Random Forest Results:', train_model(rf, X, y, kfold))
rf.fit(X, y)

predictions = rf.predict(test.drop('Id', 1))
results = pd.DataFrame(columns=['Id', 'Cover_Type'])

results['Id'] = test['Id']

results['Cover_Type'] = predictions
results.to_csv('rf_testing2.csv', index=False)