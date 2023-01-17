# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

import seaborn as sns

sns.set_style('dark')

%matplotlib inline



pf = pd.read_csv(os.path.join(dirname, filename))

pf.head()
(rows,column) = pf.shape
rows, column
# The total number of blank cells in the dataset



np.sum(pf.isnull())
pf.info()
pf.groupby("Border").sum()["Value"].sort_values()
pf.groupby("Border").sum()["Value"].sort_values().plot(kind = 'bar');

plt.title("Which Border is more exposed?")
m_val = pf.Measure.value_counts()



(m_val/rows).plot(kind = 'bar');

plt.title("Which is the most occured Measure to cross the border?");
pf.groupby("Measure").sum()["Value"].sort_values()
pf.groupby("Measure").sum()["Value"].sort_values().plot(kind = 'bar');

plt.title("Which is the most used measure to vross the border?")
pf.groupby(["Measure","Border"]).sum()["Value"].plot(kind = 'bar');

plt.title("Measures used in respective borders");
pf.groupby(["Measure","Border"]).sum()["Value"]
p_val = pf.State.value_counts()



(p_val/rows).plot(kind = 'bar');

plt.title("Which is the most found Measure to cross the border?");
pf.groupby("State").sum()["Value"].sort_values()
pf.groupby("State").sum()["Value"].sort_values().plot(kind = 'bar');

plt.title("Which state is more exposed to people crossing the border ?")
pf.groupby(["State","Border"]).sum()["Value"].plot(kind = 'bar');

plt.title("Measures used in respective borders");
pf['Date'] = pd.to_datetime(pf['Date']) # converting the date column to datetime format for ease of conversion

pf['Date'].head()

pf['year'] = pf['Date'].dt.year

pf['month'] = pf['Date'].dt.month

pf['day'] = pf['Date'].dt.day
pf.head()
sum_crossing = pf.groupby("year").sum()["Value"].reset_index()

sum_crossing
plt.figure(figsize=(15,5))

plt.grid()

sns.set_style('dark')

chart = sns.barplot(x = 'year',y = 'Value',data=sum_crossing);

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

plt.title('Amount per year');
sum_month=pf.groupby('month').sum()['Value'].reset_index()
plt.figure(figsize=(15,10))

plt.grid()

sns.set_style('dark')

sns.barplot(x='month',y='Value',data=sum_month);
sum_day=pf.groupby('day').sum()['Value'].reset_index()
plt.figure(figsize=(15,5))

plt.grid()

sns.set_style('dark')

sns.barplot(x='day',y='Value',data=sum_day)

plt.title('Amount per year')
pf.drop(columns='day',inplace=True)
pf.drop(columns='Date',inplace=True)

pf.head()
pf['year'].max() #finding the maximum year present
pf = pf[pf['year'] < 2020]
pf.describe()
pf.select_dtypes(include=['object'])
def categorical_one_hot_encoder(pf, d, col):

    '''

    Encodes the categorical values of a column and return the dataframe with the values added

    

    Parameters : The categorical_one_hot_encoder function takes following as argument

    pf - The Dataframe which contains categoricalvalues

    d -  a dictionary containing the mapping of each values in the categorical column

    col - the column to encode

    

    Returns:

    DataFrame - The dataframe with the one_hot_vectors

    

    '''

    ### integer mapping using LabelEncoder

    for label in d:

        pf[str(col)+"_"+str(label)] = np.where(pf[col] == label, 1, 0)

    

    return pf
items_border = (['US-Canada Border', 'US-Mexico Border'])

pf = categorical_one_hot_encoder(pf, items_border, col = 'Border')
items_state=(['AK', 'ND', 'ME', 'CA', 'WA', 'MT', 'NY', 'OH', 'ID', 'NM', 'MN', 'VT', 'MI', 'AZ', 'TX'])

pf = categorical_one_hot_encoder(pf, items_state, col = 'State')
items_measure=(['Trains', 'Train Passengers', 'Buses', 'Rail Containers Empty', 'Rail Containers Full','Truck Containers Empty',

             'Bus Passengers', 'Truck Containers Full', 'Trucks', 'Pedestrians', 'Personal Vehicles',

            'Personal Vehicle Passengers'])

pf = categorical_one_hot_encoder(pf, items_measure, col = 'Measure')
pf[['Port Code', 'Value', 'month', 'year']].hist();
plt.figure(figsize=(25,15))

sns.heatmap(pf.corr(), annot=True, fmt=".2f");
pf.info();
pf.columns
training_params = ['Port Code', 'year', 'month', 'Border_US-Canada Border', 'Border_US-Mexico Border', 'State_AK', 'State_ND',

                   'State_ME', 'State_CA', 'State_WA', 'State_MT', 'State_NY', 'State_OH', 'State_ID', 'State_NM', 'State_MN',

                   'State_VT', 'State_MI', 'State_AZ', 'State_TX', 'Measure_Trains', 'Measure_Train Passengers',

                   'Measure_Buses', 'Measure_Rail Containers Empty', 'Measure_Rail Containers Full',

                   'Measure_Truck Containers Empty', 'Measure_Bus Passengers', 'Measure_Truck Containers Full',

                   'Measure_Trucks', 'Measure_Pedestrians', 'Measure_Personal Vehicles', 'Measure_Personal Vehicle Passengers']
X = pf[training_params]

y = pf['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
lm = LinearRegression(normalize = True)

lm.fit(X_train, y_train)
y_pred_test = lm.predict(X_test)

y_pred_train = lm.predict(X_train)

Score_test = r2_score(y_test, y_pred_test)

Score_train = r2_score(y_train, y_pred_train)

print(Score_train, Score_test)