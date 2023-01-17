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
from glob import glob

import pandas as pdlib



def produceCombinedCSV(list_of_files):

   

   # Consolidate all CSV files into one object

   result_df = pdlib.concat([pdlib.read_csv(file).add_prefix(str(list_of_files.index(file)) + '_') for file in list_of_files], axis=1).T.drop_duplicates().T

   return result_df



# Move to the path that holds our CSV files

csv_file_path = '/kaggle/input/'



# List all CSV files in the working dir

file_pattern = "csv"

#list_of_files = [file for file in glob('*.{}'.format(file_pattern))]

list_of_files = glob(csv_file_path + "*.csv")

print(list_of_files)



df_consolidated_columnwise = produceCombinedCSV(list_of_files)
df_consolidated_columnwise.rename(columns={'0_Unnamed: 0': 'DateTime'}, inplace=True)

df_consolidated_columnwise.set_index('DateTime', inplace=True)

df_consolidated_columnwise.head()
from glob import glob

import pandas as pdlib

from os import chdir

def produceCombinedCSV(list_of_files):

   

   # Consolidate all CSV files into one object

   result_df = pdlib.concat([pdlib.read_csv(file) for file in list_of_files], keys=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])

   # Convert the above object into a csv file and export

   result_df.to_csv('/kaggle/working/ConsolidateOutput_rowwise.csv')



# Move to the path that holds our CSV files

csv_file_path = '/kaggle/input/'

chdir(csv_file_path)



# List all CSV files in the working dir

file_pattern = "csv"

list_of_files = [file for file in glob('*.{}'.format(file_pattern))]

print(list_of_files)



produceCombinedCSV(list_of_files)
import pandas as pd

df_consolidated_rowwise = pd.read_csv("/kaggle/working/ConsolidateOutput_rowwise.csv", index_col=[0]).drop('Unnamed: 1', axis=1)

df_consolidated_rowwise.rename(columns={'Unnamed: 0.1': 'DateTime'}, inplace=True)

df_consolidated_rowwise.head()
from matplotlib import pyplot

# load dataset

values = df_consolidated_columnwise.values

# specify columns to plot

groups = [0, 1, 2, 3]

i = 1

# plot each column

pyplot.figure(figsize=(24,12))

for group in groups:

    pyplot.subplot(len(groups), 1, i)

    pyplot.plot(values[:, group])

    pyplot.title(df_consolidated_columnwise.columns[group], y=0.5, loc='right')

    i += 1

pyplot.show()
sample = df_consolidated_rowwise.loc[0].copy()
sample_df = sample[(sample['0'] < 100) & (sample['0'] > -100) & (sample['1'] < 100) & (sample['1'] > -100) & (sample['2'] < 100) & (sample['2'] > -100) & (sample['3'] < 100) & (sample['3'] > -100)].copy()

sample_df.head()
from matplotlib import pyplot

# load dataset

values = sample_df.values

# specify columns to plot

groups = [1,2,3,4]

i = 1

# plot each column

pyplot.figure(figsize=(24,12))

for group in groups:

    pyplot.subplot(len(groups), 1, i)

    pyplot.plot(values[:, group])

    pyplot.xticks(np.arange(0, 3000, 100)) 

    pyplot.title(sample_df.columns[group], y=0.5, loc='right')

    i += 1

pyplot.show()
sample_df['0_sqr'] = np.square(sample_df['0'])

sample_df['1_sqr'] = np.square(sample_df['1'])

sample_df['2_sqr'] = np.square(sample_df['2'])

sample_df['3_sqr'] = np.square(sample_df['3'])



sample_df['DateTime'] = pd.to_datetime(sample_df['DateTime'])

sample_df.set_index('DateTime', inplace=True)

sample_df.head()
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(sample_df['0_sqr'], lags=50, alpha=1)
import statsmodels.api as sm

import statsmodels.tsa.api as smt

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot

import matplotlib.pyplot as plt



train, test = sample_df['0_sqr'].iloc[0:70], sample_df['0_sqr'].iloc[70:1000]

#train_log, test_log = np.log10(train), np.log10(test)

my_order = (0,0,0)

my_seasonal_order = (1, 1, 1, 12)
history = [x for x in train]

predictions = list()

predict_log=list()

for t in range(len(test)):

    model = sm.tsa.SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order,enforce_stationarity=False,enforce_invertibility=False)

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    predict_log.append(output[0])

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

print('predicted=%f, expected=%f' % (output[0], obs))

#error = math.sqrt(mean_squared_error(test_log, predict_log))

#print('Test rmse: %.3f' % error)

# plot

figsize=(24, 12)

plt.figure(figsize=figsize)

pyplot.plot(sample_df['0_sqr'].iloc[70:1000].index, test,label='Actuals')

pyplot.plot(sample_df['0_sqr'].iloc[70:1000].index, predictions, color='red',label='Predicted')

pyplot.legend(loc='upper right')

pyplot.show()
def finding_first_fault(df):

    #Noise Removal

    sample_df = df[(df['0'] < 100) & (df['0'] > -100) & (df['1'] < 100) & (df['1'] > -100) & (df['2'] < 100) & (df['2'] > -100) & (df['3'] < 100) & (df['3'] > -100)].copy()

    

    #Squaring the Waveforms

    sample_df['0_sqr'] = np.square(sample_df['0'])

    sample_df['1_sqr'] = np.square(sample_df['1'])

    sample_df['2_sqr'] = np.square(sample_df['2'])

    sample_df['3_sqr'] = np.square(sample_df['3'])



    #Setting Index

    sample_df['DateTime'] = pd.to_datetime(sample_df['DateTime'])

    sample_df.set_index('DateTime', inplace=True)



    #Windowed MAX

    sample_df['0_max'] = sample_df['0_sqr'].rolling(72).max()

    sample_df['1_max'] = sample_df['1_sqr'].rolling(72).max()

    sample_df['2_max'] = sample_df['2_sqr'].rolling(72).max()

    sample_df['3_max'] = sample_df['3_sqr'].rolling(72).max()

    

    #Removal of Blanks (Initial values of the window)

    sample_df.dropna(inplace=True)

    

    #First order difference of Rolling MAX

    sample_df['0_change'] = sample_df['0_max'].diff()

    sample_df['1_change'] = sample_df['1_max'].diff()

    sample_df['2_change'] = sample_df['2_max'].diff()

    sample_df['3_change'] = sample_df['3_max'].diff()

    

    fault_date_0 = sample_df[(sample_df['0_change']>=sample_df['0_change'].nlargest(15).mean()) | (sample_df['0_change']<=sample_df['0_change'].nsmallest(15).mean())].index[0]

    fault_date_1 = sample_df[(sample_df['1_change']>=sample_df['1_change'].nlargest(15).mean()) | (sample_df['1_change']<=sample_df['1_change'].nsmallest(15).mean())].index[0]

    fault_date_2 = sample_df[(sample_df['2_change']>=sample_df['2_change'].nlargest(15).mean()) | (sample_df['2_change']<=sample_df['2_change'].nsmallest(15).mean())].index[0]

    fault_date_3 = sample_df[(sample_df['3_change']>=sample_df['3_change'].nlargest(15).mean()) | (sample_df['3_change']<=sample_df['3_change'].nsmallest(15).mean())].index[0]

    

    date_list = [fault_date_0, fault_date_1, fault_date_2, fault_date_3]

    

    fault_induction_date = min(date_list)

    

    return date_list
machine_nos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]





df_machine_and_fault = pd.DataFrame(columns=['Machine No.', 'Fault_0', 'Fault_1', 'Fault_2', 'Fault_3', 'Fault_Inception_Date'])

df_machine_and_fault['Machine No.'] = machine_nos

for machine_no in machine_nos:

    df_machine_and_fault.iloc[machine_no,1:5] = finding_first_fault(df_consolidated_rowwise.loc[machine_no])

    

#We will now select the earliest date of the dates identified for all 4 signals, since we want to be able to identify the fault at the earliest.



#Through deductive analysis, we will be excluding the Signal 2 from this because signal 2 has been found to contain 

#values that breach the set threshold, even before the fault has set in, i.e. in the normal mode of operation itself. 

#These can be considered as outliers, and since the other signals are contributing enough to the fault identification 

#this signal can be excluded.



df_machine_and_fault['Fault_Inception_Date'] = df_machine_and_fault[['Fault_0', 'Fault_1', 'Fault_3']].min(axis=1)

df_machine_and_fault