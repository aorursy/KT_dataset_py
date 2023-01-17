# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        





# Any results you write to the current directory are saved as output.
# Import libraries



import pandas as pd

from scipy.stats import mode

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt



# Load the train and test data

train_data = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

print(train_data.describe())



test_data = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')



# Select X (will be the model's predictors)

cols_to_use = ['time', 'signal']

X_test= test_data[cols_to_use]

# Understand the data

# Select subsets of the train data based on different open channels 

data_train_subset_0= train_data.loc [train_data.open_channels == 0]

data_train_subset_1= train_data.loc [train_data.open_channels == 1]

data_train_subset_2= train_data.loc [train_data.open_channels == 2]

data_train_subset_3= train_data.loc [train_data.open_channels == 3]

data_train_subset_4= train_data.loc [train_data.open_channels == 4]

data_train_subset_5= train_data.loc [train_data.open_channels == 5]

data_train_subset_6= train_data.loc [train_data.open_channels == 6]

data_train_subset_7= train_data.loc [train_data.open_channels == 7]

data_train_subset_8= train_data.loc [train_data.open_channels == 8]

data_train_subset_9= train_data.loc [train_data.open_channels == 9]

data_train_subset_10= train_data.loc [train_data.open_channels == 10]



# Plot the different subsets based on different open channels

sns.kdeplot(data=data_train_subset_0['signal'], label='0 open channels')

sns.kdeplot(data=data_train_subset_1['signal'], label='1 open channels')

sns.kdeplot(data=data_train_subset_2['signal'], label='2 open channels')

sns.kdeplot(data=data_train_subset_3['signal'], label='3 open channels')

sns.kdeplot(data=data_train_subset_4['signal'], label='4 open channels')

sns.kdeplot(data=data_train_subset_5['signal'], label='5 open channels')

sns.kdeplot(data=data_train_subset_6['signal'], label='6 open channels')

sns.kdeplot(data=data_train_subset_7['signal'], label='7 open channels')

sns.kdeplot(data=data_train_subset_8['signal'], label='8 open channels')

sns.kdeplot(data=data_train_subset_9['signal'], label='9 open channels')

sns.kdeplot(data=data_train_subset_10['signal'], label='10 open channels')



# Add labels

plt.ylabel('Frequency')

plt.xlabel('Signal')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
# Focus on problematic area of the train data

sns.kdeplot(data=data_train_subset_4['signal'], label='4 open channels')

sns.kdeplot(data=data_train_subset_5['signal'], label='5 open channels')

sns.kdeplot(data=data_train_subset_6['signal'], label='6 open channels')

sns.kdeplot(data=data_train_subset_7['signal'], label='7 open channels')
# Extract information about where each open_channels data peaks 

chan_0=(data_train_subset_0.mode()).iloc[0]

print(chan_0)

chan_1=(data_train_subset_1.mode()).iloc[0]

print(chan_1)

chan_2=(data_train_subset_2.mode()).iloc[0]

print(chan_2)

chan_3=(data_train_subset_3.mode()).iloc[0]

print(chan_3)

chan_4=(data_train_subset_4.mode()).iloc[0]

print(chan_4)

chan_5=(data_train_subset_5.mode()).iloc[0]

print(chan_5)

chan_6=(data_train_subset_6.mode()).iloc[0]

print(chan_6)

chan_7=(data_train_subset_7.mode()).iloc[0]

print(chan_7)

chan_8=(data_train_subset_8.mode()).iloc[0]

print(chan_8)

chan_9=(data_train_subset_9.mode()).iloc[0]

print(chan_9)

chan_10=(data_train_subset_10.mode()).iloc[0]

print(chan_10)

# Extract information for each of the peaking signal

# [note to self: last 2 steps should be automated/better coded]



data_train_signal_0= train_data.loc [train_data.signal == -2.5002]

#print(data_train_signal_0)

data_train_signal_1= train_data.loc [train_data.signal == -1.2502]

#print(data_train_signal_1)

data_train_signal_2= train_data.loc [train_data.signal == -0.0002]

#print(data_train_signal_2)

data_train_signal_3= train_data.loc [train_data.signal == 1.2498]

#print(data_train_signal_3)

data_train_signal_4= train_data.loc [train_data.signal == 2.4998]

#print(data_train_signal_4)

data_train_signal_5= train_data.loc [train_data.signal == 3.7498]

#print(data_train_signal_5)

data_train_signal_6= train_data.loc [train_data.signal == 1.9815]

#print(data_train_signal_6)

data_train_signal_7= train_data.loc [train_data.signal == 3.1671]

#print(data_train_signal_7)

data_train_signal_8= train_data.loc [train_data.signal == 4.333]

#print(data_train_signal_8)

data_train_signal_9= train_data.loc [train_data.signal == 10.3080]

#print(data_train_signal_9)

data_train_signal_10= train_data.loc [train_data.signal == 11.2070]

#print(data_train_signal_10)

# Plot the data from above

sns.kdeplot(data=data_train_signal_0['open_channels'],label='signal 0 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_1['open_channels'],label='signal 1 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_2['open_channels'],label='signal 2 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_3['open_channels'],label='signal 3 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_4['open_channels'],label='signal 4 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_5['open_channels'],label='signal 5 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_6['open_channels'],label='signal 6 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_7['open_channels'],label='signal 7 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_8['open_channels'],label='signal 8 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_9['open_channels'],label='signal 9 open channels',shade=False, bw=0.1)

sns.kdeplot(data=data_train_signal_10['open_channels'],label='signal 10 open channels',shade=False, bw=0.1)





# Add labels and legend



plt.ylabel('Frequency')

plt.xlabel('Open Channels')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
# Assume that everything but the main peak is noise 

data_train_denoised= train_data.loc [(train_data.signal == -2.5002) & (train_data.open_channels == 0) |

                                     (train_data.signal == -1.2502) & (train_data.open_channels == 1) |

                                     (train_data.signal == -0.0002) & (train_data.open_channels == 2) |

                                     (train_data.signal == 1.2498) & (train_data.open_channels == 3)  |

                                     (train_data.signal == 2.4998) & (train_data.open_channels == 4)  |

                                     (train_data.signal == 3.7498) & (train_data.open_channels == 5)  |

                                     (train_data.signal == 1.9815) & (train_data.open_channels == 6)  |

                                     (train_data.signal == 3.1671) & (train_data.open_channels == 7)  |

                                     (train_data.signal == 4.333) & (train_data.open_channels == 8)   |

                                     (train_data.signal == 10.3080) & (train_data.open_channels == 9) |

                                     (train_data.signal == 11.2070) & (train_data.open_channels == 10)]  

print(data_train_denoised)



# Plot denoised data

sns.scatterplot(y=data_train_denoised.signal, x=data_train_denoised.open_channels)
# Try to fit a model using the denoised data only



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor





# Define y (will be the model's target)

y = data_train_denoised.open_channels



# Select X (will be the model's predictors)

X= data_train_denoised[cols_to_use]





# Separate data into training and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)





model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(X_train, y_train)

preds=model.predict(X_valid)



sns.scatterplot(x=y_valid, y=preds)
# Predict the test data

preds_test=model.predict(X_test)

# Save Output



sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

sub['open_channels'] =  np.array(np.round(preds_test,0), np.int) 



sub.to_csv('submission_denoised_GP.csv', index=False, float_format='%.4f')

sub.head(10)

# Visualize Output

test_RFR = pd.read_csv('submission_denoised_GP.csv')

sns.kdeplot(data=test_RFR.open_channels, shade=False)
