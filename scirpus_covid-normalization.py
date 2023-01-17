import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
states_23 = pd.read_csv('../input/covid-by-state/states_23.csv')

states_30 = pd.read_csv('../input/covid-by-state/states_30.csv')
states_23.head()
plt.figure(figsize=(25,15))

plt.plot(states_23.state,states_23.death.fillna(0),color='g')

plt.plot(states_30.state,states_30.death.fillna(0),color='b')

_ = plt.title('Number of deaths by State')
plt.figure(figsize=(25,15))

plt.plot(states_23.state,(states_23.positive.fillna(0)+states_23.negative.fillna(0)),color='g')

plt.plot(states_30.state,(states_30.positive.fillna(0)+states_30.negative.fillna(0)),color='b')

_ = plt.title('Tests by State')