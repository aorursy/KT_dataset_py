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
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['figure.figsize'] = (14, 11)
mpl.rcParams['axes.grid'] = False
csv_path = '../input/results/resultat_simple_lstm_model (1).csv'
df = pd.read_csv(csv_path)
df.head()
mpl.rcParams['figure.figsize'] = (22, 20)
mpl.rcParams['axes.grid'] = False
df.plot() #subplots=True
csv_path2 = '../input/results/resultat_single_step_model_.csv'
df2 = pd.read_csv(csv_path2)
df2.head()
mpl.rcParams['figure.figsize'] = (20, 18)
#mpl.rcParams['axes.grid'] = False
df2.plot()
#Kaggle :
#read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
csv_path3 = '../input/kaggleresults/temperature_mlp.csv'
df3 = pd.read_csv(csv_path3,header=0, index_col=0, parse_dates=True, squeeze=True)
df3.head()
mpl.rcParams['figure.figsize'] = (20, 18)
#mpl.rcParams['axes.grid'] = False
df3.plot()
csv_path4 = '../input/kaggleresults/temperature_cnn.csv'
df4 = pd.read_csv(csv_path4,header=0, index_col=0, parse_dates=True, squeeze=True)
df4.head()
mpl.rcParams['figure.figsize'] = (20, 18)
#mpl.rcParams['axes.grid'] = False
df4.plot()
csv_path5 = '../input/kaggleresults/temperature_lstm.csv'
df5 = pd.read_csv(csv_path5,header=0, index_col=0, parse_dates=True, squeeze=True)
df5.head()
mpl.rcParams['figure.figsize'] = (20, 18)
#mpl.rcParams['axes.grid'] = False
df5.plot()
csv_path6 = '../input/kaggleresults/temperature_cnn_lstm.csv'
df6 = pd.read_csv(csv_path6,header=0, index_col=0, parse_dates=True, squeeze=True)
df6.head()
mpl.rcParams['figure.figsize'] = (20, 18)
#mpl.rcParams['axes.grid'] = False
df6.plot()