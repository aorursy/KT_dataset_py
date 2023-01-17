# Imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math
#import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

#plotly.__version__

# List of Datafiles
print(os.listdir("../input"))
pitstops = pd.read_csv('../input/pitStops.csv')
results = pd.read_csv('../input/results.csv')
races = pd.read_csv('../input/races.csv')
circuits = pd.read_csv('../input/circuits.csv', encoding='latin1')
drivers = pd.read_csv('../input/drivers.csv', encoding='latin1')

#identify yellow flag
laptimes = pd.read_csv('../input/lapTimes.csv')
laptimes.head()
