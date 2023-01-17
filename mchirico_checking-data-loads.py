import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


dateparse = lambda x: datetime.datetime.strptime(x,'%m/%d/%Y %I:%M:00 %p')

# Read data 
d=pd.read_csv("../input/crime.csv",parse_dates=['incident_datetime'],date_parser=dateparse)

d.head()