import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/montcoalert/911.csv')
df.info()
df.head()
df['zip'].value_counts().head()
