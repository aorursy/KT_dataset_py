import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
data_tkd=pd.read_csv('../input/taekwondo-techniques-classification/Taekwondo_Technique_Classification_Stats.csv')
data_tkd.head(20)
data_tkd.drop([0, 1])
data_tkd.info()
data_tkd.describe()
from IPython.display import HTML,IFrame
IFrame('https://i.imgur.com/8QM16yb.png',height=500,width=1200)


IFrame('https://i.imgur.com/TqfmgUE.png',height=700,width=1200)
