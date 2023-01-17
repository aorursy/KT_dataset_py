import numpy as np
import pandas as pd
train = pd.read_csv('/kaggle/input/clean-data/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32}).drop(["open_channels"],axis=1)
test = pd.read_csv('/kaggle/input/clean-data/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
import plotly.graph_objects as go
from matplotlib import pyplot as plt
for i in range(0,2):
    x = train["time"][i*500000:(i+1)*500000]
    y = train["signal"][i*500000:(i+1)*500000]

    fig = go.Figure(data=[go.Scatter(x=x, y=y, name='Signal'),])
    fig.update_layout(title='train_No.{}'.format(i))
    fig.show()
import plotly.graph_objects as go
from matplotlib import pyplot as plt
for i in range(0,4):
    x = test["time"][i*500000:(i+1)*500000]
    y = test["signal"][i*500000:(i+1)*500000]

    fig = go.Figure(data=[go.Scatter(x=x, y=y, name='Signal'),])
    fig.update_layout(title='test_No.{}'.format(i))
    fig.show()
