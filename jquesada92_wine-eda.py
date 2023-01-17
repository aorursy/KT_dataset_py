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
white = pd.read_csv("/kaggle/input/wine-quality/wineQualityWhites.csv")

white['type'] = "White"

red = pd.read_csv("/kaggle/input/wine-quality/wineQualityReds.csv")

red['type'] = "Red"
import plotly.express as px

import plotly.graph_objects as go
df =  pd.concat([red,white],ignore_index=True)



print(df.info())
avg_quality = np.average(red.quality)

std_quality = np.std(red.quality)



UCL = avg_quality + 2.5* std_quality

LCL = avg_quality - 2.5* std_quality
fig_quality = go.FigureWidget()

points =  fig_quality.add_scattergl(x=red.index,y=red.quality,mode="markers+lines",line=dict(width=1))

mean_ = fig_quality.add_scattergl(x=red.index, y=[avg_quality]*len(red.index),mode="lines",marker=dict(color='green'),name="AVG.")

ucl = fig_quality.add_scattergl(x=red.index, y=[UCL]*len(red.index),mode="lines",marker=dict(color='red'),name="UCL")

lcl = fig_quality.add_scattergl(x=red.index, y=[LCL]*len(red.index),mode="lines",marker=dict(color='red'),name="LCL")
fig_quality