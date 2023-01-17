import os

import numpy as np

import pandas as pd

from scipy import stats, optimize

from scipy.integrate import odeint

import matplotlib.pyplot as plt

import seaborn as sns

from bokeh.io import push_notebook, show, output_notebook

from bokeh.layouts import row, gridplot

from bokeh.resources import INLINE

from bokeh.plotting import figure, output_file, save

from bokeh.models import ColumnDataSource, HoverTool

output_notebook(INLINE)
#Show all files in dir with .csv extension

work_dir = ('../input/covid19-global-forecasting-week-2/')

filenames = []

for file in os.listdir(work_dir):

    if file.endswith(".csv"):

        filenames.append(os.path.join(work_dir, file))

filenames = sorted(filenames)

for filename in filenames:

    print(filename)
Data_train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv",parse_dates=[3],header="infer")

Data_train.columns
Data_train.Country_Region.unique()
Data_train_China = Data_train[Data_train.Country_Region.str.contains('China', case=False)]

Data_train_China.Province_State.unique()
Data_train_China_Hubei = Data_train_China[Data_train_China.Province_State.str.contains('Hubei', case=False)]

Data_train_China_Hubei
Data_train_China_Hubei_grouped = Data_train_China_Hubei.groupby('Date').sum()

Data_train_China_Hubei_grouped.reset_index(inplace=True)

y=Data_train_China_Hubei_grouped.ConfirmedCases



ydiff = y.diff()

ydiff[0] = 0

y2 = ydiff.rolling(window = 3).mean()

y2[0] = ydiff[0]

y2[1] = ydiff[1]



y3 = y2.rolling(window = 7).sum()

y3[0:6] = y2[0:6]



#Plotting

_tools_to_show = 'box_zoom,pan,save,hover,reset,wheel_zoom'

p1 = figure(plot_width = 850, plot_height = 400, x_axis_label = 'Date', y_axis_label = "No. of Infections", 

            title="Hubei Infections", tools=_tools_to_show, x_axis_type='datetime')



p1.circle(x=Data_train_China_Hubei_grouped.Date, y=Data_train_China_Hubei_grouped.ConfirmedCases, color="black",size=2)

p1.line(x=Data_train_China_Hubei_grouped.Date, y=Data_train_China_Hubei_grouped.ConfirmedCases, color="red",line_width=1, 

        legend_label="Cumulative agrregate of confirmed cases")





p1.circle(x=Data_train_China_Hubei_grouped.Date, y=y3, color="black",size=2)

p1.line(x=Data_train_China_Hubei_grouped.Date, y=y3, color="blue",line_width=1, 

        legend_label="Approximation of confirmed cases")



hover = p1.select(dict(type=HoverTool))

hover.tooltips = [("Time", "@x"),  ("Value", "@y"),]

show(p1,notebook_handle=True)
#x = Data_train_China_Hubei_grouped.Date

#x1 = np.linspace(0, Data_train_China_Hubei_grouped.Id.size, Data_train_China_Hubei_grouped.Id.size)

#y = Data_train_China_Hubei_grouped.ConfirmedCases



#print(y.diff())

sub = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

sub.to_csv('submission.csv', index=False)