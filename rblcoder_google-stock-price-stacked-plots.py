from IPython.core.display import display, HTML

display(HTML("""<!DOCTYPE HTML>

<html>

<head>

  <script type="text/javascript">

  window.onload = function () {





    var chart = new CanvasJS.Chart("chartContainer",

    {

      zoomEnabled: true,

      title:{

        text: "Try Zooming And Panning"

      },

      legend: {

        horizontalAlign: "right",

        verticalAlign: "center"

      },

      axisY:{

        includeZero: false

      },

      data: data,  // random generator below



   });



    chart.render();

  }



   var limit = 1000;    //increase number of dataPoints by increasing this



    var y = 0;

    var data = []; var dataSeries = { type: "line" };

    var dataPoints = [];

    for (var i = 0; i < limit; i += 1) {

        y += (Math.random() * 10 - 5);

         dataPoints.push({

          x: i - limit / 2,

          y: y

           });

        }

     dataSeries.dataPoints = dataPoints;

     data.push(dataSeries);



  </script>

  <script type="text/javascript" src="https://canvasjs.com/assets/script/canvasjs.min.js"></script></head>

  <body>

    <div id="chartContainer" style="height: 300px; width: 100%;">

    </div>

  </body>

  </html>"""))
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from IPython.display import display

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/google-stock-price/Google_Stock_Price_Train.csv')

print(df.info())

df['Date'] = pd.to_datetime(df['Date'])

#display(df.head())



df.columns

df['Close'] = df['Close'].str.replace(',','')

df['Close'] = pd.to_numeric(df['Close'] )

df['Volume'] = df['Volume'].str.replace(',','')

df['Volume'] = pd.to_numeric(df['Volume'] )

df.info()

display(df.head())
from scipy.stats import normaltest

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:

    print(col, normaltest(df[col].pct_change().dropna()))

# Distributions are not normal https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# https://github.com/stefan-jansen/machine-learning-for-trading/blob/master/02_market_and_fundamental_data/01_NASDAQ_TotalView-ITCH_Order_Book/03_normalize_tick_data.ipynb

# https://www.packtpub.com/in/data/machine-learning-for-algorithmic-trading-second-edition



import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(15, 8))

axes[0].plot(df['Date'], df['Open'])

axes[1].plot(df['Date'], df['Volume'])

axes[0].set_title('Open Price', fontsize=14)

axes[1].set_title('Volume', fontsize=14)

fig.autofmt_xdate()

fig.suptitle('Open Price data')

fig.tight_layout()

plt.subplots_adjust(top=0.9)
import seaborn as sns

sns.set(style="ticks")

_=sns.pairplot(df)
# https://plot.ly/python/subplots/

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



from plotly.subplots import make_subplots





fig = make_subplots(rows=5, cols=1, subplot_titles=(df.columns[1:]))



fig.append_trace(go.Scattergl(

    x=df['Date'],

    y=df['Open'],

), row=1, col=1)



fig.append_trace(go.Scattergl(

    x=df['Date'],

    y=df['High'],

), row=2, col=1)



fig.append_trace(go.Scattergl(

    x=df['Date'],

    y=df['Low'],

), row=3, col=1)



fig.append_trace(go.Scattergl(

    x=df['Date'],

    y=df['Close'],

), row=4, col=1)



fig.append_trace(go.Scattergl(

    x=df['Date'],

    y=df['Volume'],

), row=5, col=1)



fig.update_layout(height=800, width=800, title_text="Google stock", showlegend=False)



fig.show()
# Plots with range slider

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



fig = go.Figure()

fig.add_trace(go.Scattergl(x=df.Date, y=df[df.columns[1]], name=df.columns[1],

                        ))



fig.add_trace(go.Scattergl(x=df.Date, y=df[df.columns[2]], name=df.columns[2],

                        ))





fig.add_trace(go.Scattergl(x=df.Date, y=df[df.columns[3]], name=df.columns[3],

                        ))



fig.add_trace(go.Scattergl(x=df.Date, y=df[df.columns[4]], name=df.columns[4],

                        ))



fig.add_trace(go.Scattergl(x=df.Date, y=df[df.columns[5]], name=df.columns[5],

                        ))



fig.update_layout(title_text='Google',

                  xaxis_rangeslider_visible=True)

fig.show()



fig = make_subplots(rows=1, cols=5, subplot_titles=("Open", "High", "Low", "Close", "Volume"))



fig.append_trace(go.Box(

    y=df['Open'],

), row=1, col=1)



fig.append_trace(go.Box(

    y=df['High'],

), row=1, col=2)



fig.append_trace(go.Box(

    y=df['Low'],

), row=1, col=3)



fig.append_trace(go.Box(

    y=df['Close'],

), row=1, col=4)



fig.append_trace(go.Box(

    y=df['Volume'],

), row=1, col=5)



fig.update_layout(height=400, width=800, title_text="Google stock", showlegend=False)

fig.show()
fig = make_subplots(rows=5, cols=1, subplot_titles=("Open", "High", "Low", "Close", "Volume"))



fig.append_trace(go.Histogram(

   

    x=df['Open'],

), row=1, col=1)



fig.append_trace(go.Histogram(

  

    x=df['High'],

), row=2, col=1)



fig.append_trace(go.Histogram(

   

    x=df['Low'],

), row=3, col=1)



fig.append_trace(go.Histogram(

  

    x=df['Close'],

), row=4, col=1)



fig.append_trace(go.Histogram(

  

    x=df['Volume'],

), row=5, col=1)



fig.update_layout(height=1000, width=400, title_text="Google stock", showlegend=False)

fig.show()
# fig = make_subplots(rows=5, cols=1, subplot_titles=(df.columns[1:]))



# fig.append_trace(go.Histogram(

   

#     x=df[df.columns[1]],

# ), row=1, col=1)



# fig.append_trace(go.Histogram(

  

#     x=df[df.columns[2]],

# ), row=2, col=1)



# fig.append_trace(go.Histogram(

   

#     x=df[df.columns[3]],

# ), row=3, col=1)



# fig.append_trace(go.Histogram(

  

#     x=df[df.columns[4]],

# ), row=4, col=1)



# fig.append_trace(go.Histogram(

  

#     x=df[df.columns[5]],

# ), row=5, col=1)



# fig.update_layout(height=1000, width=400, title_text="Google stock", showlegend=False)

# fig.show()
# xmin = df['Date'].min()

# xmax = df['Date'].max()

# ymin = df['Open'].min()

# ymax = df['Open'].max()

# import plotly.express as px



# fig = px.scatter(df, 

#                  x="Date", y="Open", 

#                  animation_frame="Date",range_x=[xmin, xmax], range_y=[ymin, ymax],

#                  # size="Not issued rate for ATVs and uniform visas ", 

#                  # color="Schengen State",

#                  # hover_name="Country where consulate is located", 

#                  # log_x=True, log_y=True, size_max=60,

#                 width=1000, height=1000)

# fig.show()
import holoviews as hv

hv.__version__
#hv.extension('bokeh', 'matplotlib')
#!conda install -y -c conda-forge hvplot 
# import hvplot.pandas

# df.iloc[:,:-1].set_index('Date').hvplot()