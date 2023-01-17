import os, time, sys, gc, pickle, warnings
from tqdm.notebook import tqdm as tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#wave analysis
import pywt
import librosa
from statsmodels.robust import mad
#import statsmodels.api as sm
import scipy
from scipy import stats
from scipy.stats.kde import gaussian_kde
from scipy import signal
from scipy.signal import hann, hilbert, convolve, butter, deconvolve
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot, column, layout, row
from bokeh.io import output_notebook, curdoc, push_notebook
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, Select, CustomJS, HoverTool
from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d
import colorcet as cc
#import lightgbm as lgb
#import xgboost as xgb
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
from tsfresh.feature_extraction import feature_calculators
output_notebook()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
%%time
INPUTDIR = '/kaggle/input/data-without-drift/'
INPUTDIR2 = '/kaggle/input/liverpool-ion-switching/'
NROWS = None
df_train = pd.read_csv(f'{INPUTDIR}/train_clean.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})
df_test = pd.read_csv(f'{INPUTDIR}/test_clean.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})
sub_df = pd.read_csv(f'{INPUTDIR2}/sample_submission.csv', nrows=NROWS)
%%time
df_train['open_channels_lag_1'] = df_train['open_channels'].shift(1).fillna(method='bfill')
df_train['transition'] =  df_train['open_channels_lag_1'].astype(int).astype(str) + '-'  + df_train['open_channels'].astype(str)
df_train.head()
df_train_agg = df_train.groupby('transition')['signal'].count()
list(df_train_agg.index)
df_train.index = ((df_train.time * 10_000) - 1).values
df_test.index = ((df_test.time * 10_000) - 1).values
df_train['GRP'] = 1+(df_train.index // 50_0000)
df_train['GRP'] = df_train['GRP'].astype('int16')
df_test.index = ((df_test.time * 10_000)).values-1
df_test['GRP'] = 1+(df_test.index // 50_0000)
df_test['GRP'] = df_test['GRP'].astype('int16')
labels = transitions = df_train.groupby('transition')['signal'].count().index
tm1 = sorted(set([int(i.split('-')[0]) for i in transitions]))
labels = [[ str(x)+'-'+str(y) for y in \
          sorted(set([int(i.split('-')[1]) for i in transitions if int(i.split('-')[0]) == x]))]\
            for x in tm1]
for label in labels:
    counts = [pd.DataFrame(df_train_agg).loc[i,'signal'] for i in label]
    plot_size_and_tools = {'plot_height': 300, 'plot_width': 1800,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
    p = figure(x_range=label, plot_height=250, title=f"Transition Counts {label}",
               toolbar_location=None, tools="")
    p.vbar(x=label, top=counts, width=0.9)
    p.add_tools(HoverTool())
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    show(p)
for b in range(10):
    BATCH = b+1
    transitions = df_train.query(f'GRP=={BATCH}').groupby('transition')['signal'].count().index
    tm1 = sorted(set([int(i.split('-')[0]) for i in transitions]))
    labels = [[ str(x)+'-'+str(y) for y in \
              sorted(set([int(i.split('-')[1]) for i in transitions if int(i.split('-')[0]) == x]))]\
                for x in tm1]
    for label in labels:
        counts = [pd.DataFrame(df_train.query(f'GRP=={BATCH}').groupby('transition')['signal'].count()).loc[i,'signal'] for i in label]
        plot_size_and_tools = {'plot_height': 300, 'plot_width': 1800,
                                'tools':['box_zoom', 'reset', 'crosshair','help']}
        p = figure(x_range=label, plot_height=250, title=f"Transition Counts {label} in Batch #{BATCH}",
                   toolbar_location=None, tools="")
        p.vbar(x=label, top=counts, width=0.9)
        p.add_tools(HoverTool())
        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        show(p)
df_train['transition_diff'] = df_train['transition'].apply(lambda x: int(x.split('-')[1])-int(x.split('-')[0]))
df_train['signal_diff_1'] = df_train['signal'].diff().fillna(method='bfill')
df_train.head()
plot_size_and_tools = {'plot_height': 600, 'plot_width': 600,
                                'tools':['box_zoom', 'reset', 'crosshair','help']}
p = figure(title=f"Signal vs. Channel Transition Difference",
                   toolbar_location='right', **plot_size_and_tools)

p.scatter(df_train.signal.values, df_train.transition_diff.values, 
          fill_color='blue', fill_alpha=0.6,
          line_color=None)
p.xaxis.axis_label = 'Signal'
p.yaxis.axis_label = 'Channel Transition Difference'
#p.add_tools(HoverTool())
show(p)
plot_size_and_tools = {'plot_height': 600, 'plot_width': 600,
                                'tools':['box_zoom', 'reset', 'crosshair','help']}
p = figure(title=f"Signal Diff vs. Channel Transition Difference",
                   toolbar_location='right', **plot_size_and_tools)

p.scatter(df_train.signal_diff_1.values, df_train.transition_diff.values, 
          fill_color='purple', fill_alpha=0.6,
          line_color=None)
p.xaxis.axis_label = 'Signal Diff'
p.yaxis.axis_label = 'Channel Transition Difference'
#p.add_tools(HoverTool())
show(p)
palette = [cc.rainbow[i*15] for i in range(15)]
x = np.linspace(-11,11, 500)
plot_size_and_tools = {'plot_height': 800, 'plot_width': 700,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p1 = figure(y_range=(-7, 9),  x_range=(-11, 11), toolbar_location='above', **plot_size_and_tools)
for i in range(13):
    diff = i-6
    pdf = gaussian_kde(df_train.query(f'transition_diff=={diff}')['signal_diff_1'].values)
    y = pdf(x) - 6 +i
    source = ColumnDataSource(data=dict(x=x, y=y))
    p1.patch('x', 'y', line_width=1, alpha = 0.6, color=palette[i], source=source, line_color='black')
p1.xaxis.ticker = FixedTicker(ticks=list(range(-11, 11, 1)))
p1.xaxis.axis_label = 'Signal Diff Lag1'
p1.yaxis.axis_label = 'Channel Transition Difference'
show(p1)
diff_dict = {}
for i in range(15):
    diff = i-6
    diff_dict[diff] = df_train.query(f'transition_diff=={diff}')['signal_diff_1'].describe()

Diff = pd.DataFrame(diff_dict)
Diff
label = list(pd.DataFrame(df_train.groupby('transition_diff')['signal'].count()).index)
counts = [pd.DataFrame(df_train.groupby('transition_diff')['signal'].count()).loc[i,'signal'] for i in label]
plot_size_and_tools = {'plot_height': 300, 'plot_width': 600,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p = figure( title=f"Transition Counts {label}",
           toolbar_location='right',  **plot_size_and_tools)
p.vbar(x=label, top=counts, width=0.5)
p.add_tools(HoverTool())
p.xgrid.grid_line_color = None
p.xaxis.ticker = FixedTicker(ticks=list(range(-8, 8, 1)))
p.y_range.start = 0

show(p)
for b in range(10):
    BATCH = b+1
    label = list(pd.DataFrame(df_train.query(f'GRP=={BATCH}').groupby('transition_diff')['signal'].count()).index)
    counts = [pd.DataFrame(df_train.query(f'GRP=={BATCH}').groupby('transition_diff')['signal'].count()).loc[i,'signal'] for i in label]
    plot_size_and_tools = {'plot_height': 300, 'plot_width': 600,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
    p = figure( title=f"Transition Counts {label} in Batch #{BATCH}",
               toolbar_location='right',  **plot_size_and_tools)
    p.vbar(x=label, top=counts, width=0.5)
    p.add_tools(HoverTool())
    p.xgrid.grid_line_color = None
    p.xaxis.ticker = FixedTicker(ticks=list(range(-8, 8, 1)))
    p.y_range.start = 0

    show(p)