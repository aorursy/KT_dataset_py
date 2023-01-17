import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from IPython.display import display
from IPython.core.display import HTML
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.rc['font.size'] = 9.0
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)
import seaborn as sns

%matplotlib inline
import subprocess
#from https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python , Olafur's answer
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

lines = file_len('../input/data.csv')
print('Number of lines in "train.csv" is:', lines)
skiplines = np.random.choice(np.arange(1, lines), size=lines-1-1000000, replace=False)
skiplines=np.sort(skiplines)
print('lines to skip:', len(skiplines))

data = pd.read_csv("../input/data.csv", skiprows=skiplines)
data.sample(5)
data.isnull().sum(0)
# Just a helper module to make visualizations more intuitive
num_to_month={
    1:"Jan",
    2:"Feb",
    3:"Mar",
    4:"Apr",
    5:"May",
    6:"June",
    7:"July",
    8:"Aug",
    9:"Sept",
    10:"Oct",
    11:"Nov",
    12:"Dec"
}
data['month'] = data.month.apply(lambda x: num_to_month[x])
gc.collect()
pivot = data.pivot_table(index='year', columns='month', values='day', aggfunc=len)
colors = ["#8B8B00", "#8B7E66", "#EE82EE", "#00C78C", 
          "#00E5EE", "#FF6347", "#EED2EE", 
          "#63B8FF", "#00FF7F", "#B9D3EE", 
          "#836FFF", "#7D26CD"]
pivot.loc[:,['Jan','Feb', 'Mar',
            'Apr','May','June',
            'July','Aug','Sept',
            'Oct','Nov','Dec']].plot.bar(stacked=True, figsize=(20,10), color=colors)
plt.xlabel("Years")
plt.ylabel("Ridership")
plt.legend(loc=10)
plt.show()
f, ax = plt.subplots(1,2, figsize=(20,7))
colors = ['#66b3ff','#ff9999']
pie = ax[0].pie(list(data['gender'].value_counts()), 
                   labels=list(data.gender.unique()),
                  autopct='%1.1f%%', shadow=True, startangle=90, colors=colors)
count = sns.countplot(x='usertype', data=data, ax=ax[1], color='g', alpha=0.75)
ax[0].set_title("Gender Distribution in Ridership")
ax[1].set_xlabel("Type of Rider")
ax[1].set_ylabel("Ridership")
ax[1].set_title("Type of Customers")
data.usertype.value_counts()
station_info = data[['from_station_name','latitude_start','longitude_start']].drop_duplicates(subset='from_station_name')
station_info.sample(5)
lat_list = list(station_info.latitude_start)
lat_list = [str(i) for i in lat_list]
lon_list = list(station_info.longitude_start)
lon_list = [str(i) for i in lon_list]
names = list(station_info.from_station_name)
display(HTML("""
<div>
    <a href="https://plot.ly/~sominw/6/?share_key=y6irxkKqSVolnuF0l4w420" target="_blank" title="Chicago Cycle Sharing Stations" style="display: block; text-align: center;"><img src="https://plot.ly/~sominw/6.png?share_key=y6irxkKqSVolnuF0l4w420" alt="Chicago Cycle Sharing Stations" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="sominw:6" sharekey-plotly="y6irxkKqSVolnuF0l4w420" src="https://plot.ly/embed.js" async></script>
</div>"""))
