# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import datetime,timedelta
import datetime as dt
df=pd.read_csv('../input/comulative-confirmed-case-in-indonesia/Comulative Confirmed Case in Indonesia.csv')
FMT = '%m/%d/%Y'
date = df['Date']
df['Date'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("3/02/2020", FMT)).days)
df.info()
def data(prov,pulau):
    dateprov=pd.DataFrame(df[['Date',prov]],)
    data=dateprov.rename(columns={prov:'Kasus'})
    prov=[prov]*len(data)
    data['Prov']=prov
    data['Pulau']=pulau
    return data
Aceh=data('Aceh','Sumatera')
Bali=data('Bali','Bali')
Jakarta=data('DKI Jakarta','Jawa')
jateng=data('jateng','Jawa')
jatim=data('jatim','Jawa')
Sulut=data('Sulut','Sulawesi')
Sulsel=data('Sulsel','Sulawesi')
Papua=data('Papua','Papua')
Jabar=data('Jabar','Jawa')
df=pd.concat([Aceh,Bali,Jakarta,Jabar,jateng,jatim,Sulut,Sulsel,Papua])
current_date = 128
dff = (df[df['Date'].eq(current_date)]
       .sort_values(by='Kasus', ascending=True)
       .head(10))
dff
colors = dict(zip(
    ['Jawa', 'Sumatera', 'Bali', 'Sulawesi',
     'Papua'],
    ['#adb0ff', '#ffb3ff', '#90d595', '#e48381',
     '#aafbff']
))
group_lk = df.set_index('Prov')['Pulau'].to_dict()
fig, ax = plt.subplots(figsize=(15, 12))
def draw_barchart(Date):
    dff = df[df['Date'].eq(Date)].sort_values(by='Kasus', ascending=True).tail(10)
    ax.clear()
    ax.barh(dff['Prov'], dff['Kasus'], color=[colors[group_lk[x]] for x in dff['Prov']])
    dx = dff['Kasus'].max() / 200
    for i, (Kasus, Prov) in enumerate(zip(dff['Kasus'], dff['Prov'])):
        ax.text(Kasus-dx, i,     Prov,           size=14, weight=600, ha='right', va='bottom')
        ax.text(Kasus-dx, i-.25, group_lk[Prov], size=10, color='#444444', ha='right', va='baseline')
        ax.text(Kasus+dx, i,     f'{Kasus:,.0f}',  size=14, ha='left',  va='center')
    # ... polished styles
    ax.text(1, 0.4, Date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Jumlah Kasus Positif', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'BarChart Race untuk beberapa provinsi',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    ax.text(1, 0, 'by @ade.kur; credit @adekurniawanputrahadi@gmail.com', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    
draw_barchart(128)
import matplotlib.animation as animation
from IPython.display import HTML
fig, ax = plt.subplots(figsize=(10, 8))
animator = animation.FuncAnimation(fig, draw_barchart, frames=range(0,128))
HTML(animator.to_html5_video()) 
# bisa menggunakan animator.to_html5_video() ataupun animator.save()
