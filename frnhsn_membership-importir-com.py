# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import datetime
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from scipy import stats

# bokeh packages
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource,CustomJS,HoverTool,CategoricalColorMapper,NumeralTickFormatter
from bokeh.layouts import row,column,gridplot
from bokeh.models.widgets import Tabs,Panel
output_notebook()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Memuat dataset
subs = pd.read_csv('/kaggle/input/importir/subs.csv')
user = pd.read_csv('/kaggle/input/importir/user.csv')

subs.head(5)
subs.describe()
user.head(5)
user.describe()
# Megubah total_price Free-Member jadi 0 rupiah
subs.loc[subs.package_name == 'Free-Member', 'total_price'] = 0

# Mengubah tipe data kolom paid_at, created_at pada dataset subs menjadi datetime
subs['paid_at'] = pd.to_datetime(subs['paid_at'], errors='coerce')
subs['created_at'] = pd.to_datetime(subs['created_at'], errors='coerce')

# Mengubah gold-3-tahun jadi Gold-3-tahun
subs.loc[subs.package_name == 'gold-3-tahun', 'package_name'] = "Gold-3-tahun"

# Menentukan waktu expired dari masing masing paket
subs.loc[subs.package_name == 'AnR-Basic-Plus', 'expired_at'] = subs['paid_at'] + DateOffset(months=6)
subs.loc[subs.package_name == 'AnR-Gold', 'expired_at'] = subs['paid_at'] + DateOffset(months=12)
subs.loc[subs.package_name == 'Basic-Harbolnas', 'expired_at'] = subs['paid_at'] + DateOffset(months=1)
subs.loc[subs.package_name == 'Basic-Plus', 'expired_at'] = subs['paid_at'] + DateOffset(months=1)
subs.loc[subs.package_name == 'Basic-Plus-24-month', 'expired_at'] = subs['paid_at'] + DateOffset(months=24)
subs.loc[subs.package_name == 'Goes to China', 'expired_at'] = subs['paid_at'] + DateOffset(months=36)
subs.loc[subs.package_name == 'Gold', 'expired_at'] = subs['paid_at'] + DateOffset(months=12)
subs.loc[subs.package_name == 'Membership 3 Tahun', 'expired_at'] = subs['paid_at'] + DateOffset(months=36)
subs.loc[subs.package_name == 'Silver', 'expired_at'] = np.datetime64('NaT')
subs.loc[subs.package_name == 'Silver', 'expired_at'] = np.datetime64('NaT')
subs['expired_at'] = pd.to_datetime(subs['expired_at'], format='%d/%m/%Y', errors='coerce')

# Menghilangkan data subscription yang membayar pada tahun 2030
subs['paid_at'] = subs['paid_at'].drop([303], axis=0)
subs['paid_at'] = subs['paid_at'].drop([10483], axis=0)


# Membuang data subs dengan tanggal bayar dan expired NaN
subs_exp = subs.dropna(subset=['paid_at', 'expired_at'])


# Membersihkan data kota misal Jakarta dan jakarta adalah kota yang sama
user['city'] = user['city'].str.lower()
subs.groupby(
    'package_name')['package_name'].count().to_frame().rename(
    columns={'package_name':'Jumlah Langganan'}).join(
    (subs.groupby(
        'package_name')['package_name'].count()/len(subs)*100).to_frame().rename(
        columns={'package_name':'Dalam persen'}))
subs['subs_length'] = subs['expired_at'] - subs['paid_at']
round(subs.dropna(subset=['subs_length']).groupby('package_name')['subs_length'].min().dt.days / 30)
subs.groupby('package_name')['price_idr'].describe()
disc = subs
disc['discount?'] = disc['promo_code_membership'].notna()
disc_group = disc.groupby(['discount?','package_name'])[['price_idr','discount_amount','total_price']].sum()
disc_group['discount_percentage'] = disc_group['discount_amount'] / disc_group['price_idr'] * 100
disc_group[['price_idr','discount_amount','total_price']].apply(
    lambda x: (x.astype(float)/1000000).round(2).astype(str) + ' juta').join(disc_group['discount_percentage'])
disc[disc['discount?'] == True].groupby('promo_code_membership')[
    ['price_idr','discount_amount','total_price']].agg('sum').sort_values(by='discount_amount',ascending=False).apply(
    lambda x: (x.astype(float)/1000000).round(2).astype(str) + ' juta')
output_file('langganan.html')

subs['paid_at'] = pd.to_datetime(subs['paid_at'], format='%Y/%m/%d')

grouped= subs.groupby(pd.Grouper(key='paid_at', freq='M'))['id'].count()

s_all = ColumnDataSource(data={
    'paid_at': grouped.index,
    'num_paid': grouped.values
})

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(
        x_axis_type="datetime",
        y_axis_type="linear",
        plot_height = 400,
        tools = TOOLS,
        plot_width = 800)

p.xaxis.axis_label = 'Bulan'
p.yaxis.axis_label = 'Jumlah langganan'

p.line(x='paid_at', y='num_paid', line_width=2, source=s_all)
p.circle(x='paid_at', y='num_paid', line_width=2, source=s_all)

p.title.text = 'Jumlah Langganan Perbulan Seluruh Paket Langganan'

hover = HoverTool(
    tooltips=[
        ('Bulan', '@paid_at{%m/%Y}'),
        ('Jumlah langganan', '@num_paid'),
    ],
    formatters={
        'paid_at': 'datetime', # use 'datetime' formatter for '@date' field
        'num_paid': 'printf', # use 'printf' formatter for '@{adj close}' field
                                  # use default 'numeral' formatter for other fields
    },
    )

p.add_tools(hover)

show(p)

from bokeh.palettes import Paired
import itertools

output_file('langganan.html')

subs['paid_at'] = pd.to_datetime(subs['paid_at'], format='%Y/%m/%d')

package = {}

for pkg in subs['package_name'].unique():
    dataset = subs[subs['package_name'] == pkg].groupby(
        pd.Grouper(key='paid_at', freq='M'))['id'].count()
    package[pkg] = ColumnDataSource(data={
        'paid_at': dataset.index,
        'num_paid': dataset.values,
        'package_name': [pkg] * len(dataset)
    })

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(
        x_axis_type="datetime",
        y_axis_type="linear",
        plot_height = 400,
        tools = TOOLS,
        plot_width = 800)

p.xaxis.axis_label = 'Bulan'
p.yaxis.axis_label = 'Jumlah langganan'

def color_gen():
    for c in itertools.cycle(Paired[12]):
        yield c
        
colors = color_gen()

for key in package.keys():
    c = next(colors)
    color = str(c)
    p.line(x='paid_at', y='num_paid', line_width=2, source=package[key], line_color=color, legend_label=key)
    p.circle(x='paid_at', y='num_paid', line_width=0.5, source=package[key], color=color, legend_label=key)

p.title.text = 'Jumlah Langganan Perbulan Perjenis Paket Langganan'

hover = HoverTool(
    tooltips=[
        ('Bulan', '@paid_at{%m/%Y}'),
        ('Jumlah langganan', '@num_paid'),
        ('Jenis Paket', '@package_name')
    ],

    formatters={
        'paid_at': 'datetime',      # use 'datetime' formatter for '@date' field
        'num_paid': 'printf',       # use 'printf' formatter for '@{adj close}' field
                                    # use default 'numeral' formatter for other fields
    },
    )

p.add_tools(hover)

show(p)
output_file('langganan.html')

subs['paid_at'] = pd.to_datetime(subs['paid_at'], format='%Y/%m/%d')

all_package = subs.groupby(pd.Grouper(key='paid_at', freq='M'))['paid_at','total_price'].sum()
s_all = ColumnDataSource(all_package)

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(
        x_axis_type="datetime",
        y_axis_type="linear",
        plot_height = 400,
        tools = TOOLS,
        plot_width = 800)

p.xaxis.axis_label = 'Bulan'
p.yaxis.axis_label = 'Jumlah pendapatan'
p.yaxis.formatter=NumeralTickFormatter(format="(Rp. 0.00 a)")

p.line(x='paid_at', y='total_price', line_width=2, source=s_all)
p.circle(x='paid_at', y='total_price', line_width=2, source=s_all)

p.title.text = 'Pendapatan Perbulan Seluruh Paket Langganan'

hover = HoverTool(
    tooltips=[
        ('Bulan', '@paid_at{%m/%Y}'),
        ('Pendapatan', 'Rp.@total_price{0.00 a}'),
    ],

    formatters={
        'paid_at': 'datetime',      # use 'datetime' formatter for '@date' field
        '@total_price': 'printf',   # use 'printf' formatter for '@{adj close}' field
                                    # use default 'numeral' formatter for other fields
    },
    )

p.add_tools(hover)

show(p)
from bokeh.palettes import Paired
import itertools

output_file('langganan.html')

subs['paid_at'] = pd.to_datetime(subs['paid_at'], format='%Y/%m/%d')

package = {}

for pkg in subs['package_name'].unique():
    dataset = subs[subs['package_name'] == pkg].groupby(
        pd.Grouper(key='paid_at', freq='M'))['total_price'].sum()
    package[pkg] = ColumnDataSource(data={
        'paid_at': dataset.index,
        'total_price': dataset.values,
        'package_name': [pkg] * len(dataset)
    })

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(
        title="Year-wise total number of crimes",
        x_axis_type="datetime",
        y_axis_type="linear",
        plot_height = 400,
        tools = TOOLS,
        plot_width = 800)

p.xaxis.axis_label = 'Bulan'
p.yaxis.axis_label = 'Jumlah Pendapatan'
p.yaxis.formatter=NumeralTickFormatter(format="(Rp. 0.00 a)")

def color_gen():
    for c in itertools.cycle(Paired[12]):
        yield c
        
colors = color_gen()

for key in package.keys():
    c = next(colors)
    color = str(c)
    p.line(x='paid_at', y='total_price', line_width=2, source=package[key], line_color=color, legend_label=key)
    p.circle(x='paid_at', y='total_price', line_width=0.5, source=package[key], color=color, legend_label=key)

p.title.text = 'Pendapatan Perbulan Perjenis Paket Langganan'

hover = HoverTool(
    tooltips=[
        ('Bulan', '@paid_at{%m/%Y}'),
        ('Pendapatan', 'Rp.@total_price{0.00 a}'),
        ('Jenis Paket', '@package_name')
    ],

    formatters={
        'paid_at': 'datetime',      # use 'datetime' formatter for '@date' field
        '@total_price': 'printf',   # use 'printf' formatter for '@{adj close}' field
                                    # use default 'numeral' formatter for other fields
    },
    )

p.add_tools(hover)

show(p)
g1 = subs.groupby('email').filter(lambda x: len(x) == 1)
g2 = subs.groupby('email').filter(lambda x: len(x) == 2)
g3 = subs.groupby('email').filter(lambda x: len(x) == 3)
g4 = subs.groupby('email').filter(lambda x: len(x) == 4)

subs.groupby('email')['id'].count().value_counts()
def selangWaktuLangganan(g):
    lst = []
    for name, group in g.groupby('email'):
        df = group['paid_at'] - group['expired_at'].shift(periods=1)
        df.loc[(df <= pd.Timedelta(0)) & (df != np.datetime64('NaT'))] = 0
        lst.extend(df)
    gd = pd.to_timedelta(pd.Series(lst).dropna(), unit='d')
    return gd

gd2 = selangWaktuLangganan(g2)
gd3 = selangWaktuLangganan(g3)
gd4 = selangWaktuLangganan(g4)
sns.distplot(gd4.dt.days, label="Frekuensi 4 kali")
sns.distplot(gd3.dt.days, label="Frekuensi 3 kali")
sns.distplot(gd2.dt.days, label="Frekuensi 2 kali")
sns.set(rc={'figure.figsize':(14,5)})
plt.legend()
plt.title("Rentang waktu berlangganan frekuensi langganan dua kali")
plt.xlim(0)
plt.xlabel("Rentang waktu (hari)")
plt.xticks(np.arange(min(gd2.dt.days.values), max(gd2.dt.days.values)+1, 30))
plt.show()
def frekuensiPaketLangganan(g):
    dct = {}
    for name, group in g.groupby('email'):
        vc = group['package_name'].value_counts().sort_values()
        k = str(list(vc.keys()))
        v = list(vc.values)
        try:
            assert dct[k]
            lst_val = dct[k]
            for i, j in enumerate(lst_val):
                lst_val[i] = lst_val[i] + v[i]
            dct[k] = lst_val
        except:
            dct.update({k: v})
    return dct
g1['package_name'].value_counts()
current = np.datetime64(datetime.datetime.now())

g1['is_active'] = g1["expired_at"] > current


active = g1[g1['is_active'] == True].dropna(subset=['expired_at']).groupby('package_name')['id'].count().to_dict()
expired = g1[g1['is_active'] == False].dropna(subset=['expired_at']).groupby('package_name')['id'].count().to_dict()


lst_active = []
lst_expired = []
for package in g1.dropna(subset=['expired_at'])['package_name'].unique():
    if package in active.keys():
        lst_active.append(active[package])
    else:
        lst_active.append(0)
    if package in expired.keys():
        lst_expired.append(expired[package])
    else:
        lst_expired.append(0)


fig, ax = plt.subplots()

fig.set_size_inches(18.5, 10.5)

width = 0.5

labels = g1.dropna(subset=['expired_at'])['package_name'].unique()
ax.bar(labels, lst_active, width, label='aktif')
ax.bar(labels, lst_expired, width, bottom=lst_active, label='berakhir')

ax.set_ylabel('Jumlah langganan')
ax.set_title('Status langganan berdasarkan jenis paket dalam frekuensi langganan paket satu kali')
ax.legend()

plt.xticks(rotation=90)

plt.show()
frekuensiPaketLangganan(g2)
frekuensiPaketLangganan(g3)
frekuensiPaketLangganan(g4)
group_paid = subs_exp.groupby(pd.Grouper(key='paid_at', freq='M')).agg(
    {'id': 'count',
     'total_price': 'sum',
     'price_idr': 'sum'
    })

group_expired = subs_exp.groupby(pd.Grouper(key='expired_at', freq='M')).agg(
    {'id': 'count',
     'total_price': 'sum',
     'price_idr': 'sum'
    })

group_expired.rename(columns = {'id':'lost'}, inplace = True)
group_paid.rename(columns = {'id':'added'}, inplace = True)

active_subs = pd.concat([group_paid['added'], group_expired['lost']], axis=1).fillna(0)

def count_active(x):
    return ((subs_exp['expired_at'].dt.normalize() > x) &\
            (subs_exp['paid_at'].dt.normalize() <= x)).sum()
    
count_active = pd.Series({x: count_active(x) for x in active_subs.index})

active_subs['current_active'] = count_active.values
output_file('active_member.html')

source = ColumnDataSource(data={
    'paid_at': active_subs.index,
    'added': active_subs.added,
    'lost': active_subs.lost,
    'current_active': active_subs.current_active,
})

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(
        x_axis_type="datetime",
        y_axis_type="linear",
        plot_height = 400,
        tools = TOOLS,
        plot_width = 800)

p.xaxis.axis_label = 'Bulan'
p.yaxis.axis_label = 'Jumlah'
p.yaxis.formatter=NumeralTickFormatter(format="(Rp. 0.00 a)")

def color_gen():
    for c in itertools.cycle(Paired[12]):
        yield c
        
colors = color_gen()

c1 = next(colors)
p.line(x='paid_at', y='added', line_width=2, source=source, line_color=c1, legend_label='langganan bertambah')
p.circle(x='paid_at', y='added', line_width=0.5, source=source, color=c1, legend_label='langganan bertambah')

c2 = next(colors)
p.line(x='paid_at', y='lost', line_width=2, source=source, line_color=c2, legend_label='langganan berakhir')
p.circle(x='paid_at', y='lost', line_width=0.5, source=source, color=c2, legend_label='langganan berakhir')

c3 = next(colors)
p.line(x='paid_at', y='current_active', line_width=2, source=source, line_color=c3, legend_label='langganan aktif')
p.circle(x='paid_at', y='current_active', line_width=0.5, source=source, color=c3, legend_label='langganan aktif')

p.title.text = 'Langganan bertambah, habis dan aktif perbulan'

hover = HoverTool(
    tooltips=[
        ('Bulan', '@paid_at{%m/%Y}'),
        ('Jumlah', '$y'),
    ],
    formatters={
        'paid_at': 'datetime',      # use 'datetime' formatter for '@date' field
        'y': 'numeral',             # use 'printf' formatter for '@{adj close}' field
                                    # use default 'numeral' formatter for other fields
    },
    mode='mouse'
    )

p.add_tools(hover)

show(p)

churn = active_subs['lost'] * 100 / (active_subs['added'] + active_subs['current_active'].shift(periods=1))

output_file('churn.html')

source = ColumnDataSource(data={
    'period': churn.index,
    'churn_rate': churn.values
})

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(
        x_axis_type="datetime",
        y_axis_type="linear",
        plot_height = 400,
        tools = TOOLS,
        plot_width = 800)

p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Total Price'

p.line(x='period', y='churn_rate', line_width=2, source=source)

p.title.text = 'Churn rate perbulan'

hover = HoverTool(
    tooltips=[
        ('Bulan',   '@period{%m/%Y}'),
        ('Churn Rate',  '@churn_rate%'),
    ],

    formatters={
        'period'       : 'datetime', # use 'datetime' formatter for '@date' field
        'churn_rate' : 'printf',   # use 'printf' formatter for '@{adj close}' field
                                     # use default 'numeral' formatter for other fields
    },

    mode='vline')

p.add_tools(hover)

show(p)
group_paid = subs_exp.groupby(pd.Grouper(key='paid_at', freq='Y')).agg(
    {'id': 'count',
     'total_price': 'sum',
     'price_idr': 'sum'
    })

group_expired = subs_exp.groupby(pd.Grouper(key='expired_at', freq='Y')).agg(
    {'id': 'count',
     'total_price': 'sum',
     'price_idr': 'sum'
    })

group_expired.rename(columns = {'id':'lost'}, inplace = True)
group_paid.rename(columns = {'id':'added'}, inplace = True)

active_subs_year = pd.concat([group_paid['added'], group_expired['lost']], axis=1).fillna(0)

def count_active(x):
    return ((subs_exp['expired_at'].dt.normalize() > x) &\
            (subs_exp['paid_at'].dt.normalize() <= x)).sum()
    
count_active = pd.Series({x: count_active(x) for x in active_subs_year.index})

active_subs_year['current_active'] = count_active.values
output_file('active_member.html')

source = ColumnDataSource(data={
    'paid_at': active_subs_year.index,
    'added': active_subs_year.added,
    'lost': active_subs_year.lost,
    'current_active': active_subs_year.current_active,
})

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(
        x_axis_type="datetime",
        y_axis_type="linear",
        plot_height = 400,
        tools = TOOLS,
        plot_width = 800)

p.xaxis.axis_label = 'Bulan'
p.yaxis.axis_label = 'Jumlah'
p.yaxis.formatter=NumeralTickFormatter(format="(Rp. 0.00 a)")

def color_gen():
    for c in itertools.cycle(Paired[12]):
        yield c
        
colors = color_gen()

c1 = next(colors)
p.line(x='paid_at', y='added', line_width=2, source=source, line_color=c1, legend_label='langganan bertambah')
p.circle(x='paid_at', y='added', line_width=0.5, source=source, color=c1, legend_label='langganan bertambah')

c2 = next(colors)
p.line(x='paid_at', y='lost', line_width=2, source=source, line_color=c2, legend_label='langganan berakhir')
p.circle(x='paid_at', y='lost', line_width=0.5, source=source, color=c2, legend_label='langganan berakhir')

c3 = next(colors)
p.line(x='paid_at', y='current_active', line_width=2, source=source, line_color=c3, legend_label='langganan aktif')
p.circle(x='paid_at', y='current_active', line_width=0.5, source=source, color=c3, legend_label='langganan aktif')

p.title.text = 'Langganan bertambah, habis dan aktif perbulan'

hover = HoverTool(
    tooltips=[
        ('Bulan', '@paid_at{%m/%Y}'),
        ('Jumlah', '$y{"0,0"}'),
    ],
    formatters={
        'paid_at': 'datetime',      # use 'datetime' formatter for '@date' field
        'y': 'numeral',             # use 'printf' formatter for '@{adj close}' field
                                    # use default 'numeral' formatter for other fields
    },
    mode='mouse'
    )

p.add_tools(hover)

show(p)

churn_year = active_subs_year['lost'] * 100 / (active_subs_year['added'] + active_subs_year['current_active'].shift(periods=1))

output_file('churn.html')

source = ColumnDataSource(data={
    'period': churn_year.index,
    'churn_rate': churn_year.values
})

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(
        x_axis_type="datetime",
        y_axis_type="linear",
        plot_height = 400,
        tools = TOOLS,
        plot_width = 800)

p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Total Price'

p.line(x='period', y='churn_rate', line_width=2, source=source)

p.title.text = 'Churn rate pertahun'

hover = HoverTool(
    tooltips=[
        ('Bulan',   '@period{%m/%Y}'),
        ('Churn Rate',  '@churn_rate%'),
    ],

    formatters={
        'period'       : 'datetime', # use 'datetime' formatter for '@date' field
        'churn_rate' : 'printf',   # use 'printf' formatter for '@{adj close}' field
                                     # use default 'numeral' formatter for other fields
    },

    mode='vline')

p.add_tools(hover)

show(p)
user_city = pd.DataFrame([user['city'].value_counts().index, 
             user['city'].value_counts(), 
              user['city'].value_counts() / len(user['city']) * 100]).transpose()
user_city.columns = ['Kota', 'Jumlah', 'Persentase']
user_city
gabung = pd.merge(subs, user, on='email')
gabung
gabung = gabung.groupby(['package_name','city'])[['id','total_price']].agg({
'id': 'count',
'total_price': 'sum', 
}).sort_values(by=['package_name','id'],ascending=False)
gabung['total_price'] = (gabung['total_price'].astype(float) / 1000000).round(2).astype(str) + ' juta'


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

gabung