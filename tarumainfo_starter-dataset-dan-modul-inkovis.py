# PEMASANGAN MODUL INKOVIS

!wget -O inkovis.py "https://github.com/hidrokit/inkovis/raw/master/notebook/inkovis.py" -q

!wget -O so.py "https://github.com/hidrokit/inkovis/raw/master/notebook/so.py" -q
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import inkovis
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# ALAMAT DATASET

DATASET_PATH = '../input/data_infeksi_covid19_indonesia.csv'
# IMPORT DATASET DARI KAGGLE



dataset = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True).drop(['catatan', 'kasus_perawatan'], axis=1)

dataset.info()
dataset.tail(5)
# PENGATURAN PARAMS VISUALISASI

FIG_SIZE = (20, 8)

FIG_SIZE_GROUP = (20, 12)



# PARAM FUNGSI

DATASET = dataset

MASK = ('2020-03-01', None)

DAYS = 3
fig, ax = plt.subplots(

    nrows=2, ncols=1, figsize=FIG_SIZE_GROUP, sharex=True,

    gridspec_kw={'height_ratios':[1, 3]})



inkovis.plot_confirmed_case(

    dataset=DATASET, ax=ax[0], mask=MASK, days=DAYS,

    show_diff_bar=False, show_info=False, show_hist=False

)



inkovis.plot_confirmed_growth(

    dataset=DATASET, ax=ax[1], mask=MASK, days=DAYS,

    show_bar=True, show_confirmed=True, 

    show_numbers=True,

    show_total_numbers=True, show_title=False, show_info=False,

    show_legend=False

)

ax[0].set_xlabel('');

fig, ax = plt.subplots(

    nrows=2, ncols=1, figsize=FIG_SIZE_GROUP, sharex=True,

    gridspec_kw={'height_ratios':[1, 3]})



inkovis.plot_testing_case(

    dataset=DATASET, ax=ax[0], mask=MASK, days=DAYS,

    show_diff_bar=False, show_info=False, show_hist=False

)



inkovis.plot_testing_growth(

    dataset=DATASET, ax=ax[1], mask=MASK, days=DAYS,

    show_bar=True, show_confirmed=True, 

    show_numbers=True,

    show_total_numbers=True, show_title=False, show_info=False,

    show_legend=False

)



ax[0].set_xlabel('');
fig, ax = plt.subplots(

    nrows=2, ncols=1, figsize=FIG_SIZE_GROUP, sharex=True,

    gridspec_kw={'height_ratios':[1, 1]})



inkovis.plot_confirmed_case(

    dataset=DATASET, ax=ax[0], mask=MASK, days=DAYS,

    show_diff_bar=False, show_info=False, show_hist=True, show_title=False

)



inkovis.plot_testing_case(

    dataset=DATASET, ax=ax[1], mask=MASK, days=DAYS,

    show_diff_bar=False, show_info=False, show_hist=True, show_title=False

)



fig.suptitle("KASUS KONFIRMASI DAN JUMLAH SPESIMEN COVID-19 DI INDONESIA", fontweight='bold', fontsize='xx-large')

fig.subplots_adjust(top=0.95)



ax[0].set_xlabel('');
fig, ax = plt.subplots(

    nrows=2, ncols=1, figsize=FIG_SIZE_GROUP, sharex=True,

    gridspec_kw={'height_ratios':[1, 1]})



inkovis.plot_confirmed_growth(

    dataset=DATASET, ax=ax[0], mask=MASK, days=DAYS,

    show_bar=True, show_confirmed=True, 

    show_numbers=True,

    show_total_numbers=True, show_title=False, show_info=False,

)

inkovis.plot_testing_growth(

    dataset=DATASET, ax=ax[1], mask=MASK, days=DAYS,

    show_bar=True, show_confirmed=True, 

    show_numbers=True,

    show_total_numbers=True, show_title=False, show_info=False,

)



fig.suptitle("PERKEMBANGAN KASUS KONFIRMASI DAN JUMLAH SPESIMEN COVID-19 DI INDONESIA", fontweight='bold', fontsize='xx-large')

fig.subplots_adjust(top=0.95)



ax[0].set_xlabel('');