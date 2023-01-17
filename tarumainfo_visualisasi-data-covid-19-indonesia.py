# PEMASANGAN MODUL INKOVIS

!wget -O inkovis.py "https://github.com/taruma/inkovis/raw/master/notebook/inkovis.py" -q

!wget -O so.py "https://github.com/taruma/inkovis/raw/master/notebook/so.py" -q
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import inkovis
# IMPORT DATASET DARI KAGGLE

dataset_kaggle = pd.read_csv('/kaggle/input/indonesia-coronavirus-cases/cases.csv', index_col=0, parse_dates=True)

dataset_kaggle.head()
_kaggle_col_names = 'acc_tested acc_confirmed acc_released acc_deceased acc_negative being_checked'.split()

_inkovis_col_names = 'jumlah_periksa konfirmasi sembuh meninggal negatif proses_periksa'.split() 



dataset_inkovis = dataset_kaggle[_kaggle_col_names].copy().dropna().astype(int)

dataset_inkovis.columns = _inkovis_col_names

dataset_inkovis.index.name = 'tanggal'

dataset_inkovis.tail()
# PENGATURAN PARAMS VISUALISASI

FIG_SIZE = (20, 8)

FIG_SIZE_GROUP = (20, 12)



# PARAM FUNGSI

DATASET = dataset_inkovis

MASK = None

DAYS = 1
fig, ax = plt.subplots(figsize=FIG_SIZE)



inkovis.plot_confirmed_case(

    dataset=DATASET, ax=ax, mask=MASK, days=DAYS,

    show_diff_numbers=False, show_hist=True,

    show_diff_bar=False,

    show_info=False

)



plt.savefig('KASUS_KONFIRMASI_HARIAN.png', dpi=150)
fig, ax = plt.subplots(figsize=FIG_SIZE)



inkovis.plot_testing_case(

    dataset=DATASET, ax=ax, mask=MASK, days=DAYS,

    show_diff_numbers=False, show_hist=True,

    show_diff_bar=False,

    show_info=False)



plt.savefig('JUMLAH_SPESIMEN_HARIAN.png', dpi=150)
fig, ax = plt.subplots(figsize=FIG_SIZE)

inkovis.plot_confirmed_growth(

    dataset=DATASET, ax=ax, mask=MASK, days=DAYS,

    show_info=False,

    show_confirmed=True, show_confirmed_numbers=True)



plt.savefig('PERKEMBANGAN_KASUS_KONFIRMASI_HARIAN.png', dpi=150)
fig, ax = plt.subplots(figsize=FIG_SIZE)

inkovis.plot_testing_growth(

    dataset=DATASET, ax=ax, mask=MASK, days=DAYS,

    show_info=False,

    show_confirmed=True)



plt.savefig('PERKEMBANGAN_JUMLAH_SPESIMEN_HARIAN.png', dpi=150)
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



plt.savefig('KONFIRMASI_PERKEMBANGAN_HARIAN.png', dpi=150)
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

plt.savefig('SPESIMEN_PERKEMBANGAN_HARIAN.png', dpi=150)
fig, ax = plt.subplots(

    nrows=2, ncols=1, figsize=FIG_SIZE_GROUP, sharex=True,

    gridspec_kw={'height_ratios':[1, 1]})



_DATASET = dataset_inkovis



inkovis.plot_confirmed_case(

    dataset=DATASET, ax=ax[0], mask=MASK, days=DAYS,

    show_diff_bar=False, show_info=False, show_hist=False, show_title=False

)



inkovis.plot_testing_case(

    dataset=DATASET, ax=ax[1], mask=MASK, days=DAYS,

    show_diff_bar=False, show_info=False, show_hist=False, show_title=False

)



fig.suptitle("KASUS KONFIRMASI DAN JUMLAH SPESIMEN COVID-19 DI INDONESIA", fontweight='bold', fontsize='xx-large')

fig.subplots_adjust(top=0.95)



ax[0].set_xlabel('');

plt.savefig('KONFIRMASI_SPESIMEN_HARIAN.png', dpi=150)
fig, ax = plt.subplots(

    nrows=2, ncols=1, figsize=FIG_SIZE_GROUP, sharex=True,

    gridspec_kw={'height_ratios':[1, 1]})



_DATASET = dataset_inkovis



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

plt.savefig('KONFIRMASI_SPESIMEN_PERKEMBANGAN_HARIAN.png', dpi=150)