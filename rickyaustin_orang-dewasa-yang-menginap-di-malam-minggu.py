import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataHotel = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

dataHotel.head()
dataHotel.columns
jumlahDewasaWeekend=dataHotel.groupby(['adults','is_repeated_guest','arrival_date_month','arrival_date_year']).stays_in_weekend_nights.sum().reset_index()

jumlahDewasaWeekend.columns=['adults','is_repeated_guest','arrival_date_month','arrival_date_year','stays_in_weekend_nights']

jumlahDewasaWeekend
jumlahDewasaWeekend=jumlahDewasaWeekend[jumlahDewasaWeekend.is_repeated_guest != 1]

jumlahDewasaWeekend=jumlahDewasaWeekend[jumlahDewasaWeekend.adults != 0]

jumlahDewasaWeekend
jumlahDewasaWeekend['time_arrival'] = pd.to_datetime(jumlahDewasaWeekend.arrival_date_year.astype(str) + '-' + jumlahDewasaWeekend.arrival_date_month.astype(str))

jumlahDewasaWeekend=jumlahDewasaWeekend.sort_values(by=['time_arrival'])

jumlahDewasaWeekend
waktuKedatanganWeekend=jumlahDewasaWeekend.groupby(['time_arrival','adults']).stays_in_weekend_nights.sum().reset_index()

waktuKedatanganWeekend.columns=['time_arrival','adults','stays_in_weekend_nights']

waktuKedatanganWeekend
waktuKedatanganWeekend['year']=pd.DatetimeIndex(waktuKedatanganWeekend['time_arrival']).year

waktuKedatanganWeekend['month']=pd.DatetimeIndex(waktuKedatanganWeekend['time_arrival']).month

waktuKedatanganWeekend
warnaPalet=sns.color_palette("cubehelix", 13)

grafik = sns.relplot(x='month',y='stays_in_weekend_nights', data=waktuKedatanganWeekend,col='year',kind='line',hue='adults', palette=warnaPalet)

grafik.fig.suptitle('Jumlah Dewasa Baru yang menginap di Malam Minggu setiap Bulan selama 3 Tahun', y=1.10)

grafik.set(xlabel="Bulan", ylabel="Menginap di Malam Minggu")

plt.show()
hitungDewasa = waktuKedatanganWeekend.groupby('adults').stays_in_weekend_nights.sum().reset_index()

hitungDewasa.columns = ['adults','stays_in_weekend_nights']

hitungDewasa