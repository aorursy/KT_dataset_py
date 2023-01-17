import pandas as pd
import numpy as np
data = pd.read_csv('../input/used-car-data/used_car_data.csv')
# Mengambil list nama mobil yang berbeda (unique)
name_list = data['Name'][:].unique()

# Menghitung kuantitas masing-masing mobil
qty_1 = []
for name in name_list:
    count = len(data[data['Name']==name][:])
    qty_1.append(count)

# Menyimpan data sebagai csv
data_save1 = pd.DataFrame({
        'Name' : name_list,
        'Quantity' : qty_1
    })
data_save1.to_csv('soal_1.csv')

print("Done")
# Mengambil list lokasi mobil yang berbeda (unique)
location_list = data['Location'][:].unique()

# Menghitung kuantitas mobil di masing-masing lokasi
qty_2 = pd.Series([], dtype='int64')
for location in location_list:
    count = len(data[data['Location']==location][:])
    qty_2 = qty_2.append(pd.Series([count]),True)

# Mengambil lokasi dengan mobil terbanyak
index_max_2 = qty_2.idxmax()
location_max = location_list[index_max_2]
print(location_max)
# Mengambil list tahun edisi mobil yang berbeda (unique)
year_list = data['Year'][:].unique()
year_list.sort()

# Menghitung kuantitas mobil di masing-masing lokasi
qty_3 = pd.Series([], dtype='int64')
for year in year_list:
    count = len(data[data['Year']==year][:])
    qty_3 = qty_3.append(pd.Series([count]),True)

# Ubah tema plotting (Opsional)
from jupyterthemes import jtplot
jtplot.style(theme = 'gruvboxd')

# Mengambil lokasi dengan mobil terbanyak
import matplotlib.pyplot as plt
plt.bar(year_list,qty_3)
plt.show()
# Menyeleksi mobil dengan total jarak pemakaian di bawah 100000 kilometer
car_driven_under_100k = data[data['Kilometers_Driven']<100000][:]

# Menampilkan jumlah mobil-mobil tersebut
print(len(car_driven_under_100k))
