#import library yang dibutuhkan

import pandas as pd

import requests

from bs4 import BeautifulSoup



#Request ke website

website_url = requests.get('https://id.wikipedia.org/wiki/Demografi_Indonesia').text

soup = BeautifulSoup(website_url, 'lxml')



#Ambil table dengan class 'wikitable sortable'

my_table = soup.find('table', {'class':'wikitable sortable'})



#cari data dengan tag 'td'

links = my_table.findAll('td')



#Membuat lists kosong

kode_bps = []

nama = []

ibu_kota = []

populasi = []

luas_km = []

pulau = []

#memasukkan data ke dalam list berdasarkan pola HTML

for i, link in enumerate(links):

    if i in range(0, len(links), 9):

        kode_bps.append(link.get_text())

    if i in range(2, len(links), 9):

        nama.append(link.get_text())

    if i in range(4, len(links), 9):

        ibu_kota.append(link.get_text())

    if i in range(5, len(links), 9):

        populasi.append(link.get_text())

    if i in range(6, len(links), 9):

        luas_km.append(link.get_text())

    if i in range(8, len(links), 9):

        pulau.append(link.get_text()[:-1])



#Membuat DatFrame

df = pd.DataFrame()

df['Kode BPS'] = kode_bps

df['Nama'] = nama

df['Ibu Kota'] = ibu_kota

df['Populasi'] = populasi

df['Luas km'] = luas_km

df['Pulau'] = pulau



print(df)

#Perintah jika ingin dimasukkan kedalam file csv

#df.to_csv('Indonesia_Demography_by_Province.csv', index=False, encoding='utf-8', quoting=1)
#import library yang dibutuhkan

import re



#function email_check

def email_check(input):

    match = re.search('(?=^((?!-).)*$)(?=[^0-9])((?=^((?!\.\d).)*$)|(?=.*_))',input )



    if match:

        print('Pass')

    else:

        print('Not Pass')



#Masukkan daftar email ke dalam list

emails = ['my-name@someemail.com', 'myname@someemail.com','my.name@someemail.com',

'my.name2019@someemail.com', 'my.name.2019@someemail.com',

'somename.201903@someemail.com','my_name.201903@someemail.com',

'201903myname@someemail.com', '201903.myname@someemail.com']



#Looping untuk pengecekan Pass atau Not Pass

for email in emails :

    email_check(email)