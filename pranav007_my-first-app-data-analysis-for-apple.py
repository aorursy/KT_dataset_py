# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import csv
import matplotlib.pyplot as plt
import seaborn as sns
data0 = "../input/AppleStore.csv"
file = open(data0)
reader = csv.reader(file)
header = next(reader)
data = [i for i in reader]

data_new = []

for row in data:
    Serial_no = int(row[0])
    ID = int(row[1])
    app_name = str(row[2])
    size = int(row[3])
    currency = str(row[4])
    price = float(row[5])
    rating_count_tot =int(row[6])
    rating_count_ver =int(row[7])
    user_rating = float(row[8])
    user_rating_ver = float(row[9])
    ver = str(row[10])
    x = (row[11]).strip('+')
    cont_rating = int(x)
    prime_gene = str(row[12])
    sup_devices_num = int(row[13])
    ipadSc_urls_num = int(row[14])
    lang_num = int(row[15])
    vpp_lic = int(row[16])
    data_new.append([Serial_no,ID,app_name,size,currency,price,rating_count_tot,rating_count_ver,
                   user_rating,user_rating_ver,ver,cont_rating,prime_gene,sup_devices_num,ipadSc_urls_num,lang_num,vpp_lic])

relevant_data_no1 = []
for line in data_new:
    relevant_data_no1.append([line[0],line[2],line[5],line[6],line[8],line[12]])
print(data[0:5])
#Colllecting prime gene list data
prime_gene_list = []
for line in relevant_data_no1:
    prime_gene_list.append(line[5])

games_list = []

for data in data_new:
    a = data[12]
    if a == 'Games':
        games_list.append(a)

game_count = int(len(games_list))


plt.figure(1)
plt.title("App Gene-type count in AppStore")
sns.countplot(y=prime_gene_list)
plt.xlabel('Total Count')
plt.ylabel('App Gene')
plt.show()

#Collecting top free games
tfg = []
for line in relevant_data_no1:
    if line[2] == 0:
        if line[5] == 'Games':
            tfg.append(line)


size = lambda tfg:tfg[3]
tfg.sort(key=size,reverse=True)

rating = lambda tfg:tfg[4]
tfg.sort(key=rating,reverse=True)

tfg_user_rating_data = []
tfg_cont_rating =[]
for x in tfg:
    x =(x[4])
    tfg_user_rating_data.append(x)

for y in tfg:
    y = (y[3])
    tfg_cont_rating.append(y)

plt.figure(2)
plt.title("Top free games by User Ratings")
sns.countplot(tfg_user_rating_data)
plt.xlabel('User app rating')
plt.ylabel("Total no.of User Rating")
plt.show()


#Plot comparing by user rating with cont_rating
plt.figure(3)
plt.title("Comparing User Rating with Content Rating in Top Free Games")
plt.xlabel("User Ratings")
plt.ylabel('Total Content Ratings')
plt.xlim(0,5.1)
plt.plot(tfg_user_rating_data,tfg_cont_rating,'og')
plt.show()

#Top paid games
tpg =[]
for line in relevant_data_no1:
    if line[2] >= 1:
        if line[5] == 'Games':
            tpg.append(line)

#print tpg[0]
tpg_prize = lambda tpg:tpg[2]
tpg.sort(key=tpg_prize,reverse=True)

tpg_prize_list = []
for line in tpg:
    tpg_prize_list.append(line[2])

tpg_rating = lambda tpg:tpg[4]
tpg.sort(key=tpg_rating,reverse=True)

tpg_rating_list = []
for line in tpg:
    tpg_rating_list.append(line[4])


tpg_cont_rating = lambda tgp:tgp[3]
tpg.sort(key=tpg_cont_rating,reverse=True)

tpg_cont_rating_list = []
for line in tpg:
    tpg_cont_rating_list.append(line[3])



#Plotting data for high paid app in AppStore

plt.figure(4)
plt.title("Games downloaded based on category of Rate")
sns.countplot(y=tpg_prize_list)
plt.xlabel('Total Count')
plt.ylabel("Prize")

plt.show()

#Plotting data for User rating list

plt.figure(5)
plt.title("Top Paid User Rating Game Apps")
sns.countplot(x=tpg_rating_list)
plt.xlabel('User Rating')
plt.ylabel("Total Count")
plt.show()

#Comparing User rating with Content Rating
plt.figure(6)
plt.title('Comparing User Rating with Content Rating in Top Paid Games')
plt.plot(tpg_rating_list,tpg_cont_rating_list,'or')
plt.xlabel("User Rating")
plt.ylabel("Total Count")
plt.xlim(0,5.1)
plt.ylim(0,800000)
plt.show()


app_data = []

for line in relevant_data_no1:
    if line[5]  != "Games":
        app_data.append(line)


#To find Top Free Apps(TFA)
tfa = []
for line in app_data:
    if line[2] == 0:
        tfa.append(line)


x = lambda tfa:tfa[4]
tfa.sort(key=x,reverse=True)

tfa_user_app_rating =[]

for line in tfa:
    tfa_user_app_rating.append(line[4])

x = lambda tfa:tfa[3]
tfa.sort(key=x,reverse=True)

tfa_cont_rating = []

for line in tfa:
    tfa_cont_rating.append(line[3])

plt.figure(7)
plt.title("Top User rating for all Free Apps")
sns.countplot(tfa_user_app_rating)
plt.xlabel("User Rating")
plt.ylabel("Total Count")
plt.show()

#Comparing User Rating with content rating

plt.figure(8)
plt.title("Comparing User Rating with Content Rating in Free Apps")
plt.plot(tfa_user_app_rating,tfa_cont_rating,'or')
plt.xlabel("Content Rating")
plt.ylabel('Total Count')
plt.xlim(0,5.1)
plt.ylim(0,3200000)

plt.show()


#To find Top Paid Apps(TFA)
tpa = []
for line in app_data:
    if line[2] != 0:
        tpa.append(line)


x = lambda tpa:tpa[4]
tpa.sort(key=x,reverse=True)

tpa_user_app_rating =[]

for line in tpa:
    tpa_user_app_rating.append(line[4])

x = lambda tpa:tpa[3]
tpa.sort(key=x,reverse=True)

tpa_cont_rating = []

for line in tpa:
    tpa_cont_rating.append(line[3])

plt.figure(9)
plt.title("Top User rating for all Paid Apps")
sns.countplot(tpa_user_app_rating)
plt.xlabel("User Rating")
plt.ylabel("Total Count")


plt.figure(10)
plt.title("Comparing User Rating with Content Rating in Paid Apps")
plt.plot(tpa_user_app_rating,tpa_cont_rating,'ob')
plt.xlabel("Content Rating")
plt.ylabel('Total Count')
plt.xlim(0,5.1)
plt.ylim(0,250000)
plt.show()


#Top 10 paid Games based on content rating
print(tpg[0:10])
#Top 10 Free Games based on content rating
print(tfg[0:10])

#Top 10 Free Apps based on content rating
print(tfa[0:10])
#Top 10 Paid Apps based on content rating
print(tpa[0:10])

