 #Adding Libraries

from matplotlib import pyplot as plt

import pandas as pd

from matplotlib import style

import seaborn as sns

# Airbnb color #FF7674



#Read data

data = pd.read_csv('../input/AirbnbIstanbul.csv')
data.dtypes
data.head()
data.tail()
#Get datas you are going to use

location = data['neighbourhood']

room_type = data['room_type']

price = data['price']

number_of_reviews= data['number_of_reviews']
#Finding room type numbers and setting up

room_type.value_counts()
Private_room = 8565

Entire_home_apt = 7191

Shared_room = 495
labels = 'Private room (8565)', 'Entire home/apt (7191)', 'Shared room(495)'

sizes = [Private_room, Entire_home_apt, Shared_room]

explode = (0.1, 0.1, 0.1)

plt.pie(sizes, labels=labels,autopct='%1.1f%%', textprops=dict(color="white"),startangle=90)

plt.title("Airbnb Istanbul Analytics")

plt.legend(labels)

plt.show()
counties=location.value_counts()

labels ='Beyoglu','Sisli','Fatih','Kadikoy','Besiktas','Uskudar','Esenyurt','Kagithane','Sariyer','Maltepe','Atasehir','Bakirkoy','Bahcelievler','Adalar','Pendik','Umraniye','Basaksehir','Eyup','Kartal','Avcilar','Kucukcekmece','Buyukcekmece','Bagcilar','Beykoz','Zeytinburnu','Beylikduzu','Sile','Gaziosmanpasa','Gungoren','Tuzla','Cekmekoy','Sancaktepe','Silivri','Esenler','Bayrampasa','Sultangazi' ,'Sultanbeyli','Arnavutkoy','Catalca'

explode = (0.1, 0.1, 0.1)

plt.figure(figsize = (15,15))

plt.pie(counties,labels=labels, autopct='%1.1f%%',pctdistance=0.9,textprops={'fontsize': 10} , startangle=90)

plt.title("Airbnb Istanbul Analytics")

plt.show()
#Location - Price

plt.scatter(price,location,color="#FF7674",marker="*")

plt.rcParams['figure.figsize'] = (25,30)

plt.xlim(left=50,right=1000)

plt.ylabel("Location")

plt.xlabel('Price')

plt.title("Airbnb Istanbul Analytics")

plt.show()