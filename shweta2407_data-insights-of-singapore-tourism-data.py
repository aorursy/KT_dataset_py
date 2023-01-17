import pandas as pd

import numpy as np

import seaborn as sns

from wordcloud import WordCloud

import matplotlib.pyplot as plt



data = pd.read_excel('../input/singapore-tourism-data/mock survey data.xlsx')

data.head()
sns.distplot(data['Year'])

plt.xticks([2014, 2015])

plt.xlabel('YEAR')

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(data['R.mth'], palette='Set2')

plt.xlabel('Month')

plt.ylabel('Visitors')

plt.title('Number of Visitors per Month')

plt.show()
city = pd.DataFrame(data['City_residence'].value_counts())[:10]

city_names = city.index

count = city['City_residence']



plt.style.use('ggplot')

plt.figure(figsize=(9,9))

plt.rc('font', size=10)

plt.pie(count, autopct='%1.0f%%', labels = city_names, pctdistance=1.1, labeldistance=1.2)

plt.title('% of the visitors from each city')

plt.show()
x = pd.DataFrame(data['City_residence'].value_counts()[:10]).index.tolist()

y = pd.DataFrame(data['City_residence'].value_counts()[:10])['City_residence'].tolist()

plt.style.use('default')

plt.figure(figsize=(13,6))

sns.barplot(x=x, y=y, palette='Set2')

plt.xlabel('Cities')

plt.ylabel('Visitors')

plt.title('Visitors Per City')

plt.show()
# top 10 purposes

ind_list = ['Holiday', 'Visit_Relatives', 'Business','Treatment','Personal', 'Accompany_patient','shopping','Bus_meetings', 'Sightseeing','Stopover']

d = dict()

for i, ind in enumerate(pd.DataFrame(data['Purpose'].value_counts()[:10]).index.tolist()):

    d[ind] = ind_list[i]



wordcloud = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 2).generate(' '.join(i for i in ind_list)) 

# plot the WordCloud image                        

plt.figure(figsize = (5, 5), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()
x = pd.DataFrame(data['Purpose'].value_counts()[:10]).rename(index=d).index.tolist()

y = pd.DataFrame(data['Purpose'].value_counts()[:10])['Purpose'].tolist()



plt.figure(figsize=(18,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Set2')

plt.xlabel('Purpose')

plt.ylabel('Visitors')

plt.title('Top Purposes of Visitors')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='Yes']['Purpose'].value_counts()[1:10]).rename(index=d).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='Yes']['Purpose'].value_counts()[1:10])['Purpose'].tolist()



plt.figure(figsize=(15,6))

plt.style.use('seaborn')

plt.rc('font', size=10)

plt.pie(y,autopct='%1.0f%%', labels =x, pctdistance=1.1, labeldistance=1.3)

plt.title('PURPOSE OF FIRST VISIT')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='No']['Purpose'].value_counts()[1:10]).rename(index=d).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='No']['Purpose'].value_counts()[1:10])['Purpose'].tolist()



plt.figure(figsize=(15,6))

plt.style.use('ggplot')

plt.rc('font', size=10)

plt.pie(y,autopct='%1.0f%%', labels =x, pctdistance=1.1, labeldistance=1.3)

plt.title('PURPOSE OF SECOND VISIT')

plt.show()
x = pd.DataFrame(data['langint'].value_counts()[:5]).index.tolist()

y = pd.DataFrame(data['langint'].value_counts()[:5])['langint'].tolist()



plt.figure(figsize=(10,6))

plt.style.use('default')

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Set2')

plt.xlabel('Langugae')

plt.ylabel('Visitors')

plt.title('Which Language do Visitors speak most ?')

plt.show()
c = pd.DataFrame(data['1st_visit'].value_counts())

m = c.index

count = c['1st_visit']



plt.style.use('default')

plt.figure(figsize=(5,5))

plt.pie(count, autopct='%1.0f%%', labels =m, pctdistance=1.1, labeldistance=1.3)

plt.legend(m, loc='upper right')

plt.title('Is it their First visit ?')

plt.show()
x = pd.DataFrame(data['length_stay'].value_counts()[:10]).index.tolist()

y = pd.DataFrame(data['length_stay'].value_counts()[:10])['length_stay'].tolist()



plt.figure(figsize=(15,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Set2')

plt.xlabel('')

plt.ylabel('Visitors')

plt.title('For how many days do the visitor stay?')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='Yes']['length_stay'].value_counts()).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='Yes']['length_stay'].value_counts())['length_stay'].tolist()



plt.figure(figsize=(15,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Set2')

plt.ylabel('Visitors')

plt.title('For how many days do the FIRST time visitor stay?')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='No']['length_stay'].value_counts()).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='No']['length_stay'].value_counts())['length_stay'].tolist()



plt.figure(figsize=(15,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Set1')

plt.ylabel('Visitors')

plt.title('For how many days do the SECOND time visitor stay?')

plt.show()
sns.countplot(data['f1_gender'], palette='Wistia')

plt.xlabel('GENDER')

plt.ylabel('Visitors')

plt.title('Gender of Visitors')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='Yes']['travel_type'].value_counts()).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='Yes']['travel_type'].value_counts())['travel_type'].tolist()



# plt.figure(figsize=(15,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Blues')

# plt.xlabel('')

plt.ylabel('Visitors')

plt.title('TRAVEL TYPE OF FIRST TIME VISITORS')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='No']['travel_type'].value_counts()).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='No']['travel_type'].value_counts())['travel_type'].tolist()



# plt.figure(figsize=(15,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Greens')

# plt.xlabel('')

plt.ylabel('Visitors')

plt.title('TRAVEL TYPE OF SECOND TIME VISITORS')

plt.show()
ind_list = ['homemaker','professional','startups','MD/CEO','exceutive','whitecollar','Others','retired','bluecollar','medium_startups','large_startups']

d = dict()

for i, ind in enumerate(pd.DataFrame(data['f3_occupation'].value_counts()[:10]).index.tolist()):

    d[ind] = ind_list[i]

x = pd.DataFrame(data['f3_occupation'].value_counts()[:10]).rename(index=d).index.tolist()

y = pd.DataFrame(data['f3_occupation'].value_counts()[:10])['f3_occupation'].tolist()



plt.figure(figsize=(15,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='autumn')

plt.xlabel('Occupation')

plt.ylabel('Visitors')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='No']['f3_occupation'].value_counts()[1:10]).rename(index=d).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='No']['f3_occupation'].value_counts()[1:10])['f3_occupation'].tolist()



plt.figure(figsize=(15,6))

plt.rc('font', size=10)

plt.pie(y, autopct='%1.0f%%', labels =x, pctdistance=1.1, labeldistance=1.2)

# sns.barplot(x=x, y=y, palette='Set2')



# plt.xlabel('Profession')

# plt.ylabel('Visitors')

plt.title('SECOND VISIT')

plt.show()
items= [ 'totshopping_$', 'totexp_$']



titles = ['Shopping','Expenditure']

plt.figure(figsize=(15,6))

plt.style.use('ggplot')

for i in range(len(items)):

    plt.subplot(1, 2, i+1)

    thing = pd.DataFrame(data[items[i]].value_counts().sort_values(ascending=False)).reset_index()[0:10]

    money = thing['index'].astype(int)

    count = thing[items[i]]

    plt.pie(count, autopct='%1.0f%%', labels =money, pctdistance=1.1, labeldistance=1.2)

    plt.title('Money Spent on {}'.format(titles[i]))

plt.show()
items= ['shop_$fash',

       'shop_$jew', 'shop_$wat', 'shop_$well', 'shop_$food', 'shop_$gift',

       'shop_$ctec', 'shop_$anti', 'shop_$oth', 'shop_$any']



titles = ['Fashion','Jewellery','Wat','Well','Food','Gift','ctech','antique','others','anything']

plt.figure(figsize=(20,30))



for i in range(len(items)):

    plt.subplot(5, 3, i+1)

    thing = pd.DataFrame(data[items[i]].value_counts().sort_values(ascending=False)).reset_index()[1:11]

    money = thing['index'].astype(int)

    count = thing[items[i]]

    plt.pie(count, autopct='%1.0f%%', labels =money, pctdistance=1.1, labeldistance=1.2)

    plt.title('Money Spent on {}'.format(titles[i]))

plt.show()
ind_list = ['Mandarin_Orchard','VHotelLavender','YorkHotel','IbisSingapore',

            'MarinaBay','TheElizabeth','OtherHotels','RoyalPlaza','ConcordeHotel','HotelMichael']

d = dict()

for i, ind in enumerate(pd.DataFrame(data['MainHotel'].value_counts()[:10]).index.tolist()):

    d[ind] = ind_list[i]



x = pd.DataFrame(data['MainHotel'].value_counts()[:10]).rename(index=d).index.tolist()

y = pd.DataFrame(data['MainHotel'].value_counts()[:10])['MainHotel'].tolist()



plt.style.use('default')

plt.figure(figsize=(25,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Set2')

plt.xlabel('Hotels')

plt.ylabel('Visitors')

plt.title('Which hotel is preferred by the visitors most?')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='Yes']['MainHotel'].value_counts()[:10]).rename(index=d).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='Yes']['MainHotel'].value_counts()[:10])['MainHotel'].tolist()



plt.figure(figsize=(15,10))

# plt.rc('font', size=12)

plt.pie(y, autopct='%1.0f%%', labels = x, pctdistance=1.1, labeldistance=1.2)

plt.title('Which Hotel visitors prefer for their first visit most?')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='No']['MainHotel'].value_counts()[:10]).rename(index=d).index.tolist()

y = pd.DataFrame(data[data['1st_visit']=='No']['MainHotel'].value_counts()[:10])['MainHotel'].tolist()



plt.figure(figsize=(15,10))

# plt.rc('font', size=12)

plt.style.use('ggplot')

plt.pie(y, autopct='%1.0f%%', labels = x, pctdistance=1.1, labeldistance=1.2)

plt.title('Which Hotel visitors prefer for their second visit most?')

plt.show()
x = pd.DataFrame(data[ (data['1st_visit']=='Yes') & (data['f3_occupation']== 'Homemaker (Full time)') ]['MainHotel'].value_counts()[:3]).rename(index=d).index.tolist()

y = pd.DataFrame(data[(data['1st_visit']=='Yes') & (data['f3_occupation']== 'Homemaker (Full time)') ]['MainHotel'].value_counts()[:3])['MainHotel'].tolist()



plt.figure(figsize=(10,5))

plt.style.use('default')

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Oranges')

plt.xlabel('Hotels')

plt.ylabel('Homemakers')

plt.title('')

plt.show()
x = pd.DataFrame(data[ (data['1st_visit']=='No') & (data['f3_occupation']== 'Homemaker (Full time)') ]['MainHotel'].value_counts()[:3]).rename(index=d).index.tolist()

y = pd.DataFrame(data[(data['1st_visit']=='No') & (data['f3_occupation']== 'Homemaker (Full time)') ]['MainHotel'].value_counts()[:3])['MainHotel'].tolist()



plt.figure(figsize=(6,5))

plt.style.use('default')

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Purples')

plt.xlabel('Hotels')

plt.ylabel('Homemakers')

plt.title('')

plt.show()
x = pd.DataFrame(data[ (data['1st_visit']=='Yes') & (data['f3_occupation']== 'Businessman (small company, <50 people)') ]['MainHotel'].value_counts()[:3]).rename(index=d).index.tolist()

y = pd.DataFrame(data[(data['1st_visit']=='Yes') & (data['f3_occupation']== 'Businessman (small company, <50 people)') ]['MainHotel'].value_counts()[:3])['MainHotel'].tolist()



plt.figure(figsize=(6,5))

plt.style.use('default')

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='twilight')

plt.title('SECOND VISIT')

plt.show()
x = pd.DataFrame(data[ (data['1st_visit']=='No') & (data['f3_occupation']== 'Businessman (small company, <50 people)') ]['MainHotel'].value_counts()[:3]).rename(index=d).index.tolist()

y = pd.DataFrame(data[(data['1st_visit']=='No') & (data['f3_occupation']== 'Businessman (small company, <50 people)') ]['MainHotel'].value_counts()[:3])['MainHotel'].tolist()



plt.figure(figsize=(6,5))

plt.style.use('default')

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Greys')

plt.title('SECOND VISIT')

plt.show()
x = pd.DataFrame(data['travel_companion.1'].value_counts()[:5]).index.tolist()

y = pd.DataFrame(data['travel_companion.1'].value_counts()[:5])['travel_companion.1'].tolist()



plt.figure(figsize=(12,6))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Set2')

plt.xlabel('Travelling Partner')

plt.ylabel('Visitors')

plt.title('Which travelling partner is preferred by visitors most?')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='Yes' ]['travel_companion.1'].value_counts()[:3]).rename(index=d).index.tolist()

y = pd.DataFrame(data [data['1st_visit']=='Yes' ]['travel_companion.1'].value_counts()[:3])['travel_companion.1'].tolist()



plt.figure(figsize=(6,5))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Wistia')

# plt.xlabel('partners')

# plt.ylabel('Visitors')

plt.title('FIRST TIME')

plt.show()
x = pd.DataFrame(data[data['1st_visit']=='No' ]['travel_companion.1'].value_counts()[:3]).rename(index=d).index.tolist()

y = pd.DataFrame(data [data['1st_visit']=='No' ]['travel_companion.1'].value_counts()[:3])['travel_companion.1'].tolist()



plt.figure(figsize=(6,5))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='PuBu')

# plt.xlabel('partners')

# plt.ylabel('Visitors')

plt.title('SECOND TIME')

plt.show()
x = pd.DataFrame(data[ (data['1st_visit']=='Yes') & (data['f3_occupation']== 'Student') ]['travel_companion.1'].value_counts()[:5]).rename(index=d).index.tolist()

y = pd.DataFrame(data[(data['1st_visit']=='Yes') & (data['f3_occupation']== 'Student') ]['travel_companion.1'].value_counts()[:5])['travel_companion.1'].tolist()



plt.figure(figsize=(8,5))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Reds')

# plt.xlabel('Travelling Partners')

# plt.ylabel('Students')

plt.title("Student's Favorite Partner FIRST VISIT")

plt.show()
x = pd.DataFrame(data[ (data['1st_visit']=='No') & (data['f3_occupation']== 'Student') ]['travel_companion.1'].value_counts()[:5]).rename(index=d).index.tolist()

y = pd.DataFrame(data[(data['1st_visit']=='No') & (data['f3_occupation']== 'Student') ]['travel_companion.1'].value_counts()[:5])['travel_companion.1'].tolist()



plt.figure(figsize=(8,5))

plt.rc('font', size=10)

sns.barplot(x=x, y=y, palette='Greens')

# plt.xlabel('Travelling Partners')

# plt.ylabel('Students')

plt.title("Student's Favorite Partner SECOND VISIT")

plt.show()
for i in data['totexp_$'].sort_values(ascending=False).head(10):

    print(data[data['totexp_$'] == i]['f3_occupation'])

    print(data[data['totexp_$'] == i]['travel_companion.1'])

    print(data[data['totexp_$'] == i]['f1_gender'])

    print(data[data['totexp_$'] == i]['City_residence'])

    print(data[data['totexp_$'] == i]['f4_industry'])

    print(data[data['totexp_$'] == i]['1st_visit'])

    print()