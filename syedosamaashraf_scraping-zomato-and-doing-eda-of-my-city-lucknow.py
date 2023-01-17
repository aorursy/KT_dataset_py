import pandas as pd                  #For data eploration.
from bs4 import BeautifulSoup        #For scraping the website.
import requests                      #For sending the requests to the website.
from nltk import word_tokenize       #For cleaning the text data.
import json                          #For reading json data
from tqdm.notebook import tqdm       #For checking the loop timinngs.
import seaborn as sns                #For visualization.
from matplotlib import pyplot as plt #For visualization.
import folium                        #For Map Visualization
import re                            #Regular Expression
import operator
headers = {'User-Agent': 'Mozilla/5.0'}
name_=[]  #For saving Restaurant's name
online_=[] #For saving Restaurant's 
title_=[]  #For saving Restaurant's type
area_=[]   #For saving Restaurant's area
rating_=[] #For saving Restaurant's rating
votes_=[]  #For saving Restaurant's votes
add_=[]    #For saving Restaurant's address
cuisines_=[]#For saving Restaurant's cuisines
cf2_=[]    #For saving Restaurant's cost for 2 persons
hours_=[]  #For saving Restaurant's timings
link_=[]   #For saving Restaurant's web link
json_=[]   #For saving Restaurant's json file
dish_=[]   #For saving Restaurant's dish   
price_=[]  #For saving Restaurant'sdish price
contact_=[] #For saving Restaurant's Contact number
url='https://www.zomato.com/lucknow/restaurants?page='
for page in tqdm(range(1,265)):
    print(f'{url}{page}')
    r=requests.get(f'{url}{page}',headers=headers)
    print(r)
    soup=BeautifulSoup(r.text,'html.parser')
    cards=soup.find_all(class_='card search-snippet-card search-card')
    for card in cards:
        try:
            contact=card.find(class_='item res-snippet-ph-info')['data-phone-no-str']
            contact_.append(contact)
        except:
            contact_.append(0)
        try:
          content=card.find(class_='content')
          online=card.find('span',class_='fontsize4 bold action_btn_icon o2_closed_now')
          if online:
            online_.append('Delivery Available')
          else:
            online_.append('Delivery Not Available')
        except:
          online_.append('Delivery Not Available')
        try:
            title=content.find(class_='res-snippet-small-establishment mt5').find('a').getText()
            title_.append(title)
        except:
            title_.append(0)
        try:
            name=content.find('a',class_='result-title hover_feedback zred bold ln24 fontsize0').getText()
            name_.append(name)
        except:
            name_.append(0)
        try:
            area=content.find(class_='row').find(class_='row').find('a').find_next('a').find_next('a').getText()
            area_.append(area)
        except:
            area_.append(0)
        try:
            rating=content.find(class_='row').find(class_='row').find(class_='ta-right floating search_result_rating col-s-4 clearfix').find('div').getText()
            rating_.append(rating)
        except:
            rating_.appned(0)
        try:
            votes=content.find(class_='row').find(class_='row').find(class_='ta-right floating search_result_rating col-s-4 clearfix').find('span').getText()
            votes_.append(votes)
        except:
            votes_.append(0)
        try:
            add=content.find(class_='row').find(class_='row').find_next(class_='row').find('div').getText()
            add_.append(add)
        except:
            add_.append(0)
        try:    
            cuisines=content.find(class_='search-page-text clearfix row').find(class_='clearfix').find('span').find_next('span').getText()
            cuisines_.append(cuisines)
        except:
            cuisines_.appned(0)
        try:
            cf2=content.find(class_='search-page-text clearfix row').find(class_='res-cost clearfix').find('span').find_next('span').getText()
            cf2_.append(cf2)
        except:
            cf2_.append(0)
        try:    
            hours=content.find(class_='search-page-text clearfix row').find(class_='res-timing clearfix')('div')[0].getText()
            hours_.append(hours)
        except:
            hours_.append(0)
        try:
            link=content.find('a',class_='result-title hover_feedback zred bold ln24 fontsize0')['href']
            link_.append(link)
        except:
            link_.appned(0)
        r=requests.get(f'{link}/order',headers=headers)
        soup=BeautifulSoup(r.text,'html.parser')
        try:
            _json=soup.find('script',type='application/ld+json').find_next('script',type='application/ld+json')
            _json=_json.getText()
            _json=json.loads(_json)
            json_.append(_json)
        except:
            json_.append(0)
        if online:
            try:
                    
                dish=soup.find(id='root').find_all('h4')
                l=[]
                for i in dish:
                    i=i.getText()
                    l.append(i)
                dish_.append(l)
                price=soup.find_all('span',class_='sc-17hyc2s-1 fnhnBd')
                l=[]
                for i in price:
                    i=i.getText()
                    l.append(i)
                price_.append(l)
            except:
                dish_.append(0)
                price_.append(0)
        else:
            
            dish_.append(0)
            price_.append(0)
data={'Name':name_,'Contact':contact_,'Online':online_,'Title':title_,'Area':area_,'Rating':rating_,'Votes':votes_,'Add':add_,'Cuisines':cuisines_,'CF2':cf2_,'Hours':hours_,'Link':link_,'Json':json_,'Dish':dish_,'Price':price_}
df=pd.DataFrame(data)
df.to_csv('Zomato_LucknowLatest.csv')
df=pd.read_csv('../input/Zomato_LucknowLatest.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)#If reading from CSV
df
df.isnull().sum()
count=0
for name,rating,vote,cf2,hours in tqdm(zip(df.Name,df.Rating,df.Votes,df.CF2,df.Hours)):
    l=word_tokenize(name)
    string=' '.join(l).replace(" 's","'s")
    df.iloc[count,0]=string
    l=word_tokenize(rating)
    string=''.join(l)
    df.iloc[count,5]=string
    try:
        string=vote.replace(' votes','')
        df.iloc[count,6]=string
    except:
        pass
    string=''.join(re.findall(r'[0-9]', cf2))
    df.iloc[count,9]=string
    string=(''.join(re.findall(r'.', hours))).strip()
    df.iloc[count,10]=string
    count+=1
df
top=[]
import ast
l=df.Dish.tolist()
for j in l:
    try:
        j=j.strip('][').split(', ')
        for i in j:
            i=i.replace("\'","").replace("\'","")
            if i=='Bestsellers' or i=='Bestseller':
                index=0
                c=0
                for k in j:
                    c+=1
                    if c>=2:
                        top.append(k)
                        index+=1
                    if index==10:
                        break
    except:
        pass
top[0:10]
tops={}
for i in top:
    tops[i]=top.count(i)
sorted_d = dict(sorted(tops.items(), key=operator.itemgetter(1),reverse=True))
sorted_d.pop('"Restaurant\'s Recommendations"')
sorted_d.pop("'Starters'")
sorted_d.pop("'Combos'")
sorted_d.pop("'Bestsellers'")
sorted_d
dish_name=[]
number=[]
for count,key in enumerate(sorted_d.keys()):
    dish_name.append(key)
    number.append(sorted_d[key])
    if count==14:
        break
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
ax=plt.figure(figsize=(12,8))
ax=sns.set(style="darkgrid")
ax=sns.barplot(y=dish_name,x=number);
ax.set(xlabel='No. of Restaurants where they are mentioned as Bestsellers',ylabel='Dishes',title="Top Dishes 15 w.r.t. Restaurant's Bestsellers");
plt.tight_layout()
df.Votes=df.Votes.astype(int)
ax=plt.figure(figsize=(12,8))
ax=sns.set(style="darkgrid")
ax=sns.barplot(y=df.sort_values(by='Votes', ascending=False).Name[0:15],x=df.sort_values(by='Votes', ascending=False).Votes[0:15])
ax.set(xlabel="No. of Votes",ylabel="Restaurant's",title="Top 15 Popular Restaurant's of the City");
plt.tight_layout()
df.Online.value_counts()
slices=[2122,1830]
plt.figure(figsize=(12,8))
plt.pie(slices, labels =['Delivery Not Available','Delivery Available'],startangle = 90,autopct='%1.0f%%', shadow = True, explode = (0, 0.1,))
plt.title('Percentage of Restaurant w.r.t. to Delivery Option',bbox={'facecolor':'0.8', 'pad':5})
plt.show()
df.CF2=df.CF2.astype(int)
ax=plt.figure(figsize=(12,8))
ax=sns.set(style="darkgrid")
ax=sns.barplot(y=df.sort_values(by='CF2', ascending=False).Name[0:15],x=df.sort_values(by='CF2', ascending=False).CF2[0:15])
ax.set(xlabel="Price for 2 Persons",ylabel="Restaurant's",title="15 Most Expensive Restaurant's in the City");
plt.tight_layout()
name=[]
num=[]
for i in df.groupby('Area').size().sort_values( ascending=False)[0:14].index:
    name.append(i)
for i in df.groupby('Area').size().sort_values( ascending=False)[0:14]:
    num.append(i)
name.pop(3)
num.pop(3)
ax=plt.figure(figsize=(12,8))
ax=sns.set(style="darkgrid")
ax=sns.barplot(y=name,x=num)
ax.set(xlabel="No. of Restaurant's",ylabel="Area",title="Top 15 Area's with the maximum Restaurant");
plt.tight_layout()
ax=plt.figure(figsize=(12,8))
ax=sns.set(style="darkgrid")
ax=sns.countplot(y=df.Title)
ax.set(xlabel="No. of Restaurant's",ylabel="Types",title="Types of Restaurant's in the City");
plt.tight_layout()

df.Json[1]
longitude=[]
latitude=[]
for c,i in enumerate(df.Json):
    try:
        j=json.loads(i.replace("\'", "\""))
        longitude.append(j['geo']['longitude'])
        latitude.append(j['geo']['latitude'])
    except:
        try:
            a=i.replace("\'","\"").replace('\"s','',1)
            a=a.replace("\'", "\"")
            j=json.loads(a)
            longitude.append(j['geo']['longitude'])
            latitude.append(j['geo']['latitude'])
        except:
            longitude.append(str(0))
            latitude.append(str(0))
for count,i in enumerate(longitude):
    if isinstance(i,int):
        longitude[count]=float(i)
    elif i.replace('.', '', 1).isdigit() :
        longitude[count]=float(i)
      
    else:
        longitude[count]=0.0
for count,i in enumerate(latitude):
    if isinstance(i,int):
        latitude[count]=float(i)
    elif i.replace('.', '', 1).isdigit() :
        latitude[count]=float(i)
      
    else:
        latitude[count]=0.0

df2=pd.concat([df,pd.DataFrame(list(zip(longitude,latitude)),columns=['Longitude','Latitude'])],axis=1)
df2=df2.drop(df2.index[df2[df2.Latitude==0.0].index.values],axis=0)
df2=df2.drop(df2.index[df2[df2.Longitude==0.0].index.values],axis=0)

map_ = folium.Map(location=[26.8542,80.9448], zoom_start=10)
locs = df2.sort_values(by='Votes', ascending=False)[['Latitude', 'Longitude']][0:50]
loc_list = locs.values.tolist()

# To display all data use the following two lines, but, since your data has
# so many points, this process will be time-consuming.
for point in range(0, len(loc_list)):
    folium.Marker(loc_list[point]).add_to(map_)

# To display first 1000 points
# for point in range(0, 1000):
#     folium.Marker(loc_list[point]).add_to(map_)

map_