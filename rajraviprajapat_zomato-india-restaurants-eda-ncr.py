# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import squarify
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
### Variable Descrpition
pd.options.mode.chained_assignment = None

# Any results you write to the current directory are saved as output

df_train = pd.read_csv('../input/zomato.csv', encoding='latin-1')
country = pd.read_excel('../input/Country-Code.xlsx', encoding='latin-1')
df_train = pd.merge(df_train,country,on='Country Code')
df_train['rating_cat'] = df_train['Rating text'].map({'Not rated': -1, 'Poor':0, 'Average':2, 'Good':3, 'Very Good':4, 'Excellent':5})
df_train.rename(columns = {'Aggregate rating':'rating_num', 'Has Table booking': 'Book', 'Has Online delivery': 'On_deliver', 'Is delivering now':'Cur_deliver', 
                          'Switch to order menu' : 'Switch_menu', 'Average Cost for two' : 'Avg_cost_two', 'Price range' : 'Pr_range'}, inplace = True)
#df_train.drop(['Rating color', 'Rating text'], axis = 1 ,inplace= True)

print('Original Train Row: ', df_train.shape[0])
#df_test = df_train.loc[df_train.rating_cat == -1, :].copy()
df_train = df_train.loc[df_train.rating_cat != -1, :].copy()
print('Train Row : ', df_train.shape[0])
#print('Test Row : ', df_test.shape[0])
India_data =df_train[df_train.Country == 'India']
result = df_train['Country'].value_counts().reset_index()
India_data.head()
Cities = India_data['City'].value_counts().reset_index()
Top_City = Cities.head(10)
plt.pie(Cities['City'],labels=Cities['index'],autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
NCR = India_data[(India_data['City']=='New Delhi') | (India_data['City']=='Gurgaon') | (India_data['City']=='Noida')]
NCR = NCR.drop(['Address','Locality','Locality Verbose','Country Code','Country'],axis=1)
NCR.head(5)
NCR.describe()
rating = ['rating_num', 'rating_cat']

f, ax = plt.subplots(1,1, figsize = (12, 4))
sns.countplot(NCR['rating_num'], ax = ax, color = 'k')
ax.tick_params('x', rotation = 70)
ax.set_title('Rating Distribtion')
plt.show()
f, ax = plt.subplots(1,1, figsize = (12, 4))
sns.barplot(NCR['rating_num'],NCR['Votes'], ax=ax)
ax.set_title('Number of vote Distribution over ratings')
plt.show()
print('ID # / Name #')
NCR[['Restaurant ID','Restaurant Name']].apply(pd.Series.nunique, axis = 0)
NCR_loc = India_data.loc[India_data['City']=='Gurgaon',['Latitude', 'Longitude']]
#NCR_loc = NCR[['Latitude', 'Longitude']]
NCR_loc = NCR_loc.loc[India_data['Latitude']!=0,['Latitude', 'Longitude']]
#Not counting for Restaurants without proper Geo Data
map_F = folium.Map( location=[28.201513, 76.989084],zoom_start = 10)
for i, (lat, lon) in enumerate(NCR_loc.values): folium.Marker([lat, lon]).add_to(map_F)
map_F



NCR['rating_cat'] = NCR['Rating text'].map({'Not rated': -1, 'Poor':0, 'Average':2, 'Good':3, 'Very Good':4, 'Excellent':5})
NCR.rename(columns = {'Aggregate rating':'rating_num', 'Has Table booking': 'Book', 'Has Online delivery': 'On_deliver', 'Is delivering now':'Cur_deliver', 
                          'Switch to order menu' : 'Switch_menu', 'Average Cost for two' : 'Avg_cost_two', 'Price range' : 'Pr_range'}, inplace = True)
#NCR.drop(['Rating color', 'Rating text'], axis = 1 ,inplace= True)

NCR_test = NCR.loc[NCR.rating_cat == -1, :].copy()
NCR = NCR.loc[NCR.rating_cat != -1, :].copy()

NCR = NCR.loc[NCR['Longitude'] != 0, :]


tmp = NCR['rating_num'].map(np.round)
a = np.full(tmp.shape[0], False, dtype = bool)
print('Round')
((tmp - NCR['rating_cat']).map(np.round)).value_counts()

sys_check = NCR[['rating_num', 'rating_cat', 'Votes']].copy()
sys_check['distorted'] = (NCR['rating_num'] - NCR['rating_cat']).map(np.round)
sys_check['diff'] = sys_check['rating_num'] - sys_check['rating_cat']
g = sns.FacetGrid(data =sys_check, col = 'distorted')
g = g.map(plt.scatter, 'diff', 'Votes', alpha = 0.5)
plt.show()
NCR = NCR.loc[sys_check['distorted'] != 2, :]
NCR['rating_cat'] = NCR['rating_num'].round(0).astype(int)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=0).fit(NCR[['Longitude', 'Latitude']])
NCR['pos'] = kmeans.labels_
pop_local = NCR.groupby('pos')['Longitude', 'Latitude', 'rating_num'].agg({'Longitude':np.mean, 'Latitude':np.mean, 'rating_num':np.median}).reset_index()

with plt.style.context('bmh', after_reset=True):
    pal = sns.color_palette('rainbow', 7)
    plt.figure(figsize = (8,6))
    for i in range(7):
        ix = NCR.pos == i
        plt.scatter(NCR.loc[ix, 'Latitude'], NCR.loc[ix, 'Longitude'], color = pal[i], label = str(i))
        plt.text(pop_local.loc[i, 'Latitude'], pop_local.loc[i, 'Longitude'], str(i) + ': '+str(pop_local.loc[i, 'rating_num'].round(2)), fontsize = 14, color = 'black')
    plt.title('KMeans New Delhi Median Rating')
    plt.legend()
    plt.show()
    
votes_area = NCR.groupby('pos').agg({'Votes': [np.sum, np.mean]})
votes_area.columns = votes_area.columns.droplevel(0)
votes_area.reset_index(inplace = True)
plt.figure(figsize = (8,4))
ax = plt.subplot(1,2,1)
sns.barplot(x = 'pos', y = 'sum', data =votes_area, palette = sns.cubehelix_palette(n_colors = 7, start = 2.4, rot = .1), ax = ax)
ax.set_title('Summation Votes')

ax = plt.subplot(1,2,2)
sns.barplot(x = 'pos', y = 'mean', data =votes_area, palette = sns.cubehelix_palette(n_colors = 7, start = 3, rot = .1), ax = ax)
ax.set_title('Mean Votes')
plt.show()
with plt.style.context('bmh', after_reset=True):
    plt.figure(figsize = (12,9))

    cat = sorted(NCR['rating_cat'].unique())
    ax = plt.subplot2grid((4,5), (0,0), colspan = 3, rowspan = 3)
    pal = sns.color_palette('Set1', len(cat))
    ix0 = NCR.rating_cat.isin((0,2))
    ix2 = NCR.rating_cat.isin((2,3))
    ix4 = NCR.rating_cat.isin((4,5))
      

    ax.scatter(NCR.loc[ix2, 'Latitude'], NCR.loc[ix2, 'Longitude'], color = 'gray', label = '2-3', alpha = 0.8)
    ax.scatter(NCR.loc[ix4, 'Latitude'], NCR.loc[ix4, 'Longitude'], color = 'red', label = '4-5', alpha = 0.8)
    ax.scatter(NCR.loc[ix0, 'Latitude'], NCR.loc[ix0, 'Longitude'], color = 'cyan', label = '0-2', alpha = 0.8,s=10)
    ax.legend()
    ax.set_title('Rating Dot')

    ax = plt.subplot2grid((4,5), (0,3), colspan = 2, rowspan = 3)
    tmp = NCR['rating_cat'].value_counts().sort_index()
    sns.barplot(tmp.index, tmp.values, palette= pal)
    ax.set_title('Rating Count')


    cat = sorted(NCR['rating_cat'].unique().tolist())
    x_lm = [NCR['Latitude'].min(), NCR['Latitude'].max()]
    y_lm = [NCR['Longitude'].min(), NCR['Longitude'].max()]
    for i, c in enumerate(reversed(cat)):
        ax = plt.subplot2grid((4,5), (3,i))
        ix = NCR.rating_cat == c
        ax.scatter(NCR.loc[ix, 'Latitude'], NCR.loc[ix, 'Longitude'], color = pal[len(cat)-1-i], alpha = 0.8)
        ax.set_xlim(x_lm)
        ax.set_ylim(y_lm)
        ax.set_title(str(c) + '_area')
    plt.subplots_adjust(hspace=0.5, wspace = 0.5)
    plt.show()

NCR['Cuisines'] = NCR['Cuisines'].astype(str)
NCR['Cuisines_num'] = NCR['Cuisines'].apply(lambda x: len(x.split(',')))

from collections import Counter
lst_cuisine = set()
Cnt_cuisine = Counter()
for cu_lst in NCR['Cuisines']:
    cu_lst = cu_lst.split(',')
    lst_cuisine.update([cu.strip() for cu in cu_lst])
    for cu in cu_lst:
        Cnt_cuisine[cu.strip()] += 1

cnt = pd.DataFrame.from_dict(Cnt_cuisine, orient = 'index')
cnt.sort_values(0, ascending = False, inplace = True)


tmp_cnt = cnt.head(10)
tmp_cnt.rename(columns = {0:'cnt'}, inplace = True)
with plt.style.context('bmh'):
    f = plt.figure(figsize = (12,8))
    ax = plt.subplot2grid((2,2), (0,0))
    sns.barplot(x = tmp_cnt.index, y = 'cnt', data = tmp_cnt, ax = ax, palette = sns.color_palette('Blues_d', 10))
    ax.set_title('# Cuisine')
    ax.tick_params(axis='x', rotation=70)
    ax = plt.subplot2grid((2,2), (0,1))
    sns.countplot(NCR['Cuisines_num'], ax=ax, palette = sns.color_palette('Blues_d', NCR.Cuisines_num.nunique()))
    ax.set_title('# Cuisine Provided')
    ax.set_ylabel('')

    ax = plt.subplot2grid((2,2), (1,0), colspan = 2)
    fusion_rate = NCR[['Cuisines_num', 'rating_cat', 'rating_num']].copy()
    fusion_rate.loc[fusion_rate['Cuisines_num'] > 5,'Cuisines_num'] = 5
    fusion_rate = fusion_rate.loc[fusion_rate.rating_cat != -1, :]
    pal = sns.color_palette('Oranges', 11)
    for i in range(1,6):
        num_ix = fusion_rate['Cuisines_num'] == i
        sns.distplot(fusion_rate.loc[num_ix, 'rating_num'], color = pal[i*2], label = str(i), ax = ax)
        ax.legend()
        ax.set_title('Rating Distribution for Cuisines_number')
        

    plt.subplots_adjust(wspace = 0.5, hspace = 0.8, top = 0.85)
    plt.suptitle('Cuisine _ Rating')
    plt.show()        
print('# Unique Cuisine: ', len(lst_cuisine))
plt.pie(tmp_cnt.cnt,labels=tmp_cnt.index,autopct='%.2f', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
south_asia = ['Afghani','Andhra','Awadhi', 'Bengali','Biryani',  'Burmese', 'Chettinad', 'Deli','Goan', 'Gujarati', 'Hyderabadi', 'Kashmiri','Kerala', 'Lebanese','Lucknowi', 'Maharashtrian','Mangalorean',
 'Mithai', 'Modern Indian', 'Moroccan', 'Mughlai','Naga', 'Nepalese', 'North Eastern', 'North Indian','Oriya', 'Parsi', 'Rajasthani','South Indian', 'Tex-Mex','Sri Lankan',
'Indian','Indonesian', 'Tibetan']
africa = ['African']
america = ['American','Mexican', 'South American']
europe = ['British', 'European', 'French','Italian', 'Mediterranean','Middle Eastern', 'Portuguese','Spanish', 'Continental']
west_asia = ['Arabian','Turkish','Iranian']
east_asia = ['Asian', 'Chinese','Japanese', 'Korean', 'Malaysian', 'Thai','Vietnamese',  'Sushi']
kind_food = set(['Bakery', 'Beverages','Burger','Cafe', 'Desserts','Drinks Only', 'Fast Food', 'Finger Food','Healthy Food', 'Ice Cream', 'Juices','Pizza','Raw Meats','Salad','Sandwich','Seafood','Street Food',
 'Tea'])
country = lst_cuisine - set(kind_food)

def kind_country_cu(lst, kind_food = kind_food, country = country):
    lst = lst.split(',')
    tmp1 = [var for var in lst if var in kind_food]
    tmp2 = [var for var in lst if var in country]
    if not tmp1: tmp1 = ['None']
    if not tmp2: tmp2 = ['None']
    return tmp1, tmp2
NCR['food_cu'], NCR['count_cu'] = zip(*NCR['Cuisines'].apply(kind_country_cu))

def get_popular(data):
    Cnt_cuisine = Counter()
    for cu_lst in data:
        for cu in cu_lst:
            if cu != 'None':
                Cnt_cuisine[cu.strip()] += 1
    cnt = pd.DataFrame.from_dict(Cnt_cuisine, orient = 'index')
    cnt.sort_values(0, ascending = False, inplace = True)
    return cnt
print(NCR['count_cu'].head(),NCR['food_cu'].head())
rat_cu = NCR['food_cu'].groupby(NCR['rating_cat'])
tmp_cnt = {}
for i in [2,3,4,5]:
    data = rat_cu.get_group(i)
    tmp_cnt[i] = get_popular(data)

f, ax = plt.subplots(1,4, figsize = (20,4))
for i in [2,3,4,5]:
    tmp = tmp_cnt[i].reset_index()
    tmp.columns = ['cuisine', 'cnt']
    sns.barplot(y = 'cuisine', x = 'cnt', data = tmp.head(10), ax = ax[i-2], palette = sns.cubehelix_palette(n_colors = tmp.shape[0], start = 1, rot = 0.1))
    ax[i-2].set_title(str(i))
    #ax[i-2].tick_params('x', rotation = 90)
plt.show()
rat_cu = NCR['count_cu'].groupby(NCR['rating_cat'])
tmp_cnt = {}
for i in [2,3,4,5]:
    data = rat_cu.get_group(i)
    tmp_cnt[i] = get_popular(data)

f, ax = plt.subplots(1,4, figsize = (12,4))
for i in [2,3,4,5]:
    tmp = tmp_cnt[i].reset_index()
    tmp.columns = ['cuisine', 'cnt']
    sns.barplot(y = 'cuisine', x = 'cnt', data = tmp.head(10), ax = ax[i-2], palette = sns.cubehelix_palette(n_colors = tmp.shape[0], start = 1, rot = 0.1))
    ax[i-2].set_title(str(i))
    #ax[i-2].tick_params('x', rotation = 90)
plt.show()
f = plt.figure(figsize = (12,8))
NCR['Avg_pr_cut'] = pd.cut(NCR['Avg_cost_two'], bins = [0, 200, 500, 1000, 3000, 5000,10000, 800000000], labels = ['<=200', '<=500', '<=1000', '<=3000', '<=5000', '<=10000', 'no limit'])
ax = plt.subplot2grid((1,2), (0,0))
sns.countplot(NCR['Avg_pr_cut'], ax = ax, palette = sns.color_palette('magma', 7))
ax.set_title('Avg Price')
ax.set_xlabel('')
ax.tick_params('x', rotation = 70)
ax = plt.subplot2grid((1,2), (0,1), colspan = 2)
sns.boxplot(x = 'Avg_pr_cut', y = 'rating_num', data = NCR, ax = ax, palette = sns.color_palette('magma', 7))


plt.show()
Cities.head()
