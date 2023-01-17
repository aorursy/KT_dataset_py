import pandas as pd
apple = pd.read_csv('../input/AppleStore.csv')
apple.columns = ['XX','id','apps','size_bytes','currency','price','rating_count_tot','rating_count_ver','user_rating','user_rating_ver','ver','cont_rating','prime_genre','sup_devices','ipadSC','lang_support','vpp_lic']
apple.drop(labels = ['XX'], axis = 1, inplace = True)
apple.head(5)
import matplotlib.pyplot as plt

x = apple['price']

apple.price.value_counts()

df = apple[apple['price'] != 0]


df.head(10)


plt.figure(figsize = (15,10))
plt.style.use("fivethirtyeight")
dfa = df[df['price'] < 49.99]
plt.xlabel('Price($)')
dfa.price.plot(kind = 'hist',log = True)
plt.figure(figsize = (25,15))
import seaborn as sns
sns.stripplot(data = dfa,y = 'price', x = 'prime_genre')
dfa.prime_genre.value_counts()
df_games = dfa[dfa['prime_genre'] == 'Games']
df_edu = dfa[dfa['prime_genre'] == 'Education']
df_pv = dfa[dfa['prime_genre'] == 'Photo & Video']
df_ut = dfa[dfa['prime_genre'] == 'Utilities']
df_hf = dfa[dfa['prime_genre'] == 'Productivity']
plt.figure(figsize = (15,15))
yrange = [0,30]

# add colors later
# add labels later

plt.subplot(1,5,1)
plt.ylim([0,25])
sns.stripplot(data = df_games, y = 'price', jitter = True, color = '#604e85', )
plt.xlabel('Games')

plt.subplot(1,5,2)
plt.ylim(yrange)
sns.stripplot(data = df_edu, y = 'price', jitter = True , color = '#c55292')
plt.xlabel('Education')

plt.subplot(1,5,3)
plt.ylim(yrange)
sns.stripplot(data = df_pv, y = 'price', jitter = True, color = '#d053cb')
plt.xlabel('Photo & Video')


plt.subplot(1,5,4)
plt.ylim(yrange)
sns.stripplot(data = df_ut, y = 'price', jitter = True, color = '#431e54')
plt.xlabel('Utilities')

plt.subplot(1,5,5)
plt.ylim(yrange)
sns.stripplot(data = df_hf, y = 'price', jitter = True, color = '#EF1A78')
plt.xlabel('Productivity')


plt.show()

l = df.prime_genre.value_counts().index[:4]

def catag(x):
    if x in l:
        return x
    else:
        return "Other"
    
    
apple['broad_genre'] = apple.prime_genre.apply(lambda x: catag(x))
apple.drop(labels = ['broad_genre'], axis = 1)

data = pd.DataFrame(apple['broad_genre'].value_counts())

data.sort_index()

free  = apple[apple['price'] == 0].broad_genre.value_counts().to_frame(name = 'free').sort_index()
free
paid = apple[apple['price']!= 0].broad_genre.value_counts().to_frame(name =  'paid').sort_index()
paid
data_price = data.join(free).join(paid)
data_price
data_price.columns = ['total','free','paid']

data_price['free%'] = data_price['free'] * 100/data_price['total']
data_price['paid%'] = data_price['paid']*100/data_price['total']


data_price
f = data_price['free'].sort_index()
p = data_price['paid'].sort_index()
df_price = data_price[['free%','paid%']]

plt.figure(figsize =(15,6))
free_tuple = tuple(data_price['free%'].tolist())
paid_tuple = tuple(data_price['paid%'].tolist())

import numpy as np
import matplotlib.pyplot as plt



N = 5

ind = np.arange(N)    # the x locations for the groups
width = 0.55      # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, free_tuple, width,color  ='#5fc497')
p2 = plt.bar(ind, paid_tuple, width,
             bottom=free_tuple, color = '#44a8c5')

plt.ylabel('Percentage')
plt.title('Scores by Categories')
plt.xticks(ind, ('Games', 'Other', 'Entertainment', 'Education', 'Photo & Video'))
plt.legend((p1[0], p2[0]), ('Free', 'Paid'))



plt.show()





df_price.T.plot.pie(subplots = True, figsize = (20,4), colors = ['#5fc497','#44a8c5'] )
plt.show()
def check(x):
    if(x == 0):
        return "free"
    else:
        return "paid"

apple['Status'] = pd.DataFrame(apple.price.apply(lambda x : check(x)))



plt.figure(figsize = (20,15))
plt.ylim(0,5)
sns.violinplot(data = apple, y = 'user_rating', x = 'broad_genre', hue = 'Status', split = True, scale = 'count', palette= ['#44a8c5','#5fc497'] )
plt.style.use("fast")
plt.xlabel('Categories')
plt.ylabel('User Rating')
plt.title('User Rating Distribution')
plt.show()
def size(x):
    mb = x/1000000
    return mb
    
apple['size_mb'] = apple.size_bytes.apply(lambda x : size(x))
apple.head(5)
plt.figure(figsize = (15,10))
paid_apps_desc = apple[apple['price'] <50]
sns.lmplot(data = paid_apps_desc, x = 'size_mb', y = 'price', col = 'broad_genre',col_wrap= 3,aspect= 1.5
           ,scatter = True,fit_reg = False, hue  = 'broad_genre', legend_out = True, palette = ['#c973d0' , '#8e52b8' , '#4a4f92', '#4a73ab' , '#649ca1'] )



plt.show()

plt.figure(figsize = (20,20))



group_names= apple.broad_genre.value_counts().sort_index().index

group_size = apple.broad_genre.value_counts().sort_index().tolist()

x = ['Free','Paid']
subgroup_names = 5*x
subgroup_size = [f[0],p[0],f[1],p[1], f[2],p[2],f[3],p[3],f[4],p[4]]

col = ['#5fc497','#44a8c5', '#5fc497','#44a8c5' , '#5fc497','#44a8c5', '#5fc497','#44a8c5', '#5fc497','#44a8c5']

# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=2.3, labels=group_names, colors = ['#4a4f92','#8e52b8','#c973d0','#4a73ab','#649ca1']) 
plt.setp( mypie, width=0.5, edgecolor='white')
 
# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius= 2.3, labels=group_names, colors = ['#4a4f92','#8e52b8','#c973d0','#4a73ab','#649ca1']) 
plt.setp( mypie, width=0.5, edgecolor='white')

# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=2.3-0.2, labels=subgroup_names, labeldistance=0.8, colors= col)
plt.setp( mypie2, width=0.5, edgecolor='white')
plt.margins(0,0)
 
# show it
plt.show()

apple[(apple.user_rating_ver  < apple.user_rating) & (apple.rating_count_ver/apple.rating_count_tot > 0.5)].prime_genre.value_counts()
apple[apple.rating_count_ver/apple.rating_count_tot > 0.5].prime_genre.value_counts()