import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/AppleStore.csv',encoding='utf-8')

data = pd.read_csv('../input/AppleStore.csv',encoding='utf-8')
print(data.shape)
data.head()
data['isNotFree'] = data['price'].apply(lambda x: 'Payantes' if x > 0 else 'Gratuites')
data['isNotFree'].value_counts().plot.bar(width=0.6,rot=0) 
plt.xlabel('Applications')
plt.ylabel('nombre d"applications')
plt.show()
#fact generator 
print('')
print('Résultats :')
print('')
print ('- '+str(sum(data.price == 0)) + ' Applications sont gratuites ' )
print('')
print ('- '+str(sum(data.price != 0)) + ' Applications sont payantes ' )
print('')
print ('- '+str(sum(data.price > 50)) +' applications parmi les 3141 applications payantes et leurs prix dépassent le 50 $ ( ≈ 145.832 DT ) ')
print('')
print ('--> Les applications les plus chéres sont d"une pourcentage de ' + str(sum(data.price > 50)/len(data.price)*100) +
       " du l'ensemble de applciations publiées ")
data['prime_genre'].value_counts(dropna=False)
s = data.prime_genre.value_counts().index[:4]
def categ(x):
    if x in s:
        return x
    else : 
        return "Others"

data['broad_genre']= data.prime_genre.apply(lambda x : categ(x))
BlueOrangeWapang = ['#fc910d','#fcb13e','#239cd3','#1674b1','#ed6d50']
plt.figure(figsize=(10,10))
label_names=data.broad_genre.value_counts().sort_index().index
size = data.broad_genre.value_counts().sort_index().tolist()
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(size, labels=label_names, colors=BlueOrangeWapang)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
outlier=data[data.price>50][['track_name','price','prime_genre','user_rating']]
outlier.sort_values(by = 'price' ,ascending = False)
# removing
paidapps =data[((data.price<50) & (data.price>0))]
print('--> Max prix : ' + str(max(paidapps.price)))
#paidapps.prime_genre.value_counts()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,15))
plt.subplot(2,1,1)

plt.hist(paidapps.price,log=True)
plt.title('Price distribution of apps ')
plt.ylabel("Frequency")
plt.xlabel("Price Distributions in ($) ")

plt.subplot(2,1,2)
plt.title('Visual price distribution')
sns.stripplot(data=paidapps,y='price',jitter= True,orient = 'h' ,size=6)
plt.show()
display(data[data.price == 49.99])
outlier=data[((data.prime_genre == 'Music')&(data.price<=49.99)&(data.price>=20)&(data.user_rating_ver!=5.0)&(data.user_rating>3.5))][['track_name','price','prime_genre','user_rating','user_rating_ver','cont_rating','size_bytes']]
outlier.sort_values(by = 'price',ascending = False)

yrange = [0,25]
fsize =15

plt.figure(figsize=(15,10))

plt.subplot(4,1,1)
plt.xlim(yrange)
paidapps =data[((data.price<50) & (data.price>0))]
games = paidapps[paidapps.prime_genre=='Games']
sns.stripplot(data=games,y='price',jitter= True , orient ='h',size=6,color='#eb5e66')
plt.title('Games',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,2)
plt.xlim(yrange)
ent = paidapps[paidapps.prime_genre=='Entertainment']
sns.stripplot(data=ent,y='price',jitter= True ,orient ='h',size=6,color='#ff8300')
plt.title('Entertainment',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,3)
plt.xlim(yrange)
edu = paidapps[paidapps.prime_genre=='Education']
sns.stripplot(data=edu,y='price',jitter= True ,orient ='h' ,size=6,color='#20B2AA')
plt.title('Education',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,4)
plt.xlim(yrange)
pv = paidapps[paidapps.prime_genre=='Photo & Video']
sns.stripplot(data=pv,y='price',jitter= True  ,orient ='h',size=6,color='#b84efd')
plt.title('Photo & Video',fontsize=fsize)
plt.xlabel('') 

plt.show()
top_trending_free_games = data[(data["prime_genre"]=="Games")][['track_name','price','prime_genre','user_rating']]
top_trending_free_games.sort_values(by = 'price' ,ascending = False).head()


catégories = data.prime_genre.value_counts().index[:4]
catégories
# reducing the number of categories
import pandas as pd
data = pd.read_csv('../input/AppleStore.csv',encoding='utf-8')
s = data.prime_genre.value_counts().index[:4]
def categ(x):
    if x in s:
        return x
    else : 
        return "Others"

data['broad_genre']= data.prime_genre.apply(lambda x : categ(x))
free = data[data.price==0].broad_genre.value_counts().sort_index().to_frame()
paid = data[data.price>0].broad_genre.value_counts().sort_index().to_frame()
total = data.broad_genre.value_counts().sort_index().to_frame()
free.columns=['free']
paid.columns=['paid']
total.columns=['total']
dist = free.join(paid).join(total)
dist ['paid_per'] = dist.paid*100/dist.total
dist ['free_per'] = dist.free*100/dist.total
dist.sort_values(by = 'total',ascending = False)
plt.figure(figsize=(10,10))
f=pd.DataFrame(index=np.arange(0,10,2),data=dist.free.values,columns=['num'])
p=pd.DataFrame(index=np.arange(1,11,2),data=dist.paid.values,columns=['num'])
final = pd.concat([f,p],names=['labels']).sort_index()
final.num.tolist()

plt.figure(figsize=(20,20))
group_names=data.broad_genre.value_counts().sort_index().index
group_size=data.broad_genre.value_counts().sort_index().tolist()
h = ['Free', 'Paid']
subgroup_names= 5*h
sub= ['#45cea2','#fdd470']
subcolors= 5*sub
subgroup_size=final.num.tolist()


# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=2.5, labels=group_names, colors=BlueOrangeWapang)
plt.setp( mypie, width=1.2, edgecolor='white')

# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.6, labels=subgroup_names, labeldistance=0.7, colors=subcolors)
plt.setp( mypie2, width=0.8, edgecolor='white')
plt.margins(0,0)

# show it
plt.show()
list_free= dist.free_per.tolist()
tuple_free = tuple(list_free)
tuple_paidapps = tuple(dist.paid_per.tolist())
from matplotlib import pyplot as plt
pies = dist[['free_per','paid_per']]
pies.columns=['free %','paid %']
plt.figure(figsize=(15,8))
pies.T.plot.pie(subplots=True,figsize=(20,4),colors=['#45cea2','#fdd470'])
plt.show()
category = list(data.prime_genre.unique())
user_rating = []
for x in category:
    user_rating.append(data[data.prime_genre == x].user_rating.mean())
df_rating = pd.DataFrame({'category': category,'user_rating':user_rating})
new_index = (df_rating['user_rating'].sort_values(ascending=False)).index.values
sorted_df_rating = df_rating.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_df_rating['category'], y=sorted_df_rating['user_rating'])
plt.xticks(rotation= 90)
plt.xlabel('Category')
plt.ylabel('Average User Rating')
plt.title('Categories and Average User Ratings')
BlueOrangeWapang = ['#fc910d','#ff0040','#239cd3','#1674b1','#8000ff']
plt.figure(figsize=(10,10))
label_names=data.cont_rating.value_counts().sort_index().index
size = data.cont_rating.value_counts().sort_index().tolist()
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(size, labels=label_names, colors=BlueOrangeWapang)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
import seaborn as sns
sns.color_palette("husl", 8)
sns.set_style("whitegrid")
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
data ['MB']= data.size_bytes.apply(lambda x : x/1048576)
paidapps_regression =data[((data.price<30) & (data.price>0))]
sns.lmplot(data=paidapps_regression,
           x='MB',y='price',size=4, aspect=2,col_wrap=2,hue='broad_genre',
           col='broad_genre',fit_reg=False,palette = sns.color_palette("husl", 5))
plt.show()
data[data.user_rating!=0.0].user_rating.value_counts().sort_index().plot.bar(width=0.5,rot=1,align='center',orientation='vertical') 
plt.xlabel('user_rating')
plt.ylabel('nombre d"applications')
plt.show()
# imports
import pandas as pd
import matplotlib.pyplot as plt

# this allows plots to appear directly in the notebook
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def visualizer(x, y, plot_type, title, xlabel, ylabel, rotation=False, rotation_value=60, figsize=(15,8)):
    plt.figure(figsize=figsize)
    
    if plot_type == "bar":  
        sns.barplot(x=x, y=y)
    elif plot_type == "count":  
        sns.countplot(x)
    elif plot_type == "reg":  
        sns.regplot(x=x,y=y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.yticks(fontsize=13)
    if rotation == True:
        plt.xticks(fontsize=13,rotation=rotation_value)
    plt.show()
visualizer(data["lang.num"], data.rating_count_tot, "reg", 
          "CORRELATION OF NUMBER OF LANGUAGES AND USERS USE", "NUMBER OF LANGUAGES",
          "USERS(TOTAL)", False)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/AppleStore.csv',encoding='utf-8')
outlier=data[(data["lang.num"]==2)][['track_name','price','lang.num','rating_count_tot']]
outlier.sort_values(by ='lang.num' ,ascending = False).head(1)
outlier=data[(data["lang.num"]==58)][['track_name','price','lang.num','rating_count_tot']]
outlier.sort_values(by ='lang.num' ,ascending = False).head(1)
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process


#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
filter=data[data.price!=0][['track_name','price','prime_genre','rating_count_tot']]
filter.sort_values(by = 'price' ,ascending = False).head()
filter2=data[data.price==0][['track_name','price','prime_genre','rating_count_tot']]
filter2.sort_values(by = 'price' ,ascending = False).head()
BlueOrangeWapang = ['#fc910d','#ff0040','#239cd3','#1674b1','#8000ff']
count = filter.groupby("prime_genre")["rating_count_tot"].count()
c = count.plot(kind='barh', color=BlueOrangeWapang, zorder=1, width=0.75,)
c.set_ylabel('catégories')
c.set_xlabel('total_count')
plt.title('Applications payantes')
fig=plt.figure(figsize=(16,8))
count2 = filter2.groupby("prime_genre")["rating_count_tot"].count()
c1 = count2.plot(kind='barh', color=BlueOrangeWapang, zorder=1, width=0.75,)
c1.set_ylabel('catégories')
c1.set_xlabel('total_count_GA')
plt.title('Applications gratuites')
fig2=plt.figure(figsize=(16,15))
data = pd.read_csv('../input/AppleStore.csv',encoding='utf-8')
data.drop_duplicates()
data ['is_duplicated'] = data.duplicated (['track_name', 'price'])[ 'is_duplicated' ] = data.duplicated([ 'track_name' , 'price' ])

data.isnull().sum()
data.shape
%matplotlib inline
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import sklearn
import pandas as pd
data = pd.read_csv('../input/AppleStore.csv',encoding='utf-8')
from numpy import *
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

lm = smf.ols(formula='price ~ user_rating_ver', data=data).fit()

lm.params
# you have to create a DataFrame since the Statsmodels formula interface expects it
X = pd.DataFrame({'user_rating_ver': [6]})
X.head()
# use the model to make predictions on a new value
lm.predict(X)
# create a DataFrame with the minimum and maximum values of TV
X = pd.DataFrame({'user_rating_ver': [data.user_rating_ver.min(), data.user_rating_ver.max()]})
X.head()
# make predictions for those x values and store them
preds = lm.predict(X)
preds
# first, plot the observed data
data.plot(kind='scatter', x='user_rating_ver', y='price')

# then, plot the least squares line
plt.plot(X, preds, c='red', linewidth=2)

# print the confidence intervals for the model coefficients
lm.conf_int()
# print the p-values for the model coefficients
lm.pvalues
lm.rsquared

