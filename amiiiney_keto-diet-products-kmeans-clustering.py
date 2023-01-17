 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec

from wordcloud import WordCloud, STOPWORDS 

plt.style.use('seaborn')

sns.set_style('whitegrid')

%matplotlib inline



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.options.mode.chained_assignment = None # Warning for chained copies disabled

a = pd.read_csv("/kaggle/input/world-food-facts/en.openfoodfacts.org.products.tsv",

                       delimiter='\t',

                       encoding='utf-8')
#Use this code to show all the 163 columns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3):

    """

    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking

    """

    

    plt.figure(figsize=(width,height))

    percentage=(data.isnull().mean())*100

    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)

    plt.axhline(y=thresh, color='r', linestyle='-')

    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )

    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, 'Columns with more than %s%s missing values' %(thresh, '%'), fontsize=12,weight='bold', color='crimson',

         ha='left' ,va='top')

    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, 'Columns with less than %s%s missing values' %(thresh, '%'), fontsize=12,weight='bold', color='blue',

         ha='left' ,va='top')

    plt.xlabel('Columns', size=15, weight='bold')

    plt.ylabel('Missing values percentage', weight='bold')

    plt.yticks(weight ='bold')

    

    return plt.show()
msv1(a,30, color=('silver', 'gainsboro', 'lightgreen', 'white', 'lightpink'))
ab=a.dropna(thresh=106800, axis=1)

print(f"Data shape before cleaning {a.shape}")

print(f"Data shape after cleaning {ab.shape}")

print(f"We dropped {a.shape[1]- ab.shape[1]} columns")
countries=ab['countries_en'].value_counts().head(10).to_frame()

s = countries.style.background_gradient(cmap='Blues')

s
brands= ab['brands'].value_counts().head(10).to_frame()

k = brands.style.background_gradient(cmap='Reds')

k
#Filter the data and keep just the Meijer brand products:

ac=ab[ab['brands']=='Meijer']
ac=ac.fillna(0, axis=1)
ac_corr=ac.corr()

f,ax=plt.subplots(figsize=(10,7))

sns.heatmap(ac_corr, cmap='viridis')

plt.title("Correlation between features", 

          weight='bold', 

          fontsize=18)

plt.xticks(weight='bold')

plt.yticks(weight='bold')



plt.show()
plt.figure(figsize=(15, 6))



plt.scatter(x=ac['nutrition-score-uk_100g'], y=ac['energy_100g'], color='deeppink', alpha=0.5)

plt.title("UK Nutrition score of Meijer's products based on calories ", 

          weight='bold', 

          fontsize=15)

plt.xlabel('Nutrition score UK', weight='bold', fontsize=14)

plt.ylabel('Calories', weight='bold', fontsize=14)

plt.xticks(fontsize=12, weight='bold')

plt.yticks(fontsize=12,weight='bold')





plt.show()
ad=ac[['product_name','energy_100g', 'fat_100g',

       'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',

       'proteins_100g']]

print(f"we have {ad.shape[0]} products in Meijer supermarkets and {ad.shape[1]} features")
keto= ad[(ad['energy_100g']<2000)&(ad['carbohydrates_100g']<40)&(ad['fat_100g']<165)&(ad['proteins_100g']<75)]

print(f'We have {keto.shape[0]} keto products in Meijer supermarkets')
plt.style.use('seaborn')

sns.set_style('whitegrid')



fig= plt.figure(figsize=(15,10))

#2 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((2,2),(0,0))

plt.hist(keto.energy_100g, bins=3, color='orange', alpha=0.7)

plt.title('Calories',weight='bold', fontsize=18)

plt.yticks(weight='bold')

plt.xticks(weight='bold')

#first row sec col

ax1 = plt.subplot2grid((2,2), (0, 1))

plt.hist(keto.fat_100g, bins=3, alpha=0.7)

plt.title('Fat',weight='bold', fontsize=18)

plt.yticks(weight='bold')

plt.xticks(weight='bold')

#Second row first column

ax1 = plt.subplot2grid((2,2), (1, 0))

plt.hist(keto.proteins_100g, bins=3, color='red', alpha=0.7)

plt.title('Protein',weight='bold', fontsize=18)

plt.yticks(weight='bold')

plt.xticks(weight='bold')

#second row second column

ax1 = plt.subplot2grid((2,2), (1, 1))

plt.hist(keto.carbohydrates_100g, bins=3, color='green', alpha=0.7)

plt.title('Carbs',weight='bold', fontsize=18)

plt.yticks(weight='bold')

plt.xticks(weight='bold')



plt.show()
da=keto.sort_values(by=['energy_100g'],ascending=False).sample(5)

n = da.style.background_gradient(cmap='Purples')

n
def label_cal (row):

   if row['energy_100g'] < 250  :

      return 'low'

   if row['energy_100g'] > 250 and row['energy_100g'] < 500 :

      return 'medium'

   if row['energy_100g'] > 500 :

      return 'high'

   

   return 'Other'





def label_fat (row):

   if row['fat_100g'] < 10 :

      return 'low'

   if row['fat_100g'] >= 10 and row['fat_100g'] < 20 :

      return 'medium'

   if row['fat_100g'] >= 20 :

      return 'high'

   

   return 'Other'





def label_pro (row):

   if row['proteins_100g'] < 10 :

      return 'low'

   if row['proteins_100g'] >= 10 and row['proteins_100g'] < 20 :

      return 'medium'

   if row['proteins_100g'] >= 20 :

      return 'high'

   

   return 'Other'





def label_carb (row):

   if row['carbohydrates_100g'] < 4 :

      return 'low'

   if row['carbohydrates_100g'] >= 4 and row['carbohydrates_100g'] < 12 :

      return 'medium'

   if row['carbohydrates_100g'] >= 12 :

      return 'high'

   

   return 'Other'



# we add those new columns to the existing keto dataset:

keto['calories'] = keto.apply (lambda row: label_cal(row), axis=1)



keto['fat'] = keto.apply (lambda row: label_fat(row), axis=1)



keto['protein'] = keto.apply (lambda row: label_pro(row), axis=1)



keto['carbs'] = keto.apply (lambda row: label_carb(row), axis=1)



#Create dataframe



db=keto.calories.value_counts().reset_index()

dd= keto.fat.value_counts().reset_index()

de=keto.protein.value_counts().reset_index()

dg=keto['carbs'].value_counts().reset_index()



#Merge them on the 'index' column:

merged=db.merge(dd,on='index').merge(de, on='index').merge(dg, on='index')

mergedstyle = merged.style.background_gradient(cmap='Greens')

mergedstyle
label1=db['index']

label2=dd['index']

label3=de['index']

label4=dg['index']





fig = plt.figure(figsize=(15,10))

#2 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((2,2),(0,0))

plt.pie(db.calories,colors=("grey","r","orange"),labels=label1, autopct='%.2f',textprops={'fontsize': 14, 'weight':'bold'})

plt.title('calories',weight='bold', fontsize=18)

#first row sec col

ax1 = plt.subplot2grid((2,2), (0, 1))

plt.pie(dd.fat,colors=("grey","r","orange"),labels=label2, autopct='%.2f',textprops={'fontsize': 14, 'weight':'bold'})

plt.title('fat',weight='bold', fontsize=18)

#Second row first column

ax1 = plt.subplot2grid((2,2), (1, 0))

plt.pie(de.protein,colors=("grey","r","orange"),labels=label3, autopct='%.2f',textprops={'fontsize': 14, 'weight':'bold'})

plt.title('protein',weight='bold', fontsize=18)

#second row second column

ax1 = plt.subplot2grid((2,2), (1, 1))

plt.pie(dg.carbs,colors=("grey","r","orange"),labels=label4, autopct='%.2f',textprops={'fontsize': 14, 'weight':'bold'})

plt.title('carbs',weight='bold', fontsize=18)

plt.show()
ketocat=keto[['product_name', 'calories', 'protein','fat','carbs']]

keto_low=ketocat.loc[ketocat['calories']=='low']

keto_medium=ketocat.loc[ketocat['calories']=='medium']

keto_high=ketocat.loc[ketocat['calories']=='high']
wordcloud1 = WordCloud(width=600, height=500, background_color='white').generate(' '.join(keto_low['product_name']))

WordCloud.generate_from_frequencies





wordcloud2 = WordCloud(width=600, height=500, background_color='white').generate(' '.join(keto_medium['product_name']))

WordCloud.generate_from_frequencies





wordcloud3 = WordCloud(width=600, height=500, background_color='white').generate(' '.join(keto_high['product_name']))

WordCloud.generate_from_frequencies





fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,8))



fig.suptitle('Low, medium and high calories products', weight='bold', fontsize=20)







ax1.set_title('Low calories products', weight='bold', fontsize=15, color='b')

# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

im1 = ax1.imshow(wordcloud1, aspect='auto')





ax2.set_title('Medium calories products', weight='bold', fontsize=15, color='b')

im4 = ax2.imshow(wordcloud2, aspect='auto')



ax3.set_title('High calories products', weight='bold', fontsize=15, color='b')

im4 = ax3.imshow(wordcloud3, aspect='auto')



# Make space for title

plt.subplots_adjust(top=0.85)

plt.show()
ketoc=keto[['energy_100g','fat_100g', 'saturated-fat_100g','carbohydrates_100g', 'sugars_100g', 'proteins_100g']]
from scipy.stats import skew



numeric_feats = ketoc.dtypes[ketoc.dtypes != "object"].index



skewed_feats = ketoc[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



ketoc[skewed_feats] = np.log1p(ketoc[skewed_feats])
from sklearn.preprocessing import RobustScaler



scaler=RobustScaler()

scaler.fit(ketoc)
from collections import defaultdict

from scipy.spatial.distance import pdist, squareform

from scipy.cluster.hierarchy import linkage, dendrogram

from matplotlib.colors import rgb2hex, colorConverter

from scipy.cluster.hierarchy import set_link_color_palette

import pandas as pd

import scipy.cluster.hierarchy as sch

%pylab inline
#ketoo=keto.drop(['calories', 'protein','fat','carbs'], axis=1 )

#keton=ketoo.set_index('product_name')

#ketonn=keton.T

#ketom=ketonn.reset_index(drop=True)
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as shc



plt.figure(figsize=(20, 7))

plt.title("Supermarket food products Dendograms")

plt.xticks(rotation='vertical')





dend = shc.dendrogram(shc.linkage(ketoc, method='ward'))
agc = AgglomerativeClustering(n_clusters=8, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func='deprecated')

pred_ag = agc.fit_predict(ketoc)
keto['ag_cluster']= agc.fit_predict(ketoc)
plt.figure(figsize=(15,5))

plt.style.use('seaborn')

sns.set_style('whitegrid')

keto['ag_cluster'].value_counts().plot(kind='bar', color=['tan', 'crimson', 'silver', 'darkcyan',

                                                          'deeppink', 'deepskyblue','lightgreen', 'orchid'])

plt.ylabel("Count",fontsize=14, weight='bold')

plt.xlabel(' Agglomerative Clusters', fontsize=14, weight='bold')

plt.show()
#Clusters column

agcluster0=keto[keto['ag_cluster']==0]

agcluster1=keto[keto['ag_cluster']==1]

agcluster2=keto[keto['ag_cluster']==2]

agcluster3=keto[keto['ag_cluster']==3]

agcluster4=keto[keto['ag_cluster']==4]

agcluster5=keto[keto['ag_cluster']==5]

agcluster6=keto[keto['ag_cluster']==6]

agcluster7=keto[keto['ag_cluster']==7]



wordcloud20 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster0['product_name']))

WordCloud.generate_from_frequencies





wordcloud21 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster1['product_name']))

WordCloud.generate_from_frequencies





wordcloud22 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster2['product_name']))

WordCloud.generate_from_frequencies





wordcloud23 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster3['product_name']))

WordCloud.generate_from_frequencies





wordcloud24 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster4['product_name']))

WordCloud.generate_from_frequencies





wordcloud25 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster5['product_name']))

WordCloud.generate_from_frequencies



wordcloud26 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster6['product_name']))

WordCloud.generate_from_frequencies





wordcloud27 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster7['product_name']))

WordCloud.generate_from_frequencies



#Create color dictionary for clusters

col_dic = {0:'darkblue',1:'green',2:'darkorange',3:'yellow',4:'magenta',5:'black', 6:'cyan', 7:'lime', 8:'red', 9:'darkviolet', 10:'grey'}

colors3 = [col_dic[x] for x in pred_ag]



#Funciton to plot the clusters

def plot_ag_cluster(keto, color):

    fig, ax = plt.subplots(2, 2, figsize=(12,11)) # define plot area         

    x_cols = ['carbohydrates_100g', 'fat_100g', 'sugars_100g', 'proteins_100g']

    y_cols = ['energy_100g', 'proteins_100g', 'energy_100g', 'carbohydrates_100g']

    for x_col,y_col,i,j in zip(x_cols,y_cols,[0,0,1,1],[0,1,0,1]):

        for x,y,c in zip(ketoc[x_col], ketoc[y_col], colors3):

            ax[i,j].scatter(x,y, color = c)

        ax[i,j].set_title('Scatter plot of ' + y_col + ' vs. ' + x_col) # Give the plot a main title

        ax[i,j].set_xlabel(x_col) # Set text for the x axis

        ax[i,j].set_ylabel(y_col)# Set text for y axis

    plt.show()
plot_ag_cluster(ketoc, colors3)
from sklearn.cluster import KMeans, AgglomerativeClustering

import numpy as np



kmeans = KMeans(n_clusters=8

                       ,init = 'k-means++'

                       , n_init = 10

                       , tol = 0.0001

                       , n_jobs = -1

                       , random_state = 1).fit(ketoc)

labels2 = kmeans.labels_



centers=kmeans.cluster_centers_

pred1 = kmeans.fit_predict(ketoc)
color_km = [col_dic[x] for x in pred1]
def plot_km_cluster(keto, color):

    fig, ax = plt.subplots(2, 2, figsize=(12,11)) # define plot area         

    x_cols = ['carbohydrates_100g', 'fat_100g', 'sugars_100g', 'proteins_100g']

    y_cols = ['energy_100g', 'proteins_100g', 'energy_100g', 'carbohydrates_100g']

    for x_col,y_col,i,j in zip(x_cols,y_cols,[0,0,1,1],[0,1,0,1]):

        for x,y,c in zip(ketoc[x_col], ketoc[y_col], color_km):

            ax[i,j].scatter(x,y, color = c)

        ax[i,j].set_title('Scatter plot of ' + y_col + ' vs. ' + x_col) # Give the plot a main title

        ax[i,j].set_xlabel(x_col) # Set text for the x axis

        ax[i,j].set_ylabel(y_col)# Set text for y axis

    plt.show()
plot_km_cluster(keto, color_km)
keto['km_cluster']= kmeans.fit_predict(ketoc)
plt.style.use('seaborn')

sns.set_style('whitegrid')





plt.subplots(0,0,figsize=(15,4))

plt.title("Comparison between Kmeans and agglomerative clusters", fontsize=20, weight='bold')



keto['ag_cluster'].value_counts().plot(kind='bar', color=['tan', 'crimson', 'silver', 'darkcyan',                                                          'deeppink', 'deepskyblue','lightgreen', 'orchid'])

plt.ylabel("Count",fontsize=14, weight='bold')

plt.xticks(weight='bold')

plt.xlabel('Agglomerative Clusters', fontsize=14, weight='bold')



plt.subplots(1,0,figsize=(15,4))

keto['km_cluster'].value_counts().plot(kind='bar', color=['tan', 'crimson', 'silver', 'darkcyan',                                                        'deeppink', 'deepskyblue','lightgreen', 'orchid'])

plt.ylabel("Count",fontsize=14, weight='bold')

plt.xticks(weight='bold')

plt.xlabel('Kmeans Clusters', fontsize=14, weight='bold')



plt.tight_layout()

plt.show()
#Create clusters column

cluster0=keto[keto['km_cluster']==0]

cluster1=keto[keto['km_cluster']==1]

cluster2=keto[keto['km_cluster']==2]

cluster3=keto[keto['km_cluster']==3]

cluster4=keto[keto['km_cluster']==4]

cluster5=keto[keto['km_cluster']==5]

cluster6=keto[keto['km_cluster']==6]

cluster7=keto[keto['km_cluster']==7]

cluster8=keto[keto['km_cluster']==8]



#Generate word clouds for each cluster

wordcloud10 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster0['product_name']))

WordCloud.generate_from_frequencies



wordcloud11 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster1['product_name']))

WordCloud.generate_from_frequencies



wordcloud12 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster2['product_name']))

WordCloud.generate_from_frequencies



wordcloud13 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster3['product_name']))

WordCloud.generate_from_frequencies



wordcloud14 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster4['product_name']))

WordCloud.generate_from_frequencies



wordcloud15 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster5['product_name']))

WordCloud.generate_from_frequencies



wordcloud16 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster6['product_name']))

WordCloud.generate_from_frequencies



wordcloud17 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster7['product_name']))

WordCloud.generate_from_frequencies







#Plot clusters

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))



fig.suptitle('Comparison between Kmeans and agglomerative clusters', weight='bold', fontsize=20)







ax1.set_title('Agglomerative cluster 7', weight='bold', fontsize=15, color='r')

# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

im1 = ax1.imshow(wordcloud27, aspect='auto')

plt.xticks(weight='bold')

plt.yticks(weight='bold')





ax2.set_title('Kmeans cluster 0', weight='bold', fontsize=15, color='g')

im4 = ax2.imshow(wordcloud10, aspect='auto')





# Make space for title

plt.subplots_adjust(top=0.85)

plt.xticks(weight='bold')

plt.yticks(weight='bold')



plt.show()
f, axarr = plt.subplots(3,2, figsize=(15,17))





fig.suptitle('Title of figure', fontsize=20)





# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

axarr[0,0].set_title('Products: CLUSTER 0', weight='bold')

axarr[0,0].imshow(wordcloud10, aspect='auto')



axarr[0,1].set_title('Products: CLUSTER 1',weight='bold')

axarr[0,1].imshow(wordcloud11, aspect='auto')



# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

axarr[1,0].set_title('Products: CLUSTER 3',weight='bold')

axarr[1,0].imshow(wordcloud13, aspect='auto')





# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

axarr[1,1].set_title('Products: CLUSTER 4',weight='bold')

axarr[1,1].imshow(wordcloud14, aspect='auto')



# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

axarr[2,0].set_title('Products: CLUSTER 5',weight='bold')

axarr[2,0].imshow(wordcloud15, aspect='auto')





# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

axarr[2,1].set_title('Products: CLUSTER 7',weight='bold')

axarr[2,1].imshow(wordcloud17, aspect='auto')



# Make space for title

plt.subplots_adjust(top=0.85)

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))





ax1.set_title('Products: cluster 2', weight='bold', fontsize=15)

# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

im1 = ax1.imshow(wordcloud12, aspect='auto')





ax2.set_title('Products: CLUSTER 6', weight='bold', fontsize=15)

im4 = ax2.imshow(wordcloud16, aspect='auto')



# Make space for title

plt.subplots_adjust(top=0.85)

plt.show()