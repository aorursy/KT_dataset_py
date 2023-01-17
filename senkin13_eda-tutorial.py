import pandas as pd

import numpy as np

import gc

import os

import glob

from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib_venn import venn2

%matplotlib inline



pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)

pd.set_option('display.width', 100)

pd.set_option('display.max_colwidth', 100)

pd.set_option("display.precision", 8)
print (os.listdir('/kaggle/input/used-car-price-forecasting'))
inputPath = '/kaggle/input/used-car-price-forecasting/'

train = pd.read_csv(inputPath + 'train.csv')

test = pd.read_csv(inputPath + 'test.csv')

sub = pd.read_csv(inputPath + 'sample_submission.csv')

print ('train shape',train.shape)

display(train.head())

print ('test shape',test.shape)

display(test.head())

print ('sub shape',sub.shape)

display(sub.head())
target = 'price'

plt.figure(figsize=(20, 10))

sns.distplot(train[target])
# transform to normal distribution

plt.figure(figsize=(20, 10))

sns.distplot(np.log1p(train[target]))
print ('train missing values percentage')

display(train.isnull().sum() * 100 / len(train))

print ('test missing values percentage')

display(test.isnull().sum() * 100 / len(test))
num_cols = ['odometer','lat','long']

print ('train numerical columns')

display(train[num_cols].describe())

print ('test numerical columns')

display(test[num_cols].describe())
cat_cols = ['region', 'year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel',

        'title_status', 'transmission', 'vin', 'drive', 'size', 'type', 'paint_color', 'state']

for i in cat_cols:

    print ('==================== ' + str(i) + ' ====================')

    print ('train unique number ', train[i].nunique())

    print ('test unique number', test[i].nunique())
for i in cat_cols:

    print ('==================== ' + str(i) + ' ====================')

    print ('train top unique number percentage')

    print (train[i].value_counts(dropna=False, normalize=True).head())

    print ('test top unique number percentage')    

    print (test[i].value_counts(dropna=False, normalize=True).head())
print ('Overlap Of Train And Test')

plt.figure(figsize=(30,30), facecolor='w')

c = 5

r = (len(cat_cols) // c) + 1

for i, col in tqdm(enumerate(cat_cols)):

    plt.subplot(r,c,i+1)

    s1 = set(train[col].unique().tolist())

    s2 = set(test[col].unique().tolist())

    venn2(subsets=[s1, s2], set_labels=['Train', 'Test'])

    plt.title(str(col), fontsize=14)

plt.show()
from mpl_toolkits.basemap import Basemap 

plt.figure(figsize=(16,8))

m = Basemap(projection='merc', # mercator projection

            llcrnrlat = 20,

            llcrnrlon = -170,

            urcrnrlat = 70,

            urcrnrlon = -60,

            resolution='l')



m.shadedrelief()

m.drawcoastlines() # drawing coaslines

m.drawcountries(linewidth=2) # drawing countries boundaries

m.drawstates(color='b') # drawing states boundaries



print ('train lat long distribution')    

for index, row in train.sample(frac=0.2).iterrows():

    latitude = row['lat']

    longitude = row['long']

    x_coor, y_coor = m(longitude, latitude)

    m.plot(x_coor,y_coor,'.',markersize=0.5,c="red")

    

    

plt.figure(figsize=(16,8))

m = Basemap(projection='merc', # mercator projection

            llcrnrlat = 20,

            llcrnrlon = -170,

            urcrnrlat = 70,

            urcrnrlon = -60,

            resolution='l')



m.shadedrelief()

m.drawcoastlines() # drawing coaslines

m.drawcountries(linewidth=2) # drawing countries boundaries

m.drawstates(color='b') # drawing states boundaries



print ('test lat long distribution')    

for index, row in test.sample(frac=0.4).iterrows():

    latitude = row['lat']

    longitude = row['long']

    x_coor, y_coor = m(longitude, latitude)

    m.plot(x_coor,y_coor,'.',markersize=0.5,c="red")
pd.set_option('display.max_colwidth', 1000)

text_col = 'description'

train[text_col].head()
train['len_chars'] = train[text_col].apply(len) # Count number of Characters

train['len_words'] = train[text_col].apply(lambda x: len(x.split())) # Count number of Words

test['len_chars'] = test[text_col].apply(len) # Count number of Characters

test['len_words'] = test[text_col].apply(lambda x: len(x.split())) # Count number of Words

print ('train characters and words length')

display(train[['len_chars','len_words']].describe())

print ('test characters and words length')

display(test[['len_chars','len_words']].describe())
from wordcloud import WordCloud , STOPWORDS

def plot_wordcloud(text,mask=None,max_words=500,max_font_size=100,figure_size=(24.0,16.0),title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    plt.imshow(wordcloud);

    plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                              'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()      

plot_wordcloud(train[text_col], title="Word Cloud of Train Text")    

plot_wordcloud(test[text_col], title="Word Cloud of Test Text")    
import cv2

from joblib import Parallel, delayed



def get_shape(img):

    path = images_path + img

    im = cv2.imread(path)

    h,w,c = im.shape

    return pd.DataFrame([[img,h,w,c]])



images_path = inputPath + 'images/train_images/'

train_imgs = os.listdir(images_path) 

my_list = Parallel(n_jobs=4, verbose=0)(delayed(get_shape)(i)for i in train_imgs) 

train_shape = pd.concat(my_list,axis=0)

train_shape.columns = ['img','height','weight','channel']

print ('train image shape')

display(train_shape.describe())



images_path = inputPath + 'images/test_images/'

test_imgs = os.listdir(images_path) 

my_list = Parallel(n_jobs=4, verbose=0)(delayed(get_shape)(i)for i in test_imgs) 

test_shape = pd.concat(my_list,axis=0)

test_shape.columns = ['img','height','weight','channel']

print ('test image shape')

display(test_shape.describe())
top_expensive = train.sort_values(['price'],ascending=False)[['price','id']].head()

top_cheap = train.sort_values(['price'],ascending=True)[['price','id']].head()

median_cheap = train.sort_values(['price'],ascending=True)[['price','id']][len(train)//2:len(train)//2+5]



print ('Most Expensive Cars')

plt.figure(figsize=(30,30), facecolor='w')

for i, z in tqdm(enumerate(list(zip(top_expensive['price'],top_expensive['id'])))):

    plt.subplot(1,5,i+1)

    im = cv2.imread(inputPath + 'images/train_images/' + str(z[1]) + '.jpg')

    plt.imshow(im)

    plt.title(str(z[0]), fontsize=14)

plt.show()    



print ('Most Cheap Cars')    

plt.figure(figsize=(30,30), facecolor='w')

for i, z in tqdm(enumerate(list(zip(top_cheap['price'],top_cheap['id'])))):

    plt.subplot(1,5,i+1)

    im = cv2.imread(inputPath + 'images/train_images/' + str(z[1]) + '.jpg')

    plt.imshow(im)

    plt.title(str(z[0]), fontsize=14)    

plt.show()  



print ('Median Cheap Cars')        

plt.figure(figsize=(30,30), facecolor='w')

for i, z in tqdm(enumerate(list(zip(median_cheap['price'],median_cheap['id'])))):

    plt.subplot(1,5,i+1)

    im = cv2.imread(inputPath + 'images/train_images/' + str(z[1]) + '.jpg')

    plt.imshow(im)  

    plt.title(str(z[0]), fontsize=14)    

plt.show()      