# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.graph_objects as go
womens_western=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/women_westernwear.csv")

mens_western=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/mens_westernwear.csv")

laptops=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/laptops.csv")

mobs=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/mobiles.csv")

baby=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/baby.csv")

food=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/food.csv")

footwear=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/women_footwear.csv")

furni=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/furn.csv")

books=pd.read_csv("/kaggle/input/flipkart-data/flipkart_data/books.csv")
womens_western.head(5)
laptops.head()
laptops['Price']=laptops['Price'].str.replace(',','')

laptops['Original Prices']=laptops['Original Prices'].str.replace(',','')

laptops['Discount rates']=laptops['Discount rates'].str.replace('% off','')
womens_western['Price']=womens_western['Price'].str.replace(',','')

womens_western['Original Prices']=womens_western['Original Prices'].str.replace(',','')

womens_western['Discount rates']=womens_western['Discount rates'].str.replace('% off','')
mens_western['Price']=mens_western['Price'].str.replace(',','')

mens_western['Original Prices']=mens_western['Original Prices'].str.replace(',','')

mens_western['Discount rates']=mens_western['Discount rates'].str.replace('% off','')
mobs['Price']=mobs['Price'].str.replace(',','')

mobs['Original Prices']=mobs['Original Prices'].str.replace(',','')

mobs['Discount rates']=mobs['Discount rates'].str.replace('% off','')
baby['Price']=baby['Price'].str.replace(',','')

baby['Original Prices']=baby['Original Prices'].str.replace(',','')

baby['Discount rates']=baby['Discount rates'].str.replace('% off','')
food['Price']=food['Price'].str.replace(',','')

food['Original Prices']=food['Original Prices'].str.replace(',','')

food['Discount rates']=food['Discount rates'].str.replace('% off','')
footwear['Price']=footwear['Price'].str.replace(',','')

footwear['Original Prices']=footwear['Original Prices'].str.replace(',','')

footwear['Discount rates']=footwear['Discount rates'].str.replace('% off','')
furni['Price']=furni['Price'].str.replace(',','')

furni['Original Prices']=furni['Original Prices'].str.replace(',','')

furni['Discount rates']=furni['Discount rates'].str.replace('% off','')
books['Price']=books['Price'].str.replace(',','')

books['Original Prices']=books['Original Prices'].str.replace(',','')

books['Discount rates']=books['Discount rates'].str.replace('% off','')
books_nona=books.dropna()

furni_nona=furni.dropna()

baby_nona=baby.dropna()
laptops_nona=laptops.dropna()

footwear_nona=footwear.dropna()

womens_western_nona=womens_western.dropna()

food_nona=food.dropna()

mobs_nona=mobs.dropna()

mens_western_nona=mens_western.dropna()

laptops_nona = laptops_nona[~laptops_nona['Discount rates'].isin(['Nan'])]

footwear_nona = footwear_nona[~footwear_nona['Discount rates'].isin(['Nan'])]

food_nona = food_nona[~food_nona['Discount rates'].isin(['Nan'])]

mobs_nona = mobs_nona[~mobs_nona['Discount rates'].isin(['Nan'])]

baby_nona = baby_nona[~baby_nona['Discount rates'].isin(['Nan'])]

furni_nona = furni_nona[~furni_nona['Discount rates'].isin(['Nan'])]

books_nona = books_nona[~books_nona['Discount rates'].isin(['Nan'])]
womens_western_nona['Discount rates']=womens_western_nona['Discount rates'].astype(str).astype(int)

footwear_nona['Discount rates']=footwear_nona['Discount rates'].astype(str).astype(int)

food_nona['Discount rates']=food_nona['Discount rates'].astype(str).astype(int)

mobs_nona['Discount rates']=mobs_nona['Discount rates'].astype(str).astype(int)

baby_nona['Discount rates']=baby_nona['Discount rates'].astype(str).astype(int)

furni_nona['Discount rates']=furni_nona['Discount rates'].astype(str).astype(int)

books_nona['Discount rates']=books_nona['Discount rates'].astype(str).astype(int)

laptops_nona['Discount rates']=laptops_nona['Discount rates'].astype(str).astype(int)

mens_western_nona['Discount rates']=mens_western_nona['Discount rates'].astype(str).astype(int)
x=['womens_westernwear','mens_westernwear','furniture','books','baby_products','footwear','laptops','mobiles', 'food']

y=[womens_western_nona['Discount rates'].mean(),mens_western_nona['Discount rates'].mean(),furni_nona['Discount rates'].mean(),books_nona['Discount rates'].mean(),baby_nona['Discount rates'].mean(),footwear_nona['Discount rates'].mean(),laptops_nona['Discount rates'].mean(),mobs_nona['Discount rates'].mean(),food_nona['Discount rates'].mean()]
fig = go.Figure(data=[go.Bar(x=x, y=y)])

# Customize aspect

fig.update_traces(marker_color='rgb(255,160,122)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(title_text='Discounts')

fig.show()
womens_western_nona['Price']=womens_western_nona['Price'].astype(str).astype(int)

womens_western_nona['Original Prices']=womens_western_nona['Original Prices'].astype(str).astype(int)

mens_western_nona['Price']=mens_western_nona['Price'].astype(str).astype(int)

mens_western_nona['Original Prices']=mens_western_nona['Original Prices'].astype(str).astype(int)
features=['Price','Original Prices','Discount rates']

fig = go.Figure(data=[

    go.Bar(name='Womens Clothing', x=features, y=[womens_western_nona['Price'].mean()

                                                  ,womens_western_nona['Original Prices'].mean()

                                                 ,womens_western_nona['Discount rates'].mean()]),

    go.Bar(name='Mens Clothing', x=features, y=[mens_western_nona['Price'].mean()

                                                  ,mens_western_nona['Original Prices'].mean()

                                                 ,mens_western_nona['Discount rates'].mean()])

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
womens_western_top10=womens_western_nona.sort_values(by='Original Prices', ascending=False)

womens_western_top10.head(10)

cols=['Product Name','Image URLS']

womens_western_top10=womens_western_top10[cols].head(10)

womens_western_top10=womens_western_top10.set_index('Product Name')
womens_western_top10=womens_western_top10.rename(columns={"Image URLS": "img"})
from IPython.display import Image, HTML

def path_to_image_html(path):

    '''

     This function essentially convert the image url to 

     '<img src="'+ path + '"/>' format. And one can put any

     formatting adjustments to control the height, aspect ratio, size etc.

     within as in the below example. 

    '''



    return '<img src="'+ path + '"width="60" height="60""/>'



HTML(womens_western_top10.to_html(escape=False ,formatters=dict(img=path_to_image_html),justify='center'))
mens_top10=mens_western_nona.sort_values(by='Original Prices', ascending=False).head(10)

mens_top10=mens_top10[cols]

mens_top10=mens_top10.set_index('Product Name')

mens_top10=mens_top10.rename(columns={"Image URLS": "img"})
HTML(mens_top10.to_html(escape=False ,formatters=dict(img=path_to_image_html),justify='center'))
mobs1=mobs.sort_values(by='Ratings', ascending=False)
top50=mobs1.head(50)

companies=['Apple','Redmi','Vivo','OnePlus','Realme','OPPO','POCO']

vals=[0,0,0,0,0,0,0]

dictionary=dict(zip(companies,vals))

for i in list(top50['Product Name']):

    for j in companies:

        if j in i:

            dictionary[j]+=1
import plotly.express as px

fig = px.pie(values=list(dictionary.values()), names=list(dictionary.keys()), color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
fig = px.line(mobs_nona, y="Discount rates", x="Ratings", title='Price VS Rating')

fig.show()
fig = px.treemap(mobs_nona, path=['Discount rates', 'Product Name'],color='Discount rates',values='Original Prices')

fig.show()
laptops1=laptops_nona.sort_values(by='Discount rates', ascending=False)
laptops1
import plotly.express as px

fig = px.histogram(laptops1, x="Discount rates",opacity=0.8,color_discrete_sequence=['indianred'])

fig.show()
fig = px.bar(laptops_nona, y="Original Prices", x="Ratings", title='Price VS Rating', color='Original Prices')

fig.show()
fig = px.treemap(footwear_nona, path=['Brand', 'Product Name'], values='Price')

fig.show()
footwear_sorted=footwear_nona.sort_values(by='Price')
footwear_sorted=footwear_sorted.rename(columns={"Image URLS": "img"})

HTML(footwear_sorted.head(10).to_html(escape=False ,formatters=dict(img=path_to_image_html),justify='center'))
baby_nona
baby_s=baby_nona.sort_values(by='Discount rates', ascending=False)

baby_s=baby_s.head(20)

colors = ['rgb(239, 243, 255)', 'rgb(189, 215, 231)', 'rgb(107, 174, 214)',

          'rgb(49, 130, 189)', 'rgb(8, 81, 156)',]

fig = go.Figure(data=[go.Table(header=dict(values=['Product Name','Discount rate'],fill_color='black',font=dict(color='white', size=12)

),

                 cells=dict(values=[list(baby_s['Product Name']),list(baby_s['Discount rates'])],fill_color='rgb(107, 174, 214)',font=dict(color='black', size=12)))

                     ])

fig.show()
list(womens_western_top10['img'])
import imageio

import matplotlib.pyplot as plt

pic = imageio.imread('https://rukminim1.flixcart.com/image/329/395/k26h0280/jacket/g/6/q/m-ttjk000814-tokyo-talkies-original-imafhhbqtxa76kf7.jpeg?q=50')

plt.figure(figsize = (5,5))

plt.imshow(pic)
print('Type of the image : ' , type(pic)) 

print('Shape of the image : {}'.format(pic.shape)) 

print('Image Height {}'.format(pic.shape[0])) 

print('Image Width {}'.format(pic.shape[1])) 

print('Dimension of Image {}'.format(pic.ndim))
print('Image size {}'.format(pic.size)) 

print('Maximum RGB value in this image {}'.format(pic.max())) 

print('Minimum RGB value in this image {}'.format(pic.min()))
print('Value of only R channel {}'.format(pic[ 100, 50, 0])) 

print('Value of only G channel {}'.format(pic[ 100, 50, 1])) 

print('Value of only B channel {}'.format(pic[ 100, 50, 2]))
plt.title('R channel') 

plt.ylabel('Height {}'.format(pic.shape[0])) 

plt.xlabel('Width {}'.format(pic.shape[1])) 

plt.imshow(pic[ : , : , 0])

plt.show()
plt.title('G channel')

plt.ylabel('Height {}'.format(pic.shape[0])) 

plt.xlabel('Width {}'.format(pic.shape[1])) 

plt.imshow(pic[ : , : , 1]) 

plt.show()
plt.title('B channel') 

plt.ylabel('Height {}'.format(pic.shape[0])) 

plt.xlabel('Width {}'.format(pic.shape[1])) 

plt.imshow(pic[ : , : , 2]) 

plt.show()
pic[50:150 , : , 0] = 255 # full intensity to those pixel's R channel 

plt.figure( figsize = (5,5)) 

plt.imshow(pic) 

plt.show()
#Note that only the last column changes to change colors

pic[200:300 , : , 1] = 255 # full intensity to those pixel's G channel



plt.figure( figsize = (5,5)) 

plt.imshow(pic) 

plt.show()
pic[350:, : , 2] = 255 # full intensity to those pixel's G channel



plt.figure( figsize = (5,5)) 

plt.imshow(pic) 

plt.show()
import numpy as np 

pic=imageio.imread('https://rukminim1.flixcart.com/image/329/395/k26h0280/jacket/g/6/q/m-ttjk000814-tokyo-talkies-original-imafhhbqtxa76kf7.jpeg?q=50')

fig, ax = plt.subplots(nrows = 1, ncols=3, figsize=(15,5))  

for c, ax in zip(range(3), ax):     

     # create zero matrix        

     split_img = np.zeros(pic.shape, dtype="uint8") 

     # 'dtype' by default: 'numpy.float64'  # assing each channel      

     split_img[ :, :, c] = pic[ :, :, c] # display each channel     

     ax.imshow(split_img)
import numpy as np

import cv2

from scipy.interpolate import interp1d

def brightness( im_file ):

    im = Image.open(im_file).convert('L')

    stat = ImageStat.Stat(im)

    return stat.mean[0]





from skimage import io

import numpy as np



def get(im):

    t=io.imread(im)

    return np.mean(t)

footwear_sorted
footwear_sorted['brightness']=footwear_sorted['img'].apply(lambda x: get(x))
grouped = footwear_sorted.groupby('Brand')

g=grouped.aggregate(np.mean)
g_dark=g.sort_values(by='brightness')

g_dark=g_dark.head(10)
g_light=g.sort_values(by='brightness', ascending=False)

g_light=g_light.head(10)
fig = px.bar(g_light, x=g_light.index, y='brightness', color='brightness')

fig.show()
fig = go.Figure(go.Bar(

            x=g_dark.index,

            y=g_dark['brightness']))



fig.show()