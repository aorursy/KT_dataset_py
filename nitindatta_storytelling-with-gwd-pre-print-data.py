!pip install pandas-profiling
import pandas as pd
import numpy as np 
import cv2
import os
import seaborn as sns
import folium
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pandas_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import warnings

warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
sns.set_style('whitegrid')
df = pd.DataFrame()
df['Sub_Dataset_name'] =  ['UTokyo_1','UTokyo_2','Arvalis_1','Arvalis_2','Arvalis_3',
                           'INRAE_1','USask_1','RRes_1','ETHZ_1','NAU_1','UQ_1']

df['Institution'] = ['NARO & UTokyo','NARO & UTokyo','Arvalis','Arvalis','Arvalis',
                     'INRAE','University of Saskatchewan','Rothamsted Research','ETHZ',
                     'Nanjing Agric.University','UQueensland']

df['Country'] = ['Japan','Japan','France','France','France',
                 'France','Canada','UK','Switzerland','China','Australia']

df['Lat'] = ['36.0N', '42.8N','43.7N','43.7N','49.7N',
             '43.5N','52.1N','51.8N','47.4N','31.6N','27.5S']

df['Long'] = ['140.0E','143.0', '5.8E','5.8E', '3.0E', '1.5E',
             '106.W', '0.36W','8.6E','119.4E','152.3E']

df['Year'] = [2018,2016,2017,2019,2019,2019,
              2019,2016,2018,2018,2016]

df['Nb_Of_Dates'] = [3,6,3,1,3,1,1,1,1,1,1]

df['Targeted_stages'] = ['Post-flowering','Flowering','Post-flowering-Ripening',
                         'Post-flowering','Post-flowering-Ripening','Post-flowering',
                         '','','',"Flowering",'Flowering-Ripening']

df['Row_Spacing'] = [15,12.5,17.5,17.5,17.5,
                     16,30.5,0,12.5,20,22]

df['Sowing_density'] = [186,200,300,300,300,300,
                       250,350,400,300,150]

df['Nb_of_Genotypes'] = [66,1,20,20,4,7,16,6,354,5,8]
stats = pd.DataFrame()
stats['Sub_Dataset_name_'] = df['Sub_Dataset_name']

stats['Nb_of_acquired_images'] = [994,30,239,51,152,
                                  44,100,72,375,20,142]

stats['Nb_patch_per_image'] = [1,4,6,4,4,4,2,6,2,1,1]

stats['Original_GSD'] = [0.43,0.6,0.23,0.56,0.56,
                         0.56,0.45,0.33,0.55,0.21,0.2]

stats['Sampling_factor'] = [1,2,0.5,2,2,2,1,1,1,1,0.5]

stats['Used_GSD'] = [0.43,0.3,0.46,0.28,0.28,0.28,
                    0.45,0.33,0.55,0.21,0.4]

stats['Nb_labelled_images'] = [994,120,1055,204,608,176,
                               200,432,747,20,142]

stats['Nb_labelled_heads'] = [29174,3263,45716,4179,
                             16665,3701,5737,20236,
                             51489,1250,7035]

stats['Average_heads_per_image'] = [29,27,43,20,27,21,
                                   29,47,69,63,50]
img = mpimg.imread('../input/sources-image/sources_image.png')
plt.figure(figsize=(20,10))
plt.imshow(img)
plt.axis('off')
plt.show()
train = pd.read_csv('../input/global-wheat-detection/train.csv')
print("Sources of train data",train['source'].unique())
print("\nSources of test data ['UTokyo1','UTolyo2','NAU_1','UQ_1']")
df.head(6).T
stats.head(8).T
print('First table shape ', df.shape)
print('Second table shape ',stats.shape)

df = pd.concat([df, stats], axis=1, sort=False)
df.head()
df.drop(['Sub_Dataset_name_'],axis=1,inplace=True)
print('Merged table shape ', df.shape)
df.head().T
df.profile_report()
wheat_count = df.groupby(['Country']).sum().reset_index()
data = dict(type ='choropleth',
            locations = wheat_count['Country'],
            locationmode = 'country names',
            colorscale='sunsetdark',
            text = wheat_count['Country'],
            z = wheat_count['Nb_labelled_heads'],
            zmin=1250,
            colorbar = {'title':'Wheathead Count'}
           )
layout = dict(title ='Wheatheads per country',title_x=0.45, 
              geo = dict(landcolor = 'rgb(250, 250, 250)',projection={'type':"natural earth"},
                   oceancolor='rgb(0,191,255)',showocean=True,showcountries=True))
choromap = go.Figure(data=[data],layout=layout)
iplot(choromap)
wheat_cont = df[['Sub_Dataset_name','Row_Spacing','Sowing_density','Nb_labelled_heads','Nb_labelled_images','Country','Year']]
wheat_cont['continent'] = ['Asia','Asia','Europe','Europe','Europe','Europe',
                           'North America','Europe','Europe','Asia','Australia']
wheat_cont = wheat_cont.sort_values(by=['continent'])

fig, ax = plt.subplots(1,1, figsize=(14, 7), dpi=100)
ax.set_ylim(0, 52000)
height = 40000
ax.bar(wheat_cont['Sub_Dataset_name'], wheat_cont['Nb_labelled_heads'],  color="#e0e0e0", width=0.52, edgecolor='black')
color =  ['green',  'blue',  'orange',  'red']
span_range = [[0, 2], [3,3], [4, 9], [10,11]]
for idx, sub_title in enumerate(['Asia', 'Aus', 'Europe', 'N America']):
    ax.annotate(sub_title,xy=(sum(span_range[idx])/2 ,height),
                    xytext=(0,0), textcoords='offset points',
                    va="center", ha="center",
                    color="w", fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round4', pad=0.4, color=color[idx], alpha=0.6))
    ax.axvspan(span_range[idx][0]-0.4,span_range[idx][1]+0.4,  color=color[idx], alpha=0.07)
    ax.set_title(f'Continent wise wheatheads', fontsize=15, fontweight='bold', position=(0.50, 1.0+0.03))
plt.show()
data = dict(type ='choropleth',
            locations = df['Country'],
            locationmode = 'country names',
            colorscale = 'hsv',
            text = df['Country'],
            z = df['Nb_labelled_images'],
            zmin=20,
            colorbar = {'title':'Number of Images'}
           )
layout = dict(title ='Images produced per country',title_x=0.45, 
              geo = dict(landcolor = 'rgb(250, 250, 250)',projection={'type':"natural earth"},
                   oceancolor='rgb(85, 173, 240)',showocean=True,showcountries=True))
choromap = go.Figure(data=[data],layout=layout)
iplot(choromap)
fig, ax = plt.subplots(1,1, figsize=(14, 7), dpi=100)
ax.set_ylim(0, 1100)
height = 900
ax.bar(wheat_cont['Sub_Dataset_name'], wheat_cont['Nb_labelled_images'],  color="#e0e0e0", width=0.52, edgecolor='black')
color =  ['green',  'blue',  'orange',  'red']
span_range = [[0, 2], [3,3], [4, 9], [10,11]]
for idx, sub_title in enumerate(['Asia', 'Aus', 'Europe', 'N America']):
    ax.annotate(sub_title,xy=(sum(span_range[idx])/2 ,height),
                    xytext=(0,0), textcoords='offset points',
                    va="center", ha="center",
                    color="w", fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round4', pad=0.4, color=color[idx], alpha=0.6))
    ax.axvspan(span_range[idx][0]-0.4,span_range[idx][1]+0.4,  color=color[idx], alpha=0.07)
    ax.set_title(f'Continent wise Images', fontsize=15, fontweight='bold', position=(0.50, 1.0+0.03))
plt.show()
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12, 12))
    ax = sns.heatmap(corr,mask=mask,square=True,annot=True,fmt='0.2f',linewidths=.8,cmap="viridis",robust=True)
fig = px.scatter_matrix(df, dimensions=["Row_Spacing", "Sowing_density","Average_heads_per_image"], color='Country',
                        size='Nb_labelled_heads')
fig.update_layout(height=800, width=800)
fig.show()
import plotly.express as px
fig = px.scatter(wheat_cont, x='Row_Spacing', y="Nb_labelled_heads",size='Sowing_density'
           , color="continent", hover_name="Country", facet_col="continent")
fig.update_layout( width=850)
fig.show()
fig = px.scatter(df, y="Row_Spacing", x="Average_heads_per_image", color="Sub_Dataset_name",
                  size='Sowing_density', hover_data=['Country'])
fig.update_layout( title_text="Row_spacing vs Average_heads_per_image")
fig.show()
TEST_DIR = '/kaggle/input/global-wheat-detection/test/'
for i,img in enumerate(os.listdir(TEST_DIR)):
    print('Image Number:',i,"---Image Name:",img)
from sklearn.cluster import KMeans

def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect
# Load image and convert to a list of pixels
for img in os.listdir(TEST_DIR):
    ax = [None for _ in range(2)]
    fig = plt.figure(figsize=(16, 8))
    image = cv2.imread(TEST_DIR+img)
#     print(TEST_DIR+img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    print('Image_name is :',img)
    # Find and display most dominant colors
    cluster = KMeans(n_clusters=5).fit(reshape)
    visualize = visualize_colors(cluster, cluster.cluster_centers_)
#     plt.imshow(image)
    ax[0] = plt.subplot2grid((5,10), (1,4), colspan=5,rowspan=2)
    plt.imshow(visualize)
    plt.axis('off')
    ax[1] = plt.subplot2grid((5,10), (0,0), colspan=4,rowspan=4)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
TRAIN_PATH = '/kaggle/input/global-wheat-detection/train/'
def show(img1, img2):
    plt.figure(figsize=(18,18))
    plt.subplot(1, 2, 1)
    plt.title('Augmented Image')
    plt.imshow(img1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Original Image')
    plt.imshow(img2)
    plt.axis('off')

def augment(aug, image):
    return aug(image=image)['image']

def strong_aug_():
    return A.Compose([A.RGBShift(r_shift_limit=29.71366007, g_shift_limit=34.93698225, b_shift_limit=13.99685498,p=1)])
aug = strong_aug_()
image= Image.open(TRAIN_PATH+'00333207f.jpg')
img= Image.fromarray(augment(aug,np.array(image)))
show(img, image)
