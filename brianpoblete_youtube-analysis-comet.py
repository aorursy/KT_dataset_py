# imports 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

ca_videos = pd.read_csv('../input/CAvideos.csv')
de_videos = pd.read_csv('../input/DEvideos.csv')
fr_videos = pd.read_csv('../input/FRvideos.csv')
gb_videos = pd.read_csv('../input/GBvideos.csv')
us_videos = pd.read_csv('../input/USvideos.csv')

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]
categories = ['Film & Animation',"Autos & Vehicles", 'none3', 'none4','none5','none6','none7','none8','none9','Music','none11','none12','none13','none14','Pets & Animals','none16','Sports','Short Movies','Travel & Events','Gaming','Videoblogging','People & Blogs','Comedy','Entertainment','News and Politics','Howto and Style','Education','Science & Technology','Nonprofit & Activism','Movies','Anime/Animation','Action/Adventure','Classics','Comedy','Documentary','Drama','Family','Foreign','Horror','Sci-fi-Fantasy','Thriller','Shorts','Shows','Trailers']
de_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for entry in de_videos['category_id']:
    de_count[(entry) - 1] += 1
    
de_updated_count = []
de_updated_cats = []
for i in range(len(de_count)):
    if(de_count[i] != 0):
        de_updated_count.append(de_count[i])
        de_updated_cats.append(categories[i])
        
index = np.arange(len(de_updated_cats))

plt.bar(index, de_updated_count)
plt.xlabel('Genre', fontsize=20)
plt.ylabel('No. of Videos', fontsize=20)
plt.xticks(index, de_updated_cats, fontsize=12, rotation=30)
plt.title('No. of YouTube Videos per Genre - Deütschland', fontsize=30)
fig = plt.gcf()
fig.set_size_inches(20,5)
plt.show()
gb_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#              1                 2                   3        4       5       6       7       8       9       10       11      12       13       14       15                16      17       18             19                20        21              22               23      24               25                 26                27          28                     29                     30        31                 32               33          34       35          36        37       38        39      40               41        42       43       44 


for entry in gb_videos['category_id']:
    gb_count[(entry) - 1] += 1

gb_updated_count = []
gb_updated_cats = []
for i in range(len(gb_count)):
    if(gb_count[i] != 0):
        gb_updated_count.append(gb_count[i])
        gb_updated_cats.append(categories[i])
    
        
#print(gb_updated_count)
#print(gb_updated_cats)

index = np.arange(len(gb_updated_cats))

plt.bar(index, gb_updated_count)
plt.xlabel('Genre', fontsize=20)
plt.ylabel('No of Videos', fontsize=20)
plt.xticks(index, gb_updated_cats, fontsize=12, rotation=30)
plt.title('No of videos per genre Great Britain', fontsize=30)
fig = plt.gcf()
fig.set_size_inches(20,5)
plt.show()
fr_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for entry in fr_videos['category_id']:
    if(entry != 129): #had to make a special case for the item in the last index because there is a problem with the dataset there
        fr_count[(entry) - 1] += 1
    
fr_updated_count = []
fr_updated_cats = []
for i in range(len(fr_count)):
    if(fr_count[i] != 0):
        fr_updated_count.append(fr_count[i])
        fr_updated_cats.append(categories[i])

index = np.arange(len(fr_updated_cats))

plt.bar(index, fr_updated_count)
plt.xlabel('Genre', fontsize=20)
plt.ylabel('No. of Videos', fontsize=20)
plt.xticks(index, fr_updated_cats, fontsize=12, rotation=30)
plt.title('No. of YouTube Videos per Genre - France', fontsize=30)
fig = plt.gcf()
fig.set_size_inches(20,5)
plt.show()
ca_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for entry in ca_videos['category_id']:
    ca_count[(entry) - 1] += 1
    
ca_updated_count = []
ca_updated_cats = []
for i in range(len(ca_count)):
    if(ca_count[i] != 0):
        ca_updated_count.append(ca_count[i])
        ca_updated_cats.append(categories[i])
        
index = np.arange(len(ca_updated_cats))

plt.bar(index, ca_updated_count)
plt.xlabel('Genre', fontsize=20)
plt.ylabel('No. of Videos', fontsize=20)
plt.xticks(index, ca_updated_cats, fontsize=12, rotation=30)
plt.title('No. of YouTube Videos per Genre - Canada', fontsize=30)
fig = plt.gcf()
fig.set_size_inches(20,5)
plt.show()
us_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for entry in us_videos['category_id']:
    us_count[(entry) - 1] += 1
    
us_updated_count = []
us_updated_cats = []
for i in range(len(us_count)):
    if(us_count[i] != 0):
        us_updated_count.append(us_count[i])
        us_updated_cats.append(categories[i])
        
index = np.arange(len(us_updated_cats))

plt.bar(index, us_updated_count)
plt.xlabel('Genre', fontsize=20)
plt.ylabel('No. of Videos', fontsize=20)
plt.xticks(index, us_updated_cats, fontsize=12, rotation=30)
plt.title('No. of YouTube Videos per Genre - USA', fontsize=30)
fig = plt.gcf()
fig.set_size_inches(20,5)
plt.show()