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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image

from IPython.core.display import HTML 

%matplotlib inline 
train_df = pd.read_csv('../input/landmark-recognition-challenge/train.csv.zip')

test_df = pd.read_csv('../input/landmark-recognition-challenge/test.csv.zip')

submission = pd.read_csv('../input/landmark-recognition-challenge/sample_submission.csv.zip')
print("Train data shape -  rows:",train_df.shape[0]," columns:", train_df.shape[1])

print("Test data size -  rows:",test_df.shape[0]," columns:", test_df.shape[1])
train_df.head()
test_df.head()
submission.head()
# missing data in training data set

missing = train_df.isnull().sum()

all_val = train_df.count()



missing_train_df = pd.concat([missing, all_val], axis=1, keys=['Missing', 'All'])

missing_train_df
# missing data in training data set

missing = test_df.isnull().sum()

all_val = test_df.count()



missing_test_df = pd.concat([missing, all_val], axis=1, keys=['Missing', 'All'])

missing_test_df
train_df.nunique()
test_df.nunique()
# concatenate train and test datasets

concatenated = pd.concat([train_df, test_df])

# print the shape of the resulted data.frame

concatenated.shape
concatenated.nunique()
plt.figure(figsize = (8, 8))

plt.title('Landmark id density plot')

sns.kdeplot(train_df['landmark_id'], color="tomato", shade=True)

plt.show()
plt.figure(figsize = (8, 8))

plt.title('Landmark id distribuition and density plot')

sns.distplot(train_df['landmark_id'],color='green', kde=True,bins=100)

plt.show()
th10 = pd.DataFrame(train_df.landmark_id.value_counts().head(10))

th10.reset_index(level=0, inplace=True)

th10.columns = ['landmark_id','count']

th10
# Plot the most frequent landmark occurences

plt.figure(figsize = (6, 6))

plt.title('Most frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=th10,

            label="Count", color="darkgreen")

plt.show()
tb10 = pd.DataFrame(train_df.landmark_id.value_counts().tail(10))

tb10.reset_index(level=0, inplace=True)

tb10.columns = ['landmark_id','count']

tb10
# Plot the least frequent landmark occurences

plt.figure(figsize = (6,6))

plt.title('Least frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=tb10,

            label="Count", color="orange")

plt.show()
# Extract repositories names for train data

ll = list()

for path in train_df['url']:

    ll.append((path.split('//', 1)[1]).split('/', 1)[0])

train_df['site'] = ll

# Extract repositories names for test data

ll = list()

for path in test_df['url']:

    ll.append((path.split('//', 1)[1]).split('/', 1)[0])

test_df['site'] = ll
print("Train data shape -  rows:",train_df.shape[0]," columns:", train_df.shape[1])

print("Test data size -  rows:",test_df.shape[0]," columns:", test_df.shape[1])
train_df.head()
test_df.head()
train_site = pd.DataFrame(train_df.site.value_counts())

test_site = pd.DataFrame(test_df.site.value_counts())
train_site
trsite = pd.DataFrame(list(train_site.index),train_site['site'])

trsite.reset_index(level=0, inplace=True)

trsite.columns = ['count','site']

plt.figure(figsize = (6,6))

plt.title('Sites storing images - train dataset')

sns.set_color_codes("pastel")

sns.barplot(x = 'site', y="count", data=trsite, color="blue")

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

plt.show()
test_site
tesite = pd.DataFrame(list(test_site.index),test_site['site'])

tesite.reset_index(level=0, inplace=True)

tesite.columns = ['count','site']

plt.figure(figsize = (6,6))

plt.title('Sites storing images - test dataset')

sns.set_color_codes("pastel")

sns.barplot(x = 'site', y="count", data=tesite, color="magenta")

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

plt.show()
def displayLandmark(urls, landmarkName):

    

    img_style = "height: 60px; margin: 2px; float: left; border: 1px solid blue;"

    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.iteritems()])



    display(HTML(images_list))
IMAGES_NUMBER = 60

landmarkId = train_df['landmark_id'].value_counts().keys()[5]

urls = train_df[train_df['landmark_id'] == landmarkId]['url'].head(IMAGES_NUMBER)

displayLandmark(urls, "Petronas")
urls.head(1)
from PIL import Image

from PIL.ExifTags import TAGS, GPSTAGS





class ImageMetaData(object):

    '''

    Extract the exif data from any image. Data includes GPS coordinates, 

    Focal Length, Manufacture, and more.

    '''

    exif_data = None

    image = None



    def __init__(self, img_path):

        self.image = Image.open(img_path)

        print(self.image._getexif())

        #self.get_exif_data()

        #super(ImageMetaData, self).__init__()



    def get_exif_data(self):

        """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""

        exif_data = {}

        info = self.image._getexif()

        if info:

            for tag, value in info.items():

                decoded = TAGS.get(tag, tag)

                if decoded == "GPSInfo":

                    gps_data = {}

                    for t in value:

                        sub_decoded = GPSTAGS.get(t, t)

                        gps_data[sub_decoded] = value[t]



                    exif_data[decoded] = gps_data

                else:

                    exif_data[decoded] = value

        self.exif_data = exif_data

        return exif_data



    def get_if_exist(self, data, key):

        if key in data:

            return data[key]

        return None



    def convert_to_degress(self, value):



        """Helper function to convert the GPS coordinates 

        stored in the EXIF to degress in float format"""

        d0 = value[0][0]

        d1 = value[0][1]

        d = float(d0) / float(d1)



        m0 = value[1][0]

        m1 = value[1][1]

        m = float(m0) / float(m1)



        s0 = value[2][0]

        s1 = value[2][1]

        s = float(s0) / float(s1)



        return d + (m / 60.0) + (s / 3600.0)



    def get_lat_lng(self):

        """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""

        lat = None

        lng = None

        exif_data = self.get_exif_data()

        #print(exif_data)

        if "GPSInfo" in exif_data:      

            gps_info = exif_data["GPSInfo"]

            gps_latitude = self.get_if_exist(gps_info, "GPSLatitude")

            gps_latitude_ref = self.get_if_exist(gps_info, 'GPSLatitudeRef')

            gps_longitude = self.get_if_exist(gps_info, 'GPSLongitude')

            gps_longitude_ref = self.get_if_exist(gps_info, 'GPSLongitudeRef')

            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:

                lat = self.convert_to_degress(gps_latitude)

                if gps_latitude_ref != "N":                     

                    lat = 0 - lat

                lng = self.convert_to_degress(gps_longitude)

                if gps_longitude_ref != "E":

                    lng = 0 - lng

        return lat, lng

# submit the most freq label

submission['landmarks'] = '%d %2.2f' % (freq_label.index[0], freq_label.values[0])

submission.to_csv('submission.csv', index=False)



np.random.seed(2018)

r_idx = lambda : np.random.choice(freq_label.index, p = freq_label.values)



r_score = lambda idx: '%d %2.4f' % (freq_label.index[idx], freq_label.values[idx])

submission['landmarks'] = submission.id.map(lambda _: r_score(r_idx()))

submission.to_csv('rand_submission.csv', index=False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv('../input/landmark-recognition-challenge/train.csv.zip')

test_data = pd.read_csv('../input/landmark-recognition-challenge/test.csv.zip')

submission = pd.read_csv("../input/landmark-recognition-challenge/sample_submission.csv.zip")



print("Training data size",train_data.shape)

print("test data size",test_data.shape)

submission.head()
test_data.head()
# now open the URL

temp = 4444

print('id', train_data['id'][temp])

print('url:', train_data['url'][temp])

print('landmark id:', train_data['landmark_id'][temp])
train_data['landmark_id'].value_counts().hist()
# missing data in training data 

total = train_data.isnull().sum().sort_values(ascending = False)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)

missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
# missing data in test data 

total = test_data.isnull().sum().sort_values(ascending = False)

percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending = False)

missing_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_test_data.head()
# Occurance of landmark_id in decreasing order(Top categories)

temp = pd.DataFrame(train_data.landmark_id.value_counts().head(8))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp
# Plot the most frequent landmark_ids

plt.figure(figsize = (9, 8))

plt.title('Most frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

plt.show()
# Occurance of landmark_id in increasing order

temp = pd.DataFrame(train_data.landmark_id.value_counts().tail(8))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp


# Plot the least frequent landmark_ids

plt.figure(figsize = (9, 8))

plt.title('Least frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

plt.show()
# Unique URL's

train_data.nunique()
#Class distribution

plt.figure(figsize = (10, 8))

plt.title('Category Distribuition')

sns.distplot(train_data['landmark_id'])



plt.show()


print("Number of classes under 20 occurences",(train_data['landmark_id'].value_counts() <= 20).sum(),'out of total number of categories',len(train_data['landmark_id'].unique()))
from IPython.display import Image

from IPython.core.display import HTML 



def display_category(urls, category_name):

    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"

    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])



    display(HTML(images_list))
category = train_data['landmark_id'].value_counts().keys()[0]

urls = train_data[train_data['landmark_id'] == category]['url']

display_category(urls, "")
# Extract site_names for train data

temp_list = list()

for path in train_data['url']:

    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])

train_data['site_name'] = temp_list

# Extract site_names for test data

temp_list = list()

for path in test_data['url']:

    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])

test_data['site_name'] = temp_list
print("Training data size",train_data.shape)

print("test data size",test_data.shape)
train_data.head(8)
test_data.head()
# Occurance of site in decreasing order(Top categories)

temp = pd.DataFrame(train_data.site_name.value_counts())

temp.reset_index(inplace=True)

temp.columns = ['site_name','count']

temp
# Plot the Sites with their count

plt.figure(figsize = (9, 8))

plt.title('Sites with their count')

sns.set_color_codes("pastel")

sns.barplot(x="site_name", y="count", data=temp,

            label="Count")

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

plt.show()
# Occurance of site in decreasing order(Top categories)

temp = pd.DataFrame(test_data.site_name.value_counts())

temp.reset_index(inplace=True)

temp.columns = ['site_name','count']

temp
# Plot the Sites with their count

plt.figure(figsize = (9, 8))

plt.title('Sites with their count')

sns.set_color_codes("pastel")

sns.barplot(x="site_name", y="count", data=temp,

            label="Count")

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

plt.show()
# take the most frequent label

freq_label = train_df['landmark_id'].value_counts()/train_df['landmark_id'].value_counts().sum()



# submit the most freq label

submission['landmarks'] = '%d %2.2f' % (freq_label.index[0], freq_label.values[0])

submission.to_csv('submission.csv', index=False)



np.random.seed(2018)

r_idx = lambda : np.random.choice(freq_label.index, p = freq_label.values)



r_score = lambda idx: '%d %2.4f' % (freq_label.index[idx], freq_label.values[idx])

submission['landmarks'] = submission.id.map(lambda _: r_score(r_idx()))

submission.to_csv('rand_submission.csv', index=False)