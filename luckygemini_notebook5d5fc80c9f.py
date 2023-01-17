# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
train_photos = pd.read_csv('../input/train_photo_to_biz_ids.csv')

import os

print(''.join([str(train_photos.photo_id[0]),'.jpg']))



from PIL import Image

im = Image.open(os.path.join('../input/','train_photos',''.join([str(train_photos.photo_id[5]),'.jpg'])))

plt.imshow(im)





#0: good_for_lunch

#1: good_for_dinner

#2: takes_reservations

#3: outdoor_seating

#4: restaurant_is_expensive

#5: has_alcohol

#6: has_table_service

#7: ambience_is_classy

#8: good_for_kids
train_attributes = pd.read_csv('../input/train.csv')



list(train_attributes)
train_attributes['labels_list'] = train_attributes['labels'].str.split(' ')

train_attributes['outdoor'] = train_attributes['labels'].str.contains('3')

outdoor_businesses = train_attributes[train_attributes.outdoor==True].business_id.tolist()

outdoor_photos = train_photos[train_photos.business_id.isin(outdoor_businesses)].photo_id.tolist()
num_images_for_show = 5



photos_to_show = np.random.choice(outdoor_photos,num_images_for_show**2)



for x in range(num_images_for_show ** 2):

        

        plt.subplot(num_images_for_show, num_images_for_show, x+1)

        im = Image.open(os.path.join('../input/','train_photos',''.join([str(photos_to_show[x]),'.jpg'])))

        plt.imshow(im)

        plt.axis('off')
#0: good_for_lunch

#1: good_for_dinner

#2: takes_reservations

#3: outdoor_seating

#4: restaurant_is_expensive

#5: has_alcohol

#6: has_table_service

#7: ambience_is_classy

#8: good_for_kids



train_attributes['labels_list'] = train_attributes['labels'].str.split(' ')

train_attributes['kids'] = train_attributes['labels'].str.contains('8')

kids_businesses = train_attributes[train_attributes.kids==True].business_id.tolist()

kidsRes_photos = train_photos[train_photos.business_id.isin(kids_businesses)].photo_id.tolist()
num_images_for_show = 5



photos_to_show = np.random.choice(kidsRes_photos,num_images_for_show**2)



for x in range(num_images_for_show ** 2):

        

        plt.subplot(num_images_for_show, num_images_for_show, x+1)

        im = Image.open(os.path.join('../input/','train_photos',''.join([str(photos_to_show[x]),'.jpg'])))

        plt.imshow(im)

        plt.axis('off')
#create a new dataset
