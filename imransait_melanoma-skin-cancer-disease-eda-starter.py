import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# pd.set_option('display.max_rows', None)

import gc

import os



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

import random

import cv2



base_data_folder = "../input/siim-isic-melanoma-classification"

def read_train():

    train=pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

    print('Train set : Total Images are {} and columns are {}'.format(train.shape[0], train.shape[1])) 

    return train



def read_test():

    test=pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

    print('Train set : Total Images are {} and columns are {}'.format(test.shape[0], test.shape[1])) 

    return test



train_df = read_train()

test_df = read_test()
train_df.tail(5)
test_df.tail(5)
import os

print(os.listdir("../input/siim-isic-melanoma-classification"))


folder_path = os.path.join(base_data_folder, "jpeg")

all_data = os.listdir(folder_path)

for folders in all_data:

    files_in_path = os.path.join(folder_path, folders)

    count = len([cnt for cnt in os.scandir(files_in_path)])

    print('Jpeg Images from {} are : {}'.format(files_in_path, count)) 

    print("Size is " + str(round(os.path.getsize(files_in_path) / 1000000, 2)) + 'MB')



    



train_folder_path = os.path.join(base_data_folder, "train")

count = len([cnt for cnt in os.scandir(train_folder_path)])

print('\n\nDICOM Images from {} are : {}'.format(train_folder_path, count)) 

print("Size is " + str(round(os.path.getsize(train_folder_path) / 1000000, 2)) + 'MB')





test_folder_path = os.path.join(base_data_folder, "test")

count = len([cnt for cnt in os.scandir(test_folder_path)])

print('DICOM Images from {} are : {}'.format(test_folder_path, count)) 

print("Size is " + str(round(os.path.getsize(test_folder_path) / 1000000, 2)) + 'MB')









tf_folder_path = os.path.join(base_data_folder, "tfrecords")

count = len([cnt for cnt in os.scandir(tf_folder_path)])

print('\n\ntfrecords from {} are : {}'.format(tf_folder_path, count)) 

train_df.info()
train_df.isna().sum()
test_df.info()
test_df.isna().sum()
train_df[train_df.sex.isna() == True]
train_df[train_df.sex.isna() == True].patient_id.unique()

train_df[train_df.age_approx.isna() == True].patient_id.unique()

train_df[train_df.anatom_site_general_challenge.isna() == True].patient_id.count()

train_df[train_df.anatom_site_general_challenge.isna() == True].patient_id.unique()

train_df['anatom_site_general_challenge'].value_counts(normalize=True).sort_values().plot(kind='bar')

train_df.head()
for cols in train_df.columns:

    if (cols != 'patient_id') and (cols != 'image_name'):

        plt.figure()

        train_df[cols].value_counts().head(10).plot(kind='bar',title=cols)
# 1. Which gender is highly impacted? 

# 2. Which age people highly impacted?

# 3. by location where issue comes in?

# 4. what daignosis been done to them?
temp=train_df.groupby(['target','sex'])['benign_malignant'].count().to_frame().reset_index()

sns.catplot(x='target',y='benign_malignant', hue='sex',data=temp,kind='bar')
sns.stripplot(x="sex", y="age_approx", hue="benign_malignant", data=train_df,jitter=0.20);
temp=train_df.groupby(['anatom_site_general_challenge','sex','benign_malignant'])['target'].count().to_frame().reset_index()

temp
temp=train_df.groupby(['anatom_site_general_challenge','sex','benign_malignant','diagnosis'])['target'].count().to_frame().reset_index()

temp
train_df.head()
def show_images(df, base_data_folder, what_to_show):

    n_row, n_col = 3,3

    img_dir = base_data_folder+what_to_show

    

    _, axs = plt.subplots(n_row, n_col, figsize=(10, 10))

    axs = axs.flatten()

    

    for ax in axs:

        random_image = np.random.choice(df+'.jpg')

        img = cv2.imread(os.path.join(img_dir, random_image),cv2.IMREAD_COLOR)

        ax.imshow(img)

    plt.show()

    plt.tight_layout()  

    

    

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    gc.collect()
what_to_show = "/jpeg/train"

what_images = train_df[train_df['benign_malignant'] == 'benign']['image_name'].values

show_images(what_images, base_data_folder, what_to_show)



what_to_show = "/jpeg/train"

what_images = train_df[train_df['benign_malignant'] == 'malignant']['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'benign') & (train_df['sex'] == 'male')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)

what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'benign') & (train_df['sex'] == 'female')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)

what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'malignant') & (train_df['sex'] == 'male')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'malignant') & (train_df['sex'] == 'female')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
train_df.head(5)
train_df['anatom_site_general_challenge'].value_counts()
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'benign') & (train_df['anatom_site_general_challenge'] == 'oral/genital')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'malignant') & (train_df['anatom_site_general_challenge'] == 'oral/genital')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'benign') & (train_df['anatom_site_general_challenge'] == 'palms/soles')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'malignant') & (train_df['anatom_site_general_challenge'] == 'palms/soles')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'benign') & (train_df['anatom_site_general_challenge'] == 'head/neck')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'malignant') & (train_df['anatom_site_general_challenge'] == 'head/neck')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'benign') & (train_df['anatom_site_general_challenge'] == 'upper extremity')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'malignant') & (train_df['anatom_site_general_challenge'] == 'upper extremity')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'benign') & (train_df['anatom_site_general_challenge'] == 'lower extremity')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'malignant') & (train_df['anatom_site_general_challenge'] == 'lower extremity')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'benign') & (train_df['anatom_site_general_challenge'] == 'torso')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
what_to_show = "/jpeg/train"

what_images = train_df[(train_df['benign_malignant'] == 'malignant') & (train_df['anatom_site_general_challenge'] == 'torso')]['image_name'].values

show_images(what_images, base_data_folder, what_to_show)
train_df.head()
malignant_images = train_df[train_df['benign_malignant'] == 'malignant']
what_to_show = "/jpeg/train"

what_images = malignant_images['image_name'].values

show_images(what_images, base_data_folder, what_to_show)



# import numpy 

# from matplotlib import pyplot as plt



# def gaussian_kernel(size, size_y=None):

#     size = int(size)

#     if not size_y:

#         size_y = size

#     else:

#         size_y = int(size_y)

#     x, y = numpy.mgrid[-size:size+1, -size_y:size_y+1]

#     g = numpy.exp(-(x**2/float(size)+y**2/float(size_y)))

#     return g / g.sum()
# malignant_images.groupby(['sex','age_approx'])['target'].value_counts()



file_path = "../input/siim-isic-melanoma-classification//jpeg/train/ISIC_0080817.jpg"

img = cv2.imread(file_path,cv2.IMREAD_COLOR)



kernel = np.ones((3,3), np.float32) / 9

filt_2D = cv2.filter2D(img, -1, kernel)

blur = cv2.blur(img,(3,3))

gaussian_blur = cv2.GaussianBlur(img, (3,3),0)

median_blur = cv2.medianBlur(img, 3)

bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)



n_row, n_col = 2,3



_, axs = plt.subplots(n_row, n_col, figsize=(11, 5))

axs = axs.flatten()

    

axs[0].imshow(img)

axs[1].imshow(filt_2D)

axs[2].imshow(blur)

axs[3].imshow(gaussian_blur)

axs[4].imshow(median_blur)

axs[5].imshow(bilateral_blur)







# malignant_images.groupby(['sex','age_approx'])['target'].value_counts()



img = cv2.imread(file_path,cv2.IMREAD_COLOR)

img_edges = cv2.Canny(img, 75, 75)





n_row, n_col = 1,2



_, axs = plt.subplots(n_row, n_col, figsize=(11, 5))

axs = axs.flatten()

    

axs[0].imshow(img)

axs[1].imshow(img_edges)






