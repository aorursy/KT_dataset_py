from pathlib import Path

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image
# modified from https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resnet50-binary-cropped

import cv2

def resize_images(location, name, extension, resize_location, desired_size = 1024):

    img = cv2.imread(f"{location}/{name}.{extension}")

    

    img = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)

    

    contours,hierarchy = cv2.findContours(gray,

                                          cv2.RETR_EXTERNAL,

                                          cv2.CHAIN_APPROX_SIMPLE)

    contours = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(contours)



    if w>200 and h>200:

        new_img = img[y:y+h,x:x+w]

        height, width, _= new_img.shape



        if max([height, width]) > desired_size:

            ratio = float(desired_size/max([height, width]))

            new_img = cv2.resize(new_img, 

                                 tuple([int(width*ratio), int(height*ratio)]), 

                                 interpolation = cv2.INTER_CUBIC)

            

        cv2.imwrite(f'{resize_location}/{name}.jpg', new_img)

    else:

        print(f'No bounding for {name}')

        cv2.imwrite(f'{resize_location}/{name}.jpg', img)
path = Path("../input")

train_19_df = pd.read_csv(path/'labels'/'trainLabels19.csv')

test_19_df = pd.read_csv(path/'labels'/'testImages19.csv')

train_15_df = pd.read_csv(path/'labels'/'trainLabels15.csv')

test_15_df = pd.read_csv(path/'labels'/'testLabels15.csv')



train_19_df.head() 
test_19_df.head() 
train_15_df.head() 
test_15_df.head() 
test_15_df.rename(index=str, columns={"image": "id_code", "level":"diagnosis"}, inplace=True)

train_15_df.rename(index=str, columns={"image": "id_code", "level":"diagnosis"}, inplace=True)



data_df = train_19_df.append([test_15_df.drop('Usage', axis=1), train_15_df], ignore_index = True)
def add_counts_to_bars(ax):

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x() + p.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom')

    return ax



plt.figure(figsize=(12,10))



plt.subplot(2,2,1)

ax1 = sns.countplot('diagnosis', data=train_19_df)

ax1 = add_counts_to_bars(ax1)

plt.ylabel("Count")

plt.xlabel("Diagnosis")

plt.title('2019 Training Set')



plt.subplot(2,2,2)

ax2 = sns.countplot('diagnosis', data=train_15_df)

ax2 = add_counts_to_bars(ax2)

plt.ylabel("Count")

plt.xlabel("Diagnosis")

plt.title('2015 Training Set')



plt.subplot(2,2,3)

ax3 = sns.countplot('diagnosis', data=test_15_df)

ax3 = add_counts_to_bars(ax3)

plt.ylabel("Count")

plt.xlabel("Diagnosis")

plt.title('2015 Test Set')



plt.subplot(2,2,4)

ax4 = sns.countplot('diagnosis', data=data_df)

ax4 = add_counts_to_bars(ax4)

plt.ylabel("Count")

plt.xlabel("Diagnosis")

plt.title('All Labeled Images')



sns.despine()

plt.tight_layout(h_pad=2)

plt.show()
def plot_images(df, location, title):

    plt.figure(figsize=(16,12))



    i = 0

    sample = df.sample(12)

    for row in sample.iterrows():

        name = row[1][0]

        with Image.open(f'{location}/{name}.jpg') as img:

            i += 1

            plt.subplot(3,4,i)

            img_title = ' '.join(['Diagnosis:', str(row[1][1]), name])

            plt.title(img_title, fontsize=10)

            plt.imshow(img)



    plt.subplots_adjust(top=1.25)

    plt.tight_layout()

    plt.suptitle(title, fontsize=14)

    return plt
plt = plot_images(train_19_df, str(path/'resized train 19'), 'Sample of 2019 Training Images')

plt.show()
location = str(path/'resized test 19')

title = 'Sample of 2019 Test Images'



plt.figure(figsize=(16,12))



i = 0

sample = test_19_df.sample(12)

for row in sample.iterrows():

    name = row[1][0]

    with Image.open(f'{location}/{name}.jpg') as img:

        i += 1

        plt.subplot(3,4,i)

        img_title = name

        plt.title(img_title, fontsize=10)

        plt.imshow(img)



plt.subplots_adjust(top=1.25)

plt.tight_layout()

plt.suptitle(title, fontsize=14)

plt.show()
plt = plot_images(train_15_df, str(path/'resized train 15'), 'Sample of 2015 Training Images')

plt.show()
plt = plot_images(test_15_df, str(path/'resized test 15'), 'Sample of 2015 Test Images')

plt.show()