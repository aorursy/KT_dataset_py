# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import random
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import random
import os
print(os.listdir("../input/retail-product-checkout-dataset"))
import json
#os.chdir('../input/retail-product-checkout-dataset')
with open('../input/retail-product-checkout-dataset/instances_train2019.json') as json_data:
    data = json.load(json_data)
    
type(data)

data
raw_Chinese_name_df = pd.DataFrame(data['__raw_Chinese_name_df'])
print(raw_Chinese_name_df)
categories_df = pd.DataFrame(data['categories'])
categories_df = categories_df[["supercategory", "name"]]
print(categories_df)
categories_df = pd.DataFrame(data['categories'])
categories_df = categories_df.rename(columns={'id': 'category_id'})

print(categories_df)
retail_product_info = pd.merge(categories_df, raw_Chinese_name_df, on=["category_id"])
images_df = pd.DataFrame(data['images'])
print(images_df)
import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.image import imread
def display_sample_data(image_data):
  # plot first few images
  for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    image = imread("../input/retail-product-checkout-dataset/train2019/"+image_data[i])
    plt.imshow(image)
    print(image_data[i])
  # show the figure
  plt.show()
display_sample_data(images_df['file_name'])
def display_image(image_data):
    interp = 'bilinear'
    fig, axs = plt.subplots(nrows=4, sharex=True, figsize=(5, 5))
    for i in range(4):
        image = imread("../input/retail-product-checkout-dataset/train2019/"+image_data[i])
        axs[i].set_title(image_data[i])
        axs[i].imshow(image, origin='upper', interpolation=interp)
    plt.show()
    
display_image(images_df['file_name'])
print(retail_product_info)
retail_product_info = retail_product_info.drop(['name_x','ind','sku_class'], axis = 1)
retail_product_info.head()
retail_product_info.info()
retail_product_info.describe().transpose()
retail_product_info.isnull().sum()

plt.figure(figsize=(15,5))
sns.heatmap(retail_product_info.corr(), cbar = False,annot=True,cmap='viridis')
supercategory_groupby = categories_df.groupby( [ "supercategory"]).count().reset_index()
print(supercategory_groupby)
plt.figure(figsize=(30, 6))
sns.set(style="whitegrid")
g = sns.catplot(x="name", y="supercategory", data=supercategory_groupby,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Super Category wise")

name_groupby = categories_df.groupby( [ "name"]).count().reset_index()
print(name_groupby)