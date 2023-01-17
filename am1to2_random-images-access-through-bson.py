from __future__ import print_function



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import io

import bson

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

import random
def extract_categories_df(num_images):

    img_category = list()

    item_locs_list = list()

    items_len_list = list()

    pic_ind_list = list()

    prod_id_list = list()



    with open('../input/train.bson', 'rb') as f:

        data = bson.decode_file_iter(f)

        last_item_loc = 0

        item_len = 0

        for c, d in enumerate(data):

            loc = f.tell()

            item_len = loc - last_item_loc

            category_id = d['category_id']

            prod_id = d["_id"]



            for e, pic in enumerate(d['imgs']):

                prod_id_list.append(prod_id)

                img_category.append(category_id)

                item_locs_list.append(last_item_loc)

                items_len_list.append(item_len)

                pic_ind_list.append(e)

                

                if num_images is not None:

                    if len(img_category) >= num_images:

                        break

            

            last_item_loc = loc

            

            if num_images is not None:

                if len(img_category) >= num_images:

                    break

    

    f.close()

    df_dict = {

        'category': img_category,

        "prod_id": prod_id_list,

        "img_id": range(len(img_category)),

        "item_loc": item_locs_list,

        "item_len": items_len_list,

        "pic_ind": pic_ind_list

    }

    df = pd.DataFrame(df_dict)

    df.to_csv("all_images_categories.csv", index=False)

        

    return df



def get_image(image_id,data_df,fh):

    img_info = data_df[data_df["img_id"] == image_id]

    item_loc = img_info["item_loc"].values[0]

    item_len = img_info["item_len"].values[0]

    pic_ind = img_info["pic_ind"].values[0]

    fh.seek(item_loc)

    item_data = fh.read(item_len)

    d = bson.BSON.decode(item_data)

    

    picture = imread(io.BytesIO(d["imgs"][pic_ind]['picture']))

    return picture
cat_df = extract_categories_df(None)
print(cat_df.iloc[0])
train_fh = open('../input/train.bson', 'rb')
pic = get_image(0,cat_df,train_fh)

plt.imshow(pic);

plt.show()

pic = np.rot90(pic)

plt.imshow(pic);

plt.show()

pic = np.flip(pic,axis=0)

plt.imshow(pic);

plt.show()
pic = get_image(500,cat_df,train_fh)

plt.imshow(pic);
pic = get_image(20,cat_df,train_fh)

plt.imshow(pic);
for i in random.sample(range(len(cat_df)),20):

    print(i)

    pic = get_image(i,cat_df,train_fh)

    plt.imshow(pic);

    plt.show();
def extract_test_df(num_images):

    prod_id_list = list()

    item_locs_list = list()

    items_len_list = list()

    pic_ind_list = list()



    with open('../input/test.bson', 'rb') as f:

        data = bson.decode_file_iter(f)

        last_item_loc = 0

        item_len = 0

        for c, d in enumerate(data):

            loc = f.tell()

            item_len = loc - last_item_loc

            prod_id = d["_id"]



            for e, pic in enumerate(d['imgs']):

                prod_id_list.append(prod_id)

                item_locs_list.append(last_item_loc)

                items_len_list.append(item_len)

                pic_ind_list.append(e)

                

                if num_images is not None:

                    if len(prod_id) >= num_images:

                        break

            

            last_item_loc = loc

            

            if num_images is not None:

                if len(prod_id) >= num_images:

                    break

    

    f.close()

    df_dict = {

        'prod_id': prod_id_list,

        "img_id": range(len(prod_id_list)),

        "item_loc": item_locs_list,

        "item_len": items_len_list,

        "pic_ind": pic_ind_list

    }

    df = pd.DataFrame(df_dict)

    df.to_csv("all_test_images_categories.csv", index=False)

        

    return df
test_cat_df = extract_test_df(None)
test_fh = open('../input/test.bson', 'rb')

for i in random.sample(range(len(test_cat_df)),20):

    print(i)

    pic = get_image(i,test_cat_df,test_fh)

    plt.imshow(pic);

    plt.show();
img_num_train = cat_df["pic_ind"].value_counts()

img_num_train.plot(kind="bar")

plt.show()

img_num_train = cat_df["category"].value_counts()

img_num_train.plot(kind="bar")

plt.show()
img_num_test = test_cat_df["pic_ind"].value_counts()

img_num_test.plot(kind="bar")

plt.show()
## Data Statistics

print("## Total number of images in train = {:d}".format(len(cat_df)))

print("## Total number of products in train = {:d}".format(len(pd.unique(cat_df["prod_id"]))))

print("## Total number of categories in train = {:d}".format(len(pd.unique(cat_df["category"]))))

print("## Total number of images in test = {:d}".format(len(test_cat_df)))

print("## Total number of products in test = {:d}".format(len(pd.unique(test_cat_df["prod_id"]))))