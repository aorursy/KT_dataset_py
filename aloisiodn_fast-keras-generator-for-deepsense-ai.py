import os, sys, math, io

import numpy as np

import pandas as pd

import multiprocessing as mp

import bson

import struct



%matplotlib inline

import matplotlib.pyplot as plt



import keras

from keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf



from collections import defaultdict

from tqdm import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_dir = "../input/"



train_bson_path = os.path.join(data_dir, "train.bson")

num_train_products = 7069896



# train_bson_path = os.path.join(data_dir, "train_example.bson")

# num_train_products = 82



test_bson_path = os.path.join(data_dir, "test.bson")

num_test_products = 1768182
categories_path = os.path.join(data_dir, "category_names.csv")

categories_df = pd.read_csv(categories_path, index_col="category_id")



# Maps the category_id to an integer index. This is what we'll use to

# one-hot encode the labels.

categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)



categories_df.to_csv("categories.csv")

categories_df.head()
def make_category_tables():

    cat2idx = {}

    idx2cat = {}

    for ir in categories_df.itertuples():

        category_id = ir[0]

        category_idx = ir[4]

        cat2idx[category_id] = category_idx

        idx2cat[category_idx] = category_id

    return cat2idx, idx2cat
cat2idx, idx2cat = make_category_tables()
# Test if it works:

cat2idx[1000012755], idx2cat[4]
def read_bson(bson_path, num_records, with_categories):

    rows = {}

    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:

        offset = 0

        while True:

            item_length_bytes = f.read(4)

            if len(item_length_bytes) == 0:

                break



            length = struct.unpack("<i", item_length_bytes)[0]



            f.seek(offset)

            item_data = f.read(length)

            assert len(item_data) == length



            item = bson.BSON.decode(item_data)

            product_id = item["_id"]

            num_imgs = len(item["imgs"])



            row = [num_imgs, offset, length]

            if with_categories:

                row += [item["category_id"]]

            rows[product_id] = row



            offset += length

            f.seek(offset)

            pbar.update()



    columns = ["num_imgs", "offset", "length"]

    if with_categories:

        columns += ["category_id"]



    df = pd.DataFrame.from_dict(rows, orient="index")

    df.index.name = "product_id"

    df.columns = columns

    df.sort_index(inplace=True)

    return df
%time train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)
train_offsets_df.head()
train_offsets_df.to_csv("train_offsets.csv")
# How many products?

len(train_offsets_df)
# How many categories?

len(train_offsets_df["category_id"].unique())
# How many images in total?

train_offsets_df["num_imgs"].sum()
def make_val_set(df, split_percentage=0.2, drop_percentage=0.):

    # Find the product_ids for each category.

    category_dict = defaultdict(list)

    for ir in tqdm(df.itertuples()):

        category_dict[ir[4]].append(ir[0])



    train_list = []

    val_list = []

    with tqdm(total=len(df)) as pbar:

        for category_id, product_ids in category_dict.items():

            category_idx = cat2idx[category_id]



            # Randomly remove products to make the dataset smaller.

            keep_size = int(len(product_ids) * (1. - drop_percentage))

            if keep_size < len(product_ids):

                product_ids = np.random.choice(product_ids, keep_size, replace=False)



            # Randomly choose the products that become part of the validation set.

            val_size = int(len(product_ids) * split_percentage)

            if val_size > 0:

                val_ids = np.random.choice(product_ids, val_size, replace=False)

            else:

                val_ids = []



            # Create a new row for each image.

            for product_id in product_ids:

                row = [product_id, category_idx]

                for img_idx in range(df.loc[product_id, "num_imgs"]):

                    if product_id in val_ids:

                        val_list.append(row + [img_idx])

                    else:

                        train_list.append(row + [img_idx])

                pbar.update()

                

    columns = ["product_id", "category_idx", "img_idx"]

    train_df = pd.DataFrame(train_list, columns=columns)

    val_df = pd.DataFrame(val_list, columns=columns)   

    return train_df, val_df
train_images_df, val_images_df = make_val_set(train_offsets_df, split_percentage=0.2, 

                                              drop_percentage=0.9)
train_images_df.head()
val_images_df.head()
print("Number of training images:", len(train_images_df))

print("Number of validation images:", len(val_images_df))

print("Total images:", len(train_images_df) + len(val_images_df))
train_images_df.to_csv("train_images.csv")

val_images_df.to_csv("val_images.csv")
categories_df = pd.read_csv("categories.csv", index_col=0)

cat2idx, idx2cat = make_category_tables()



train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)

train_images_df = pd.read_csv("train_images.csv", index_col=0)

val_images_df = pd.read_csv("val_images.csv", index_col=0)
train_offsets_df.head()
train_images_df.head()
train_images_df = train_images_df.sample(frac=1).reset_index(drop=True)
train_images_df.head()
val_images_df.head()
#Uses LabelEncoder for class_id encoding

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(pd.read_csv(categories_path).category_id)
#Testing the encoder

original=le.classes_[:5]

print("5 original classes:", original)

encoded=le.transform(original)

print("5 encoded classes:",encoded)

print("getting back the original classes:", le.inverse_transform(encoded))

classes = train_images_df.category_idx.value_counts()[-400:].index.values
def create_bin_file2(images_df, offsets_df, bson_file_name, bin_file_name, encoder):

    with open(bson_file_name, 'rb') as bson_file, open(bin_file_name, 'wb') as bin_file:    

        #the line above was modified to get only less frequent classes

        for index, row in images_df.loc[images_df.category_idx.isin(classes)].iterrows():

            offset_row = offsets_df.loc[row.product_id]

            bson_file.seek(offset_row["offset"])

            item_data = bson_file.read(offset_row["length"])



            # Grab the image from the product.

            item = bson.BSON.decode(item_data)

            img_idx = row["img_idx"]

            bson_img = item["imgs"][img_idx]["picture"]



            #write down the encoded class, the size of the img and the img it self 

            encoded_class = encoder.transform([offset_row.category_id])[0]

            img_size = len(bson_img)

            bin_file.write(struct.pack('<ii', encoded_class, img_size))   

            bin_file.write(bytes(bson_img))   

        bin_file.close()

        bson_file.close()
#test function

def bin_file_test(file_name, encoder, n=3):

    with open(file_name, 'rb') as bin_file:    

        count = 0

        while count<n:

            count += 1 

            buffer=bin_file.read(8)

            encoded_class, length = struct.unpack("<ii", buffer)

            bson_img = bin_file.read(length)

            img = load_img(io.BytesIO(bson_img), target_size=(180,180))

            plt.figure()

            plt.imshow(img)

            plt.text(5,20,

                     "%d Class: %s (size: %d)" %(count, encoder.inverse_transform(encoded_class), length),

                    backgroundcolor='0.75',alpha=.5)

#create train bin file and test it!!!

img_df = train_images_df 

create_bin_file2(img_df, train_offsets_df, train_bson_path, 'train_sample.bin', le)

bin_file_test('train_sample.bin', le, n=9)
#create val bin file and test it

img_df = val_images_df 

create_bin_file(img_df, train_offsets_df, train_bson_path, 'val_sample.bin', le)

bin_file_test('val_sample.bin', le)