import os

import numpy as np

import pandas as pd
num_test_images = len(os.listdir("../input/test"))

num_train_images = len(os.listdir("../input/train"))



print("Number of images in test set: {}".format(num_test_images))

print("Number of images in train set: {}".format(num_train_images))
train_df = pd.read_csv("../input/train.csv")

train_df.head()
avg_class_per_image = np.round(train_df.shape[0]/num_train_images, 2)

print("Average number of classes per image: {}".format(avg_class_per_image))



assert len(train_df["ImageId"].value_counts()) == num_train_images

print("Every image has at least 1 class")
train_df["fine_grained"] = train_df["ClassId"].apply(lambda x: len(x.split("_"))) > 1

train_df["main_class"] = train_df["ClassId"].apply(lambda x: x.split("_")[0])



fine_grained_obj_perc = np.round(train_df["fine_grained"].mean()*100, 1)

print("{}% of the objects are fine-grained.".format(fine_grained_obj_perc))
fine_grained_img_perc = np.round((train_df.groupby("ImageId")["fine_grained"].sum() > 0).mean()*100, 1)

print("{}% of the images have at least one fine-grained object.".format(fine_grained_img_perc))
class_df = train_df.groupby("main_class").agg({"fine_grained": "mean", "ImageId": "count"}).reset_index()

class_df = class_df.rename(columns={"ImageId": "img_count"})

print("Number of classes: {}".format(class_df.shape[0]))
print("{} of the classes are never fine-grained.".format((class_df["fine_grained"] == 0).sum()))
perc = np.round(100*class_df[class_df["fine_grained"] == 0]["img_count"].sum()/train_df.shape[0], 1)

print("{}% of the objects are from non-fine-grained classes.".format(perc))