import pandas as pd
import csv
import re
import os
print(os.listdir('../input/'))
print(os.listdir('../input/cvpr-2018-autonomous-driving/'))
pd.read_csv("../input/cvpr-database-detail/train_database.csv").head()
pd.read_csv("../input/cvpr-database-detail/test_database.csv").head()
# Base path where are stored the unzip dataset
database_path       = "../inputs/"

# Images location subpath 
train_image_subpath = "train_color/"
train_label_subpath = "train_label/"
test_image_subpath  = "test/"

# Text description file sub path
list_train_subpath        = "train_video_list/"
list_test_subpath         = "test_video_list_and_name_mapping/list_test/"
list_test_mapping_subpath = "test_video_list_and_name_mapping/list_test_mapping/"



def extractMapping(filename, path):
    df = pd.read_csv(path + filename ,sep="\t" , quoting=csv.QUOTE_NONE,header=None, names=["md5", "image_name"])
    return df

test_md5_files = os.listdir(database_path + list_test_mapping_subpath)
md5_info  = pd.concat([extractMapping(filename, database_path + list_test_mapping_subpath) for filename in test_md5_files], ignore_index=True, copy=False)    
index_md5 = {row[1]["image_name"] : row[1]["md5"] for row in md5_info.iterrows() }


def extractImageListInfo(filename, file_path):
    filepath = file_path + filename

    df = pd.read_csv(filepath ,sep="\t" , quoting=csv.QUOTE_NONE,header=None,names=["image_name", "ids_name"])
    
    # We have an ID in the filename, probably the video ID
    df['VideoID'] = filename.split('_')[4]    
    
    # We extract information from file name
    df['Road'], _, df['Record'], df['Camera'], df['ImageFileName']  = df['image_name'].str.split('\\').str
    df['Road'] = df['Road'].apply(lambda x: re.sub("road([0-9]*)_.*",r'\1',x))
    df['Record'] = df['Record'].apply(lambda x: re.sub("Record([0-9]*)",r'\1',x))
    df['Camera'] = df['Camera'].apply(lambda x: re.sub("Camera([0-9]*)",r'\1',x))
    df['CarID'], df['TimeStamp'],_, _  = df['ImageFileName'].str.split('_').str
    return df


train_list_files = os.listdir(database_path + list_train_subpath)
train_info = pd.concat([extractImageListInfo(filename, database_path + list_train_subpath) for filename in train_list_files], ignore_index=True, copy=False)    

_, _, _, _, train_info['IdsFileName']  = train_info['ids_name'].str.split('\\').str
train_info['ImageFileName'] = train_info['ImageFileName'].apply(lambda x: train_image_subpath + x)
train_info['IdsFileName'] = train_info['IdsFileName'].apply(lambda x: train_label_subpath + x) 
del train_info['ids_name']
del train_info['image_name']

train_info.to_csv("train_database.csv")
train_info.head()

test_list_files = os.listdir(database_path + list_test_subpath)
test_info  = pd.concat([extractImageListInfo(filename, database_path + list_test_subpath) for filename in test_list_files], ignore_index=True, copy=False)    

test_info['ImageFileName'] = test_info['image_name'].apply(lambda x: test_image_subpath + index_md5[x] + ".jpg")
del test_info['image_name']
del test_info['ids_name']

train_info.to_csv("test_database.csv")
test_info.head()
