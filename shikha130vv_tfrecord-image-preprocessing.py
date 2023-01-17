import tensorflow as tf

import pandas as pd

import time

import os

import pydicom

from pydicom.pixel_data_handlers.util import convert_color_space

import matplotlib.pyplot as plt

import cv2

import numpy as np

from sklearn.model_selection import train_test_split

from multiprocessing import Pool

import pickle
base_dir = "/kaggle/input/siim-isic-melanoma-classification/"
def decode_jpeg(filename):

  bits = tf.io.read_file(filename)

  image = tf.image.decode_jpeg(bits)

  #label = train_data[train_data["image_name"] == filename[-16:-4]]["target"].values

  start = tf.strings.length(filename) - 16

  filename = tf.strings.substr(filename, start, 12)

  label = target_list[tf.where(tf.equal(image_name_list, filename))[0,0]]

  return image, label

  

train_data = pd.read_csv(base_dir + "train.csv")

image_name_list = tf.constant(train_data["image_name"].values)

target_list = tf.constant(train_data["target"].values)



filenames_dataset = tf.data.Dataset.list_files(base_dir + "jpeg/train/*.jpg")



image_dataset = filenames_dataset.map(decode_jpeg)

t0 = time.process_time()

pos_label_count = 0

for image, label in image_dataset.take(50):

    pos_label_count = label + pos_label_count

t1 = time.process_time()

print("Elapsed time:", t1-t0)
def parse_rec(data):           

    feature_set = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'target': tf.io.FixedLenFeature([], tf.int64)

    }

    features = tf.io.parse_single_example(data, features= feature_set )

    image = tf.image.decode_image(features["image"])

    target = features["target"]

    return image, target



def parse_rec_target(data):           

    feature_set = {

        'target': tf.io.FixedLenFeature([], tf.int64)

    }

    features = tf.io.parse_single_example(data, features= feature_set )

    target = features["target"]

    return target



def parse_test_rec(data):           

    feature_set = {

        'image': tf.io.FixedLenFeature([], tf.string),

    }

    features = tf.io.parse_single_example(data, features= feature_set )

    image = tf.image.decode_image(features["image"])

    return image



tfrec_files_train = [base_dir + "/tfrecords/" + file for file in os.listdir(base_dir + "tfrecords/") if "test" not in file]





tfrec_dataset = tf.data.TFRecordDataset(tfrec_files_train)



image_dataset = tfrec_dataset.map(parse_rec)

t2 = time.process_time()

pos_label_count = 0

for image, label in image_dataset.take(50):

    pos_label_count = label + pos_label_count

t3 = time.process_time()

print("Elapsed time:", t3-t2)
import seaborn as sns

sns.set_palette("hls")



sns.barplot(x=["File Dataset","TFRec Dataset"], y=[t1-t0, t3-t2])
def _bytestring_feature(list_of_bytestrings):

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))



def _int_feature(list_of_ints): # int64

  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))



def _float_feature(list_of_floats): # float32

  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))



def read_dcm_image(file_path):

    dataset = pydicom.dcmread(file_path)

    image = dataset.pixel_array

    image = convert_color_space(image, "YBR_FULL_422", "RGB")

    return image, dataset



def crop_image(image):

    lower = [0,0,0]

    upper = [int(image[:,:,0].mean()), int(image[:,:,1].mean()), int(image[:,:,2].mean())]

    lower = np.array(lower, dtype="uint8")

    upper = np.array(1.01*np.array(upper, dtype="uint8"), dtype="uint8")

    mask = cv2.inRange(image, lower, upper)



    output = cv2.bitwise_and(image, image, mask=mask)



    ret,thresh = cv2.threshold(mask, 40, 255, 0)



    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



    if len(contours) != 0:

        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)



        image = image[y:y+h, x:x+w]

        image = cv2.resize(image,(224,224))

    return image



def is_tf_file_valid(tfrec_path):

    if tfrec_path[-1] == "_":

        return False

    print(tfrec_path)

    i = int(tfrec_path.split("_")[-1])

    filename = "train_" + str(i)

    try:

        num_tfrec = len(list(tf.data.TFRecordDataset("../input/croppedskincancerimagestrain/" + filename).map(parse_rec_target).as_numpy_iterator()))

        num_df = arr_dic[i]["df"].shape[0]

        print(num_tfrec, num_df)

        return num_tfrec == num_df

    except:

        return False

         

    

def write_all_tfrec(tfrec_path, train_data, bln_crop=False):

    t0 = time.process_time()

    tfrec_full_path = "../input/croppedskincancerimagestrain/" + tfrec_path

    is_valid_file = False

    if os.path.exists(tfrec_full_path):

        print("File exists!")

        if is_tf_file_valid(tfrec_path):

            print("File is valid!")

            os.popen('cp ' + tfrec_full_path + ' ' + tfrec_path)

            is_valid_file = True

        

    if is_valid_file == False:

        print(tfrec_path, ": Valid file not found!")

        with tf.io.TFRecordWriter(tfrec_path) as out_file:

            for idx,row in train_data.iterrows():

                if "train" in tfrec_path:

                    image_path = base_dir + "train/" + row["image_name"] + ".dcm"

                else:

                    image_path = base_dir + "test/" + row["image_name"] + ".dcm"

                img, dataset = read_dcm_image(image_path)

                if bln_crop:

                    img = crop_image(img)

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

                feature = {

                    "image": _bytestring_feature([img]),

                }

                if "train" in tfrec_path:

                    feature["target"] = _int_feature([row["target"]])

                tf_record = tf.train.Example(features=tf.train.Features(feature=feature))

                out_file.write(tf_record.SerializeToString())

    else:

        print("File exists!")

    t1 = time.process_time()

    print("Process time:", t1-t0)

        

        

train_data = pd.read_csv(base_dir + "train.csv").head(24)

tfrec_path = "train_"

write_all_tfrec(tfrec_path, train_data)
def show_img(img_list1, img_list2):

    row=3; col=8;

    plt.figure(figsize=(20,row*12/col))

    x = 1

    

    for img1, img2 in zip(img_list1, img_list2):

        plt.subplot(row,col,x)

        plt.imshow(img1)

        x = x + 1

        plt.subplot(row,col,x)

        plt.imshow(img2)

        x = x + 1

        

def peek_dataset(filename):

    tfrec_dataset = tf.data.TFRecordDataset(filename)

    image_dataset = tfrec_dataset.map(parse_rec)

    arr_img1 = []

    arr_img2 = []

    for img, label in image_dataset.take(12):

        arr_img1.append(img)

    for img, label in image_dataset.skip(12).take(12):

        arr_img2.append(img)

    show_img(arr_img1, arr_img2)

    

def peek_test_dataset(filename):

    tfrec_dataset = tf.data.TFRecordDataset(filename)

    image_dataset = tfrec_dataset.map(parse_test_rec)

    arr_img1 = []

    arr_img2 = []

    for img in image_dataset.take(12):

        arr_img1.append(img)

    for img in image_dataset.skip(12).take(12):

        arr_img2.append(img)

    show_img(arr_img1, arr_img2)

    

peek_dataset("train_")
tfrec_dataset = tf.data.TFRecordDataset("train_")

image_dataset = tfrec_dataset.map(parse_rec)

   

for image, label in image_dataset.take(1):

    image = image.numpy()

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 

    edges = cv2.Canny(gray,50,150,apertureSize = 3) 

  

    lines = cv2.HoughLines(edges,1,np.pi/180, 200) 

  



    for r,theta in lines[0]: 

        a = np.cos(theta) 

        b = np.sin(theta) 

        x0 = a*r 

        y0 = b*r 

        x1 = int(x0 + 1000*(-b)) 

        y1 = int(y0 + 1000*(a)) 

        x2 = int(x0 - 1000*(-b)) 

        y2 = int(y0 - 1000*(a))  

        line_color = (int(image[:,:,0].mean()), int(image[:,:,1].mean()), int(image[:,:,2].mean()))

        cv2.line(image,(x1,y1), (x2,y2), line_color,50) 



    plt.imshow(image)
train_data.head(1)
def create_stratify_col():

    train_data = pd.read_csv(base_dir + "train.csv")

    train_data[["target","sex","anatom_site_general_challenge", "image_name"]].groupby(["target","sex","anatom_site_general_challenge"]).count().unstack(-1).unstack(-1)



    train_data["anatom_site_general_challenge"].fillna("", inplace=True)

    train_data["bodypart"] = train_data["anatom_site_general_challenge"].map(lambda x: x if x not in ["head/neck","palms/soles","oral/genital",""] else "other")

    train_data["stratify"] = train_data.apply(lambda row: str(row["target"]) + "_" + str(row["sex"]) + "_" + str(row["bodypart"]), axis=1)



    low_freq_val = train_data["stratify"].value_counts().tail(2).index.values

    train_data["stratify"] = train_data["stratify"].map(lambda x: "other" if x in low_freq_val else x)

    train_data.fillna(0, inplace=True)

    return train_data

train_data = create_stratify_col()
def split_train_data():

    pickle_filename = "../input/data-pipeline/train_dic.pkl"

    if not os.path.exists(pickle_filename):

        print("Pickle file not found!")

        pickle_filename = "train_dic.pkl"

    if not os.path.exists(pickle_filename):

        print("Pickle file not found!")

        size = train_data.shape[0]//16

        arr_data = []

        split_data = train_data

        for i in range(15):

            split_data, test_data = train_test_split(split_data, test_size=size, stratify= split_data["stratify"])

            arr_data.append(test_data)

        arr_data.append(split_data)

        for df in arr_data[-3:]:

            print(df.shape)



        i  = 0

        arr_dic = []

        for df in arr_data:

            tfrec_path = "train_" + str(i)

            arr_dic.append({"tfrec_path":tfrec_path, "df":df, "bln_crop":True})

            i = i + 1

        with open(pickle_filename, 'wb') as file:

            pickle.dump(arr_dic, file)

        

    else:

        print("Pickle file found!")

        with open(pickle_filename, 'rb') as file:

            arr_dic = pickle.load(file)

    return arr_dic
arr_dic = split_train_data()
def mp_write_all_tfrec(param):

    write_all_tfrec(param["tfrec_path"], param["df"], param["bln_crop"])



for elem in arr_dic:

    mp_write_all_tfrec(elem)

if 1==2:

    p = Pool(1)

    p.map(mp_write_all_tfrec, arr_dic)

    p.close()

    p.join() 
peek_dataset("train_15")
if 1==2:

    pickle_filename = "../input/data-pipeline/test_dic.pkl"

    if not os.path.exists(pickle_filename):

        print("Pickle file not found!")

        pickle_filename = "test_dic.pkl"

    if not os.path.exists(pickle_filename):

        print("Pickle file not found!")

        test_data = pd.read_csv(base_dir + "test.csv")

        size = test_data.shape[0]//16

        arr_data = []

        split_data = test_data

        for i in range(15):

            split_data, data = train_test_split(split_data, test_size=size)

            arr_data.append(data)

        arr_data.append(split_data)

        for df in arr_data[-3:]:

            print(df.shape)

        i  = 0

        arr_dic = []

        for df in arr_data:

            tfrec_path = "test_" + str(i)

            arr_dic.append({"tfrec_path":tfrec_path, "df":df, "bln_crop":True})

            i = i + 1

        with open(pickle_filename, 'wb') as file:

            pickle.dump(arr_dic, file)
if 1==2:

    p = Pool(4)

    p.map(mp_write_all_tfrec, arr_dic)

    p.close()

    p.join() 
if 1==2:

    peek_test_dataset("test_9")
