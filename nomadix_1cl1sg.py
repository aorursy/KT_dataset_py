# ===========================================================================================

# In kernel used segmentation model library (https://github.com/qubvel/segmentation_models)

# ===========================================================================================

import os

import cv2

import time

import pandas as pd

import numpy as np

from os import listdir

from os.path import isfile, join

from keras import backend as K

from keras.models import load_model

from keras.optimizers import Adam

from skimage.morphology import remove_small_objects

from skimage import measure



def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def preprocess(X):

    return (X - 110.0) / 1.0



test_images_path = "/kaggle/input/severstal-steel-defect-detection/test_images/"

classifier_model_folder = "/kaggle/input/1cl1sg-a/"

segmentator_model_folder = "/kaggle/input/1cl1sg-d/"

bin_classifier_file = "model_data_v1.3.epoch06-loss0.00882-accuracy0.9973.hdf5" # "rn34-bcl-v10.0a1-E35_loss-0.0105_val_loss-0.0004.h5"

segmentator_4C_file = "rn34-sg-v2.1a-E23_loss-0.0093_val_loss-0.0012.h5" # "rn34-sg4c-v0.7a-E38_loss-0.0023_val_loss-0.0044.h5" # "rn34-sg4c-v0.7a1-E12_loss-0.0025_val_loss-0.0014.h5"

custom_objects = { "dice_coef": dice_coef }



img_w = 1600

img_h = 256



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
# ===================================================================

# Load Keras model (binary classifier and segmentator)

def LoadModels():

    bin_classifier = load_model(classifier_model_folder + bin_classifier_file, custom_objects, compile=False)

    bin_classifier.compile(optimizer = Adam(lr=0.00075), loss = "binary_crossentropy", metrics = ["acc"])



    segmentator_4C = load_model(segmentator_model_folder + segmentator_4C_file, custom_objects, compile=False)

    segmentator_4C.compile(optimizer = Adam(lr=0.00075), loss = "binary_crossentropy", metrics = [dice_coef])

    print("Models are loaded succesfully")



    return bin_classifier, segmentator_4C
# ===================================================================

# Create list of files for prediction

# Returned list contains full path for all files

def CreateFilesListInFolder(images_path=test_images_path):

    files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    files = list(map(lambda x: images_path + x, files))

    files_count = len(files)

    print("Total count of files in folder: {0}".format(files_count))

    return files





# ===================================================================

# Returns run length as string formated

def mask2rle(img):

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)





# min_size_arr4D = [85, 200, 20, 20]

min_size_arr4D = [43, 100, 10, 10]

# ===================================================================

# Remove small objects

# data_tensor - 4D-tensor with shape (samples, height, width, depth)

# min_size_arr - length equal to data_tensor depth

def RemoveSmallObjectsIn4D(data_tensor, min_size_arr=min_size_arr4D):

    res_tensor = data_tensor.copy()

    samples, height, width, depth = data_tensor.shape

    for ix in range(samples):

        for idp in range(depth):

            img = measure.label(data_tensor[ix, :, :, idp], background=0)

            remove_small_objects(img, min_size=min_size_arr[idp], in_place=True)

            res_tensor[ix, :, :, idp] = img

    return res_tensor





# =============================================================================

# Prepare one image for test

# fname - full path to file, img_shape = (height, width)

def PrepareImageForTest(fname, img_shape, color_mode, use_median_blur=False):

    h, w = img_shape

    name = os.path.basename(fname)

    X = cv2.imread(fname, color_mode)

    if color_mode != cv2.IMREAD_GRAYSCALE:

        X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)



    if use_median_blur: X = cv2.medianBlur(X, 3)

    X = cv2.resize(X, (w, h))

    return X





# =============================================================================

# Prepare test batch of images for prediction

# files_list contains full path to file

def CreateTestBatch(files_list, batch_size, img_shape, color_mode):

    last_ix = batch_size if len(files_list) > batch_size else len(files_list)



    if color_mode == cv2.IMREAD_GRAYSCALE:

        color_planes = 1

    else:

        color_planes = 3



    h, w = img_shape

    files = files_list[:last_ix]

    # X = np.empty((batch_size, h, w, color_planes), dtype = np.float32)

    X = np.empty((len(files), h, w, color_planes), dtype = np.float32)

    for ix, fname in enumerate(files):

        Xi = PrepareImageForTest(fname, img_shape, color_mode, use_median_blur=True)

        if color_mode == cv2.IMREAD_GRAYSCALE:

            Xi = Xi.reshape(h, w, 1)

        X[ix,] = Xi

    return X, files





# ====================================================================

# Predict defected / non-defected samples

def PredictDefectedSamples(files, model, batch_size = 250, threshold = 0.5):

    print("Classification defected/ non defected samples has started")

    files_count = len(files)

    batches_count = files_count // batch_size

    if (batches_count * batch_size) != files_count: batches_count += 1

    print("Total count of files for classification: {0}".format(files_count))



    df_non_defected = pd.DataFrame(columns=["ImageId_ClassId", "EncodedPixels"])

    defected_images = []



    for i in range(batches_count):

        start_pos = i * batch_size

        end_pos = start_pos + batch_size

        if end_pos > files_count: end_pos = files_count



        filesi = files[start_pos : end_pos]

        X, fls = CreateTestBatch(files_list=filesi, batch_size=batch_size, img_shape=(128, 800), color_mode=cv2.IMREAD_COLOR)



        print("Predicting files: from {0} to {1}".format(start_pos, end_pos))

        prediction_vector = model.predict(preprocess(X))



        for pr_ix in range(len(filesi)):

            if prediction_vector[pr_ix] >= threshold:

                # Image predicted as contains defect

                defected_images.append(filesi[pr_ix])

            else:

                # Image predicted as non-defected

                fn = os.path.basename(filesi[pr_ix])

                file_name, file_extension = os.path.splitext(fn)

                for clsid in range(4):

                    img_class = file_name + file_extension + "_" + str(clsid + 1)

                    df_non_defected = df_non_defected.append(

                        {'ImageId_ClassId': img_class, 'EncodedPixels': ""},

                        ignore_index=True)



    print("Classification defected/ non defected samples has finished")

    return df_non_defected, defected_images





# ====================================================================

# Make segmentation of defected samples by classes

def SegmentDefectsByClasses(files, model, batch_size=250, threshold=0.5, visualize=False):

    print("Segmentation has started")

    files_count = len(files)

    batches_count = files_count // batch_size

    if (batches_count * batch_size) != files_count: batches_count += 1

    df_submission = pd.DataFrame(columns=["ImageId_ClassId", "EncodedPixels"])

    print("Total count of files for segmentation: {0}".format(files_count))





    for i in range(batches_count):

        start_pos = i * batch_size

        end_pos = start_pos + batch_size

        if end_pos > files_count: end_pos = files_count



        filesi = files[start_pos: end_pos]

        X, fls = CreateTestBatch(files_list=filesi, batch_size=batch_size, img_shape=(128, 800),

                                 color_mode=cv2.IMREAD_COLOR)



        print("Predicting files: from {0} to {1}".format(start_pos, end_pos))

        prediction_tensor = model.predict(preprocess(X))

#         prediction_tensor = RemoveSmallObjectsIn4D(prediction_tensor, min_size_arr4D)



        # Save predictions

        for ix, file in enumerate(filesi):

            for clsid in range(4):

                fn = os.path.basename(file)

                file_name, file_extension = os.path.splitext(fn)

                img_class = file_name + file_extension + "_" + str(clsid + 1)

                mask = prediction_tensor[ix, :, :, clsid]                   # 128 x  800

                maskb = cv2.resize(mask, (img_w, img_h), cv2.INTER_CUBIC)   # 256 x 1600

                maskb[maskb > threshold] = 1

                maskb[maskb <= threshold] = 0

                rle_mask = mask2rle(maskb)

                df_submission = df_submission.append(

                    {'ImageId_ClassId': img_class, 'EncodedPixels': rle_mask},

                    ignore_index=True)



                # Visualize prediction

                if visualize:

                    vimg = cv2.imread(file)

                    vimg = cv2.cvtColor(vimg, cv2.COLOR_BGR2RGB)

                    for clp in range(3):

                        vimg[maskb == 1, clp] = color_masks1[clsid][clp]

                    cv2.imwrite(visualize_prediction_path + "c" + str(clsid + 1) + "/" + file_name + ".png", vimg)



    print("Segmentation has finished")

    return df_submission





def mask_threshold_check(prediction: pd.DataFrame):

    threshold = {'1': 1062, '2': 1212, '3': 2785, '4': 3500}

    th = prediction['EncodedPixels'].apply(lambda x: x if pd.isna(x) else sum([int(elem) for elem in x.split(' ')[1::2]]))

    cl = prediction['ImageId_ClassId'].str.split('_', expand=True)[1]

    counter = 0

    for i in prediction.index:

        if th[i] < threshold[cl[i]]:

            prediction.at[i, 'EncodedPixels'] = np.nan

            counter += 1

    print(f'Count of removed masks = {counter}')

    return prediction
    start_time = time.time()

    bin_classifier, segmentator_4C = LoadModels()

    

    test_files = CreateFilesListInFolder(images_path=test_images_path)

    df_non_defected, list_defected_samples = PredictDefectedSamples(test_files, bin_classifier, batch_size=250, threshold=0.3)

    end_time = time.time()

    print("Defecten/ non defected images prediction time: {0} min".format((end_time - start_time) / 60))

    print("Non-defected samples found: {0}".format(len(df_non_defected) / 4))

    print("Defected samples found: {0}".format(len(list_defected_samples)))

    df_non_defected.head()
# # Rewrite results for binary classification

# df = pd.read_csv("/kaggle/input/1cl1sg-b/reschinas.csv")

# df0 = df[df["ClassId"] == 0]

# df1 = df0.copy()

# df2 = df0.copy()

# df3 = df0.copy()

# df4 = df0.copy()

# df1["ImageId_ClassId"] = df1["ImageId"] + "_1"

# df2["ImageId_ClassId"] = df1["ImageId"] + "_2"

# df3["ImageId_ClassId"] = df1["ImageId"] + "_3"

# df4["ImageId_ClassId"] = df1["ImageId"] + "_4"

# df_non_defected = pd.concat([df1, df2, df3, df4], ignore_index=True).reset_index(drop=True)

# df_non_defected["EncodedPixels"] = ""

# df_non_defected = df_non_defected.drop(["ClassId", "ImageId"], 1)

# df_non_defected = df_non_defected.sort_values(by=["ImageId_ClassId"])



# list_defected_samples = df[df["ClassId"] == 1]["ImageId"].to_list()

# list_defected_samples = list(map(lambda x: test_images_path + x, list_defected_samples))

# # df_non_defected.head()

# print(list_defected_samples)
    start_time = time.time()

    df_segmentation = SegmentDefectsByClasses(list_defected_samples, segmentator_4C, batch_size=250, threshold=0.6, visualize=False)

    end_time = time.time()

    print("Segmentation time: {0} min".format((end_time - start_time) / 60))
    df_submission = pd.concat([df_non_defected, df_segmentation], ignore_index=True).reset_index(drop=True)

    df_submission = df_submission.sort_values(by=["ImageId_ClassId"])

    

#     df1 = df_submission[df_submission["EncodedPixels"] != ""]

#     df2 = df_submission[df_submission["EncodedPixels"] == ""]

#     df1["EncodedPixels"] = "1 1"

#     df_submission = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)

#     df_submission["EncodedPixels"] = "1 1"



#     df = df_submission.copy()

#     df["ClassId"] = df["ImageId_ClassId"].str[-1:].astype(int)

#     df1 = df[df["ClassId"] == 4]

#     df2 = df[df["ClassId"] != 4]

#     df1["EncodedPixels"] = ""

#     df_submission = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)

#     df_submission = df_submission.drop('ClassId', 1)

#     df_submission = df_submission.sort_values(by=["ImageId_ClassId"])

#     df_submission["EncodedPixels"] = "1 1"

    

#     df_submission = mask_threshold_check(df_submission)

    df_submission.to_csv("submission.csv", index=False)
# df_submission.head(100)

# df = df_submission[df_submission["EncodedPixels"] == ""]

# df.head(50)

# df_submission.head(50)

# df_segmentation.head()

# df_non_defected.head()