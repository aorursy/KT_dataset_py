# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



from skimage import io



from tqdm.notebook import tqdm



from  scipy import misc as npimsv



import cv2



import shutil
PATH_TO_DATASET="/kaggle/input/prostate-cancer-grade-assessment"

PATH_TO_TRAINING_IMAGES=os.path.join(PATH_TO_DATASET,"train_images")

PATH_TO_TRAINING_MASKS=os.path.join(PATH_TO_DATASET,"train_label_masks")

PATH_TO_TEST_IMAGES=os.path.join(PATH_TO_DATASET,"test_images")



PATH_TO_REDUCED_TRAIN_IMAGES=os.path.join(".","reduced", "images")



PATH_TO_REDUCED_TRAIN_MASKS=os.path.join(".","reduced", "masks")



PATH_TO_TRAIN_CLASSES=os.path.join(".","classes")



PATH_TO_VALIDATION_IMAGES=os.path.join(".","val")



#Checking the locations existance before reading from the directory

path_to_dataset_exists=os.path.exists(PATH_TO_DATASET)

path_to_training_images_exists=os.path.exists(PATH_TO_TRAINING_IMAGES)

path_to_training_masks_exists=os.path.exists(PATH_TO_TRAINING_MASKS)





path_to_test_images_exists=os.path.exists(PATH_TO_TEST_IMAGES)









#if successful pointing to those directories then we will set the path for training csv file



if path_to_dataset_exists and path_to_training_images_exists and path_to_training_masks_exists:

    print("Training images and masks are reachable.")

    PATH_TO_TRAINING_LABELS_CSV=os.path.join(PATH_TO_DATASET,"train.csv")

    path_to_training_labels_csv_exists=os.path.exists(PATH_TO_TRAINING_LABELS_CSV)

    print(PATH_TO_TRAINING_LABELS_CSV, path_to_training_labels_csv_exists)

    

    

    

if path_to_test_images_exists:

    

    path_to_test_csv_file=os.path.join(PATH_TO_DATASET, "test.csv")

    path_to_test_csv_file_exists=os.path.exists(path_to_test_csv_file)

    print("Test image set is reachable.")

    

    if path_to_test_csv_file_exists:

        print("Test csv file is reachable.")

        

else:

    

    path_to_test_csv_file=os.path.join(PATH_TO_DATASET, "test.csv")

    path_to_test_csv_file_exists=os.path.exists(path_to_test_csv_file)

    



if not os.path.exists(PATH_TO_REDUCED_TRAIN_IMAGES):    

    os.makedirs(PATH_TO_REDUCED_TRAIN_IMAGES, exist_ok=True)

    

if not os.path.exists(PATH_TO_REDUCED_TRAIN_MASKS):    

    os.makedirs(PATH_TO_REDUCED_TRAIN_MASKS, exist_ok=True)

    



if os.path.exists(PATH_TO_REDUCED_TRAIN_IMAGES):

    print("Path to reduced training images is present.")

    

if os.path.exists(PATH_TO_REDUCED_TRAIN_MASKS):

    print("Path to reduced training masks is present.")

    

    

if not os.path.exists(PATH_TO_VALIDATION_IMAGES):

    os.makedirs(PATH_TO_VALIDATION_IMAGES, exist_ok=True)



if os.path.exists(PATH_TO_VALIDATION_IMAGES):

    print("Path to validation images is present")

    

    

#Global variables used in the following code

OPTIMAL_PATCH_SIZE=256

PATCH_SIZE=128

MODEL_INPUT_PATCH_SIZE=224

DEGREE_OF_ROTATION=30

PATCH_SIZE_HALF=PATCH_SIZE//2

NO_OF_CLASSES=4

NO_OF_EPOCHS=50



WHICH_LAYER=-1



BATCH_SIZE=128

NUM_OF_WORKER=32
path_to_training_labels_csv_exists
if path_to_training_labels_csv_exists:





        for file_name in tqdm(os.listdir(PATH_TO_TRAINING_IMAGES)):

            if file_name.endswith("tiff"):

                file_name_prefix=file_name.split()[0]

                

            

                

#            



                image_filename=os.path.join(PATH_TO_TRAINING_IMAGES,file_name)



                #             mask_filename=os.path.join(PATH_TO_TRAINING_MASKS, fileid+"_mask.tiff" )

        

                







                try:

                    image_slide=io.MultiImage(image_filename)

                    image_slide_cropped=image_slide[WHICH_LAYER]

                    image_slide.close()



                except:

                #                 print(fileid)

                    image_slide_okay=False



                #                 try:

                #                     mask_slide=io.MultiImage(mask_filename)

                #                     mask_slide_cropped=mask_slide[WHICH_LAYER]

                #         #             print(mask_slide_cropped.shape)



                #                 except:

                #     #                 print("Problem with Mask of {}".format(fileid))

                #                     #mask_slide.close()

                #                     mask_slide_okay=False







                path_to_reduced_image_slide=os.path.join(PATH_TO_REDUCED_TRAIN_IMAGES,\

                file_name_prefix +".png"



                )



                #                         path_to_reduced_mask_slide=os.path.join(PATH_TO_REDUCED_TRAIN_IMAGES,\

                #                                                                  fileid+"_mask.png")





                #                         image_slide_cropped=padded_image(image_slide_cropped,patch_size=PATCH_SIZE)



                #                         mask_slide_cropped=padded_image(mask_slide_cropped,patch_size=PATCH_SIZE)



                cv2.imwrite(path_to_reduced_image_slide,image_slide_cropped)



                #                         cv2.imwrite(path_to_reduced_mask_slide,mask_slide_cropped)



                #                         list_image_id_gleason_score_same.append([fileid, gleason_score_splited[0],\

                #                                                                  path_to_reduced_image_slide,\

                #                                                                  path_to_reduced_mask_slide

                #                                                                 ])





if path_to_training_labels_csv_exists:





        for file_name in tqdm(os.listdir(PATH_TO_TRAINING_MASKS)):

            if file_name.endswith("tiff"):

                file_name_prefix=file_name.split()[0]

                

            

                

#            



#                 image_filename=os.path.join(PATH_TO_TRAINING_IMAGES,file_name)



                mask_filename=os.path.join(PATH_TO_TRAINING_MASKS, file_name )

        

                







#                 try:

#                     image_slide=io.MultiImage(image_filename)

#                     image_slide_cropped=image_slide[WHICH_LAYER]

#                     image_slide.close()



#                 except:

#                 #                 print(fileid)

#                     image_slide_okay=False



                try:

                    mask_slide=io.MultiImage(mask_filename)

                    mask_slide_cropped=mask_slide[WHICH_LAYER]

                #             print(mask_slide_cropped.shape)

                    mask_slide.close()



                except:

                #                 print("Problem with Mask of {}".format(fileid))

                    #mask_slide.close()

                    mask_slide_okay=False







#                 path_to_reduced_image_slide=os.path.join(PATH_TO_REDUCED_TRAIN_IMAGES,\

#                 file_name_prefix +".png"



#                 )



                path_to_reduced_mask_slide=os.path.join(PATH_TO_REDUCED_TRAIN_MASKS,\

                                                 file_name_prefix+".png")





#                                         image_slide_cropped=padded_image(image_slide_cropped,patch_size=PATCH_SIZE)



#                                         mask_slide_cropped=padded_image(mask_slide_cropped,patch_size=PATCH_SIZE)



#                 cv2.imwrite(path_to_reduced_image_slide,image_slide_cropped)



                cv2.imwrite(path_to_reduced_mask_slide,mask_slide_cropped)



                #                         list_image_id_gleason_score_same.append([fileid, gleason_score_splited[0],\

                #                                                                  path_to_reduced_image_slide,\

                #                                                                  path_to_reduced_mask_slide

                #                                                                 ])





!ls -1 reduced/images/ |wc -l 

!ls -1 reduced/masks/ |wc -l 
shutil.copyfile("../input/prostate-cancer-grade-assessment/sample_submission.csv", "submission.csv")