# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import KFold

from skimage.morphology import dilation, erosion, opening, closing, convex_hull_image
from skimage.util import invert

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, layers, optimizers

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.resnet import ResNet152V2
!pip install -q efficientnet
from efficientnet.tfkeras import EfficientNetB5 , EfficientNetB7 , EfficientNetB3
def convex_crop ( img, pad = 20 ) :
    convex = convex_hull_image ( img )
    r,c = np.where ( convex )
    while ( min ( r ) - pad < 0 ) or ( max ( r ) + pad > img.shape [ 0 ] ) or ( min ( c ) - pad < 0 ) or ( max ( c )  + pad > img.shape [ 1 ] ) :
        pad = pad - 1
    return img [ min ( r ) - pad : max ( r ) + pad, min ( c ) - pad : max ( c ) + pad ]
image_size = 64
def threshold_resize ( img, threshold = 40 ) :
    img = cv2.resize ( img, ( 256, 256 ) )
    img = ( ( img > threshold ) * 255 ).astype ( np.uint8 ) # astype ( np.uint8 ) เพื่อให้ค่าสูงสุดไม่เกิน 255
    img = cv2.resize ( img, ( 128, 128 ) )
    img = ( ( img > threshold ) * 255 ).astype ( np.uint8 ) # astype ( np.uint8 ) เพื่อให้ค่าสูงสุดไม่เกิน 255
    img = cv2.resize ( img, ( 64, 64 ) )
    img = ( ( img > threshold ) * 255 ).astype ( np.uint8 ) # astype ( np.uint8 ) เพื่อให้ค่าสูงสุดไม่เกิน 255
    return img
def image_preprocessing ( img ) :
    # img = cv2.bitwise_not ( img )
    img = invert ( img )
    img = dilation ( img )
    img = convex_crop ( img, pad = 10 )
    img = threshold_resize ( img )
    return img
df_train_map = pd.read_csv ( "/kaggle/input/thai-mnist-classification/mnist.train.map.csv" )  
df_train_rule = pd.read_csv ( "/kaggle/input/thai-mnist-classification/train.rules.csv" )  
cv2_image = cv2.imread ( "/kaggle/input/thai-mnist-classification/train/75606737-d17d-43eb-86e6-6735b6f45a52.png", cv2.IMREAD_GRAYSCALE )
cv2_image = image_preprocessing ( cv2_image )

plt.imshow ( cv2_image, cmap = "gray" )
plt.show ( )
df_train_rule = pd.merge ( df_train_rule, df_train_map, left_on = [ "feature1" ], right_on = [ "id" ], how = "left", suffixes = [ "", "_feature1" ] )
df_train_rule = df_train_rule.drop ( [ "id_feature1" ], axis = "columns" )
df_train_rule = df_train_rule.rename ( columns = { "category" : "feature1_category" } )

df_train_rule = pd.merge ( df_train_rule, df_train_map, left_on = [ "feature2" ], right_on = [ "id" ], how = "left", suffixes = [ "", "_feature2" ] )
df_train_rule = df_train_rule.drop ( [ "id_feature2" ], axis = "columns" )
df_train_rule = df_train_rule.rename ( columns = { "category" : "feature2_category" } )

df_train_rule = pd.merge ( df_train_rule, df_train_map, left_on = [ "feature3" ], right_on = [ "id" ], how = "left", suffixes = [ "", "_feature3" ] )
df_train_rule = df_train_rule.drop ( [ "id_feature3" ], axis = "columns" )
df_train_rule = df_train_rule.rename ( columns = { "category" : "feature3_category" } )

df_train_rule = df_train_rule.drop ( [ "id", "feature1", "feature2", "feature3" ], axis = "columns" )
df_train_rule.head ( )
# f1 = NaN
def f1_null_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ].isnull ( ) ]
    df_temp [ "predict_calculated" ] = df_temp [ "feature2_category" ] + df_temp [ "feature3_category" ]
    return df_temp

df_train_rule_temp = f1_null_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 0
def f1_0_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 0 ]
    df_temp [ "predict_calculated" ] = df_temp [ "feature2_category" ] * df_temp [ "feature3_category" ]
    return df_temp

df_train_rule_temp = f1_0_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 1
def f1_1_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 1 ]
    df_temp [ "predict_calculated" ] = abs ( df_temp [ "feature2_category" ] - df_temp [ "feature3_category" ] )
    return df_temp

df_train_rule_temp = f1_1_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 2
def f1_2_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 2 ]
    df_temp [ "predict_calculated" ] = ( df_temp [ "feature2_category" ] + df_temp [ "feature3_category" ] ) * abs ( df_temp [ "feature2_category" ] - df_temp [ "feature3_category" ] )
    return df_temp

df_train_rule_temp = f1_2_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 3
def f1_3_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 3 ]
    df_temp [ "tmp_left" ] = df_temp [ "feature3_category" ] * ( df_temp [ "feature3_category" ] + 1 )
    df_temp [ "tmp_right" ] = df_temp [ "feature2_category" ] * ( df_temp [ "feature2_category" ] - 1 )
    df_temp [ "predict_calculated" ] = abs ( ( df_temp [ "tmp_left" ] - df_temp [ "tmp_right" ] ) / 2 )
    df_temp = df_temp.drop ( [ "tmp_left", "tmp_right" ], axis = "columns" )
    return df_temp

df_train_rule_temp = f1_3_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 4
def f1_4_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 4 ]
    df_temp [ "predict_calculated" ] = 50 + ( df_temp [ "feature2_category" ] - df_temp [ "feature3_category" ] )
    return df_temp
    
df_train_rule_temp = f1_4_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 5
def f1_5_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 5 ]
    df_temp [ "predict_calculated" ] = np.minimum ( df_temp [ "feature2_category" ], df_temp [ "feature3_category" ] )
    return df_temp

df_train_rule_temp = f1_5_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 6
def f1_6_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 6 ]
    df_temp [ "predict_calculated" ] = np.maximum ( df_temp [ "feature2_category" ], df_temp [ "feature3_category" ] )
    return df_temp

df_train_rule_temp = f1_6_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 7
def f1_7_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 7 ]
    df_temp [ "predict_calculated" ] = ( ( df_temp [ "feature2_category" ] * df_temp [ "feature3_category" ] ) % 9 ) * 11
    return df_temp

df_train_rule_temp = f1_7_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 8
def f1_8_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 8 ]
    df_temp [ "tmp_left" ] = ( ( df_temp [ "feature2_category" ] * df_temp [ "feature2_category" ] ) + 1 ) * df_temp [ "feature2_category" ]
    df_temp [ "tmp_right" ] = df_temp [ "feature3_category" ] * ( df_temp [ "feature3_category" ] + 1 )
    df_temp [ "predict_calculated" ] = ( df_temp [ "tmp_left" ] + df_temp [ "tmp_right" ] ) % 99
    df_temp = df_temp.drop ( [ "tmp_left", "tmp_right" ], axis = "columns" )
    return df_temp

df_train_rule_temp = f1_8_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
# f1 = 9
def f1_9_calculate ( df ) :
    df_temp = df [ df [ "feature1_category" ] == 9 ]
    df_temp [ "predict_calculated" ] = 50 + df_temp [ "feature2_category" ]
    return df_temp

df_train_rule_temp = f1_9_calculate ( df_train_rule )
df_train_rule_temp [ df_train_rule_temp [ "predict" ] != df_train_rule_temp [ "predict_calculated" ] ]
list_baddata = [ ] 
list_baddata.append ( "113ab6a1-4c67-4807-9bea-7775d83e0439.png" )
list_baddata.append ( "091b70f6-33cf-4732-9aa6-b8fbabcc3e2c.png" )
list_baddata.append ( "5f542797-660a-4e7c-8716-2b373a2dddf8.png" )
list_baddata.append ( "4777156d-6e58-44d8-97ed-eb6c117a9ae0.png" )
list_baddata.append ( "62e2ae06-cec4-41ad-ab9b-0cbe9708d575.png" )
list_baddata.append ( "5d672be1-8c16-42a6-891f-5d220aea1144.png" )
list_baddata.append ( "be73bbaf-a3a0-488d-b6ab-56c817b4e0f1.png" )
list_baddata.append ( "057b69fc-3e8a-4329-b852-ae8f99a0518e.png" )
list_baddata.append ( "8b0d6250-946b-45ce-946d-83dff0f36c82.png" )
list_baddata.append ( "0f339947-7644-4ceb-b498-5e8088ef71d0.png" )
list_baddata.append ( "0721b846-06a6-45af-8b38-9886f516cfc3.png" )
list_baddata.append ( "6d2144b5-43aa-46cd-9c34-a7a8cdbcc24a.png" )
list_baddata.append ( "e248f46e-f628-4318-a112-b1dcf1d68a61.png" )
list_baddata.append ( "1ecd3833-307a-48ac-9967-b3eb49d3df7f.png" )
list_baddata.append ( "3ae5f59d-6ca0-4d0f-8c5e-792dd727c7cd.png" )
list_baddata.append ( "aed13482-d91d-46fe-a86b-287b4281be83.png" )
list_baddata.append ( "760911cc-2867-4765-b0be-60f84b68a596.png" )
list_baddata.append ( "b9036353-4600-4909-b82f-8fc9e5410a70.png" )
list_baddata.append ( "f60223a4-7b0a-4cb2-9afa-23a18e1a0f7e.png" )
list_baddata.append ( "669754bf-1b79-426c-9c97-8bfd4d436eeb.png" )
list_baddata.append ( "476ec014-c431-4c80-90d0-42b8d76b9b47.png" )
list_baddata.append ( "d637c8e9-38e2-47d3-8306-a274c4c143dc.png" )
list_baddata.append ( "2c4b690d-e181-4853-8c7d-e1a5f0b02e54.png" )
list_baddata.append ( "26549ad2-ac6e-446a-962d-45a0617b374c.png" )
list_baddata.append ( "a6e34d37-9f82-43a2-8815-197002348f3c.png" )
list_baddata.append ( "a9f12b73-702d-4584-bac0-2687da7be42b.png" )
list_baddata.append ( "053d37a3-d8e9-4c5c-931c-be23e3d44838.png" )
list_baddata.append ( "09a58894-52b4-437c-b8a5-13de60988ed0.png" )
list_baddata.append ( "4c8fa098-f333-4de0-8a59-82d70672aa8c.png" )
list_baddata.append ( "07d5dc30-420a-47fc-bce2-c9a5d21aba1b.png" )
list_baddata.append ( "91c52d69-275e-4c91-a216-62522333152b.png" )
list_baddata.append ( "78750541-7e91-4f6a-95f9-cb6cab53e459.png" )
list_baddata.append ( "784d9ee3-28f7-4e0c-9162-c69775405a52.png" )
list_baddata.append ( "1637f8f3-2cd1-42b8-b23b-30ac8fae0d29.png" )
list_baddata.append ( "50cbd184-8455-4a28-9d10-e98a46a3f59e.png" )
list_baddata.append ( "bec13d1a-1c93-4ac1-a178-566da5b132c4.png" )
list_baddata.append ( "2327bf93-0eaf-49ef-9b6d-61c06e1ccafc.png" )
list_baddata.append ( "34274de5-c532-4546-b9ca-8721ebd74353.png" )
list_baddata.append ( "a48ff1e4-c0f4-482d-beb3-21acf8597a05.png" )
list_baddata.append ( "6eacf518-abb9-4775-a56d-7235b55b3821.png" )
list_baddata.append ( "322fdb1f-fc30-4f84-8ad6-27352bf7d4b7.png" )
list_baddata.append ( "dc9801a7-5df0-4629-9772-022274e3378b.png" )
list_baddata.append ( "42b8de34-11f2-467d-8f3a-5a3637640483.png" )
list_baddata.append ( "f83fed6a-fc2d-4c74-abe6-755376f477a8.png" )
list_baddata.append ( "5c2ef093-4040-4463-816c-dfc83883ae5e.png" )
list_baddata.append ( "019bb3e2-ad8c-47ee-9692-e5d473494740.png" )
list_baddata.append ( "32e044ce-0aef-42a0-a4a4-93a9350eaa5a.png" )
list_baddata.append ( "83a0a5de-dc3f-44bf-a097-8c98670751ee.png" )
list_baddata.append ( "5cd0aaf8-7766-45bc-8af7-883924714759.png" )
list_baddata.append ( "623e5a2a-d3fd-4ef0-9a2d-8d58187c7a0b.png" )
list_baddata.append ( "f0883378-c9c7-40f8-95d1-7daa5392fc57.png" )
list_baddata.append ( "b3af5dbb-dfb8-400a-b6e7-c5c0e73f5a7a.png" )
list_baddata.append ( "c9323156-4ef8-4f8e-b91c-492cc96272f3.png" )
list_baddata.append ( "bbd472b4-100e-40eb-8b22-475ab18d9858.png" )
list_baddata.append ( "d8ad0b26-893a-4197-a019-8d440ee030aa.png" )
list_baddata.append ( "06d939cf-8c34-4207-8baa-1fda197f793e.png" )
list_baddata.append ( "121d5f9b-688e-468e-bde8-923f2f5e70ca.png" )
list_baddata.append ( "09cbceb4-c127-42be-a293-03b416d01af9.png" )
list_baddata.append ( "bc2d9cae-c51b-4baf-9192-955f2c69a5be.png" )
list_baddata.append ( "0aff6fc4-e8ad-4e0e-a7d7-70845918e396.png" )
list_baddata.append ( "5f73f0f4-59c5-45c4-87b6-f01c4e509489.png" )
list_baddata.append ( "6bbdb42f-8de7-4d14-bc11-48c2c64180c3.png" )
list_baddata.append ( "dff99b33-a097-4408-bff8-2f2c1234b93e.png" )
list_baddata.append ( "8304a775-aad5-408b-9a47-7e20211bd8fe.png" )
list_baddata.append ( "026564bc-335d-4eb2-92e8-406537541eeb.png" )
list_baddata.append ( "574a1bc7-40b6-46b4-8713-f4dd8b87ca55.png" )
df_train_map = df_train_map [ ~ df_train_map [ "id" ].isin ( list_baddata ) ]
df_train_map_count = df_train_map.groupby ( "category" ).count ( ).reset_index ( ).rename ( columns = { "id" : "count" } )

max_cateogry = max ( df_train_map_count [ "count" ] )
min_category = min ( df_train_map_count [ "count" ] )

df_train_map_count [ "count" ]
"""

# up sample

df_category_0 = df_train_map [ df_train_map [ "category" ] == 0 ]
df_category_1 = df_train_map [ df_train_map [ "category" ] == 1 ]
df_category_2 = df_train_map [ df_train_map [ "category" ] == 2 ]
df_category_3 = df_train_map [ df_train_map [ "category" ] == 3 ]
df_category_4 = df_train_map [ df_train_map [ "category" ] == 4 ]
df_category_5 = df_train_map [ df_train_map [ "category" ] == 5 ]
df_category_6 = df_train_map [ df_train_map [ "category" ] == 6 ]
df_category_7 = df_train_map [ df_train_map [ "category" ] == 7 ]
df_category_8 = df_train_map [ df_train_map [ "category" ] == 8 ]
df_category_9 = df_train_map [ df_train_map [ "category" ] == 9 ]
 
# df_category_0_downsample = resample ( df_category_0, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_1_downsample = resample ( df_category_1, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_2_downsample = resample ( df_category_2, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_3_downsample = resample ( df_category_3, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_4_downsample = resample ( df_category_4, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_5_downsample = resample ( df_category_5, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_6_downsample = resample ( df_category_6, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_7_downsample = resample ( df_category_7, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_8_downsample = resample ( df_category_8, replace = True, n_samples = max_cateogry, random_state = 123 ) 
df_category_9_downsample = resample ( df_category_9, replace = True, n_samples = max_cateogry, random_state = 123 ) 
 
df_train_map_downsample = pd.concat ( [ df_category_0, df_category_1_downsample, df_category_2_downsample, df_category_3_downsample, df_category_4_downsample, df_category_5_downsample, df_category_6_downsample, df_category_7_downsample, df_category_8_downsample, df_category_9_downsample ] )
 
df_train_map_downsample.groupby ( "category" ).count ( )

"""

df_category_0 = df_train_map [ df_train_map [ "category" ] == 0 ]
df_category_1 = df_train_map [ df_train_map [ "category" ] == 1 ]
df_category_2 = df_train_map [ df_train_map [ "category" ] == 2 ]
df_category_3 = df_train_map [ df_train_map [ "category" ] == 3 ]
df_category_4 = df_train_map [ df_train_map [ "category" ] == 4 ]
df_category_5 = df_train_map [ df_train_map [ "category" ] == 5 ]
df_category_6 = df_train_map [ df_train_map [ "category" ] == 6 ]
df_category_7 = df_train_map [ df_train_map [ "category" ] == 7 ]
df_category_8 = df_train_map [ df_train_map [ "category" ] == 8 ]
df_category_9 = df_train_map [ df_train_map [ "category" ] == 9 ]
 
df_category_0_downsample = resample ( df_category_0, replace = False, n_samples = min_category, random_state = 123 ) 
df_category_1_downsample = resample ( df_category_1, replace = False, n_samples = min_category, random_state = 123 ) 
df_category_2_downsample = resample ( df_category_2, replace = False, n_samples = min_category, random_state = 123 ) 
df_category_3_downsample = resample ( df_category_3, replace = False, n_samples = min_category, random_state = 123 ) 
df_category_4_downsample = resample ( df_category_4, replace = False, n_samples = min_category, random_state = 123 ) 
df_category_5_downsample = resample ( df_category_5, replace = False, n_samples = min_category, random_state = 123 ) 
df_category_6_downsample = resample ( df_category_6, replace = False, n_samples = min_category, random_state = 123 ) 
df_category_7_downsample = resample ( df_category_7, replace = False, n_samples = min_category, random_state = 123 ) 
df_category_8_downsample = resample ( df_category_8, replace = False, n_samples = min_category, random_state = 123 ) 
# df_category_9_downsample = resample ( df_category_9, replace = False, n_samples = min_category, random_state = 123 ) 
 
df_train_map_downsample = pd.concat ( [ df_category_0_downsample, df_category_1_downsample, df_category_2_downsample, df_category_3_downsample, df_category_4_downsample, df_category_5_downsample, df_category_6_downsample, df_category_7_downsample, df_category_8_downsample, df_category_9 ] )
 
df_train_map_downsample.groupby ( "category" ).count ( )
x_train = [ ]
x_train_filename = [ ]
y_train = [ ]

image_rootpath = "/kaggle/input/thai-mnist-classification/train"

round = 0
for index, record in df_train_map_downsample.iterrows ( ) :
    image_fullpath = os.path.join ( image_rootpath, record [ "id" ] )
    cv2_image = cv2.imread ( image_fullpath, cv2.IMREAD_GRAYSCALE )
    cv2_image = image_preprocessing ( cv2_image )
    x_train.append ( [ cv2_image ] )
    y_train.append ( [ record [ "category" ] ] )
    x_train_filename.append ( record [ "id" ] )
    round = round + 1
    print ( round, record [ "id" ], record [ "category" ] )
x_train = np.concatenate ( x_train, axis = 0 )
x_train.shape
x_train = x_train.astype ( np.float32 ) / 255.0
y_train = np.concatenate ( y_train, axis = 0 )
y_train.shape
plt.imshow ( x_train [ 0 ], cmap = "gray" )
plt.show ( )
y_train [ 0 ]
def trainmodel_kfold ( np_x, np_y ) :    
    list_model = [ ]
    list_history = [ ]
    
    kf = KFold ( n_splits = 5, shuffle = True, random_state = 123 )
    for train_index, test_index in kf.split ( np_x ) :
        
        #app = DenseNet201 ( include_top = False, weights = "imagenet" )
        app = EfficientNetB5(weights='imagenet', include_top=False)
        #app = NASNetLarge(include_top=False)
        #app = InceptionV3(weights='imagenet', include_top=False)
        x_in = layers.Input ( shape = ( image_size, image_size, 1 ) )
        x = layers.Conv2D ( 3, 1 ) ( x_in )
        x = app ( x )

        x = layers.Flatten ( ) ( x )
        x = layers.Dense ( 4096, activation = "relu" ) ( x )
        x = layers.Dense ( 4096, activation = "relu" ) ( x )
        x = layers.Dense ( 10, activation = "softmax" ) ( x )
    
        model = Model ( x_in, x )
        model.summary ( )

        x_train, x_test = np_x [ train_index ], np_x [ test_index ]
        y_train, y_test = np_y [ train_index ], np_y [ test_index ]
        
        callback_learningratereduction = tf.keras.callbacks.ReduceLROnPlateau ( monitor = "val_loss", patience = 3, verbose = 1, factor = 0.5, min_lr = 0.0000001 )
        callback_earlystopping = tf.keras.callbacks.EarlyStopping ( monitor = "val_loss", patience = 10, verbose = 1 )
        optimizer_adam = optimizers.Adam ( learning_rate = 0.00001 )
        model.compile ( loss = "sparse_categorical_crossentropy", optimizer = optimizer_adam, metrics = [ "accuracy" ] )
        history = model.fit ( x_train, y_train, epochs = 100, batch_size = 32, verbose = 1, validation_data = ( x_test, y_test ), callbacks = [ callback_learningratereduction, callback_earlystopping ] )
        
        list_model.append ( model )
        list_history.append ( history )
        
    return list_model, list_history
list_model, list_history = trainmodel_kfold ( x_train, y_train )
result = { }

fold_index = 0
for each_history in list_history : 
    loss = each_history.history [ "loss" ] [ len ( each_history.history [ "loss" ] ) - 1 ]
    accuracy = each_history.history [ "accuracy" ] [ len ( each_history.history [ "accuracy" ] ) - 1 ]
    val_loss = each_history.history [ "val_loss" ] [ len ( each_history.history [ "val_loss" ] ) - 1 ]
    val_accuracy = each_history.history [ "val_accuracy" ] [ len ( each_history.history [ "val_accuracy" ] ) - 1 ]
    
    fold_index = fold_index + 1
    result [ "Fold " + str ( fold_index ) ] = list ( [ loss, accuracy, val_loss, val_accuracy ] )
    
pd.DataFrame ( result, ).rename ( index = { 0 : "loss", 1 : "accuracy", 2 : "val_loss", 3 : "val_accuracy" } ).T
model = list_model [ 2 ]
df_test_submit = pd.read_csv ( "/kaggle/input/thai-mnist-classification/submit.csv" )  
df_test_rule = pd.read_csv ( "/kaggle/input/thai-mnist-classification/test.rules.csv" )  
x_test = [ ]
x_test_filename = [ ]

round = 0
image_rootpath = "/kaggle/input/thai-mnist-classification/test"
for image_filename in os.listdir ( image_rootpath ) :
    image_fullpath = os.path.join ( image_rootpath, image_filename )
    cv2_image = cv2.imread ( image_fullpath, cv2.IMREAD_GRAYSCALE )
    cv2_image = image_preprocessing ( cv2_image )
    x_test.append ( [ cv2_image ] )
    x_test_filename.append ( image_filename )
    round = round + 1
    print ( round, image_filename )
x_test = np.concatenate ( x_test, axis = 0 )
x_test.shape
x_test = x_test.astype ( np.float32 ) / 255.0
y_predict = model.predict ( x_test )
y_predict = y_predict.argmax ( axis = 1 )
y_predict
df_predict = pd.DataFrame ( list ( zip ( x_test_filename, y_predict ) ), columns = [ "id", "category" ] ) 
df_predict
df_submit = pd.merge ( df_test_rule, df_predict, left_on = [ "feature1" ], right_on = [ "id" ], how = "left", suffixes = [ "", "_feature1" ] )
df_submit = df_submit.drop ( [ "id_feature1" ], axis = "columns" )
df_submit = df_submit.rename ( columns = { "category" : "feature1_category" } )

df_submit = pd.merge ( df_submit, df_predict, left_on = [ "feature2" ], right_on = [ "id" ], how = "left", suffixes = [ "", "_feature2" ] )
df_submit = df_submit.drop ( [ "id_feature2" ], axis = "columns" )
df_submit = df_submit.rename ( columns = { "category" : "feature2_category" } )

df_submit = pd.merge ( df_submit, df_predict, left_on = [ "feature3" ], right_on = [ "id" ], how = "left", suffixes = [ "", "_feature3" ] )
df_submit = df_submit.drop ( [ "id_feature3" ], axis = "columns" )
df_submit = df_submit.rename ( columns = { "category" : "feature3_category" } )

df_submit = df_submit.drop ( [ "feature1", "feature2", "feature3", "predict" ], axis = "columns" )
df_submit
df_submit_temp_null = f1_null_calculate ( df_submit )
df_submit_temp_null = df_submit_temp_null [ [ "id", "predict_calculated" ] ]

df_submit_temp_0 = f1_0_calculate ( df_submit )
df_submit_temp_0 = df_submit_temp_0 [ [ "id", "predict_calculated" ] ]

df_submit_temp_1 = f1_1_calculate ( df_submit )
df_submit_temp_1 = df_submit_temp_1 [ [ "id", "predict_calculated" ] ]

df_submit_temp_2 = f1_2_calculate ( df_submit )
df_submit_temp_2 = df_submit_temp_2 [ [ "id", "predict_calculated" ] ]

df_submit_temp_3 = f1_3_calculate ( df_submit )
df_submit_temp_3 = df_submit_temp_3 [ [ "id", "predict_calculated" ] ]

df_submit_temp_4 = f1_4_calculate ( df_submit )
df_submit_temp_4 = df_submit_temp_4 [ [ "id", "predict_calculated" ] ]

df_submit_temp_5 = f1_5_calculate ( df_submit )
df_submit_temp_5 = df_submit_temp_5 [ [ "id", "predict_calculated" ] ]

df_submit_temp_6 = f1_6_calculate ( df_submit )
df_submit_temp_6 = df_submit_temp_6 [ [ "id", "predict_calculated" ] ]

df_submit_temp_7 = f1_7_calculate ( df_submit )
df_submit_temp_7 = df_submit_temp_7 [ [ "id", "predict_calculated" ] ]

df_submit_temp_8 = f1_8_calculate ( df_submit )
df_submit_temp_8 = df_submit_temp_8 [ [ "id", "predict_calculated" ] ]

df_submit_temp_9 = f1_9_calculate ( df_submit )
df_submit_temp_9 = df_submit_temp_9 [ [ "id", "predict_calculated" ] ]


df_submit_final = pd.concat ( [ df_submit_temp_null, df_submit_temp_0, df_submit_temp_1, df_submit_temp_2, df_submit_temp_3, df_submit_temp_4, df_submit_temp_5, df_submit_temp_6, df_submit_temp_7, df_submit_temp_8, df_submit_temp_9 ] )
df_submit_final
df_submit_final = df_submit_final.rename ( columns = { "predict_calculated" : "predict" } )
df_submit_final
df_submit_final.to_csv ( "22p22c0689_val.csv", index = False )