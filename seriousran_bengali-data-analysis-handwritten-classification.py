import numpy as np 

import pandas as pd 



from matplotlib import pyplot as plt

import cv2







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

df_test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

df_class = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

df_submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')



df_train_img_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

df_train_img_1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

df_train_img_2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

df_train_img_3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')

df_test_img_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_0.parquet')

df_test_img_1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_1.parquet')

df_test_img_2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_2.parquet')

df_test_img_3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_3.parquet')
print('shape of df_train:', df_train.shape)

print('shape of df_test:', df_test.shape)

print('shape of df_shape:', df_class.shape)

print('shape of df_submission:', df_submission.shape)



print('shape of df_train_img_0:', df_train_img_0.shape)

print('shape of df_train_img_1:', df_train_img_1.shape)

print('shape of df_train_img_2:', df_train_img_2.shape)

print('shape of df_train_img_3:', df_train_img_3.shape)

print('shape of df_test_img_0:', df_test_img_0.shape)

print('shape of df_test_img_1:', df_test_img_1.shape)

print('shape of df_test_img_2:', df_test_img_2.shape)

print('shape of df_test_img_3:', df_test_img_3.shape)
df_train_imgs = [df_train_img_0, df_train_img_1, df_train_img_2, df_train_img_3]

df_train_img = pd.concat(df_train_imgs)



df_test_imgs = [df_test_img_0, df_test_img_1, df_test_img_2, df_test_img_3]

df_test_img = pd.concat(df_test_imgs)



print('shape of df_train_img:', df_train_img.shape)

print('shape of df_test_img:', df_test_img.shape)
# memory release



del df_train_img_0

del df_train_img_1

del df_train_img_2

del df_train_img_3

del df_test_img_0

del df_test_img_1

del df_test_img_2

del df_test_img_3
df_train.head()
df_test.head()
df_class.head()
df_submission.head()
df_train_img.iloc[0].values[1:].astype(np.uint8)
for i in range(10):

    plt.imshow(df_train_img.iloc[i].values[1:].astype(np.uint8).reshape(137,236), cmap='gray')

    plt.show()
for i in range(3):

    plt.imshow(df_test_img.iloc[i].values[1:].astype(np.uint8).reshape(137,236), cmap='gray')

    plt.show()
df_submission['target'] = 4

df_submission.to_csv("submission.csv", index=False)