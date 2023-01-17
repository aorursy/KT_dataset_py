import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import cv2
import random
plt.show()


path = '../input/global-wheat-detection/'
path_img = '../input/global-wheat-detection/train/'
def Df_info(df):
    print('Size = ', *df.shape)
    print('Count null values', df.isnull().sum().sum())
    print('Count unique Id', len(df.image_id.unique()))
    
def random_images(df):   
    fig , ax = plt.subplots(2, 2, figsize = (12,12))
    name = random.choice(df.image_id.unique())
    ax[0][0].imshow(cv2.imread(path_img + name + '.jpg')[...,[2,1,0]])
    ax[0][0].set_title('Id: '+ name, fontsize=14)
    name = random.choice(df.image_id.unique())
    ax[0][1].imshow(cv2.imread(path_img + name + '.jpg')[...,[2,1,0]])
    ax[0][1].set_title('Id: '+ name, fontsize=14)
    name = random.choice(df.image_id.unique())
    ax[1][0].imshow(cv2.imread(path_img + name + '.jpg')[...,[2,1,0]])
    ax[1][0].set_title('Id: '+ name, fontsize=14)
    name = random.choice(df.image_id.unique())
    ax[1][1].imshow(cv2.imread(path_img + name + '.jpg')[...,[2,1,0]])
    ax[1][1].set_title('Id: '+ name, fontsize=14)    
    plt.show()
    
def images_from_sourse(source, df):  
    df = df.loc[df['source'] == source]
    fig , ax = plt.subplots(1, 4, figsize = (35,35))
    name = random.choice(df.image_id.unique())
    ax[0].imshow(cv2.imread(path_img + name + '.jpg')[...,[2,1,0]])
    ax[0].set_title(source + ' id: '+ name, fontsize=14)
    name = random.choice(df.image_id.unique())
    ax[1].imshow(cv2.imread(path_img + name + '.jpg')[...,[2,1,0]])
    ax[1].set_title(source + ' id: '+ name, fontsize=14)
    name = random.choice(df.image_id.unique())
    ax[2].imshow(cv2.imread(path_img + name + '.jpg')[...,[2,1,0]])
    ax[2].set_title(source + ' id: '+ name, fontsize=14)
    name = random.choice(df.image_id.unique())
    ax[3].imshow(cv2.imread(path_img + name + '.jpg')[...,[2,1,0]])
    ax[3].set_title(source + ' id: '+ name, fontsize=14)
    plt.show()
train_df = pd.read_csv(path + 'train.csv')
sample_submission = pd.read_csv(path + 'sample_submission.csv')
Df_info(train_df)
random_images(train_df)
for source in train_df.source.unique():
    print(source)
    images_from_sourse(source, train_df)
def R_G_B_statics(source, df):
    df = df.loc[df['source'] == source]
    R = 0
    G = 0
    B = 0
    for id in df.image_id.unique():
        img = cv2.imread(path_img + id + '.jpg')
        R += np.mean(img[:,:,2])
        G += np.mean(img[:,:,1])
        B += np.mean(img[:,:,0])
    R = R/len(df.image_id.unique())
    G = G/len(df.image_id.unique())
    B = B/len(df.image_id.unique())
    return R, G, B
for source in train_df.source.unique():
    print(source)
    print(R_G_B_statics(source, train_df))
    print('*'*100)
R_mean = 0
G_mean = 0
B_mean = 0
for id in train_df.image_id.unique():
    img = cv2.imread(path_img + id + '.jpg')
    R_mean += np.mean(img[:,:,2])
    G_mean += np.mean(img[:,:,1])
    B_mean += np.mean(img[:,:,0])
R_mean = R_mean/len(train_df.image_id.unique())
G_mean = G_mean/len(train_df.image_id.unique())
B_mean = B_mean/len(train_df.image_id.unique())
def color_conversion(img, R_mean, B_mean, G_mean):
    img = img
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]
    mean_blue = np.mean(blue)
    mean_red = np.mean(red)
    mean_green = np.mean(green)
    for i in range(0, blue.shape[0]):
        for j in range(0, blue.shape[1]):
            blue[i][j] = np.minimum((blue[i][j]/mean_blue)*B_mean, 255)
            green[i][j] = np.minimum((green[i][j]/mean_green)*G_mean, 255)
            red[i][j] = np.minimum((red[i][j]/mean_red)*R_mean, 255)
    img = cv2.merge((blue, green, red))
    return img
fig, ax = plt.subplots(2,3, figsize=(15,15))
img_1 = cv2.imread(path_img + '4c572095f' + '.jpg')
img_2 =  cv2.imread(path_img + '00b70a919' + '.jpg')
img_3 = cv2.imread(path_img +  '6c9cf179f' + '.jpg') 
ax[0][0].imshow(img_1[...,[2,1,0]])
ax[0][0].set_title('original_img '+ ' id: '+ '00e903abe', fontsize=14)
ax[0][1].imshow(img_2[...,[2,1,0]])
ax[0][1].set_title('original_img '+ ' id: '+ '00b70a919', fontsize=14)
ax[0][2].imshow(img_3[...,[2,1,0]])
ax[0][2].set_title('original_img '+ ' id: '+ '6c9cf179f', fontsize=14)
ax[1][0].imshow(color_conversion(img_1, R_mean, B_mean, G_mean)[...,[2,1,0]])
ax[1][0].set_title('color_conversion_img '+ ' id: '+ '00e903abe', fontsize=14)
ax[1][1].imshow(color_conversion(img_2, R_mean, B_mean, G_mean)[...,[2,1,0]])
ax[1][1].set_title('color_conversion_img '+ ' id: '+ '00e903abe', fontsize=14)
ax[1][2].imshow(color_conversion(img_3, R_mean, B_mean, G_mean)[...,[2,1,0]])
ax[1][2].set_title('color_conversion_img '+ ' id: '+ '6c9cf179f', fontsize=14)