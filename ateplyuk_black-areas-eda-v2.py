import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import gc

import cv2

import os



import warnings

warnings.filterwarnings("ignore")
def rle2mask(rle, imgshape = (256,1600)):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )
path = '../input/severstal-steel-defect-detection/'
tr = pd.read_csv(path + 'train.csv')

print(tr.shape)

tr.head()
df = tr[tr['EncodedPixels'].notnull()]

df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

print(len(df))

df.head()
def ShowImgMask(df, sub = 'train',  columns = 1, rows = 4):

    fig = plt.figure(figsize=(20,columns*rows+6))

    for i in range(1,columns*rows+1):

        fn = df['ImageId_ClassId'].str[:-2].iloc[i]

        fig.add_subplot(rows, columns, i).set_title(fn)

        img = cv2.imread( path + sub + '_images/'+fn )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = rle2mask(df['EncodedPixels'].iloc[i])

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i in range(0, len(contours)):

            cv2.polylines(img, contours[i], True, 0, 2)

        plt.imshow(img)

    plt.show()
ShowImgMask(df)
def GetLftRgtBl(img_nm, sub = 'train'):

    img = plt.imread(path + sub + '_images/' + img_nm)[:,:,0][:1][0]

    

    bgn_lf = 0

    for i, x in enumerate(img):

        if x > 15:

            bgn_lf = i

            break

    

    bgn_rg = 0

    for i, x in reversed(list(enumerate(img))):

        if x > 15:

            bgn_rg = i

            break

    return bgn_lf, bgn_rg
%%time

df_cut2 = df.copy(deep=True).reset_index(drop=True)

df_cut2['BgnLeft'] = 0

df_cut2['BgnRigth'] = 0

df_cut2.head()



for i, row in df_cut2.iterrows():    

    df_cut2.at[i,'BgnLeft'], df_cut2.at[i,'BgnRigth'] = GetLftRgtBl(row['ImageId'])
df_cut2.head(10)
df_bl2 = df_cut2[(df_cut2['BgnLeft'] > 0) | (df_cut2['BgnRigth']  < 1599)]

print(len(df_bl2))                                             

df_bl2.head()
ShowImgMask(df_bl2)
def ShowDistBlc(df_black, df_all):

    lbls = ('has_black_area', 'no_black_area')

    y_pos = np.arange(len(lbls))

    cnt = [len(df_black),(len(df_all) - len(df_black))]

    print(cnt)



    plt.bar(y_pos, cnt, align='center', alpha=0.5)

    plt.xticks(y_pos, lbls)

    plt.ylabel('Count')

    plt.title('Distribution of images with black areas and without')



    plt.show()
ShowDistBlc(df_bl2, df)
def ShowDist(df, isblk = ''):

    lbls = ('1', '2', '3', '4')

    y_pos = np.arange(len(lbls))



    cnt = df.groupby('ClassId')['ImageId_ClassId'].count()

    print(cnt)



    plt.bar(y_pos, cnt, align='center', alpha=0.5)

    plt.xticks(y_pos, lbls)

    plt.ylabel('Count')

    plt.title('Distribution of deffect classes among images with ' + isblk +' black areas')



    plt.show()
ShowDist(df_bl2)
# No black



df_no_bl = df_cut2[~df_cut2.index.isin(df_bl2.index)]

ShowDist(df_no_bl, 'no')
def ShowPlt(side):                    

    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    axs = axs.ravel()



    for i in range(4):

        df1 = df_bl2[df_bl2['ClassId'] == str(i + 1)]



        if side == 0: # Rigth side area

            df_rgt =df1[df1['BgnLeft'] == 0]

            cnt1 = 1599 - df_rgt['BgnRigth']

        else: # Left side area

            df_rgt =df1[df1['BgnRigth'] == 1599]

            cnt1 = 0 + df_rgt['BgnLeft']



        axs[i].hist(cnt1)

        axs[i].set_title('id = ' + str(i + 1))



    axs[0].set_xlabel('Length of black area')

    axs[0].set_ylabel('Count')

    plt.show()
ShowPlt(side=0)
ShowPlt(side=1)
files = []

for file in os.listdir(path + 'test_images/'):

    files.append(file)



df_tst = pd.DataFrame(files, columns=['ImageId'])

print(len(df_tst))

df_tst.head()
%%time

df_tst2 = df_tst.copy(deep=True).reset_index(drop=True)

df_tst2['BgnLeft'] = 0

df_tst2['BgnRigth'] = 0

df_tst2.head()



for i, row in df_tst2.iterrows():    

    df_tst2.at[i,'BgnLeft'], df_tst2.at[i,'BgnRigth'] = GetLftRgtBl(row[0], 'test')
df_tst2.head()
df_tst_bl2 = df_tst2[(df_tst2['BgnLeft'] > 0) | (df_tst2['BgnRigth']  < 1599)]

print(len(df_tst_bl2))                                             

df_tst_bl2.head()
ShowDistBlc(df_tst_bl2, df_tst)
plt.imshow(plt.imread(path + 'test_images/' + df_tst_bl2.iloc[4]['ImageId']))
def ShowLenDistTst(side):

    

    if side == 0: # Rigth side black area

        df_rgt = df_tst_bl2[df_tst_bl2['BgnLeft'] == 0]

        cnt = 1599 - df_rgt['BgnRigth']

    elif side == 1: # Left side black area

        df_rgt =df_tst_bl2[df_tst_bl2['BgnRigth'] == 1599]

        cnt = 0 + df_rgt['BgnLeft']    



    plt.hist(cnt)

    plt.xlabel('Length of black areas')

    plt.ylabel('Count')

    plt.title('Distribution of lengths of black areas')



    plt.show()
ShowLenDistTst(0)
ShowLenDistTst(1)
ShowPlt(1)
df_tst_bl2.head()
df_lft_67 = df_tst_bl2[(df_tst_bl2['BgnRigth'] == 1599) & 

                       (df_tst_bl2['BgnLeft'] > 590) &

                       (df_tst_bl2['BgnLeft'] < 710)]





print(len(df_lft_67))

df_lft_67.head()
plt.figure(figsize=(20, 2))

plt.imshow(plt.imread(path + 'test_images/' + df_lft_67.iloc[4]['ImageId']))

plt.show()