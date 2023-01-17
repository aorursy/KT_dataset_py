import pandas as pd

from os import listdir

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import pydicom
dir_top = '../input/siim-isic-melanoma-classification/'



dir_ts_dcm = dir_top + 'test/'

dir_tr_dcm = dir_top + 'train/'



dir_ts_jpg = dir_top + 'jpeg/test/'

dir_tr_jpg = dir_top + 'jpeg/train/'



file_skin = "https://lipy.us/docs/SkinAreas.png"
df_ts = pd.read_csv(dir_top + 'test.csv')

df_tr = pd.read_csv(dir_top + 'train.csv')

df_ts = df_ts.rename(columns={'anatom_site_general_challenge':'anatom_site','age_approx':'age_apx','benign_malignant':'b_m'})

df_tr = df_tr.rename(columns={'anatom_site_general_challenge':'anatom_site','age_approx':'age_apx','benign_malignant':'b_m'})

df_tr.head()
df_tr_b = df_tr[ df_tr['b_m']=='benign' ]

df_tr_m = df_tr[ df_tr['b_m']=='malignant' ]
def miss_one(df, s):

    

    ros = len(df)

    mis = df.isnull().sum()

    new = df.fillna(0).astype('str')

    

    tab = pd.concat([mis, round(100*mis/ros,1)], axis=1)

    tab = tab.rename(columns = {0:s+' Mis', 1:s+' Mis%'})

    tab = tab[ tab.iloc[:,1] != 0 ]

    

    msg = s + " has " + str(ros) + " rows and " + str(df.shape[1]) + " columns. "

    msg = msg + str(tab[tab.iloc[:,1]!=0].shape[0]) + " columns have missing values."    

    return new, tab, msg
df_ts, tab_ts, msg_ts = miss_one(df_ts, "Test")

df_tr, tab_tr, msg_tr = miss_one(df_tr, "Train")

df_tr_b, tab_tr_b, msg_tr_b = miss_one(df_tr_b, "Train Ben")

df_tr_m, tab_tr_m, msg_tr_m = miss_one(df_tr_m, "Train Mal")



tab_all = pd.concat([tab_ts, tab_tr, tab_tr_b, tab_tr_m], axis=1)

tab_all = tab_all.sort_values('Train Mis%', ascending=False)



print(msg_ts + "\n" + msg_tr + "\n" + msg_tr_b + "\n" + msg_tr_m)

tab_all
def divby(a, b):

    if b==0:

        return 0

    else:

        return round(100 * a / b, 0)

    

def gdfplot(gpc, ref, val):

    

    if gpc not in df_ts.columns:

        df_ts[gpc] = '?'

    

    dfTs = df_ts.groupby([gpc]).size()

    dfTr = df_tr.groupby([gpc]).size()

    dfTrM = df_tr_m.groupby([gpc]).size()    

    

    dfC = pd.concat([dfTs, dfTr, dfTrM], axis=1).reset_index()

    dfC = dfC.rename(columns={0:'Test', 1:'Train', 2:'MalTrain', 'index':gpc}).fillna(0)

    dfC['MalPerc'] = dfC.apply(lambda x: divby(x.MalTrain, x.Train), axis = 1)

    if ref != '':

        dfC[ref] = val

        

    print(dfC)

    

    dfD = dfC.drop([gpc], axis=1)

    dfD = round(100 * (dfD - dfD.min())/(dfD.max()-dfD.min()), 0)

    dfD[gpc] = dfC[gpc].apply(lambda x: x[:10])

    print("\nNormalized values")

    print(dfD)

    

    plt.figure(figsize=(13, 2), dpi= 80, facecolor='w', edgecolor='k')    

    plt.plot(gpc, 'Test', data=dfD, marker='', color='red', linewidth=1, label='Test')

    plt.plot(gpc, 'Train', data=dfD, marker='', color='yellow', linewidth=1, label='Train')

    plt.plot(gpc, 'MalTrain', data=dfD, marker='', color='green', linewidth=1, label='MalTrain')

    plt.plot(gpc, 'MalPerc', data=dfD, marker='', color='blue', linewidth=1, label='MalTrain')

    if ref != '':

        plt.plot(gpc, ref, data=dfD, marker='', color='olive', linewidth=1, label=ref)

    plt.legend(bbox_to_anchor=(1.15, 1.0))

    plt.show()
plt.figure(figsize=(8, 8))

plt.imshow(mpimg.imread(file_skin))

plt.show()
gdfplot('anatom_site', 'SitePerc', [0, 8, 32, 2, 11, 32, 15])
gdfplot('age_apx', 'AgePerc', [0,12.2,6.4,6.4,6.6,7.2,6.8,6.6,6.0,6.3,6.3,6.5,6.3,5.4,4.4,2.9,1.9,1.8])
gdfplot('sex', 'SexPerc', [0, 50, 50])
gdfplot('diagnosis', '', [])
jpg_ts = dir_ts_jpg + df_ts['image_name'] + '.jpg'

jpg_tr_b = dir_tr_jpg + df_tr_m['image_name'] + '.jpg'

jpg_tr_m = dir_tr_jpg + df_tr_m['image_name'] + '.jpg'



dcm_ts = dir_ts_dcm + df_ts['image_name'] + '.dcm'

dcm_tr_b = dir_tr_dcm + df_tr_m['image_name'] + '.dcm'

dcm_tr_m = dir_tr_dcm + df_tr_m['image_name'] + '.dcm'



sam_jpg_tr_b = jpg_tr_b.sample(n=6, replace=True, axis=0).reset_index()

sam_jpg_tr_m = jpg_tr_m.sample(n=6, replace=True, axis=0).reset_index()



sam_dcm_tr_b = dcm_tr_b.sample(n=6, replace=True, axis=0).reset_index()

sam_dcm_tr_m = dcm_tr_m.sample(n=6, replace=True, axis=0).reset_index()



print(sam_jpg_tr_b.iloc[0,1])
fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(15, 12))



def jpgplot(H, r, c):

    

    img = mpimg.imread(H.iloc[c, 1])    

    axs[r, c].imshow(img, cmap='gray')

    axs[r, c].set_xticklabels([])

    axs[r, c].set_yticklabels([])

    

    axs[r+2, c].hist(img[:, :, 0].ravel(), bins = 256, color = 'Red', alpha = 0.5)

    axs[r+2, c].hist(img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

    axs[r+2, c].hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

    axs[r+2, c].set_xticklabels([])

    axs[r+2, c].set_yticklabels([])

    

def dcmplot(H, r, c):

    

    img = pydicom.dcmread(H.iloc[c, 1])

    axs[r, c].imshow(-img.pixel_array, cmap=plt.cm.bone)

    axs[r, c].set_xticklabels([])

    axs[r, c].set_yticklabels([])

    

for c in range(5):

    

    jpgplot(sam_jpg_tr_b, 0, c)

    jpgplot(sam_jpg_tr_m, 1, c)

    dcmplot(sam_dcm_tr_b, 4, c)

    dcmplot(sam_dcm_tr_m, 5, c)

    

for axe, col in zip(axs[0], ['Sample 1','Sample 2','Sample 3','Sample 4','Sample 5']):

    axe.set_title(col, rotation=0, size='small')

    

for axe, row in zip(axs[:,0], ['Benign Jpg','Malign Jpg','Benign Histo','Malign Histo','Benign Dcm','Malign Histo']):

    axe.set_ylabel(row, rotation=90, size='small')



plt.show()