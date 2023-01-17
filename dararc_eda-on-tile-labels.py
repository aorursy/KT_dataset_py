import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from subprocess import check_output

import skimage.io

import matplotlib

import matplotlib.pyplot as plt
os.mkdir('/kaggle/working/masks/')
import zipfile

with zipfile.ZipFile("../input/panda-step-1-tiling/masks.zip","r") as z:

    z.extractall("../masks")
# loading and using our training data

MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'

MAIN_DIR = '../input/prostate-cancer-grade-assessment'

# load data

train = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv'))

K_0 = train[(train.data_provider == 'karolinska') & (train.gleason_score == '0+0')].iloc[:,0]

K_1 = train[(train.data_provider == 'karolinska') & (train.gleason_score == '3+3')].iloc[:,0]

K_2 = train[(train.data_provider == 'karolinska') & (train.gleason_score == '4+4')].iloc[:,0]

K_3 = train[(train.data_provider == 'karolinska') & (train.gleason_score == '5+5')].iloc[:,0]

R_All = train[train.data_provider == 'radboud'].iloc[:,0]

list_of_dfs = [K_0, K_1, K_2, K_3, R_All]

for item in list_of_dfs:

    print(len(item))

names = [name[:-10] for name in os.listdir(MASKS)]

def drop_id(df_of_ids):

    for i in df_of_ids.index:

        if df_of_ids[i] not in names:

            df_of_ids = df_of_ids.drop([i])

    return(df_of_ids)

new_list_of_dfs = [drop_id(item) for item in list_of_dfs]            

for item in new_list_of_dfs:

    print(len(item))

Kar_0 = pd.Series(['karolinska'])

Rad = pd.Series(['radboud'])

Rad = Rad.repeat(len(new_list_of_dfs[4])*12)

Kar_0 = Kar_0.repeat((len(new_list_of_dfs[0])+len(new_list_of_dfs[1])+len(new_list_of_dfs[2])+len(new_list_of_dfs[3]))*12)

dp_s = pd.concat([Kar_0, Rad])
images = [skimage.io.imread("../masks/"+K_3.iloc[0]+"_"+str(i)+".png") for i in range(10,22)]

plt.imshow(images[10])

num_canc = [(image == 2).sum() for image in images]


K_0_S = pd.Series(new_list_of_dfs[0])

K_0_S = K_0_S.repeat(12)

K_1_S = pd.Series(new_list_of_dfs[1])

K_1_S = K_1_S.repeat(12)

K_2_S = pd.Series(new_list_of_dfs[2])

K_2_S = K_2_S.repeat(12)

K_3_S = pd.Series(new_list_of_dfs[3])

K_3_S = K_3_S.repeat(12)

R_All_S = pd.Series(new_list_of_dfs[4])

R_All_S = R_All_S.repeat(12)



ids_s = pd.concat([K_0_S, K_1_S, K_2_S, K_3_S, R_All_S])



tile_ind = [i for i in range(0,12)] * (int(len(ids_s)/12))

tile_ind_s = pd.Series(tile_ind)

non_existent = []

vals = np.zeros((len(ids_s), 5))

for i in range(0,len(ids_s)):

        idx = tile_ind_s[i] + 10

        if os.path.isfile("../masks/"+ids_s.iloc[i]+"_"+str(idx)+".png"):

            image = skimage.io.imread("../masks/"+ids_s.iloc[i]+"_"+str(idx)+".png")

            if i < len(K_0_S):

                for j in range(0,2):

                    vals[i,j] = (image == j).sum()

            else: 

                if i < (len(K_0_S)+len(K_1_S)):

                    for j in range(0,3):

                        vals[i,j] = (image == j).sum()

                else: 

                    if i < (len(K_0_S)+len(K_1_S)+len(K_2_S)):

                        for j in range(0,2):

                            vals[i,j] = (image == j).sum()

                        vals[i,3] = (image == 2).sum()

                    else:

                        if i < (len(K_0_S)+len(K_1_S)+len(K_2_S)+len(K_3_S)):

                            for j in range(0,2):

                                vals[i,j] = (image == j).sum()

                            vals[i,4] = (image == 2).sum()

                        else:

                            for j in [3,4,5]:

                                vals[i,(j-1)] = (image == j).sum()

                            vals[i,0] = (image == 0).sum()

                            vals[i,1] = (image == 1).sum() + (image == 2).sum()

        else:

            non_existent.append(ids_s.iloc[i]+"_"+str(idx)+".png does not exist")

                                

vals = (vals * 100) / (128 ** 2)
df2 = pd.DataFrame(vals, columns=['%background', '%benign', '%Gleason3', '%Gleason4', '%Gleason5'])
gs_l = [train[(train.image_id == ids_s.iloc[i])].iloc[0]['gleason_score'] for i in range(0,len(ids_s))]

ids_l = ids_s.tolist()

dp_l = dp_s.tolist()

tile_ind_l = tile_ind_s.tolist()



# maybe lists instead of series maybe multi index is goin mad


d = {'image_id': ids_l, 'data_provider': dp_l, 'Gleason_score': gs_l, 'tile_index': tile_ind_s}

df1 = pd.DataFrame(data=d)

df = pd.concat([df1, df2], axis=1)

df.to_csv('tile_info.csv', index=False)
import shutil

shutil.rmtree("../masks")