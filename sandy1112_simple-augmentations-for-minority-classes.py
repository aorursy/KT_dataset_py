# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image



from PIL import ImageFilter,ImageEnhance,ImageChops,ImageOps

from PIL import ImageEnhance

from tqdm import tqdm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DIR = '../input/landmark-retrieval-2020/'

#path_to_train = DIR + '/train/'

train = pd.read_csv("../input/landmark-retrieval-2020/train.csv")
def get_paths(sub):

    index = ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]



    paths = []



    for a in index:

        for b in index:

            for c in index:

                try:

                    paths.extend([f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}/" + x for x in os.listdir(f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}")])

                except:

                    pass



    return paths
train_path = train



rows = []

for i in tqdm(range(len(train))):

    row = train.iloc[i]

    path  = list(row["id"])[:3]

    temp = row["id"]

    row["id"] = f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{temp}.jpg"

    rows.append(row["id"])

    

rows = pd.DataFrame(rows)

train_path["id"] = rows
k =train[['id','landmark_id']].groupby(['landmark_id']).agg({'id':'count'})

k.rename(columns={'id':'Count_class'}, inplace=True)

k.reset_index(level=(0), inplace=True)

freq_ct_df = pd.DataFrame(k)

freq_ct_df.head()
train_labels = pd.merge(train,freq_ct_df, on = ['landmark_id'], how='left')

train_labels.head()
train_labels_lt3 = train_labels[train_labels['Count_class']<3]

train_labels_lt3.shape
##lets take a sample of 100 images

aug_sample = train_labels_lt3.sample(100)

img_list = aug_sample['id'].tolist()

id_list = aug_sample['landmark_id'].tolist()
train_labels_aug = pd.DataFrame(columns=['id','landmark_id'])
def update_file_path(filename, sufx):

    parts = filename.split('.')

    return "".join(parts[:-1])+ '_' + sufx + '.' + parts[-1]

##Creating some sample transformations. One can add many more in a similar manner

for imagefile, IdFile in zip(img_list, id_list):

    im=Image.open(imagefile)

    im_blur=im.filter(ImageFilter.GaussianBlur)

    im_unsharp=im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    im_edgeenhance = im.filter(ImageFilter.EDGE_ENHANCE_MORE)

    #im_invert = im.transpose(Image.FLIP_LEFT_RIGHT)

    im_rot30=im.rotate(30)

    os.chdir('../working/')

    _blur_path=update_file_path(imagefile, 'bl')

    im_blur.save("".join(os.path.splitext(os.path.basename(_blur_path))))

    _unsharp_path=update_file_path(imagefile, 'un')

    im_unsharp.save("".join(os.path.splitext(os.path.basename(_unsharp_path))))

    _edgeenhance_path=update_file_path(imagefile, 'edgenh')

    im_edgeenhance.save("".join(os.path.splitext(os.path.basename(_edgeenhance_path))))

    _imrot30_path=update_file_path(imagefile, 'rot30')

    im_rot30.save("".join(os.path.splitext(os.path.basename(_imrot30_path))))

    train_labels_aug= train_labels_aug.append({'id' : "".join(os.path.splitext(os.path.basename(_blur_path))) , 'landmark_id' : IdFile} , ignore_index=True)

    train_labels_aug = train_labels_aug.append({'id' : "".join(os.path.splitext(os.path.basename(_unsharp_path))), 'landmark_id' : IdFile} , ignore_index=True)

    train_labels_aug = train_labels_aug.append({'id' : "".join(os.path.splitext(os.path.basename(_edgeenhance_path))) , 'landmark_id' : IdFile} , ignore_index=True)

    train_labels_aug = train_labels_aug.append({'id' : "".join(os.path.splitext(os.path.basename(_imrot30_path))) , 'landmark_id' : IdFile} , ignore_index=True)

train_labels_aug.to_csv('train_labels_aug.csv', index=False)