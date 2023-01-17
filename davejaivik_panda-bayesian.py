# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib

import matplotlib.pyplot as plt

from IPython.display import Image, display

import openslide

import PIL



import tensorflow as tf

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten, Conv3D,MaxPooling3D, BatchNormalization

from tensorflow.keras import optimizers

from sklearn.metrics import roc_auc_score

from mpl_toolkits import mplot3d

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(os.listdir('../input/prostate-cancer-grade-assessment/train_images'))-len(os.listdir('../input/prostate-cancer-grade-assessment/train_label_masks'))
data_path = "../input/prostate-cancer-grade-assessment/"

img_path = data_path + "train_images/"

mask_path = data_path + "train_label_masks/"

trn = pd.read_csv(data_path+"train.csv").set_index('image_id')

trn.head()
trn.info()
samples = trn.sample(frac=0.01, replace=True, random_state=7)

samples.info()
for i in samples.index:

    path = mask_path+i+"_mask.tiff"

    if(os.path.exists(path)):

        continue

    else:

        samples.drop(i,inplace=True)



samples.info()
slide = openslide.OpenSlide(img_path+"0005f7aaab2800f6170c399693a96917.tiff")

print(slide.level_count)

print(slide.level_dimensions)

f1,ax1 = plt.subplots(3,5,figsize=(15,11))

for i in range(slide.level_count):

    im = slide.read_region((0,0),slide.level_count - (i+1), slide.level_dimensions[-1])

    imn = np.asarray(im)

    ax1[i,0].imshow(im)

    ax1[i,0].set_title("Level: {} \tDimension:{}".format(i+1,slide.level_dimensions[-1]))

    ax1[i,0].axis('off')

    for j in range(imn.shape[2]):

        ax1[i,j+1].imshow(imn[:,:,j])

        ax1[i,j+1].set_title("Slice: {}".format(j+1))

        ax1[i,j+1].axis('off')

f1.tight_layout()

f1.suptitle("Dimensions:{}".format(slide.level_dimensions[-1]))

plt.show()
f1,ax1 = plt.subplots(3,5,figsize=(15,12))

for i in range(slide.level_count):

    im = slide.read_region((0,0),slide.level_count - (i+1), slide.level_dimensions[-2])

    imn = np.asarray(im)

    ax1[i,0].imshow(im)

    ax1[i,0].set_title("Level: {}".format(i+1))

    ax1[i,0].axis('off')

    for j in range(imn.shape[2]):

        ax1[i,j+1].imshow(imn[:,:,j])

        ax1[i,j+1].set_title("Slice: {}".format(j+1))

        ax1[i,j+1].axis('off')

f1.tight_layout()

f1.suptitle("Dimensions:{}".format(slide.level_dimensions[-2]))

plt.show()

images = []

for i in samples.index:

    #slide = openslide.OpenSlide(img_path+"0005f7aaab2800f6170c399693a96917.tiff")

    path = img_path+i+".tiff"

    if(os.path.exists(path)):

        slide = openslide.OpenSlide(path)

        im = slide.read_region((0,0),slide.level_count - 1, slide.level_dimensions[-1])

        images.append( (i,np.asarray(im),samples.loc[i,'isup_grade'],samples.loc[i,'gleason_score']) )

        slide.close()

    else:

         images.append( (i,None,samples.loc[i,'isup_grade'],samples.loc[i,'gleason_score']) )

    

f,ax = plt.subplots(3,3,figsize=(10,15))

for i, im in enumerate(images[:9]):

    ax[i//3,i%3].imshow(im[1])

    ax[i//3, i%3].axis('off')  

    ax[i//3,i%3].set_title('ISUP: {}  Gleason: {}'.format(im[2],im[3]))

f.tight_layout()

plt.show()
slide = openslide.OpenSlide(mask_path+"0005f7aaab2800f6170c399693a96917_mask.tiff")

print(slide.level_count)

print(slide.level_dimensions)

#cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

f1,ax1 = plt.subplots(3,5,figsize=(15,12))

for i in range(slide.level_count):

    im = slide.read_region((0,0),slide.level_count - (i+1), slide.level_dimensions[-1])

    imn = np.asarray(im)

    ax1[i,0].imshow(im)

    ax1[i,0].set_title("Level: {}".format(i+1))

    ax1[i,0].axis('off')

    for j in range(imn.shape[2]):

        ax1[i,j+1].imshow(imn[:,:,j])

        ax1[i,j+1].set_title("Slice: {}".format(j+1))

        ax1[i,j+1].axis('off')

f1.tight_layout()

f1.suptitle("Dimensions:{}".format(slide.level_dimensions[-1]))

plt.show()
masks = []

for i in samples.index:

    path = mask_path+i+"_mask.tiff"

    if(os.path.exists(path)):

        slide = openslide.OpenSlide(path)

        #print(slide.level_count - 1, slide.level_dimensions[-1])

        im = slide.get_thumbnail(size=slide.level_dimensions[-1])

        imn = np.asarray(im)[:,:,0]

        masks.append( (i,imn,samples.loc[i,'isup_grade'],samples.loc[i,'gleason_score']) )

        slide.close()

    else:

        masks.append( (i,None,samples.loc[i,'isup_grade'],samples.loc[i,'gleason_score']) )

    

f,ax = plt.subplots(3,3,figsize=(10,15))

cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

for i, im in enumerate(masks[:9]):

    ax[i//3,i%3].imshow(im[1],cmap=cmap)

    ax[i//3,i%3].axis('off')  

    ax[i//3,i%3].set_title('ISUP: {}  Gleason: {}'.format(im[2],im[3]))

f.tight_layout()

plt.show()
def overlay(ind, center='radboud', alpha=0.8, max_size=(1024, 1024)):

    

    ov_img = []

    ptl = []

    for i in ind:

        slide = openslide.OpenSlide(img_path+i+".tiff")

        path = mask_path+i+"_mask.tiff"

        if(os.path.exists(path)):

            mask = openslide.OpenSlide(mask_path+"{}_mask.tiff".format(i))

            slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

            mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

            mask_data = mask_data.split()[0]

    

            alpha_int = int(round(255*alpha))

            if center == 'radboud':

                alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)

            elif center == 'karolinska':

                alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)



            alpha_content = PIL.Image.fromarray(alpha_content)

            preview_palette = np.zeros(shape=768, dtype=int)



            if center == 'radboud':

                # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

                preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

            elif center == 'karolinska':

                # Mapping: {0: background, 1: benign, 2: cancer}

                preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)



            mask_data.putpalette(data=preview_palette.tolist())

                #mask_data.putpalette(data=preview_palette.tolist())

            mask_rgb = mask_data.convert(mode='RGB')

            overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)

            overlayed_image.thumbnail(size=max_size, resample=0)

            ov_img.append(overlayed_image)

            ptl.append([i,np.asarray(slide_data),np.asarray(mask_data),np.asarray(overlayed_image)])

            #print(np.asarray(alpha_content).shape)

            #ax[i//3, i%3].imshow(overlayed_image) 

            slide.close()

            mask.close()       



        else:

            print("in")

            slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

            ov_img.append(np.asarray(slide_data))

            ptl.append([i,np.asarray(slide_data),None,np.asarray(slide_data)])

            plt.imshow(np.asarray(slide_data))

            slide.close()

    return ov_img,ptl

    
#print(samples.index[0])

mx = 600

ovr,ptl = overlay(samples.index,max_size=(mx,mx))
f,ax = plt.subplots(3,3, figsize=(15,10))

for i, enm in enumerate(ptl[:3]):

    #print([i,i%3],[i,i%3+1],[i,i%3+2])

    for j in range(3):

        ax[i,j].imshow(enm[j+1])

        ax[i,j].axis("off")

        if(j==0):

            ax[i,j].set_title("Image")

        elif(j==1):

            ax[i,j].set_title("Mask")

        else:

            ax[i,j].set_title("Overlay Image")

        

        #ax[i,j].imshow(j[1])

        #ax[i,j].set_title("Mask")

        #ax[i,j].axis("off")

        #ax[i,j].imshow(j[2])

        #ax[i,j].set_title("Overlay Image")

        #ax[i,j].axis("off")
def get_pad(imc,mx):

    im1 = np.zeros([mx,mx,3]).astype('uint8')

    #print(imc.shape)

    if(imc.shape[0]==mx):

        for i in range(im1.shape[2]):

            im1[:imc.shape[0],:imc.shape[1],i] = imc[:,:,i]

        #f,ax = plt.subplots(1,2,figsize=(7,3))

        #ax[0].imshow(imc)

        #ax[1].imshow(im1)

    if(imc.shape[1]==mx):

        for i in range(im1.shape[2]):

            im1[:imc.shape[0],:imc.shape[1],i] = imc[:,:,i]

        #f,ax = plt.subplots(1,2,figsize=(7,3))

        #ax[0].imshow(imc)

        #ax[1].imshow(im1)

    

    return im1





i = 0

while (i<len(ptl)):

    if(max(ptl[i][3].shape)<mx):

        samples.drop(ptl[i][0],inplace=True)

        del ptl[i]

    else:

        ptl[i].append(get_pad(ptl[i][3],max(ptl[i][3].shape)))

        i+=1
#print(np.array(ptl)[:,4].shape)

X_train = []

#mp = 0

for i in ptl:

#    print(i[4].shape)

    #print(i[4])

    #tmp.shape = [tmp.shape[0],tmp.shape[1],tmp.shape[2],1]

    X_train.append(i[4])





X_train = np.array(X_train)

samples['gleason_code'] = samples['gleason_score'].astype("category").cat.codes

y1_train = samples['isup_grade'].values

y2_train = samples['gleason_code'].values



#print(X_train.shape)
print(X_train.shape)

print(y1_train.shape)

print(y2_train.shape)
import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

in_shape = X_train[0].shape

inpt = keras.Input(shape=in_shape,name = 'inputs')

cnv1 = layers.Conv2D(16, 3, activation="relu")(inpt)

mxp1 = layers.MaxPooling2D(2)(cnv1)

cnv2 = layers.Conv2D(32, 3, activation="relu")(mxp1)

mxp2 = layers.MaxPooling2D(2)(cnv2)

flt = layers.Flatten()(mxp2)

#drop = layers.Dropout(0.1)(flt)



#D1_1 = layers.Dense(64, activation="relu")(flt)

#D1_2 = layers.Dense(128, activation="relu")(D1_1)



#D1 = layers.Dense(64, activation="relu")(flt)

D2 = layers.Dense(128, activation="relu")(flt)



out_1 = layers.Dense(6,activation='softmax',name= 'isup')(D2)



#D2_1 = layers.Dense(64, activation="relu")(flt)

#D2_2 = layers.Dense(128, activation="relu")(D2_1)

out_2 = layers.Dense(11,activation='softmax',name = 'gleason')(D2)



model = keras.Model(

    inputs=[inpt],

    outputs=[out_1,out_2],

)



keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)



model.compile(

    optimizer=keras.optimizers.Adam(lr=0.01),

    loss={

        "isup":"sparse_categorical_crossentropy", #keras.losses.SparseCategoricalCrossentropy(),#

        "gleason":"sparse_categorical_crossentropy",#keras.losses.SparseCategoricalCrossentropy(), #

    }, metrics = ['accuracy']

)





import matplotlib.image as mpimg

plt.figure(figsize=(10,20))

img = mpimg.imread("./multi_input_and_output_model.png")

plt.imshow(img)

plt.axis('off')

plt.show()
#model.fit(X_train,y_train,batch_size = 1,epochs = 5,shuffle=True)    

model.fit({'inputs':X_train},{'gleason':y1_train, 'isup':y2_train},batch_size = 8,epochs = 10, shuffle = True)
from keras import backend as K



#for l in model.layers:

#    print(l.name,l.output_shape)



get_vec = K.function([model.layers[0].input],

                      [model.layers[6].output])



#get_out1 = K.function([model.layers[0].input],

#                      [model.layers[9].output])



out_vec = np.array(get_vec([X_train]) )

out_vec = np.squeeze(out_vec)

print(out_vec.shape)



#lyr_7 = np.array(get_6rd_layer_output([X_train]))

#lyr_8 = np.array(get_6rd_layer_output([X_train]))

#lyr_7.shape = [lyr_7.shape[1],lyr_7.shape[2]]



df = pd.DataFrame(out_vec)

df['gleason_code'] = samples['gleason_code'].values

df['isup_grade'] = samples['isup_grade'].values

print(df.info())

#print(os.path.exists("../input/output"))

df.to_csv("test4.csv", index=False, float_format='%.4f')

#df.head()

#print(os.getcwd())

f,ax = plt.subplots(3,3,figsize=(15,10))

clr = ['r','g','b']

#imn = []

#for i in range(len(ovr[:9])):

    #im = np.array(ovr[i])

    #imn.append(256*( (im-np.min(im))/(np.max(im)-np.min(im)) ))

    

#imn = np.array(imn)



    

for i in range(len(ovr[:9])):

    imn = np.array(ovr[i])

    for j in range(3):

        uq = len(np.unique(imn[:,:,j]))

        ax[i//3,i%3].hist(imn[:,:,j].ravel(),bins=uq,range=[0,uq],color=clr[j])

        mn = np.mean(imn[:,:,j])

        std = np.std(imn[:,:,j])

        #print(mn,std)

        #ax[i//3,i%3].set_ylim([0,])

        #ax.set_title("Histogram of Image:{}")

plt.show()
f,ax = plt.subplots(3,3,figsize=(15,10))

clr = ['r','g','b']

#imn = []

#for i in range(len(ovr[:9])):

    #im = np.array(ovr[i])

    #imn.append(256*( (im-np.min(im))/(np.max(im)-np.min(im)) ))

    

#imn = np.array(imn)



    

for i in range(len(ovr[:9])):

    im = np.array(ovr[i])

    for j in range(3):

        uq = len(np.unique(im[:,:,j]))

        imh = cv2.calcHist([im[:,:,j]],[0],None,[uq],[0,uq])

        #np.where(imh<=50,0,imh)

        imh.shape = imh.T.shape

        ax[i//3,i%3].plot(range(uq),imh[0],color=clr[j])

        #print(mn,std)

        #ax[i//3,i%3].set_ylim([0,])

        #ax.set_title("Histogram of Image:{}")

plt.show()
f,ax = plt.subplots(3,3,figsize=(10,7))

clr = ['r','g','b']

#imhr = cv2.calcHist([imsr],[0],None,[256],[0,256])

#imh = cv2.calcHist([ovrn],[3],None,[256],[0,256])

for i in range(len(ovr[:3])):

    im = np.array(ovr[i])

    for j in range(3):

        uq = len(np.unique(im[:,:,j]))

        imh = cv2.calcHist([im[:,:,j]],[0],None,[uq],[0,uq])

        np.where(imh<=50,0,imh)

        imh.shape = imh.T.shape

        ax[i,j].plot(imh[0],color=clr[j])

plt.show()
# UDF

def thresh(arr,th):

    for i in range(len(arr)):

        if(arr[i]<=th):

            arr[i] = 0

    

    return arr



def get_zeros(arr,th):

    #cnt = 0; st = 0; end = 0;

    arr = thresh(arr,th)

    rng_dict = {}

    def rngs(arr,j):

        rng ={}

        cnt = 0; end = 0; k = j; flag = 0

        while ( k < len(arr) ):

            if (arr[k]==0):

                cnt+=1

            else:

                end = k;flag = 1

                break

            k+=1

        if(flag==0):

            end = k

        rng[(j,end-1)] = cnt

        return rng,end

    i = 0

    while ( i < len(arr) ):

        if (arr[i]==0):

            zrng,ind = rngs(arr,i) 

            rng_dict.update(zrng)

            i = ind

        i+=1

    return rng_dict   



def get_hist2(im):

    uq = len(np.unique(im))

    return cv2.calcHist([im],[0],None,[uq],[0,uq])



def get_segments(im,*args,th=10):

    seg = {}

    clr = ['r','g','b']

    #hst_im = []

    def segs(im,dct):

        #rng = len(list(dct.jeys()):)

        sg = []

        keys = np.array(list(dct.keys()))

        for k in range(keys.shape[0]):

            if( k == (keys.shape[0]-1) ):

                sg.append( np.where( im>keys[k][1],im,0 ) )

                continue

            #f,ax = plt.subplots(1,2,figsize=(7,3))

            #ax[0].imshow(im)

            #ax[1].imshow(np.where( ( (im>=keys[k][1]) & (im<=keys[k+1][0]) ), im, 0))

            sg.append( np.where( ( (im>=keys[k][1]) & (im<=keys[k+1][0]) ), im, 0) )

        sg = np.array(sg)

        #print("Sg:",sg.shape)

        return sg



    for i in range(im.shape[2]):

        hst_im = get_hist2(im[:,:,i]).T[0]

        hst_zeros = get_zeros(hst_im,th)

        seg[clr[i]] = segs(im,hst_zeros)

        #seg.append( np.where( ( (im>=j[1]) & (im<=j[0]) ), a, 0) )

    return seg

    

def get_seg_data(ptl,*args,th=10):

    x_train = []

    y1_train = []

    y2_train = []

    #clr = ['r','g','b']

    #hst_im = []

    def segs(im,dct,plit):

        #rng = len(list(dct.jeys()):)

        #print(im.shape)

        keys = np.array(list(dct.keys()))

        for k in range(keys.shape[0]):

            if( k == (keys.shape[0]-1) ):

                tmp_img = np.where( im>keys[k][1],im,0 )

                if(tmp_img.mean()>=50):

                    x_train.append( tmp_img )

                    y1_train.append(samples.loc[plit[0]]['gleason_code'])

                    y2_train.append(samples.loc[plit[0]]['isup_grade'])

                continue

            tmp_im = np.where( ( (im>=keys[k][1]) & (im<=keys[k+1][0]) ), im, 0)

            if(tmp_im.mean()>=50):

                x_train.append( tmp )

                y1_train.append(samples.loc[plit[0]]['gleason_code'])

                y2_train.append(samples.loc[plit[0]]['isup_grade'])

        

        #print("Sg:",sg.shape)

    

    for i in ptl:#range(len(ptl)):

        im = i[4]

        for j in range(im.shape[2]):

            hst_im = get_hist2(im[:,:,j]).T[0]

            hst_zeros = get_zeros(hst_im,th)

            segs(im[:,:,j],hst_zeros,i) 

    

    x_train = np.array(x_train)

    y1_train = np.array(y1_train)

    y2_train = np.array(y2_train)

        #seg[clr[i]] = segs(im,hst_zeros)

        #seg.append( np.where( ( (im>=j[1]) & (im<=j[0]) ), a, 0) )

    return x_train,y1_train,y2_train

    
imh1 = []

imx = np.array(ovr[1])

for j in range(imx.shape[2]):

    uq = len(np.unique(imx[:,:,j]))

    #print(im[:,:,j])

    imh = cv2.calcHist([im[:,:,j]],[0],None,[uq],[0,uq])

    imh.shape = imh.T.shape

    imh1.append(imh[0])

    #ax[i,j].bar(range(uq),imh[0],color=clr[j])



imh1 = np.array(imh1)



imhz = []

for i in range(imh1.shape[0]):

    imhz.append(get_zeros(imh1[i],10))

imhz = np.array(imhz)



print(imhz)
hovr = {}

hovr['image_id'] = []

hovr['hist_r'] = []

hovr['hist_g'] = []

hovr['hist_b'] = []



for i in (ptl):

    hso = []

    #print(i[3].shape)

    for j in range(i[3].shape[2]):

        #print(i[3][:,:,j].shape)

        hso.append(get_hist2(i[3][:,:,j]).T[0])

    #print(np.array(hso).shape)

    hovr['image_id'].append(i[0])

    hovr['hist_r'].append(np.array(hso[0]))

    hovr['hist_g'].append(np.array(hso[1]))

    hovr['hist_b'].append(np.array(hso[2]))



img_hist = pd.DataFrame(hovr)

img_hist
segs = {}

for i in ptl:

    segs[i[0]] = get_segments(i[4])

#get_segments(ptl[0][4])

#plt.imshow(ptl[0][3])

#print("See")

X_trains = None;y1_trains=None;y2_trains=None;

X_trains,y1_trains,y2_trains = get_seg_data(ptl)
X_trains.shape = [ X_trains.shape[0], X_trains.shape[1], X_trains.shape[2], 1]

print(X_trains.shape)

print(y1_trains.shape)

print(y2_trains.shape)
f,ax = plt.subplots(3,3,figsize=(15,10))

for i in range(len(X_trains[3:12])):

    ax[i//3,i%3].imshow(np.squeeze(X_trains[i*i]),cmap='gray')

    ax[i//3,i%3].axis("off")

#plt.figure(figsize=(10,15))

#plt.imshow(X_train[0])
import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



inpt = keras.Input(shape=X_trains[0].shape,name = 'inputs')

cnv1 = layers.Conv2D(16, 3, activation="relu")(inpt)

mxp1 = layers.MaxPooling2D(2)(cnv1)

cnv2 = layers.Conv2D(32, 3, activation="relu")(mxp1)

mxp2 = layers.MaxPooling2D(2)(cnv2)

flt = layers.Flatten()(mxp2)



#D1_1 = layers.Dense(256, activation="relu")(flt)

#D1_2 = layers.Dense(512, activation="relu")(D1_1)



#D1 = layers.Dense(128, activation="relu")(flt)

D2 = layers.Dense(128, activation="relu")(flt)

out_1 = layers.Dense(6,activation='softmax',name= 'isup')(D2)



#D2_1 = layers.Dense(256, activation="relu")(flt)

#D2_2 = layers.Dense(512, activation="relu")(D2_1)

out_2 = layers.Dense(11,activation='softmax',name = 'gleason')(D2)



model = keras.Model(

    inputs=[inpt],

    outputs=[out_1,out_2],

)





keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)



model.compile(

    optimizer=keras.optimizers.Adam(lr=0.01),

    loss={

        "isup":"sparse_categorical_crossentropy", #keras.losses.SparseCategoricalCrossentropy(),#

        "gleason":"sparse_categorical_crossentropy",#keras.losses.SparseCategoricalCrossentropy(), #

    }, metrics = ['accuracy']

)

import matplotlib.image as mpimg

plt.figure(figsize=(10,20))

img = mpimg.imread("./multi_input_and_output_model.png")

plt.imshow(img)

plt.show()
model.fit({'inputs':X_trains},{'gleason':y1_trains, 'isup':y2_trains},batch_size=4,epochs = 10, shuffle = True)
from keras import backend as K



#for l in model.layers:

#    print(l.name,l.output_shape)



get_vec = K.function([model.layers[0].input],

                      [model.layers[6].output])



#get_out1 = K.function([model.layers[0].input],

#                      [model.layers[9].output])



out_vec = np.array(get_vec([X_trains]) )

out_vec = np.squeeze(out_vec)

print(out_vec.shape)



#lyr_7 = np.array(get_6rd_layer_output([X_train]))

#lyr_8 = np.array(get_6rd_layer_output([X_train]))

#lyr_7.shape = [lyr_7.shape[1],lyr_7.shape[2]]



df = pd.DataFrame(out_vec)

df['gleason_code'] = y1_trains#samples['gleason_code'].values

df['isup_grade'] = y2_trains#samples['isup_grade'].values

print(df.info())

#print(os.path.exists("../input/output"))

df.to_csv("test5.csv", index=False, float_format='%.4f')

#df.head()

#print(os.getcwd())
