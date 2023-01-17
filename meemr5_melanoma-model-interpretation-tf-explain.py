!pip install -q efficientnet

!pip install -q tf-explain
import os

import cv2

import time

import numpy as np

import pandas as pd

from datetime import date



from scipy.stats import rankdata



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from kaggle_datasets import KaggleDatasets



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras import optimizers



from sklearn.metrics import roc_auc_score, roc_curve



from tf_explain.core.grad_cam import GradCAM



# from tensorflow.keras.mixed_precision import experimental as mixed_precision

# policy = mixed_precision.Policy('mixed_bfloat16')

# mixed_precision.set_policy(policy)



import efficientnet.tfkeras as efn
SEED = 2020



LABEL_SMOOTHING = 0.05

WEIGHT_DECAY = 0



IMG_R = 256

IMG_C = 250

IMG_N = 224



ROT = 180.0

SHR = 2.0

HZOOM = 6.0

WZOOM = 6.0

HSHIFT = 8.0

WSHIFT = 8.0



R, C = 5,5
DATA_PATH = '../input/siim-isic-melanoma-classification'

MODEL_PATH = '../input/melanoma-efficientnet-b6-tpu-tta-saved-models'

TRAIN_IMG_PATH = '../input/siim-isic-melanoma-classification/jpeg/train'

TEST_IMG_PATH = '../input/siim-isic-melanoma-classification/jpeg/test'



train_df = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))

test_df = pd.read_csv(os.path.join(DATA_PATH,'test.csv'))



sgkf = pd.read_csv('../input/siimisic-stratified-group-kfold-traindata/train_StratifiedGroupK(5)Fold(SEED2020)(Group_sex_anatomsite_target).csv')



val_pred = pd.read_csv(f'{MODEL_PATH}/{[filename for filename in os.listdir(MODEL_PATH) if "VALPREDS" in filename][0]}')



train_pred = pd.read_csv(f'{MODEL_PATH}/{[filename for filename in os.listdir(MODEL_PATH) if "TRAINPREDS" in filename][0]}')

train_logs = pd.read_csv(f'{MODEL_PATH}/{[filename for filename in os.listdir(MODEL_PATH) if "TRAINING_LOGS" in filename][0]}')



test_pred = pd.read_csv(f'{MODEL_PATH}/{[filename for filename in os.listdir(MODEL_PATH) if "TESTPREDS_2" in filename][0]}')

test_pred_mean = pd.read_csv(f'{MODEL_PATH}/{[filename for filename in os.listdir(MODEL_PATH) if "TESTPREDS_MEAN" in filename][0]}')

test_pred_noaug = pd.read_csv(f'{MODEL_PATH}/{[filename for filename in os.listdir(MODEL_PATH) if "TESTPREDS_NOAUG" in filename][0]}')

test_pred_rank = pd.read_csv(f'{MODEL_PATH}/{[filename for filename in os.listdir(MODEL_PATH) if "TESTPREDS_RANK" in filename][0]}')



FOLD = int(os.listdir(MODEL_PATH)[0].split("_")[-1][0])

print("Fold: ",FOLD)



EFNMODEL = [filename for filename in os.listdir(MODEL_PATH) if 'hdf5' in filename][0].split("_")[2][-1]

print("EfficientNet Model Num: ",EFNMODEL)



val_merged = sgkf[sgkf.fold==FOLD].merge(val_pred,on='image_name')

train_merged = sgkf[sgkf.fold!=FOLD].merge(train_pred,on='image_name')
modelName = f'EfficientNetB{EFNMODEL}'



MODEL_WEIGHTS = [filename for filename in os.listdir(MODEL_PATH) if 'hdf5' in filename][0]



# model_input = tf.keras.Input(shape=(IMG_N, IMG_N, 3), name='imgInput')

    

constructor = getattr(efn, modelName)



base_model = constructor(include_top=False,

                         weights='imagenet',

                         input_shape=(IMG_N, IMG_N, 3),

                         pooling='avg')



output = tf.keras.layers.Dense(1, activation='sigmoid',dtype='float32')(base_model.output)



model = tf.keras.Model(base_model.input, output, name=modelName)

        

# model.compile(

#     optimizer='adam',

#     loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING),

#     metrics=[

#         'binary_accuracy',

#         tf.keras.metrics.AUC(name='auc'),

#         tf.keras.metrics.Recall(name='recall'),

#         tf.keras.metrics.TruePositives(name="TP"),

#         tf.keras.metrics.FalseNegatives(name="FN"),

#         tf.keras.metrics.FalsePositives(name="FP")

#     ]

# )



model.summary()



model.load_weights(f'{MODEL_PATH}/{MODEL_WEIGHTS}')
def getImage(image_name,train=True):

    PATH = TRAIN_IMG_PATH if train else TEST_IMG_PATH

    

    image = cv2.imread(f'{PATH}/{image_name}.jpg')

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    

    # Preprocessing

    image = cv2.resize(image,(IMG_R,IMG_R))

    image = tf.image.central_crop(image,IMG_C/IMG_R).numpy()

    image = cv2.resize(image,(IMG_N,IMG_N))

    

#     print(image)

    

    return image



def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):

    # apply the supplied color map to the heatmap and then

    # overlay the heatmap on the input image

    heatmap = cv2.applyColorMap(heatmap, colormap)

    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

    # return a 2-tuple of the color mapped heatmap and the output,

    # overlaid image

    return heatmap,output
fig, ax = plt.subplots(2,2,figsize=(20,15))



val_merged['ranks'] = rankdata(val_merged.target_y.values)/len(val_merged.target_y.values)

train_merged['ranks'] = rankdata(train_merged.target_y.values)/len(train_merged.target_y.values)





# sns.kdeplot(val_merged[val_merged.target_x==1].ranks,label='malignant(Rank wise)',shade=True,ax=ax[0,0])

# sns.kdeplot(val_merged[val_merged.target_x==0].ranks,label='beingn(Rank wise)',shade=True,ax=ax[0,0])

# sns.kdeplot(val_merged.target_y,label='Combined',shade=True,ax=ax[0,0])

sns.kdeplot(val_merged[val_merged.target_x==1].target_y,label='malignant',shade=True,ax=ax[0,0])

sns.kdeplot(val_merged[val_merged.target_x==0].target_y,label='benign',shade=True,ax=ax[0,0])

ax[0,0].set_title(f'Distribution of Validation set Predictions (FOLD:{FOLD})')



# sns.kdeplot(train_merged.target_y,label='Combined',shade=True,ax=ax[0,1])

sns.kdeplot(train_merged[train_merged.target_x==1].target_y,label='malignant',shade=True,ax=ax[0,1])

sns.kdeplot(train_merged[train_merged.target_x==0].target_y,label='benign',shade=True,ax=ax[0,1])

ax[0,1].set_title(f'Distribution of Training set Predictions (FOLD:{FOLD})')



fpr, tpr, thresholds = roc_curve(val_merged.target_x, val_merged.target_y)

roc_auc = roc_auc_score(val_merged.target_x, val_merged.target_y)

ax[1,0].plot(fpr,tpr,linestyle='--')

ax[1,0].set_title(f"ROC Curve of Validation set predictions (AUC: {roc_auc})")



fpr, tpr, thresholds = roc_curve(train_merged.target_x, train_merged.target_y)

roc_auc = roc_auc_score(train_merged.target_x, train_merged.target_y)

ax[1,1].plot(fpr,tpr,linestyle='--')

ax[1,1].set_title(f"ROC Curve of Training set predictions (AUC: {roc_auc})")
fig, ax = plt.subplots(1,3,figsize=(20,5))



sns.kdeplot(test_pred_mean.target,shade=True,ax=ax[0])

ax[0].set_title(f'Distribution of Test set (TTA-Mean) Predictions')



sns.kdeplot(test_pred_noaug.target,shade=True,ax=ax[1])

ax[1].set_title(f'Distribution of Test set (No-AUG) Predictions')



sns.kdeplot(test_pred_rank.target,shade=True,ax=ax[2])

ax[2].set_title(f'Distribution of Test set (TTA-Mean by Ranks) Predictions')
def displayImages(df,target):

    

    image_names = list(set(df.image_name).intersection(set(df[df.target_x==target].image_name)))

    print("Total Images: ",len(image_names))

    

    r, c = min(int(np.ceil(len(image_names)/R)),R), C

    

    if r==1:

        r=2

    

    fig, ax = plt.subplots(r,c,figsize=(c*4*2,r*6))



    layer_name = 'top_conv'



    # fig.suptitle(f'GridCAMs of layer: {layer_name} of train images',fontsize=15)



    explainer = GradCAM()



    i = 0



    for image_name in np.random.choice(image_names,replace=False,size=min(r*c,len(image_names))):



        prediction = df[df.image_name==image_name]['target_y'].values[0]

        actual = df[df.image_name==image_name]['target_x'].values[0]



        color = 'green' if int(prediction>=0.5)==actual else 'red'



        title = f'Image Name: {image_name}\nActual: {actual}\nPrediction: {prediction}'



        image = getImage(image_name)



        heatmap1 = explainer.explain(([image/255.0], None), model, layer_name = layer_name, class_index=0)



        heatmap2, output = overlay_heatmap(heatmap1,image)



        output = np.hstack([image,output])



        ax[i//c,i%c].imshow(output)

        ax[i//c,i%c].axis('off')

        ax[i//c,i%c].set_title(title,fontsize=15,color=color)



        i += 1
displayImages(train_merged,1)
displayImages(train_merged,0)
displayImages(val_merged,1)
displayImages(val_merged,0)
layer_name = 'top_conv'

explainer = GradCAM()



patient_id = np.random.choice(val_merged.patient_id.unique(),size=1,replace=False)[0]



x = val_merged[val_merged.patient_id == patient_id].sort_values(['age_approx','anatom_site_general_challenge','image_name'])



r, c = int(np.ceil(x.shape[0]/5)), 5



fig, ax = plt.subplots(r,c, figsize=(c*4*2,6*r))



# fig = fig.suptitle(f'{patient_id} Sex: {x.sex.values[0]}',fontsize=20)

print(f'{patient_id} Sex: {x.sex.values[0]}')



for i, image_name in enumerate(x.image_name.values):

    

    prediction = val_pred[val_pred.image_name==image_name]['target'].values[0]

    actual = train_df[train_df.image_name==image_name]['target'].values[0]

    age = train_df[train_df.image_name==image_name]['age_approx'].values[0]

    site = train_df[train_df.image_name==image_name]['anatom_site_general_challenge'].values[0]

    

    color = 'green' if int(prediction>=0.5)==actual else 'red'

    

    title = f'Image Name: {image_name}\nActual: {actual}\nPrediction: {prediction}\nAge:{age}\nAnatom site:{site}'

    

    image = getImage(image_name)

    

    heatmap1 = explainer.explain(([image/255.0], None), model, layer_name = layer_name, class_index=0)

    

    heatmap2, output = overlay_heatmap(heatmap1,image)

    

    output = np.hstack([image,output])

    

    if r>1:

        ax[i//c,i%c].imshow(output)

        ax[i//c,i%c].set_title(title,color=color)

        ax[i//c,i%c].axis('off')

    else:

        ax[i%c].imshow(output)

        ax[i%c].set_title(title,color=color)

        ax[i%c].axis('off')



# plt.savefig(f'{x.target.sum()}_{patient_id}_{x.sex.values[0]}.png')



plt.show()
val_merged.anatom_site_general_challenge.value_counts()
def displayAnatomsite(df,site,target,layer_name = 'top_conv'):

    image_names = list(df[(df.target_x==target) & (df.anatom_site_general_challenge==site)].image_name)

    print("Total Images: ",len(image_names))

    

    if len(image_names)==0:

        return

    

    r, c = 5, 5

    

    r, c = min(int(np.ceil(len(image_names)/5)),5), 5

    if r==1:

        r=2



    fig, ax = plt.subplots(r,c,figsize=(c*4*2,r*6))



    # fig.suptitle(f'GridCAMs of layer: {layer_name} of train images',fontsize=15)



    explainer = GradCAM()



    i = 0



    for image_name in np.random.choice(image_names,replace=False,size=min(r*c,len(image_names))):



        prediction = df[df.image_name==image_name]['target_y'].values[0]

        actual = df[df.image_name==image_name]['target_x'].values[0]



        color = 'green' if int(prediction>=0.5)==actual else 'red'



        title = f'Image Name: {image_name}\nActual: {actual}\nPrediction: {prediction}'



        image = getImage(image_name)



        heatmap1 = explainer.explain(([image/255.0], None), model, layer_name = layer_name, class_index=0)



        heatmap2, output = overlay_heatmap(heatmap1,image)



        output = np.hstack([image,output])



        ax[i//c,i%c].imshow(output)

        ax[i//c,i%c].axis('off')

        ax[i//c,i%c].set_title(title,fontsize=15,color=color)



        i += 1
displayAnatomsite(val_merged,'torso',1)
displayAnatomsite(val_merged,'torso',0)
displayAnatomsite(val_merged,'lower extremity',1)
displayAnatomsite(val_merged,'lower extremity',0)
displayAnatomsite(val_merged,'upper extremity',1)
displayAnatomsite(val_merged,'upper extremity',0)
displayAnatomsite(val_merged,'head/neck',1)
displayAnatomsite(val_merged,'head/neck',0)
displayAnatomsite(val_merged,'palms/soles',1)
displayAnatomsite(val_merged,'palms/soles',0)
displayAnatomsite(val_merged,'oral/genital',1)
displayAnatomsite(val_merged,'oral/genital',0)
layer_name = 'top_conv'



image_names = list(val_merged[(val_merged.target_x==1)&(val_merged.target_y<0.5)].image_name)

print("Total Images: ",len(image_names))



if len(image_name)!=0:



    r, c = 5, 5



    r, c = min(int(np.ceil(len(image_names)/5)),5), 5

    if r==1:

        r=2



    fig, ax = plt.subplots(r,c,figsize=(c*4*2,r*6))



    # fig.suptitle(f'GridCAMs of layer: {layer_name} of train images',fontsize=15)



    explainer = GradCAM()



    i = 0



    for image_name in np.random.choice(image_names,replace=False,size=min(r*c,len(image_names))):



        prediction = val_merged[val_merged.image_name==image_name]['target_y'].values[0]

        actual = val_merged[val_merged.image_name==image_name]['target_x'].values[0]



        color = 'green' if int(prediction>=0.5)==actual else 'red'



        title = f'Image Name: {image_name}\nActual: {actual}\nPrediction: {prediction}'



        image = getImage(image_name)



        heatmap1 = explainer.explain(([image/255.0], None), model, layer_name = layer_name, class_index=0)



        heatmap2, output = overlay_heatmap(heatmap1,image)



        output = np.hstack([image,output])



        ax[i//c,i%c].imshow(output)

        ax[i//c,i%c].axis('off')

        ax[i//c,i%c].set_title(title,fontsize=15,color=color)



        i += 1
layer_name = 'top_conv'



image_names = list(val_merged[(val_merged.target_x==0)&(val_merged.target_y>=0.5)].image_name)

print("Total Images: ",len(image_names))



if len(image_name)!=0:

    

    r, c = 5, 5



    r, c = min(int(np.ceil(len(image_names)/5)),5), 5

    if r==1:

        r=2



    fig, ax = plt.subplots(r,c,figsize=(c*4*2,r*6))



    # fig.suptitle(f'GridCAMs of layer: {layer_name} of train images',fontsize=15)



    explainer = GradCAM()



    i = 0



    for image_name in np.random.choice(image_names,replace=False,size=min(r*c,len(image_names))):



        prediction = val_merged[val_merged.image_name==image_name]['target_y'].values[0]

        actual = val_merged[val_merged.image_name==image_name]['target_x'].values[0]



        color = 'green' if int(prediction>=0.5)==actual else 'red'



        title = f'Image Name: {image_name}\nActual: {actual}\nPrediction: {prediction}'



        image = getImage(image_name)



        heatmap1 = explainer.explain(([image/255.0], None), model, layer_name = layer_name, class_index=0)



        heatmap2, output = overlay_heatmap(heatmap1,image)



        output = np.hstack([image,output])



        ax[i//c,i%c].imshow(output)

        ax[i//c,i%c].axis('off')

        ax[i//c,i%c].set_title(title,fontsize=15,color=color)



        i += 1
image_names = test_pred_mean.image_name.values



r, c = min(int(np.ceil(len(image_names)/R)),R), C

    

if r==1:

    r=2



fig, ax = plt.subplots(r,c,figsize=(c*4*2,r*6))



layer_name = 'top_conv'



# fig.suptitle(f'GridCAMs of layer: {layer_name} of train images',fontsize=15)



explainer = GradCAM()



i = 0



for image_name in np.random.choice(image_names,replace=False,size=min(r*c,len(image_names))):



    prediction = test_pred_mean[test_pred_mean.image_name==image_name]['target'].values[0]

    rank = test_pred_rank[test_pred_rank.image_name==image_name]['target'].values[0]



    title = f'Image Name: {image_name}\nRank: {rank}\nPrediction: {prediction}'



    image = getImage(image_name,train=False)



    heatmap1 = explainer.explain(([image/255.0], None), model, layer_name = layer_name, class_index=0)



    heatmap2, output = overlay_heatmap(heatmap1,image)



    output = np.hstack([image,output])



    ax[i//c,i%c].imshow(output)

    ax[i//c,i%c].axis('off')

    ax[i//c,i%c].set_title(title,fontsize=15)



    i += 1