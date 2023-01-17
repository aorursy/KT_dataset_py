import os

# There are two ways to load the data from the PANDA dataset:
# Option 1: Load images using openslide
import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
import random
import seaborn as sns
import cv2

# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display
import skimage.io
from tqdm.notebook import tqdm
import random
import zipfile
plt.rcParams['figure.figsize'] = [15,8]

# Plotly for the interactive viewer (see last section)
import plotly.graph_objs as go
import re
# Location of the training images

BASE_PATH = '/kaggle/input/prostate-cancer-grade-assessment/'

# image and mask directories
train_dir = f'{BASE_PATH}/train_images/'
mask_dir = f'{BASE_PATH}/train_label_masks/' #'/kaggle/input/prostate-cancer-grade-assessment/train_images/'

# Location of training labels
train = pd.read_csv(f'{BASE_PATH}train.csv').set_index('image_id')
test = pd.read_csv(f'{BASE_PATH}test.csv')
submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()
    
def plot_relative_distribution(df, feature, hue, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(x=feature, hue=hue, data=df, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()
def display_images(slides): 
    f, ax = plt.subplots(5,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        image = openslide.OpenSlide(os.path.join(train_dir, f'{slide}.tiff'))
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region((0,0), 0, (256, 256))
        ax[i//3, i%3].imshow(patch) 
        image.close()       
        ax[i//3, i%3].axis('off')
        image_id = slide
        data_provider = train.loc[slide, 'data_provider']
        isup_grade = train.loc[slide, 'isup_grade']
        gleason_score = train.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
    plt.show() 
def overlay_mask_on_slide(images, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""
    f, ax = plt.subplots(5,3, figsize=(18,22))
    
    
    for i, image_id in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        mask_data = mask_data.split()[0]
        
        
        # Create alpha mask
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
        mask_rgb = mask_data.convert(mode='RGB')
        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
        overlayed_image.thumbnail(size=max_size, resample=0)

        
        ax[i//3, i%3].imshow(overlayed_image) 
        slide.close()
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        data_provider = train.loc[image_id, 'data_provider']
        isup_grade = train.loc[image_id, 'isup_grade']
        gleason_score = train.loc[image_id, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
def tile(img):
    sz = 128
    bs = 2
    N = 12
    nworkers = 2
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5871, 0.1140])/255.0


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5871, 0.1140])/255.0
    #plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    #plt.show()

def rmpad(img,seed=5):
    t=pd.DataFrame(img[:,:,0]<200)
    t1=t.sum(axis=0)
    t1=t1[t1>seed]
    col1,col2 = t1.index[0], t1.index[-1]
    
    t1=t.sum(axis=1)
    t1=t1[t1>seed]
    row1,row2 = t1.index[0], t1.index[-1]
    return img[row1:row2,col1:col2,:]

def rmpad2(img,seed=10,lb=10,ub=250):
    t=pd.DataFrame((img[:,:,0]<ub) & (img[:,:,0]>lb))
    t1=t.sum(axis=0)
    t1=t1[t1>seed]
    col1,col2 = t1.index[0], t1.index[-1]
    
    t1=t.sum(axis=1)
    t1=t1[t1>seed]
    row1,row2 = t1.index[0], t1.index[-1]
    return img[row1:row2,col1:col2,:]

def display(img_name='',sub=-1,path='/kaggle/input/prostate-cancer-grade-assessment/train_images/'):
    img=skimage.io.MultiImage(path +  img_name + '.tiff')[sub]
    plt.imshow(img) 
def display_sharp(img_name,pad_rem=True,enh=True,unsharp=True,sub=-1,path='/kaggle/input/prostate-cancer-grade-assessment/train_images/'):
    img = skimage.io.MultiImage(path +  img_name + '.tiff')[sub]
    if pad_rem:
        img=rmpad2(img)
    if  enh:
        img=enhance_image(img)
    if unsharp:
        img=unsharp_masking(img)
    plt.imshow(img)  
def sharp(img,pad_rem=True,enh=True,unsharp=True):
    if pad_rem:
        img=rmpad2(img)
    if  enh:
        img=enhance_image(img)
    if unsharp:
        img=unsharp_masking(img)
    return img
def enhance_image(img, contrast=1, brightness=15):
    """
    Enhance constrast and brightness of images
    """
    img = cv2.addWeighted(img, contrast, img, 0, brightness)
    return img
def unsharp_masking(img):
    """ Unsharp masking of an RGB image"""
    img_gaussian = cv2.GaussianBlur(img, (21,21), 10.0)
    return cv2.addWeighted(img, 1.8, img_gaussian, -0.75, -0.05, img)

find=lambda x:list(np.where(x)[0][:]) if isinstance(x,np.ndarray) else (
            list(np.where(x)[0][:]) if isinstance(x,list) else (
            list(np.where(np.array(x)[0][:])) if not isinstance(x,bool) else np.where(x)[0]))

#######################################Function###################################################
def flatten(y):
    t=lambda x: [x] if type(x) is not list else x
    z=lambda x: sum([t(i) for i in x],[])
    for i,j in enumerate(y):
        if isinstance(j,np.ndarray):
            y[i]=list(j.flatten())
        elif isinstance(j,tuple):
            y[i]=list(j)
    h=t(z(y));
    if any([isinstance(i,list) | isinstance(i,tuple)  for i in h ]):
        return flatten(h)
    else:
        return h;
    
def randi(x,p=1,full_rep_prop=.85):
    
    from random import shuffle,sample
    import numpy as np
    
    if isinstance(x,int) or isinstance(x,float):
        x=[i for i in range(round(x))]
    
    if isinstance(x,np.ndarray):
        x=list[x]

    if not isinstance(p,float):
        if len(x)==p:
            shuffle(x)
            return x
        if len(x)>p:
            return sample(x,p)
        else: # over sampling
            len_x=len(x)
            list0=[]
            while p>len_x:
                p=p-len_x
                shuffle(x)
                list0.extend(x)
            list0.extend(sample(x,p))
            return list0
   
    # if it is a proportion
    if isinstance(p,float):
        len_x=len(x)
        list0=[]
        while p>1:
            p=p-1
            if full_rep_prop==1:
                shuffle(x)
                list0.extend(x)
            else:
                list0.extend(sample(x,int(round(full_rep_prop*len_x))))  
        list0.extend(sample(x,int(round(p*len_x))))
        return list0
    
def splitstrat(X,Y,alpha=.65,verbose=True,data_proportion=1.,random_seed=7): 
    import random 
    data_proportion=float(data_proportion) 

    # select proportion of data 
    num_samples=X.shape[0] 

    u=np.unique(Y) 
    num_class=len(u) 
    idx=[list(np.array(np.where(Y==u[i])).flatten()) for i in range(len(u))] 
    idx=[random.Random(random_seed).sample(idx[i],int(round(data_proportion*len(idx[i])))) 
         for i in range(num_class)]  
    factors=[len(idx[i]) for i in range(num_class)] 


    if isinstance (alpha,float): 
        alpha=[alpha]*num_class 
    if isinstance(alpha,list): 
        if len(alpha)!= num_class: 
            alpha=[alpha[0]]*num_class


    f=[int(round(factors[i]*alpha[i])) for i in range(num_class) ]

    # create indexes 
    idx_train=[idx[i][0:f[i]] for i in range(num_class)]
    idx_val=[idx[i][f[i]:] for i in range(num_class) ]

    # flatten list of arrays
    idx_train,idx_val=list(),list()
    [idx_train.extend(idx[i][0:f[i]]) for i in range(num_class)]
    [idx_val.extend(idx[i][f[i]:]) for i in range(num_class)]

    # create datasets 
    x_train=X.iloc[idx_train]
    x_val=X.iloc[idx_val]
    y_train=Y[idx_train] 
    y_val=Y[idx_val] 
    
    #update date frame with new field
    #X["split"].iloc[idx_train]="train"
    #X["split"].iloc[idx_val]="val" 
    
    # to avoid panda warning we use loc instead of iloc
    X["split"]="not_used"
    X.loc[X.index[idx_train],"split"]="train"
    X.loc[X.index[idx_val],"split"]="val"
    
    #X["split"].iloc[idx_val].apply(lambda x:[x+"val"])

    if verbose==1: 
            print('class names: %s' % [i for i in u])  
            print('each class#: %s' % [i for i in factors])  
            print('num of samples of each class in training samples...')
            print('-->>>>>>>>: %s' % [i for i in f])    
            print('<<<<<<<<-----x_val, x_train is %s  ------------>>>>>>' % [len(idx_val),len(idx_train)]) 
            print(f'all data is  {X.shape}  and all lables is {Y.shape}') 
            #print('alpha and data proportion is %s'% [alpha,data_proportion] ) 
            print(f'{data_proportion*100}% of data were used to create stratified samples,')
            print(f'{alpha[0]*100}% of employed samples is for training and the rest if for validation')
    return (x_train,y_train,x_val,y_val) 

# create new binary lable
train.isnull().sum()
train["blabel"]=train.isup_grade>3

# create ordered lables from 0:8 corresponding to 9 levels of severity from Normal to End-level
levels=list(np.unique(train.gleason_score))
levels.insert(0,levels.pop(-1))# move the last item in the list ("negative") to the first position
train["glabel"]=train.apply(lambda x: levels.index(x.gleason_score),axis=1)

train.head()
pen_marked_images = [
    'fd6fe1a3985b17d067f2cb4d5bc1e6e1',
    'ebb6a080d72e09f6481721ef9f88c472',
    'ebb6d5ca45942536f78beb451ee43cc4',
    'ea9d52d65500acc9b9d89eb6b82cdcdf',
    'e726a8eac36c3d91c3c4f9edba8ba713',
    'e90abe191f61b6fed6d6781c8305fe4b',
    'fd0bb45eba479a7f7d953f41d574bf9f',
    'ff10f937c3d52eff6ad4dd733f2bc3ac',
    'feee2e895355a921f2b75b54debad328',
    'feac91652a1c5accff08217d19116f1c',
    'fb01a0a69517bb47d7f4699b6217f69d',
    'f00ec753b5618cfb30519db0947fe724',
    'e9a4f528b33479412ee019e155e1a197',
    'f062f6c1128e0e9d51a76747d9018849',
    'f39bf22d9a2f313425ee201932bac91a',
]
idx_to_include=[i for i in train.index if i not in pen_marked_images]
mask_image_names=[re.sub('_mask.*','',name) for name in os.listdir(mask_dir)]
idx_with_masks=[i for i in mask_image_names if i in idx_to_include]
print("number of masks:",len(mask_image_names))
print("number of image with masks and no pen marks:",len(idx_with_masks))
print("number of images in training data:",len(train.index))

random_seed=7
random.Random(random_seed).shuffle(idx_with_masks)
idx=idx_with_masks[:100]# select only 50 images for now

train2=train.loc[idx] 
train2.head(5)

train2_data=list()
for i in range(len(idx)):
    train2_data.append(sharp(skimage.io.MultiImage(train_dir + train2.index[i] + '.tiff')[-1]))
    #print(f'image:{i}',i+1)
    
print(f'Reading {len(train2_data)} images finished')
plot_relative_distribution(df=train2, feature='data_provider', hue='blabel', 
                           title = 'relative count plot of isup_grade with data_provider', size=3)
plot_relative_distribution(df=train, feature='data_provider', hue='blabel', 
                           title = 'relative count plot of isup_grade with data_provider', size=3)
plot_relative_distribution(df=train2, feature='glabel', hue='data_provider', 
                           title = 'relative count plot of isup_grade with data_provider', size=3)
plot_relative_distribution(df=train, feature='glabel', hue='data_provider', 
                           title = 'relative count plot of isup_grade with data_provider', size=3)

(x_train,y_train,x_val,y_val) = splitstrat(train2,train2.glabel,.65,random_seed= random_seed)
plot_relative_distribution(df=train2, feature='split', hue='blabel', 
                           title = 'relative count plot of binary label over data splits', size=3)
plot_relative_distribution(df=train2, feature='split', hue='glabel', 
                           title = 'relative count plot of all labels over data splits', size=3)
plot_relative_distribution(df=train2, feature='split', hue='data_provider', 
                           title = 'relative count plot of splits over data provider', size=3)

plot_count(df=train, feature='glabel', 
                           title = ' count plot of all labels over original data', size=3)
X=train2.tail(5)
X.iloc[[1,2,3],[3,4]]=None
X.iloc[[4],[3,5]]=None

t=X.copy()


def repall(X,val=None,rep=None,replace=False):
    #replace different values in provided columns
    #repall(X,rep={'split':{'val':'validation',None:'not_used'},'glabel':{0:1,5:25,None:-10}},replace=True)
    if val==None:
        x=X.isnull().sum(axis=0).reset_index()
    else:
        x=X.applymap(lambda x: [True if type(val)==type(x) and val==x else False][0]).sum(axis=0).reset_index()
    x.rename(columns={"index":"col_name",0:"hit_num"},inplace=True)
    x["hit_rate"]=x.miss_num/X.shape[0]
    
    if  sum(x.hit_num)>0 and replace:
        if type(rep)!=dict : # all data
            if rep is not None:
                X.replace(rep)
            else:
                X.fillna(rep) 
        else: # some columns
            cols=list(rep.keys())
            rep_val=list(rep.values())
            _=[X[cols[i]].replace(rep_val[i],inplace=True) for i,j in enumerate(cols)]
            #if val==None:
            #X.fillna(rep,inplace=True) 
        #else: #type(rep)!=dict:
            #else:
            #X=X.applymap(lambda x: [rep if type(val)==type(x) and val==x else x][0])
            

    print(f'number of search results: {sum(x.hit_num)} ')
    print(f'number of columns having search value: {sum(x.hit_num>0)}')
    print(f'columns are: {x.col_name[x.hit_num>0].values}')
    return x
def rmnull(X,rep=None):
    #example:t=rmnull(t,{'blabel':0,'split':'not_used'})
    x=X.isnull().sum(axis=0).reset_index()
    x.rename(columns={"index":"col_name",0:"miss_num"},inplace=True)
    x["miss_rate"]=x.miss_num/X.shape[0]
    
    if  sum(x.miss_num)>0 and (rep is not None):
        #X=X.applymap(lambda x: [rep if x is None else x][0])
        X.fillna(rep,inplace=True)
    
    print(f'number of null values: {sum(x.miss_num)} ')
    print(f'number of columns having nulls: {sum(x.miss_num>0)}')
    print(f'columns are: {x.col_name[x.miss_num>0].values}')
    return (x,X)
rep={'split':{'val':'validation',None:'not_used'},'glabel':{0:1,5:25,None:-10}}
rep_key=list(rep.keys())
rep_val=list(rep.values())
rep_val[0]
y=X.copy()
[y[rep_key[i]].replace(rep_val[i],inplace=True) for i,j in enumerate(rep_val)]
X
y

y.head()

t=X.copy()
tt=repall(t,{'blabel':0,'split':'not_used'})
t=X.copy()
tt=rmnull(t,{'blabel':0,'split':'not_used'})
X.isna()
tt[1].head(5)
X.iloc[0,1]=24
X.iloc[1,1]=27
X.iloc[2,1]=-37
X.iloc[3,1]=None
X.iloc[4,1]=None
t=X
t
a=[X[X.columns[i]].dtype in [type(1),type(.1)] for i in range(len(X.columns))]
a
t=X
X[X.columns[a]]
t[t.columns[a]].apply(lambda x:[x.mode() if x>20 else x],axis=0)

#X.apply(lambda x:x.fillna([-11 if type(x) in [type(1),type(.1)]  else "fuck"][0]),axis=0)
#x=X[X.columns[ i in X.columns if  X.i.dtype=type(1)]].apply(lambda x:[x.mode() if x is None else x])
#x
#X.iloc[:,a]
t
X.loc[:,a]
X.isup_grade.median()
X.split.mode()

X


X.isup_grade
rep=-1
val=None
x=X.applymap(lambda x: [True if type(val)==type(x) and val==x else False][0]).sum().reset_index()
x.rename(columns={"index":"col_name",0:"miss_num"},inplace=True)
x["miss_rate"]=x.miss_num/X.shape[0]
#sum(x.miss_num)
x
img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_karolinska.image_id[800] + '.tiff')[-1]
img[:4,1,:]
img=np.array(img)
img[:4,1,:]

#Let's check for relative distribution of isup_grade and gleason_score¶
plot_relative_distribution(df=train, feature='isup_grade', hue='gleason_score', 
                           title = 'relative count plot of isup_grade with gleason_score', size=2)
plot_relative_distribution(df=train, feature='isup_grade', hue='data_provider', 
                           title = 'relative count plot of isup_grade with data_provider', size=3)
plot_count(df=train[train.data_provider=="karolinska"], feature='gleason_score', title = 'gleason_score count for karolinska')
plot_count(df=train[train.data_provider=="radboud"], feature='gleason_score', title = 'gleason_score count for radboud')
plot_count(df=train[train.data_provider=="radboud"], feature='isup_grade', title = 'isup_grade count for radboud')
plot_count(df=train[train.data_provider=="karolinska"], feature='isup_grade', title = 'isup_grade count for karolinska')
plot_count(df=train, feature='isup_grade', title = 'isup_grade count and %age plot')
plot_count(df=train, feature='data_provider', title = 'data_provider count and %age plot')

#plot over new created labels
plot_count(df=train[train.data_provider!="karolinska"], feature='label', title = 'label count for Radbound')
plot_count(df=train[train.data_provider=="karolinska"], feature='label', title = 'label count for karolinska')
plot_count(df=train, feature='label', title = 'label count')
plot_count(df=train, feature='glabel', title = 'glabel count ')


display(train.head())
print("Shape of training data :", train.shape)
print("unique data provider :", len(train.data_provider.unique()))
print("unique isup_grade(target) :", len(train.isup_grade.unique()))
print("unique gleason_score :", len(train.gleason_score.unique()))

train.isna().sum()



