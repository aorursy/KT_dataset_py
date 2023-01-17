import pandas as pd

import numpy as np

import os

from glob import glob

import random

import matplotlib.pylab as plt

import cv2
PATH = os.path.abspath(os.path.join('..', 'input'))



SOURCE_IMAGES001 = os.path.join(PATH, "images_001", "images")

SOURCE_IMAGES002 = os.path.join(PATH, "images_002", "images")

SOURCE_IMAGES003 = os.path.join(PATH, "images_003", "images")

SOURCE_IMAGES004 = os.path.join(PATH, "images_004", "images")

SOURCE_IMAGES005 = os.path.join(PATH, "images_005", "images")

SOURCE_IMAGES006 = os.path.join(PATH, "images_006", "images")

SOURCE_IMAGES007 = os.path.join(PATH, "images_007", "images")

SOURCE_IMAGES008 = os.path.join(PATH, "images_008", "images")

SOURCE_IMAGES009 = os.path.join(PATH, "images_009", "images")

SOURCE_IMAGES010 = os.path.join(PATH, "images_010", "images")

SOURCE_IMAGES011 = os.path.join(PATH, "images_011", "images")

SOURCE_IMAGES012 = os.path.join(PATH, "images_012", "images")



images001 = glob(os.path.join(SOURCE_IMAGES002, "*.png"))

images002 = glob(os.path.join(SOURCE_IMAGES002, "*.png"))

images003 = glob(os.path.join(SOURCE_IMAGES003, "*.png"))

images004 = glob(os.path.join(SOURCE_IMAGES004, "*.png"))

images005 = glob(os.path.join(SOURCE_IMAGES005, "*.png"))

images006 = glob(os.path.join(SOURCE_IMAGES006, "*.png"))

images007 = glob(os.path.join(SOURCE_IMAGES002, "*.png"))

images008 = glob(os.path.join(SOURCE_IMAGES002, "*.png"))

images009 = glob(os.path.join(SOURCE_IMAGES002, "*.png"))

images010 = glob(os.path.join(SOURCE_IMAGES002, "*.png"))

images011 = glob(os.path.join(SOURCE_IMAGES002, "*.png"))

images012 = glob(os.path.join(SOURCE_IMAGES002, "*.png"))



print(images001[0:10],"\n")

print(images002[0:10],"\n")

print(images003[0:10],"\n")

print(images004[0:10],"\n")

print(images005[0:10],"\n")

print(images006[0:10],"\n")

print(images007[0:10],"\n")

print(images008[0:10],"\n")

print(images009[0:10],"\n")

print(images010[0:10],"\n")

print(images011[0:10],"\n")

print(images012[0:10],"\n")

labels = pd.read_csv('../input/Data_Entry_2017.csv')

labels.head(10)
%matplotlib inline

image_name = "/kaggle/input/images_003/images/00006329_004.png" #Image to be used as query

def plotImage(image_location):

    image = cv2.imread(image_name)

    image = cv2.resize(image, (512,512))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis('off')

    return

plotImage(image_name)
# Plot Multiple Images

xrays = glob('/kaggle/input/images_002/images/**')

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in xrays[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (64, 64)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
r = random.sample(images002, 3)

plt.figure(figsize=(16,16))

plt.subplot(131)

plt.imshow(cv2.imread(r[0]))

plt.subplot(132)

plt.imshow(cv2.imread(r[1]))

plt.subplot(133)

plt.imshow(cv2.imread(r[2])); 
import matplotlib.gridspec as gridspec

import seaborn as sns





#drop unused columns

labels = labels[['Image Index','Finding Labels','Follow-up #','Patient ID','Patient Age','Patient Gender']]

#create new columns for each decease

pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']

for pathology in pathology_list :

    labels[pathology] = labels['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

#remove Y after age

labels['Age']=labels['Patient Age'].apply(lambda x: x[:-1]).astype(int)



plt.figure(figsize=(15,10))

gs = gridspec.GridSpec(8,1)

ax1 = plt.subplot(gs[:7, :])

ax2 = plt.subplot(gs[7, :])

data1 = pd.melt(labels,

             id_vars=['Patient Gender'],

             value_vars = list(pathology_list),

             var_name = 'Category',

             value_name = 'Count')

data1 = data1.loc[data1.Count>0]

g=sns.countplot(y='Category',hue='Patient Gender',data=data1, ax=ax1, order = data1['Category'].value_counts().index)

ax1.set( ylabel="",xlabel="")

ax1.legend(fontsize=20)

ax1.set_title('X Ray partition (total number = 121120)',fontsize=18);



labels['Nothing']=labels['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)



data2 = pd.melt(labels,

             id_vars=['Patient Gender'],

             value_vars = list(['Nothing']),

             var_name = 'Category',

             value_name = 'Count')

data2 = data2.loc[data2.Count>0]

g=sns.countplot(y='Category',hue='Patient Gender',data=data2,ax=ax2)

ax2.set( ylabel="",xlabel="Number of decease")

ax2.legend('')

plt.subplots_adjust(hspace=.5)
f, axarr = plt.subplots(7, 2, sharex=True,figsize=(15, 20))



i=0

j=0

x=np.arange(0,100,10)

for pathology in pathology_list :

    g=sns.boxplot(x='Age', hue="Patient Gender",data=labels[labels['Finding Labels']==pathology], ax=axarr[i, j])

    axarr[i, j].set_title(pathology)   

    g.set_xlim(0,90)

    g.set_xticks(x)

    g.set_xticklabels(x)

    j=(j+1)%2

    if j==0:

        i=(i+1)%7

f.subplots_adjust(hspace=0.3)
f, axarr = plt.subplots(7, 2, sharex=True,figsize=(15, 20))



i=0

j=0

x=np.arange(0,100,10)

for pathology in pathology_list :

    g=sns.countplot(x='Age', hue="Patient Gender",data=labels[labels['Finding Labels']==pathology], ax=axarr[i, j])

    axarr[i, j].set_title(pathology)   

    g.set_xlim(0,90)

    g.set_xticks(x)

    g.set_xticklabels(x)

    j=(j+1)%2

    if j==0:

        i=(i+1)%7

f.subplots_adjust(hspace=0.3)
def proc_images(folder):

    """

    Returns two arrays: 

        x is an array of resized images

        y is an array of labels

    """

    NoFinding = "No Finding"

    Consolidation="Consolidation"

    Infiltration="Infiltration"

    Pneumothorax="Pneumothorax"

    Edema="Edema"

    Emphysema="Emphysema"

    Fibrosis="Fibrosis"

    Effusion="Effusion"

    Pneumonia="Pneumonia"

    Pleural_Thickening="Pleural_Thickening"

    Cardiomegaly="Cardiomegaly"

    NoduleMass="Nodule"

    Hernia="Hernia"

    Atelectasis="Atelectasis"

    RareClass=["Emphysema","Edema","Fibrosis","Pneumonia","Hernia"]

    x = [] # images as arrays

    y = [] # labels Infiltration or Not_infiltration

    WIDTH = 64

    HEIGHT = 64



    for img in folder:

        base = os.path.basename(img)

        finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]



        # Read and resize image

        full_size_image = cv2.imread(img)

        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))



        # Labels

        if NoFinding in finding:

            finding = 0

            y.append(finding)        

        elif Consolidation in finding:

            finding = 1

            y.append(finding)    

        elif Infiltration in finding:

            finding = 2

            y.append(finding)           

        elif Pneumothorax in finding:

            finding = 3

            y.append(finding)

        elif Edema in finding:

            finding = 9##

            y.append(finding)

        elif Emphysema in finding:

            finding = 9##

            y.append(finding)

        elif Fibrosis in finding:

            finding = 9##

            y.append(finding) 

        elif Effusion in finding:

            finding = 4

            y.append(finding)             

        elif Pneumonia in finding:

            finding = 9##

            y.append(finding)   

        elif Pleural_Thickening in finding:

            finding = 5

            y.append(finding) 

        elif Cardiomegaly in finding:

            finding = 6

            y.append(finding) 

        elif NoduleMass in finding:

            finding = 7

            y.append(finding) 

        elif Hernia in finding:

            finding = 9##

            y.append(finding) 

        elif Atelectasis in finding:

            finding = 8

            y.append(finding) 

        else:

            finding = 9

            y.append(finding)



    return x,y

# use for loop and dictionary
X001,y001 = proc_images(images001)

df001 = pd.DataFrame()

df001["images"]=X001

df001["labels"]=y001

print(len(df001), df001.images[0].shape)
X002,y002 = proc_images(images002)

df002 = pd.DataFrame()

df002["images"]=X002

df002["labels"]=y002

print(len(df002), df002.images[0].shape)
X003,y003 = proc_images(images003)

df003 = pd.DataFrame()

df003["images"]=X003

df003["labels"]=y003

print(len(df003), df003.images[0].shape)
X004,y004 = proc_images(images004)

df004 = pd.DataFrame()

df004["images"]=X004

df004["labels"]=y004

print(len(df004), df004.images[0].shape)
# X005,y005 = proc_images(images005)

# df005 = pd.DataFrame()

# df005["images"]=X005

# df005["labels"]=y005

# print(len(df005), df005.images[0].shape)
# X006,y006 = proc_images(images006)

# df006 = pd.DataFrame()

# df006["images"]=X006

# df006["labels"]=y006

# print(len(df006), df006.images[0].shape)
# X007,y007 = proc_images(images007)

# df007 = pd.DataFrame()

# df007["images"]=X007

# df007["labels"]=y007

# print(len(df007), df007.images[0].shape)
# X008,y008 = proc_images(images008)

# df008 = pd.DataFrame()

# df008["images"]=X008

# df008["labels"]=y008

# print(len(df008), df008.images[0].shape)
# X009,y009 = proc_images(images009)

# df009 = pd.DataFrame()

# df009["images"]=X009

# df009["labels"]=y009

# print(len(df009), df009.images[0].shape)
# X010,y010 = proc_images(images010)

# df010 = pd.DataFrame()

# df010["images"]=X010

# df010["labels"]=y010

# print(len(df010), df010.images[0].shape)
# X011,y011 = proc_images(images011)

# df011 = pd.DataFrame()

# df011["images"]=X011

# df011["labels"]=y011

# print(len(df011), df011.images[0].shape)
# X012,y012 = proc_images(images012)

# df012 = pd.DataFrame()

# df012["images"]=X012

# df012["labels"]=y012

# print(len(df012), df012.images[0].shape)
dfCombined = pd.DataFrame()

dfCombined["images"]=X001+X002+X003+X004

dfCombined["labels"]=y001+y002+y003+y004

print(len(dfCombined), dfCombined.images[0].shape)



# dfCombined = pd.DataFrame()

# dfCombined["images"]=X001+X002+X003+X004+X005+X006+X007+X008+X009+X010+X011+X012

# dfCombined["labels"]=y001+y002+y003+y004+y005+y006+y007+y008+y009+y010+y011+y012

# print(len(dfCombined), dfCombined.images[0].shape)
X = X001+X002+X003+X004

y = y001+y002+y003+y004



# X = X001+X002+X003+X004+X005+X006+X007+X008+X009+X010+X011+X012

# y = y001+y002+y003+y004+y005+y006+y007+y008+y009+y010+y011+y012
# dict_characters = {0: 'No Finding', 1: 'Consolidation', 2: 'Infiltration', 

#         3: 'Pneumothorax', 4: 'Edema', 5: 'Emphysema', 6: 'Fibrosis', 7:'Effusion',

#         8: 'Pneumonia', 9: 'Pleural_Thickening',10:'Cardiomegaly', 11: 'Nodule Mass', 

#         12: 'Hernia', 13: 'Atelectasis'}



dict_characters = {0: 'No Finding', 1: 'Consolidation', 2: 'Infiltration', 3: 'Pneumothorax', 4:'Effusion', 5: 'Pleural_Thickening',6:'Cardiomegaly', 7: 'Nodule Mass', 8: 'Atelectasis', 9: 'Other Rare Pathology'}



print(dfCombined.head(10))

print("")

print(dict_characters)
# We look at RGB histograms (represent colors). Histogram counts the number of pixels with a certain intensity

# between 0 and 255 for each color red, green and blue. A peak at 255 for all colors mean a lot of white ! 



i= 1 # Try 0, 1, 2.. for negative images and -1, -2, -3 for positive images and compare the histograms.

xi = X[i]





def plotArray(array):

    image = cv2.imread(image_name)

    image = cv2.resize(image, (512,512))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis('off')

    return

plotArray(xi)





# plt.imshow(xi)

# plt.axis('off')

plt.title('Representative Image')

plt.figure(figsize=(20,5))

n_bins = 50

plt.hist(xi[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);

plt.figure(figsize=(20,5))

plt.hist(xi[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);

plt.figure(figsize=(20,5))

plt.hist(xi[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);
X=np.array(X)

X=X/255.0
i= 1 # Try 0, 1, 2.. for negative images and -1, -2, -3 for positive images and compare the histograms.

xi = X[i]





def plotArray(array):

    image = cv2.imread(image_name)

    image = cv2.resize(image, (512,512))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis('off')

    return

plotArray(xi)





# plt.imshow(xi)

# plt.axis('off')

plt.title('Representative Image')

plt.figure(figsize=(20,5))

n_bins = 50

plt.hist(xi[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);

plt.figure(figsize=(20,5))

plt.hist(xi[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);

plt.figure(figsize=(20,5))

plt.hist(xi[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);
lab = dfCombined['labels']

dist = lab.value_counts()

sns.countplot(lab)

print(dict_characters)

#print(dist)
import zlib

A=X

dtype=A.dtype

B=zlib.compress(A,7) # number 0 to 9 for level of compression w/ being 9 highest and slowest

y=np.array(y)

A2=y

dtype=A2.dtype

B2=zlib.compress(A2,7) # .gzip unfortunately instead of .zip

np.savez("X_images_np_zip", B)

np.savez("Y_labels_np_zip", B2)

!ls -1