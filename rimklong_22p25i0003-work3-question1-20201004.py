# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        a=0
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#  OK 1
#filenames = os.listdir("../input/train/train")
#categories = []
#for filename in filenames:
#    category = filename.split('.')[0]
#    if category == 'dog':
#        categories.append(1)
#    else:
#        categories.append(0)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os

import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


FAST_RUN = False
IMAGE_WIDTH=32
IMAGE_HEIGHT=32
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

output_train_dir = 'output_dir_for_train'
input_dir_for_train = '/kaggle/input/super-ai-image-classification/train/train/images'


output_val_dir = 'output_dir_for_val'
input_dir_for_val = '/kaggle/input/super-ai-image-classification/val/val/images'

if not os.path.lexists(output_train_dir):
    os.mkdir(output_train_dir)
else:
    #clear old files
    for dirname, _, filenames in os.walk(output_train_dir):
        for filename in filenames:             
            #print(os.path.join(dirname, filename))
            if os.path.exists(os.path.join(dirname, filename)):
                os.remove(os.path.join(dirname, filename))

    

file_for_train = pd.read_csv('/kaggle/input/super-ai-image-classification/train/train/train.csv')
file_for_train.head()
file_for_train.info()
filenames=[]
categories=[]
vCount_Old_Files_For_Train = 0
for ind in file_for_train.index: 
    #print(file_for_train['id'][ind], file_for_train['category'][ind]) 
    train_name = file_for_train['id'][ind]   
    n_sp = train_name.split('.')
    n_sp_name = n_sp[0]
    n_sp_surname = n_sp[1]    
    vcategory = file_for_train['category'][ind]
    if vcategory ==1:        
        #new_name = 'nobath'+ file_for_train['id'][ind]
        new_id = n_sp_name+str(ind)+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id 
        filenames.append(new_id)
        categories.append(vcategory)
    else:
        #new_name = 'bath'+'.'+file_for_train['id'][ind] 
        new_id = n_sp_name+str(ind)+str(vcategory)+'.'+n_sp_surname
        filenames.append(new_id)
        categories.append(vcategory)      
        
    img = load_img(input_dir_for_train+'/'+train_name)
    img_array = img_to_array(img)
    if not os.path.lexists(output_train_dir):
        os.mkdir(output_train_dir)   
    #save old input train to new directory
    img.save(output_train_dir+'/'+new_id)
    vCount_Old_Files_For_Train += 1
    
df = pd.DataFrame({
    'id': filenames,
    'category': categories
})

print('old files for train :',vCount_Old_Files_For_Train)


#  ****MAKE BALANCE DATA TO NOT OVER FIT****
# OK 2
import cv2
from PIL import Image

vcountbath = 0
vcountnobath=0
i=0
for i in range(len(df)):
    vid = df.id[i]
    vcategory = df.category[i]    
    if vcategory == 0:
        vcountbath += 1
    else:
        vcountnobath += 1        
        #vsample_image_nobath= '/kaggle/input/super-ai-image-classification/train/train/images/'+vid

print('count bath=',vcountbath)
print('count no bath=',vcountnobath)
print('total picture=',vcountbath+vcountnobath)
vdiff_bath_nobath = abs(vcountbath-vcountnobath)
print('difference=',abs(vcountbath-vcountnobath))

listid_nobath=[]
listcategory_nobath=[]
 
i=0
if  vcountnobath < vcountbath :
    vcountnobath_add = 0
    
    for i in range(len(df)):    
        vid = df.id[i]
        vcategory = df.category[i]   
        n_sp = vid.split('.')
        n_sp_name = n_sp[0]
        n_sp_surname = n_sp[1]
        
        #print(i,vcategory)
        if vcategory == 1:    
           
                         
            if vcountnobath_add <= vdiff_bath_nobath:
                
                #nobath               
                vsample_image_nobath=  output_train_dir+'/'+vid
                print(vsample_image_nobath)
                img_nobath = cv2.imread(vsample_image_nobath)
                img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
                img_nobath = img_nobath.reshape(1,img_nobath.shape[0],img_nobath.shape[1],img_nobath.shape[2])
                #vertical shift
                datagen = ImageDataGenerator(height_shift_range=0.2)
                aug_iter = datagen.flow(img_nobath, batch_size=1)
                #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,15))
                #img_nobath = cv2.imread(vsample_image_nobath)
                #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
                k=0
                for j in range(3):
                    if j>0:
                        image = next(aug_iter)[0].astype('uint8')                         
                        #ax[j].imshow(image)
                        #ax[j].axis('off')                      
                        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
                        n_sp_name_surname = new_id                                   
                        #print(new_id)                         
                        listid_nobath.append(new_id)
                        listcategory_nobath.append(1)
                                                              
                        if not os.path.lexists(output_train_dir):
                            os.mkdir(output_train_dir)   
                            #save new image Augmentation train to new directory
                        arr_2_image = Image.fromarray(image, 'RGB') 
                        arr_2_image.save(output_train_dir+'/'+new_id)
                        print(vcountnobath_add,' save=',output_train_dir+'/'+new_id)
                        vcountnobath_add   += 1
            else:
                #add image from countbath
                a='nothing do'
            
print('count no bath add=',vcountnobath_add)
print('old+add=',vcountnobath+vcountnobath_add)

print('out',len(listid_nobath),len(listcategory_nobath))
if len(listid_nobath) > 0 and len(listcategory_nobath) > 0:
    print('ok')
    df2 = pd.DataFrame({
    'id': listid_nobath,
    'category': listcategory_nobath
    })
    df3 = df2.append(df, ignore_index=True)
    #to_append = [listid_nobath, listcatgory_nobath]
    #a_series = pd.Series(to_append, index = df.columns)
    #df = df.append(a_series, ignore_index=True)
     
    print(df3.head())
    print(df3.tail())
    
            
            
 
#check new data train file
vcount_file = 0
for dirname, _, filenames in os.walk(output_train_dir):
    for filename in filenames:             
        print(vcount_file,os.path.join(dirname, filename))
        vcount_file = vcount_file + 1




print(len(df3))
print(df3.head())
print(df3.tail())
print(df3.info())
df3['category'].value_counts().plot.bar()
filenames=[]
categories=[]
i=0
for ind in df3.index:     
    print(i,df3['id'][ind],df3['category'][ind])
    filenames.append(df3['id'][ind])
    categories.append(df3['category'][ind])
    i=i+1
 
          
#OK 3
i=0

vcount_add = 0
listid_=[]
listcategory_=[]

for i in range(len(df3)):    
    vid = df3.id[i]
    vcategory = df3.category[i]   
    n_sp = vid.split('.')
    n_sp_name = n_sp[0]
    n_sp_surname = n_sp[1]

                 
    vsample_image_=  output_train_dir+'/'+vid
    print(i,vsample_image_)
    img_ = cv2.imread(vsample_image_)
    img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR) 
    img_ = img_.reshape(1,img_.shape[0],img_.shape[1],img_.shape[2])
    
    #vertical shift  0.3
    datagen = ImageDataGenerator(height_shift_range=0.2)
    aug_iter = datagen.flow(img_, batch_size=1)
    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,15))    
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #vertical shift 0.4
    datagen = ImageDataGenerator(height_shift_range=0.4)
    aug_iter = datagen.flow(img_, batch_size=1)
    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,15))    
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i+1)+str(j+1)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #horizon shift
    datagen = ImageDataGenerator(width_shift_range=0.2)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #horizon shift 0.4
    datagen = ImageDataGenerator(width_shift_range=0.4)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i+1)+str(j+1)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #random shear  shift 20
    datagen = ImageDataGenerator(shear_range=20)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #random shear  shift 30
    datagen = ImageDataGenerator(shear_range=30)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i+1)+str(j+1)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #zoom_range 0.2
    datagen = ImageDataGenerator(zoom_range=0.2)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #zoom_range 0.4
    datagen = ImageDataGenerator(zoom_range=0.4)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
        
    #vertical_flip=True
    datagen = ImageDataGenerator(vertical_flip=True)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #horizontal_flip=True
    datagen = ImageDataGenerator(horizontal_flip=True)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #rotation_range
    datagen = ImageDataGenerator(rotation_range=10)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #rotation_range
    datagen = ImageDataGenerator(rotation_range=15)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i+1)+str(j+1)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
        
        
print('old=',len(df3))
print('count add=',vcount_add)
print('old+add=',len(df3)+vcount_add)

print('out',len(listid_),len(listcategory_))
if len(listid_) > 0 and len(listcategory_) > 0:
    print('ok')
    df4 = pd.DataFrame({
    'id': listid_,
    'category': listcategory_
    })
    df5 = df4.append(df3, ignore_index=True)
    #to_append = [listid_nobath, listcatgory_nobath]
    #a_series = pd.Series(to_append, index = df.columns)
    #df = df.append(a_series, ignore_index=True)
     
    print(df5.head())
    print(df5.tail())
    
#check new dataframe to train

print(len(df5))
print(df5.head())
print(df5.tail())
print(df5.info())
df5['category'].value_counts().plot.bar()
filenames=[]
categories=[]
i=0
for ind in df5.index:     
    print(i,df5['id'][ind],df5['category'][ind])
    filenames.append(df5['id'][ind])
    categories.append(df5['category'][ind])
    i=i+1
#check new data train file
vcount_file = 0
for dirname, _, filenames in os.walk(output_train_dir):
    for filename in filenames:             
        print(vcount_file,os.path.join(dirname, filename))
        vcount_file = vcount_file + 1
#OK 4
i=0

vcount_add = 0
listid_=[]
listcategory_=[]

for i in range(len(df5)):    
    vid = df5.id[i]
    vcategory = df5.category[i]   
    n_sp = vid.split('.')
    n_sp_name = n_sp[0]
    n_sp_surname = n_sp[1]

                 
    vsample_image_=  output_train_dir+'/'+vid
    print(i,vsample_image_)
    img_ = cv2.imread(vsample_image_)
    img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR) 
    img_ = img_.reshape(1,img_.shape[0],img_.shape[1],img_.shape[2])
    
    #vertical shift  0.3
    datagen = ImageDataGenerator(height_shift_range=0.2)
    aug_iter = datagen.flow(img_, batch_size=1)
    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,15))    
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #vertical shift 0.4
    datagen = ImageDataGenerator(height_shift_range=0.4)
    aug_iter = datagen.flow(img_, batch_size=1)
    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,15))    
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i+1)+str(j+1)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #horizon shift
    datagen = ImageDataGenerator(width_shift_range=0.2)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #horizon shift 0.4
    datagen = ImageDataGenerator(width_shift_range=0.4)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i+1)+str(j+1)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #random shear  shift 20
    datagen = ImageDataGenerator(shear_range=20)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #random shear  shift 30
    datagen = ImageDataGenerator(shear_range=30)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i+1)+str(j+1)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #zoom_range 0.2
    datagen = ImageDataGenerator(zoom_range=0.2)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
    
    #zoom_range 0.4
    datagen = ImageDataGenerator(zoom_range=0.4)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
        
    #vertical_flip=True
    datagen = ImageDataGenerator(vertical_flip=True)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #horizontal_flip=True
    datagen = ImageDataGenerator(horizontal_flip=True)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #rotation_range
    datagen = ImageDataGenerator(rotation_range=10)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i)+str(j)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
    #rotation_range
    datagen = ImageDataGenerator(rotation_range=15)
    aug_iter = datagen.flow(img_, batch_size=1)     
    #img_nobath = cv2.imread(vsample_image_nobath)
    #img_nobath = cv2.cvtColor(img_nobath, cv2.COLOR_RGB2BGR) 
    k=0
    for j in range(3):

        image = next(aug_iter)[0].astype('uint8')                         
        #ax[j].imshow(image)
        #ax[j].axis('off')                      
        new_id = n_sp_name+str(i+1)+str(j+1)+'-'+str(vcategory)+'.'+n_sp_surname
        n_sp_name_surname = new_id                                   
        #print(new_id)                         
        listid_.append(new_id)
        listcategory_.append(vcategory)

        if not os.path.lexists(output_train_dir):
            os.mkdir(output_train_dir)   
            #save new image Augmentation train to new directory
        arr_2_image = Image.fromarray(image, 'RGB') 
        arr_2_image.save(output_train_dir+'/'+new_id)
        #print('save=',output_train_dir+'/'+new_id)
        vcount_add   += 1
        
        
        
print('old=',len(df5))
print('count add=',vcount_add)
print('old+add=',len(df3)+vcount_add)

print('out',len(listid_),len(listcategory_))
if len(listid_) > 0 and len(listcategory_) > 0:
    print('ok')
    df6 = pd.DataFrame({
    'id': listid_,
    'category': listcategory_
    })
    df7 = df6.append(df5, ignore_index=True)
    #to_append = [listid_nobath, listcatgory_nobath]
    #a_series = pd.Series(to_append, index = df.columns)
    #df = df.append(a_series, ignore_index=True)
     
    print(df7.head())
    print(df7.tail())
sample = random.choice(filenames)
#sample = 'output_dir_for_train/94179052-4d73-4672-a02b-a261dd942ac010122.jpg'
image = load_img(output_train_dir+'/'+sample)
plt.imshow(image)
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
df5["category"] = df5["category"].replace({0: 'bath', 1: 'nobath'}) 

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
train_df, validate_df = train_test_split(df5, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=32
train_datagen = ImageDataGenerator(
    rotation_range=10,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    output_train_dir, 
    x_col='id',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
     output_train_dir, 
    x_col='id',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    output_train_dir, 
    x_col='id',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
print(example_generator)
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
epochs=30 if FAST_RUN else 50
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)
model.save_weights("model20201008.h5")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
test_filenames = os.listdir("/kaggle/input/super-ai-image-classification/val/val/images/")
test_df = pd.DataFrame({
    'id': test_filenames
})
nb_samples = test_df.shape[0]
#test_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/kaggle/input/super-ai-image-classification/val/val/images/", 
    x_col='id',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
        
)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'nobath': 1, 'bath': 0 })
test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(10)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['id']
    category = row['category']
    img = load_img("/kaggle/input/super-ai-image-classification/val/val/images/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()
print(test_df)
submission_df = test_df.copy()
submission_df['id'] = submission_df['id'].str.split('.').str[0]
submission_df['category'] = submission_df['category']
#submission_df.drop(['filename', 'category'], axis=1, inplace=True)
#submission_df.to_csv('/kaggle/input/super-ai-image-classification/val/val/val2.csv', index=False)
print(submission_df.head())
submission_df.drop
from IPython.display import HTML
import pandas as pd
import numpy as np
#dftest =   test_df
dftest = test_df
print(test_df.head())
print(test_df.tail())


dftest.to_csv('val.csv',index = False)

def create_download_link(title = "Download CSV file", filename = "val.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='val.csv')
