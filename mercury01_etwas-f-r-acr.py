import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numba import jit, prange

from skimage.transform import resize



import json

import os

import tqdm.notebook as tqdm



#from tqdm import tqdm_notebook



from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.tree import DecisionTreeClassifier as DT



np.seterr(divide='ignore', invalid='ignore')

train_path =  '/kaggle/input/abstraction-and-reasoning-challenge/training/'

evaluation_path =  '/kaggle/input/abstraction-and-reasoning-challenge/evaluation/'

test_path =  '/kaggle/input/abstraction-and-reasoning-challenge/test/'
data_path = evaluation_path





same_shape = []

for ex in tqdm.tqdm(os.listdir(data_path)):# get exampels with same out- and input shape

    with open(data_path + ex, 'r') as  train_file:

        all_im = json.load(train_file)

        im_in = np.array(all_im['train'][0]['input'])

        im_out = np.array(all_im['train'][0]['output'])

        if im_in.shape == im_out.shape:

            same_shape.append(ex)

            

print("Same:",len(same_shape),"All:", len(os.listdir(data_path)))
def get_im_with_same_ioshape(file_path, name, show=False, mode='train'):

    train = []

    test = []

    

    with open(file_path+name, 'r') as  train_file:

        all_im = json.load(train_file)

        im_in = np.array(all_im['train'][0]['input'])

        im_out = np.array(all_im['train'][0]['output'])

        

        if im_in.shape != im_out.shape:

            return None

            

        for im in all_im['train']:

            

            im_in = np.array(im['input'])

            im_out = np.array(im['output'])

            mask = np.asarray(np.nan_to_num((im_in-im_out)/(im_in-im_out), 0), 'int8')

            train.append((im_in, im_out, mask))

            

            if show:

                print("NAME:\n",name)

                print("IN:\n")

                plt.imshow(im_in)

                plt.show()

                print("OUT:\n")

                plt.imshow(im_out)

                plt.show()

                print("MASK:\n")

                plt.imshow(mask)

                plt.show()

                

        if mode=='train':

            for im in all_im['test']:



                im_in = np.array(im['input'])

                im_out = np.array(im['output'])

                test.append((im_in, im_out))

        if mode=='test':

            for im in all_im['test']:



                im_in = np.array(im['input'])

                test.append((im_in))

            

      

    return train, test

               

train, test = get_im_with_same_ioshape(data_path,same_shape[0], show=False)
#@jit(nopython=False)

def get_features(input_, flipping=False, rotate=False, center=True, da=1):#get features form ech pixels

                                        # a1   a2   a3

                                        # a4   pix  a5

                                        # a6   a7   a8

    im_in,im_out, mask = input_

    size=sum(sum(mask))

    

    if flipping:

        size*=4

    if rotate:

        size*=(7/4)

        size = int(size)

        

    features = np.zeros((size,9))

    colors = np.zeros(size)

    f=0

    for y in range(mask.shape[0]):

        for x in range(mask.shape[1]):



            if mask[y,x]==1:

                pix_exp = np.zeros((2*da+1)**2)

                n_p=0

                for dy in range(-da,da+1):

                    for dx in range(-da,da+1):

                        

                        if dy!=0 or dx!=0:

                            if dx+x>=0 and dy+y>=0 and dx+x<mask.shape[1] and dy+y<mask.shape[0]:

                                pix_exp[n_p]=im_in[y+dy, x+dx]

                            else:

                                pix_exp[n_p]=-1

                        else:

                            if center:

                                pix_exp[n_p]=im_in[y, x]#-2

                            else:

                                pix_exp[n_p]=-2

                        

                        n_p+=1



                features[f] = pix_exp

                colors[f] = im_out[y, x]

                f+=1

                

                if flipping:

                    features[f] = np.flipud(pix_exp.reshape(3,3)).flatten()

                    colors[f] = im_out[y, x]

                    f+=1

                    features[f] = np.fliplr(pix_exp.reshape(3,3)).flatten()

                    colors[f] = im_out[y, x]

                    f+=1

                    features[f] = np.flip(pix_exp.reshape(3,3), (0, 1)).flatten()

                    colors[f] = im_out[y, x]

                    f+=1

                if rotate:

                    features[f] = np.rot90(pix_exp.reshape(3,3), 1).flatten()

                    colors[f] = im_out[y, x]

                    f+=1

                    features[f] = np.rot90(pix_exp.reshape(3,3), 2).flatten()

                    colors[f] = im_out[y, x]

                    f+=1

                    features[f] = np.rot90(pix_exp.reshape(3,3), 3).flatten()

                    colors[f] = im_out[y, x]

                    f+=1

                

    return features, colors

@jit(nopython=False)

def get_cf(train, flipping=False, rotate=False, center=True):#mining features from each train example and stacking of them

    features_set = []

    colors_set = []



    for in_out_mask in train:

        features, colors = get_features(in_out_mask, flipping, rotate, center)

        features_set+=list(features)

        colors_set+=list(colors)

    

    features_set_min = np.unique(np.array(features_set), axis = 0)

    colors_min = np.zeros(len(features_set_min))

    

    for n, feature in enumerate(features_set):#ToDo make adequater

            for i ,feature_uniq in enumerate(features_set_min):

                if str(feature_uniq)==str(feature):

                    colors_min[i]=colors_set[n]

                    break



            

    return colors_min, features_set_min



colors_min, features_set_min = get_cf(train,  flipping=False, rotate=False, center=True)

print(colors_min)

print(features_set_min)
#@jit(nopython=False)

def make_pred(im_in, features, colors, it=1, solver=None, center=True):  

    if solver=="KNN":

            model = KNN(2)

            model.fit(X=features, y=colors)

    elif solver=="RF":

            model = RF()

            model.fit(X=features, y=colors)

    elif solver=="DT":

            model = DT()

            model.fit(X=features, y=colors)

    

        



    for epoch in range(it):

        im_out = im_in.copy()

        f=0

        for y in range(im_in.shape[0]):

            for x in range(im_in.shape[1]):



                pix_exp = np.zeros(9)

                n_p=0

                for dy in range(-1,2):

                    for dx in range(-1,2):



                        if dy!=0 or dx!=0:



                            if dx+x>=0 and dy+y>=0 and dx+x<im_in.shape[1] and dy+y<im_in.shape[0]:

                                pix_exp[n_p]=im_in[y+dy, x+dx]

                            else:

                                pix_exp[n_p]=-1

                        else:

                            if center:

                                pix_exp[n_p]=im_in[y, x]#-2

                            else:

                                pix_exp[n_p]=-2



                        n_p+=1

                        

                if solver==None:

                    for n, f in enumerate(features):

                        if str(f)==str(pix_exp):

                            im_out[y,x]=colors[n]

                else:

                    im_out[y,x]=model.predict([pix_exp])

                        

                        

                        

        

        im_in=im_out.copy()

    

                    

    return im_out

pred=make_pred(test[0][0], features_set_min, colors_min, 1, "DT", True)

print("INPUT")

plt.imshow(test[0][0])

plt.show()

print("PREDICT")

plt.imshow(pred)

plt.show()

print("OUTPUT")

plt.imshow(test[0][1])

plt.show()

data_path = evaluation_path # evaluation_path or train_path





same_shape = []

for ex in tqdm.tqdm(os.listdir(data_path)):

    with open(data_path + ex, 'r') as  train_file:

        all_im = json.load(train_file)

        im_in = np.array(all_im['train'][0]['input'])

        im_out = np.array(all_im['train'][0]['output'])

        if im_in.shape == im_out.shape:

            same_shape.append(ex)



            

print("Same:",len(same_shape),"All:", len(os.listdir(data_path)))





solved=0

for name in tqdm.tqdm(same_shape):

    data = get_im_with_same_ioshape(data_path, name)

    if data!=None:

        train, test = data



        colors, features = get_cf(train, True, True, True) 

        pred = str(make_pred(test[0][0],  features,  colors, 1, None, True))

        

        colors1, features1 = get_cf(train, False, False, True)

        pred1=str(make_pred(test[0][0], features1, colors1, 1, "DT", True))

        

        colors2, features2 = get_cf(train, False, False, False)

        pred2=str(make_pred(test[0][0], features2, colors2, 1, None, False)) 

     

        vorbild =str( test[0][1])

        if  pred==vorbild or pred1==vorbild or pred2==vorbild:

            

            solved+=1

            print('*************\nUhu!!!\n'+str(solved)+"\n"+name)

            

print("Same_solved:", solved)
sample = pd.read_csv("/kaggle/input/abstraction-and-reasoning-challenge/sample_submission.csv")
# Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation

def flattener(grid):

    grid = grid.astype('uint8').tolist()    

    str_pred = str([row for row in grid])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
num=0

for test_name in tqdm.tqdm(sample.output_id):

        name = test_name.split('_')[0]+'.json'

        index=int(test_name.split('_')[1]) 

        

        data = get_im_with_same_ioshape(test_path, name, False, 'test')

        if data!=None:

                train, test = data

                colors, features = get_cf(train, True, True, True) 

                pred = make_pred(test[index],  features,  colors, 1, None, True)



                colors1, features1 = get_cf(train, False, False, True)

                pred1 = make_pred(test[index], features1, colors1, 1, "DT", True)



                colors2, features2 = get_cf(train, False, False, False)

                pred2 = make_pred(test[index], features2, colors2, 1, None, False)  



                sample.output[num] = ' '.join([flattener(pred), flattener(pred1), flattener(pred2)])



        num+=1



sample.to_csv('submission.csv', index = False)