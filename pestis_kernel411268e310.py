from keras.applications.vgg19 import VGG19

from keras.preprocessing import image

from keras.applications.vgg19 import preprocess_input

from keras.models import Model

import numpy as np

import os

import pandas as pd

import cv2



base_model = VGG19(weights='imagenet')

model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)



res = []

res2 = []

res3 = []

res4 = []

df = pd.read_csv('/kaggle/input/evohackaton/train.csv')

for index, row in df.iterrows():

    img = image.load_img("../input/evohackaton/train/train/" + row['name'], target_size=(224, 224))

    ar = image.img_to_array(img)

    x = np.expand_dims(ar, axis=0)

    x = preprocess_input(x)



    

    block4_pool_features = model.predict(x)

    res.append(block4_pool_features[0])

    ar2 = image.random_rotation(ar, 90)



    x = np.expand_dims(ar2, axis=0)

    x = preprocess_input(x)



    block4_pool_features = model.predict(x)

    res2.append(block4_pool_features[0])

    

    ar3 = image.random_rotation(ar, 90)



    x = np.expand_dims(ar3, axis=0)

    x = preprocess_input(x)



    block4_pool_features = model.predict(x)

    res3.append(block4_pool_features[0])

    

    ar4 = image.random_rotation(ar, 90)



    x = np.expand_dims(ar4, axis=0)

    x = preprocess_input(x)



    block4_pool_features = model.predict(x)

    res4.append(block4_pool_features[0])

res = np.array(res)

res2 = np.array(res2)

res3 = np.array(res3)

res4 = np.array(res4)





from annoy import AnnoyIndex

t = AnnoyIndex(4096, metric='euclidean')

for i, r in enumerate(res, start=1):

    t.add_item(i, r)



for i, r in enumerate(res2, start=1):

    t.add_item(i, r)



for i, r in enumerate(res3, start=1):

    t.add_item(i, r)



for i, r in enumerate(res4, start=1):

    t.add_item(i, r)

t.build(350)



nns = []

for i, r in enumerate(res):

    nns.append(t.get_nns_by_vector(r, 30, search_k=-1, include_distances=False))





    

cats = []

for arr in nns:

    a = []

    for o in arr[1:]:

        a.append(df.iloc[o-1]['category'])

    cats.append(a)

    

def most_frequent(List): 

    counter = 0

    num = List[0] 

      

    for i in List: 

        curr_frequency = List.count(i) 

        if(curr_frequency> counter): 

            counter = curr_frequency 

            num = i 

  

    return num 



c = []

for arr in cats:

    c.append(most_frequent(arr))

    

df['cats'] = c

C = np.where(df['category'] == df['cats'], 1, 0)



print(np.count_nonzero(np.where(df['category'] == df['cats'], 1, 0))/df.shape[0])
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.models import Model

import numpy as np

import os

import pandas as pd



base_model = ResNet50(weights='imagenet')

model_net = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)



res_net = []

df = pd.read_csv('/kaggle/input/evohackaton/train.csv')

for index, row in df.iterrows():

    img = image.load_img("../input/evohackaton/train/train/" + row['name'], target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)



    block4_pool_features = model_net.predict(x)

    res_net.append(block4_pool_features[0])

        

res_net = np.array(res_net)



from annoy import AnnoyIndex

t_net = AnnoyIndex(2048, metric='hamming')

for i, r in enumerate(res_net, start=1):

    t_net.add_item(i, r)

    

t_net.build(150)



nns_net = []

for i, r in enumerate(res_net):

    nns_net.append(t_net.get_nns_by_vector(r, 30, search_k=-1, include_distances=False))

    

cats_net = []

for arr in nns_net:

    a = []

    for o in arr[1:]:

        a.append(df.iloc[o-1]['category'])

    cats_net.append(a)



def most_frequent(List): 

    counter = 0

    num = List[0] 

      

    for i in List: 

        curr_frequency = List.count(i) 

        if(curr_frequency> counter): 

            counter = curr_frequency 

            num = i 

  

    return num 



c_net = []

for arr in cats_net:

    c_net.append(most_frequent(arr))



df['cats'] = c_net

C = np.where(df['category'] == df['cats'], 1, 0)



print(np.count_nonzero(np.where(df['category'] == df['cats'], 1, 0))/df.shape[0])
cn = np.array(cats_net)

vgn = np.array(cats)



merged = np.concatenate((cn, vgn), 1)



merged = merged.tolist()



c_merged = []

for arr in merged:

    c_merged.append(most_frequent(arr))
c_merged

df['cats'] = c_merged

C = np.where(df['category'] == df['cats'], 1, 0)



print(np.count_nonzero(np.where(df['category'] == df['cats'], 1, 0))/df.shape[0])
df
res_test = []

res_net_test = []

df_test = df.append({'name' : '16857.jpg' , 'category' : 1} , ignore_index=True)



for index, row in df_test.iterrows():

    img = image.load_img("../input/evohackaton/test/test/" + row['name'], target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)



    block4_pool_features = model.predict(x)

    res_test.append(block4_pool_features[0])

    block4_pool_features = model_net.predict(x)

    res_net_test.append(block4_pool_features[0])

        

res_test = np.array(res_test)

res_net_test = np.array(res_net_test)
nns_net_test = []

for i, r in enumerate(res_net_test):

    nns_net_test.append(t_net.get_nns_by_vector(r, 30, search_k=-1, include_distances=False))



cats_net_test = []

for arr in nns_net_test:

    a = []

    for o in arr[1:]:

        a.append(df.iloc[o-1]['category'])

    cats_net_test.append(a)



    

nns_test = []

for i, r in enumerate(res_test):

    nns_test.append(t.get_nns_by_vector(r, 30, search_k=-1, include_distances=False))

    

cats_test = []

for arr in nns_test:

    a = []

    for o in arr[1:]:

        a.append(df.iloc[o-1]['category'])

    cats_test.append(a)    



cn = np.array(cats_net_test)

vgn = np.array(cats_test)



merged_test = np.concatenate((cn, vgn), 1)



merged_test = merged_test.tolist()



def most_frequent(List): 

    counter = 0

    num = List[0] 

      

    for i in List: 

        curr_frequency = List.count(i) 

        if(curr_frequency> counter): 

            counter = curr_frequency 

            num = i 

  

    return num 



c_merged_test = []

for arr in merged_test:

    c_merged_test.append(most_frequent(arr))

    

df_test['category'] = c_merged_test
df_test.to_csv('submission.csv', header=True, index=False)
df_test
test_nns = []

for i, r in enumerate(res_test):

    test_nns.append(t.get_nns_by_vector(r, 30, search_k=-1, include_distances=False))

    

cats = []

for arr in test_nns:

    a = []

    for o in arr[1:]:

        a.append(df.iloc[o-1]['category'])

    cats.append(a)

    

def most_frequent(List): 

    counter = 0

    num = List[0] 

      

    for i in List: 

        curr_frequency = List.count(i) 

        if(curr_frequency> counter): 

            counter = curr_frequency 

            num = i 

  

    return num 



c = []

for arr in cats:

    c.append(most_frequent(arr))
test_df['category'] = c

test_df.to_csv('result.csv')
np.array(nns).shape
test_df
df