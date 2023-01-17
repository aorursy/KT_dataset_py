# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import seaborn as sns

import os

import re

import copy

import math

from decimal import *

import tensorflow as tf

import cv2

import scipy

import scipy.spatial.distance as sp

import pathlib

import imageio

import json

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
missing_values = ["n/a", "na", "--"]

data = pd.read_csv('../input/spamemail/spam.csv', encoding = "ISO-8859-1")

data = data.dropna(axis=1)

data = data.replace({'v1': {'ham': 0, 'spam': 1}}).dropna(axis=1)

data.head()




def preprocess_string(str_arg):

    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE)

    cleaned_str=re.sub('(\s+)',' ',cleaned_str)

    cleaned_str=cleaned_str.lower()

    

    return cleaned_str







data['text'] = data['v2'].apply(preprocess_string)

data.head()







def prepare_count(df):

  word_dict = {0: {}, 1: {}}

  sample_count = {0: 0, 1: 0}

  word_count = {0: 0 , 1: 0}

  for _, r in df.iterrows():

    cls = r['v1']

    sample_count[cls] += 1

    for w in r['text'].split():

      word_count[cls] += 1

      word_dict[cls][w] = word_dict[cls].get(w, 0) + 1

  

  return word_dict, sample_count, word_count



train_data, test_data = train_test_split(data, test_size=0.3, shuffle=True)

print(train_data.size, test_data.size)
word_dict,sample_count,word_count = prepare_count(train_data)

spam_dict = word_dict[1]

ham_dict = word_dict[0]

spam_count = sample_count[1]

ham_count = sample_count[0]

spam_wc = word_count[1]

ham_wc = word_count[0]

# print(ham_wc,spam_wc)

print(len(ham_dict))
test_data.head()
def get_probability(df,cls):

    probabilities = []

    class_prob = Decimal(sample_count[cls])/Decimal((spam_count+ham_count))

    new_dict = copy.deepcopy(spam_dict)

    new_dict.update(ham_dict)

    numer = class_prob

#     numer = np.log(spam_prob)

    denom = 1

    for _, r in df.iterrows():

        #one particular email

        cls = r['v1']

        #numerator calculation

        for w in r['text'].split():

            if w in word_dict[cls].keys():

                prob = Decimal((word_dict[cls][w]+1))/Decimal((word_count[cls]+1+len(word_dict[cls])))

                numer *= Decimal(prob)

#                 numer += np.log(prob)

            else:

                prob = Decimal(1)/Decimal(word_count[cls]+1+len(word_dict[cls]))

                numer *= Decimal(prob)

#                 numer += np.log(prob)

        #denominator calculation

#         for w in r['text'].split():

#             c1 = 1

#             c2 = 1

#             if w in spam_dict.keys():

#                 c1 += spam_dict[w]

#             if w in ham_dict.keys():

#                 c2 += ham_dict[w]

# #             prob = (c1 + c2) / (spam_count + ham_count + 1 + len(new_dict))

#             prob1 = Decimal((c1/(spam_wc+1+len(new_dict))))*Decimal((spam_count/(spam_count+ham_count)))

#             prob2 = Decimal((c2/(ham_wc+1+len(new_dict))))*Decimal((ham_count/(spam_count+ham_count)))

#             prob = Decimal(prob1)+Decimal(prob2)

# #             print(prob)

#             denom *= Decimal(prob)

# #             denom += np.log(prob)

# #         sample_probability = Decimal(numer)/Decimal((denom))

        sample_probability = Decimal(numer)

#         sample_probability = numer - denom

#         probabilities.append(np.exp(sample_probability))

        probabilities.append(sample_probability)

    return probabilities
probabilities_spam = get_probability(test_data,1)

probabilities_ham = get_probability(test_data,0)

# print(probabilities)
Y_pred = []

length = len(probabilities_spam)

for s in range(length):

    if probabilities_ham[s] > probabilities_spam[s]:

        Y_pred.append(0)

    else:

        Y_pred.append(1)

    





Y_actual = test_data['v1']

cm = confusion_matrix(Y_actual, Y_pred)

print("Confusion Matrix",cm)

ax = sns.heatmap(confusion_matrix(Y_actual, Y_pred))

acc = accuracy_score(Y_pred, Y_actual)

print(acc, 'is the accuracy')




data = []

paths = pathlib.Path('../input/riverdataset/images').glob('*/*.jpg')

for x in paths:

    val = cv2.imread(str(x),0)

    data.append(val)

print(paths)    

data_arr = np.asarray(data)

print(data_arr.shape)

print(os.listdir("../input/jsonfiles"))







#df = pd.read_json("/input/river-ds/send-archive/river.json", lines = True)

with open("../input/jsonfiles/send-archive/river.json") as df_river:

    data = json.load(df_river)

df_river = pd.DataFrame(data)

df_river = df_river.values

#dataframe







Dict_river = {} 

for i in range (256):

    Dict_river[i] = 0

#Dict

for k in range (4):

    for i in range (76):

        m = df_river[i][0]

        n = df_river[i][1]

        Dict_river[data_arr[k][m][n]] = Dict_river[data_arr[k][m][n]] + 1

Dict_river



with open("../input/jsonfiles/send-archive/not-river.json") as df_not_river:

    data = json.load(df_not_river)

df_not_river = pd.DataFrame(data)

df_not_river = df_not_river.values

Dict_not_river = {} 

for i in range (256):

    Dict_not_river[i] = 0

#Dict

for k in range (4):

    for i in range(197):

        m = df_not_river[i][0]

        n = df_not_river[i][1]

        Dict_not_river[data_arr[k][m][n]] = Dict_not_river[data_arr[k][m][n]] + 1

           

Dict_not_river
def get_probability_river(val1,val2,val3,val4):

#     class_prob = Decimal(len(df_river))/Decimal(len(df_not_river)+len(df_river))

    class_prob = Decimal(1/2)

    numer = class_prob

    denom = 1

    count = 0

    for value in Dict_river.values():

        if value!=0:

            count = count+1

    prob1 = Decimal((Dict_river[val1]+1))/Decimal(76*4+1+count)

    prob2 = Decimal((Dict_river[val2]+1))/Decimal(76*4+1+count)

    prob3 = Decimal((Dict_river[val3]+1))/Decimal(76*4+1+count)

    prob4 = Decimal((Dict_river[val4]+1))/Decimal(76*4+1+count)

    numer *= Decimal(prob1)*Decimal(prob2)*Decimal(prob3)*Decimal(prob4)

        #denominator calculation

#     prob = Decimal(Dict_river[val]+2+Dict_not_river[val])/Decimal(76*4+197*4+255+2)

#     denom *= Decimal(prob)

#     sample_probability = Decimal(numer)/Decimal((denom))

    sample_probability = Decimal(numer)

    return sample_probability  
def get_probability_notriver(val1,val2,val3,val4):

    class_prob = Decimal(1/2)

#     class_prob = Decimal(len(df_not_river))/Decimal(len(df_not_river)+len(df_river))

    numer = class_prob

    denom = 1

    count = 0

    for value in Dict_not_river.values():

        if value!=0:

            count = count+1

    prob1 = Decimal((Dict_not_river[val1]+1))/Decimal(197*4+1+count)

    prob2 = Decimal((Dict_not_river[val2]+1))/Decimal(197*4+1+count)

    prob3 = Decimal((Dict_not_river[val3]+1))/Decimal(197*4+1+count)

    prob4 = Decimal((Dict_not_river[val4]+1))/Decimal(197*4+1+count)

    numer *= Decimal(prob1)*Decimal(prob2)*Decimal(prob3)*Decimal(prob4)

        #denominator calculation

#     prob = Decimal(Dict_river[val]+2+Dict_not_river[val])/Decimal(76*4+197*4+255+2)

#     denom *= Decimal(prob)

#     sample_probability = Decimal(numer)/Decimal((denom))

    sample_probability = Decimal(numer)

    return sample_probability 

data = []

output = []

paths = pathlib.Path('../input/riverdataset/images').glob('*/*.jpg')

for x in paths:

    val = cv2.imread(str(x),0)

    val = val.flatten()

    data.append(val)

data = np.asarray(data)

print(data.shape)

for i in range(262144):

    if get_probability_river(data[0][i],data[1][i],data[2][i],data[3][i]) > get_probability_notriver(data[0][i],data[1][i],data[2][i],data[3][i]):

        output.append(255)

#         print("yay")

    else:

#         print("nay")

        output.append(0)

        

output = np.reshape(output, [512, 512])

print(output)

plt.imshow(output,cmap="gray")
import glob

# river_dataset_path = '../input/riverdataset2/images'

# raw_image_files = pathlib.Path('../input/riverdataset/images').glob('*/*.jpg')

# raw_image_files = glob.glob(f'{river_dataset_path}/PDCV*.jpeg')

# raw_image_files = glob.glob(f'{river_dataset_path}/UDCV*.jpeg')

data = []

paths = pathlib.Path('../input/riverdataset2/images').glob('*/*.jpg')

for x in paths:

    val = cv2.imread(str(x),0)

    data.append(val)

print(paths)    

images = np.asarray(data)

print(data_arr.shape)

# paths = pathlib.Path('../input/riverdataset2/images').glob('*/*.jpg')

masked_image = ''

paths2 = pathlib.Path('../input/riverdataset/images').glob('*/*.jpg')

for x in paths2:

    masked_image = cv2.imread(str(x),0)

    break
n_river_pts = 50

n_not_river_pts = 100

print(masked_image)

# this needs to be thresholded

_, masked_image = cv2.threshold(masked_image, 127, 255, cv2.THRESH_BINARY)



# prepare samples of points having coordinates of river and not-river

river_coords = np.argwhere(masked_image == 255)

not_river_coords = np.argwhere(masked_image == 0)

assert(len(river_coords) + len(not_river_coords) == 512*512)



river_pts = river_coords[

    np.random.choice(len(river_coords), n_river_pts)

]

not_river_pts = not_river_coords[

    np.random.choice(len(not_river_coords), n_not_river_pts)

]



river_samples = np.array([images[:, x, y] for x, y in river_pts]).T

not_river_samples = np.array([images[:, x, y] for x, y in not_river_pts]).T



print(river_samples.shape, not_river_samples.shape)
mean_river = river_samples.mean(axis=1).reshape((4,1))

mean_not_river = not_river_samples.mean(axis=1).reshape((4, 1))

cov_river = np.cov(river_samples)

cov_not_river = np.cov(not_river_samples)



# calculating inverses and determinants : pre-computation

cov_river_inv = np.linalg.pinv(cov_river)

cov_not_river_inv = np.linalg.pinv(cov_not_river)

det_cov_river = np.linalg.det(cov_river)

det_cov_not_river = np.linalg.det(cov_not_river)
import itertools 



def predict(p_prior_river, p_prior_not_river):

  result = np.zeros((512, 512))

  xs = list(range(512))

  for x, y in itertools.product(xs, xs):

    cur_sample = images[:, x, y].reshape((4, 1))

    diff = cur_sample - mean_river

    river_cls = np.linalg.multi_dot([diff.T, cov_river_inv, diff]).item()

    

    diff = cur_sample - mean_not_river

    not_river_cls = np.linalg.multi_dot([diff.T, cov_not_river_inv, diff]).item()

    

    p_river = -0.5 * np.sqrt(det_cov_river) / np.exp(river_cls)

    p_not_river = -0.5 * np.sqrt(det_cov_not_river) / np.exp(not_river_cls)

    

    result[x, y] = 0 if p_river*p_prior_river >= p_not_river*p_prior_not_river else 255

  return result

test_cases = [(0.3, 0.7),

              (0.7, 0.3),

              (0.5, 0.5)

             ]



results = []

for t in test_cases:

  results.append(predict(*t))





y_actual = masked_image.flatten()

# compute accuracy

acc = [accuracy_score(r.flatten(), y_actual) for r in results]

  

plt.figure(figsize=(20, 20))

for i, r in enumerate(results):

  ax = plt.subplot(f'13{i+1}')

  ax.grid(False)

  ax.imshow(r, cmap='gray')

  ax.set_title(f'Accuracy = {acc[i]}')