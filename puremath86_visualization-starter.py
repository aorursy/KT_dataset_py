import numpy as np

import pandas as pd 



import random



%matplotlib inline

import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns; sns.set_style("white")



import os

print(os.listdir("../input"))
print(os.listdir("../input/train_images")[:5])
df_train = pd.read_csv("../input/train.csv")

print(df_train.head())

print(df_train.diagnosis.unique())
PATH = "../input/train_images"





for i in range(10):

    plt.figure(figsize=(10,10))

    _id = random.choice(os.listdir(PATH))

    id_code = _id.split(".")[0]

    pil_im = Image.open(os.path.join(PATH, _id))

    print(id_code, df_train.loc[df_train.id_code == id_code, 'diagnosis'])

    plt.imshow(np.asarray(pil_im))    

    plt.show()
plt.figure(figsize=(10, 6))

df_train.diagnosis.hist()

plt.show()
df_train.diagnosis.value_counts()
sub = pd.read_csv("../input/sample_submission.csv")

sub.head()
len(sub.diagnosis)
n = 1928

a = [0]*1805

b = [1]*370

c = [2]*999

d = [3]*193

e = [4]*295

_list = a + b + c + d + e

random_guess = [random.choice(_list) for i in range(n)]

random_guess[:10]
sub['diagnosis'] = random_guess

sub.head()
# random guess --nothing interesting here... :'(

sub.to_csv("submission.csv", index=False)