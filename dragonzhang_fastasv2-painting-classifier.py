#!zip output.zip *.csv
!cp -rf /kaggle/input/painting/* .
 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



 

import os

  
!pip install --upgrade fastai
from fastai.vision.all import * 

from fastai import *

from fastai.vision import *

from fastai.metrics import *
df=pd.read_csv("train.csv")

df.head()
newdf=df.copy()



for i in df.index:

  newdf["filename"][i]=str(df["filename"][i])+".jpg"



newdf.head()



newdf.to_csv("my_train.csv",index=None)

 
 

data_folder = Path(".")

total_test_num=len(sorted(os.listdir("./test")))



train_df = df 

test_df = pd.DataFrame({"filename": [f'{i}.jpg' for i in range(total_test_num)]})

#test_img = ImageList.from_df(test_df, path=data_folder, folder='test')

 
dls = ImageDataLoaders.from_df(newdf, path=".", folder='train', label_delim=None,

                               item_tfms=Resize(300), bs=16)
#learn = cnn_learner(dls, xresnext101, metrics=accuracy)
learn = cnn_learner(dls, resnet101, metrics=accuracy,pretrained=True)
learn.fit(240)
#learn.unfreeze()

#learn.fit(2, max_lr=slice(1e-5,1e-4))

#learn.fine_tune(1)
learn.export()

learn.predict('./test/0.jpg')
imgs=sorted(os.listdir("./test"))

print(imgs[0])
sub2=[]

pred2=[]

 

i=0

for imgname in imgs:

    i+=1

    n = imgname[:-4]

    print(n)

    

    img="./test/"+imgname

    pred2 = learn.predict(img)

    print(pred2[0])



    

     

    sub2.append([n,pred2[0]])

     



    #if i==10:

    #  break



    

df2=pd.DataFrame(sub2)

# convert column "a" to int64 dtype and "b" to complex type

#df = df.astype({"a": int, "b": complex})

df2[[0]]=df2[[0]].astype(int)

df2[[1]]=df2[[1]].astype(int)



df2.sort_values(by=[0],inplace=True,ascending=True)



dt=datetime.now()

df2.to_csv("painting_{}_submission.csv".format(dt.strftime("%m%d%H")),index=None,header=None)
!zip output.zip  /kaggle/working/*.csv