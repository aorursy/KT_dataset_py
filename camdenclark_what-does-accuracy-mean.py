import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv("../input/train.csv")
train = train[train.x<=.2]
train = train[train.y<=.2]
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.scatter(train.x,train.y,c=train.place_id,s=train.accuracy,alpha=0.5)
#for accuracy, x, y in zip(small_train.accuracy,small_train.x,small_train.y):
 #   plt.annotate(
  #      accuracy,
   #     xy = (x,y), xytext = (-20,20),
    #    textcoords = 'offset points', ha = 'right', va = 'bottom',
     #   bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
      #  arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.show()
small_train.place_id.value_counts()
small_train_largest=pd.concat([small_train[small_train.place_id==5445221293],
                               small_train[small_train.place_id==3804306710],
                               small_train[small_train.place_id==1006316884],
                               small_train[small_train.place_id==2148728558],
                               small_train[small_train.place_id==3027578816],
                               small_train[small_train.place_id==4492862780],
                               small_train[small_train.place_id==8370753254],
                               small_train[small_train.place_id==6007444822],
                               small_train[small_train.place_id==8555966805],
                               small_train[small_train.place_id==7065354365],
                               small_train[small_train.place_id==9727638738],
                              ])
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.scatter(small_train_largest.x,small_train_largest.y,c=small_train_largest.place_id,s=small_train_largest.accuracy,alpha=0.5)
plt.show()