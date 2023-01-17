import matplotlib.pyplot as plt

import numpy as np
a = np.array([[0,1,2,3,4]])

plt.imshow(a,cmap = 'gray')
a = np.array([[0,1,2,3,4,7,6,9,5],

              [0,1,2,3,4,5,8,9,6],

              [0,1,2,3,4,5,8,9,6]

             ])

plt.imshow(a,cmap = 'gray')
import matplotlib.pyplot as plt

import numpy as np

a = np.zeros([100,100], dtype ='int32')

a[0:50,0:50] =1

a[50:100,50:100] =1

plt.imshow(a, cmap ='gray')

from sklearn import datasets
digits = datasets.load_digits()
digits
type(digits)
digits.data    #vector representation  8*8
digits.data[1]
digits.images[1]
digits.target
digits.target_names
import pandas as pd

df =pd.DataFrame(digits.data)   

df
df.head()
df.tail()

df['Target'] = digits.target
df
plt.imshow(digits.images[8],cmap ='gray')
plt.figure(figsize=(20,5))

for index, (image, label) in enumerate(zip(digits.data[23:28], digits.target[23:28])):

    plt.subplot(1, 5, index +1)

    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)

    plt.title('Image :%i\n' %label, fontsize =20)