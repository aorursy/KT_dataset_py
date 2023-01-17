# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from collections import OrderedDict
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
# Any results you write to the current directory are saved as output.
classdic = OrderedDict([('CALsuburb', 9),#0
                        ('PARoffice', 7),#1
                        ('bedroom', 12),#2
                        ('coast', 10),#3
                        ('forest',4),#4
                        ('highway', 14),#5
                        ('industrial', 2),#6
                        ('insidecity', 3),#7
                        ('kitchen', 0),#8
                        ('livingroom', 5),#9
                        ('mountain', 8),#10
                        ('opencountry', 6),#11
                        ('store', 11),#12
                        ('street', 1),#13
                        ('tallbuilding', 13)])#14
from sklearn.metrics import confusion_matrix
# ary = confusion_matrix(validation_generator.classes, predicted_classes)
ary = np.array([[30,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0, 21,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  5,  0,  0,  0,  1,  0,  2,  9,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 39,  0,  3,  0,  0,  0,  0,  0,  5,  0,  0,  0],
       [ 0,  0,  0,  0, 37,  0,  0,  0,  0,  0,  2,  2,  0,  0,  0],
       [ 0,  0,  0,  0,  0, 29,  0,  0,  1,  1,  0,  2,  1,  0,  0],
       [ 1,  1,  1,  0,  0,  1, 23,  7,  2,  0,  0,  0,  6,  2,  1],
       [ 0,  0,  0,  0,  0,  0,  0, 31,  0,  0,  0,  0,  2,  1,  1],
       [ 0,  0,  0,  0,  0,  0,  0,  0, 20,  3,  0,  0,  1,  0,  0],
       [ 0,  4,  3,  0,  0,  0,  1,  2,  5, 28,  0,  0,  3,  0,  0],
       [ 0,  0,  0,  1,  1,  1,  0,  0,  0,  0, 63,  2,  0,  0,  0],
       [ 0,  0,  0,  2,  0,  1,  0,  0,  0,  0,  6, 47,  0,  1,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0, 31,  2,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0, 42,  0],
       [ 0,  0,  1,  0,  0,  0,  2,  4,  0,  0,  0,  1,  0,  0, 45]])

df_cm = pd.DataFrame(ary, index = [ c for c in classdic.keys()],
                  columns = [ c for c in classdic.keys()])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)




