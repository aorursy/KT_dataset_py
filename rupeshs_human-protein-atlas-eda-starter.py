dicts={
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles"   ,
5:  "Nuclear bodies"   ,
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus"   ,
8:  "Peroxisomes"   ,
9:  "Endosomes"   ,
10:  "Lysosomes"   ,
11:  "Intermediate filaments",   
12:  "Actin filaments"   ,
13:  "Focal adhesion sites",   
14:  "Microtubules"   ,
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle"   ,
18:  "Microtubule organizing center" ,  
19:  "Centrosome"   ,
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions"  , 
23:  "Mitochondria"   ,
24:  "Aggresome"   ,
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_train.info()
df_train.head()
#cheking for nulls
df_train["Target"].isnull().values.sum()
from collections import Counter, defaultdict
labels = df_train['Target'].apply(lambda x: x.split(' '))
labels[:5]
counts = defaultdict(int)
for l in labels:
    for l2 in l:
        counts[l2] += 1
strs=[]
for count in counts.keys(): strs.append(dicts[int(count)])
import seaborn as sns
sns.set(style="whitegrid")
df_count = pd.DataFrame({'ptype': strs,
     'Count': list(counts.values())})
df_count.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.subplots(figsize=(20,15))
ax = sns.barplot(x="ptype", y="Count", data=df_count)
for item in ax.get_xticklabels():
    item.set_rotation(90)
import cv2
from PIL import Image
#opening a green image
img = Image.open('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png')
img
print("Image size :",img.size)
#opening a neuclus blue image
img = Image.open('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png')
img
#opening a microtubles red image
img = Image.open('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png')
img
 #opening a endoplasmic reticulum (yellow)
img = Image.open('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png')
img
    