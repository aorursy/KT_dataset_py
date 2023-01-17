# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install dlib
!pip install face_recognition
from PIL import Image
import face_recognition
import numpy as np
import pandas as pd
import cv2
dataset=[]
Root_Folder_Path='/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled'
k=0
for filename in os.listdir(Root_Folder_Path):
    for subfolder in os.listdir(Root_Folder_Path+'/'+filename):
        person_name=filename
        path=Root_Folder_Path+'/'+filename+'/'+subfolder
        dataset.append({"Person_Name":person_name, "Path": path})



df=pd.DataFrame(dataset)
df.head()
df.shape
df=df.groupby('Person_Name').filter(lambda x:len(x)>10)
df.shape
from sklearn.model_selection import train_test_split
train, test=train_test_split(df,test_size=0.2,random_state=26)
import matplotlib.pyplot as plt
import random
plt.figure(figsize=(10,6))
for i in range(9):
    index = random.randint(0,df.shape[0])
    img = plt.imread(df.Path.iloc[index])
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(df.Person_Name.iloc[index])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from skimage.feature import hog
from skimage import data, exposure
%matplotlib inline


num=random.randint(0,df.shape[0])
image=cv2.imread(df.Path.iloc[num])
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#plt.title(df.Person_Name.iloc[num])
#plt.imshow(image)


fd,hog_image=hog(image, orientations=8, pixels_per_cell=(16,16),
                    cells_per_block=(1,1),visualize=True, multichannel=True)

fig, (ax1, ax2)=plt.subplots(1,2,figsize=(8,4),sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image,cmap=plt.cm.gray)
ax1.set_title('Input image')


hog_image_rescaled=exposure.rescale_intensity(hog_image,in_range=(0,10))


ax2.axis('off')
ax2.imshow(hog_image,cmap=plt.cm.gray)
ax2.set_title('Histogram of oriented Gradients')
plt.show()
len(fd)
image.shape

face_locations=face_recognition.face_locations(image)
number_of_faces=len(face_locations)
print("No of Faces found in Input Image are {}".format(number_of_faces))

num=random.randint(0,df.shape[0])
image=cv2.imread(df.Path.iloc[num])
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image)
ax=plt.gca()


for face_location in face_locations:
    
    top, right, bottom, left=face_location
    x,y,x1,y1=left,top, right, bottom
    
    rect=Rectangle((x,y),x1-x,y1-y,fill=False,color='red')
    ax.add_patch(rect)
    
plt.show()
train['Person_Name'].value_counts().head()
george_bush_index_list=train[train['Person_Name']=='George_W_Bush']
num=list(george_bush_index_list.sample().index.values.astype(int))
random_number=random
image=cv2.imread(train.Path.iloc[i])
print(type(num))
george_bush_index_list=train[train['Person_Name']=='George_W_Bush']
print(george_bush_index_list.Path.iloc[2])
george_bush_index_list=train[train['Person_Name']=='George_W_Bush']
george_bush_index=list(george_bush_index_list.sample().index)
image=cv2.imread('/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/George_W_Bush/George_W_Bush_0077.jpg')
face_demo=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

Colin_Powell_index_list=train[train['Person_Name']=='Colin_Powell']
Colin_Powell_index=list(Colin_Powell_index_list.sample().index)
image=cv2.imread(train.Path.iloc[1527])
Colin_Powell_face_demo=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


Tony_Blair_index_list=train[train['Person_Name']=='Tony_Blair']
Tony_Blair_index=list(Tony_Blair_index_list.sample().index)
image=cv2.imread(train.Path.iloc[1528])
Tony_Blair_face_demo=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)



george_bush_encoding=face_recognition.face_encodings(face_demo)[0]
Colin_Powell_encoding=face_recognition.face_encodings(Colin_Powell_face_demo)[0]
Tony_Blair_encoding=face_recognition.face_encodings(Tony_Blair_face_demo)[0]
known_face_encodings=[george_bush_encoding,Colin_Powell_encoding,Tony_Blair_encoding]

image=cv2.imread('/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/George_W_Bush/George_W_Bush_0154.jpg')
face_demo=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

'''Colin_Powell_index_list=train[train['Person_Name']=='Colin_Powell']
Colin_Powell_index=list(Colin_Powell_index_list.sample().index)
image=cv2.imread(train.Path.iloc[1530])
Colin_Powell_face_demo=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


Tony_Blair_index_list=train[train['Person_Name']=='Tony_Blair']
Tony_Blair_index=list(Tony_Blair_index_list.sample().index)
image=cv2.imread(train.Path.iloc[1531])
Tony_Blair_face_demo=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)'''



george_bush_encoding=face_recognition.face_encodings(face_demo)[0]
#Colin_Powell_encoding=face_recognition.face_encodings(Colin_Powell_face_demo)[0]
#Tony_Blair_encoding=face_recognition.face_encodings(Tony_Blair_face_demo)[0]
unknown_face_encodings=[george_bush_encoding]

test.shape

len(unknown_face_encodings)
from scipy.spatial import distance

for unknown_face_encoding in unknown_face_encodings:
    
    results=[]
    for face_encoding in known_face_encodings:
        d=distance.euclidean(face_encoding,unknown_face_encoding)
        results.append(d)
    thresold=0.6
    results=np.array(results)<=thresold
    name ='UnKnown'
    
    if results[0]:
        name='George Bush'
    elif results[1]:
        name= 'one'
    elif results[2]:
        name='three'
    
    print(f"Found {name } in the photo!")
num=random.randint(0,df.shape[0])
image=cv2.imread(df.Path.iloc[num])
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
face_landmarks_list=face_recognition.face_landmarks(image)
plt.imshow(image)
import matplotlib.lines as mlines
from  matplotlib.patches import Polygon

plt.imshow(image)
ax=plt.gca()

for face_landmark in face_landmarks_list:
    
    left_eyebrow_pts=face_landmark['left_eyebrow']
    pre_x,pre_y=left_eyebrow_pts[0]
    for (x,y) in left_eyebrow_pts[1:]:
        l=mlines.Line2D([pre_x,x],[pre_y,y],color='red')
        ax.add_line(l)
        pre_x,pre_y=x,y
    
    right_eyebrow_pts=face_landmark['right_eyebrow']
    pre_x,pre_y=right_eyebrow_pts[0]
    for (x,y) in right_eyebrow_pts[1:]:
        l=mlines.Line2D([pre_x,x],[pre_y,y],color='red')
        ax.add_line(l)
        pre_x,pre_y=x,y
        
    p=Polygon(face_landmark['top_lip'],facecolor='lightsalmon',edgecolor='orangered')
    ax.add_patch(p)
    p=Polygon(face_landmark['bottom_lip'],facecolor='lightsalmon',edgecolor='orangered')
    ax.add_patch(p)

plt.show()
        


