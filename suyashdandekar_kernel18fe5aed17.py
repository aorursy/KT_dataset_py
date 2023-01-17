from tensorflow import keras 
from tensorflow import keras
model = keras.models.load_model('/content/drive/My Drive/modeltrained (1).h5')
import zipfile
with zipfile.ZipFile('/content/drive/My Drive/710024_1246711_compressed_Medical mask.zip', 'r') as zip_ref:
    zip_ref.extractall('unzipped')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
img = mpimg.imread('/content/unzipped/Medical mask/Medical Mask/images/0001.jpg')
imgplot = plt.imshow(img)
!pip install mtcnn
from mtcnn import MTCNN
import os
images_dir='/content/unzipped/Medical mask/Medical Mask/images'
a = os.listdir(images_dir)
a.sort()
test_images = a[:1800]
len(test_images)
detector = MTCNN()
test_df = []
for image in test_images:
    img = plt.imread(os.path.join(images_dir, image))
    faces = detector.detect_faces(img)
    test = []
    for face in faces:
        bounding_box = face['box']
        a=bounding_box[0]
        b=bounding_box[1]
        c=bounding_box[2]
        d=bounding_box[3]
        test.append([image, a,b,c,d])
    test_df.append(test)
test=[]
for i in test_df:
    if len(i)>0:
        if len(i)==1:
            test.append(i[0])
        else:
            for j in i:
                test.append(j)
import pandas as pd
demo=pd.DataFrame(test)
demo.to_csv('/mtcnn',index=False)
len(test)
import cv2
test[:5]
sub=[]
rest_image=[]
for i in test:
    sub.append(i[0])
for image in test_images:
    if image not in sub:
        rest_image.append(image)
test_df_=[]
for image in rest_image:
    img=cv2.imread(os.path.join('/content/unzipped/Medical mask/Medical Mask/images',image))
    faces=detector.detect_faces(img)
    test_=[]
    for face in faces:
        bounding_box=face['box']
        test_.append([image,bounding_box])
    test_df_.append(test_)
listt=[]
anss=[]
images_names=[]
coordinates=[]
for i in test:
  try:
      for j in i:
          listt.append(j)
      s=listt[0]
      x=listt[1]
      y=listt[2]
      w=listt[3]
      h=listt[4]
      coordinates.append([x,y,x+w,y+h])
      listt.clear()
      img=plt.imread(os.path.join('/content/unzipped/Medical mask/Medical Mask/images',s))
      output=img[y:y+h,x:x+w]
      output=cv2.resize(output,(150,150))
      anss.append(output)
      images_names.append(s)
  except Exception as e:
    print(str(e))
anss
import numpy as np
ansss=np.array(anss)
ansss.shape
op=model.predict(ansss)
oppp=np.round(op)
opp=pd.DataFrame(data=oppp,columns=['classname'])
opp.head()
opp['classname'].value_counts()
opp.to_csv('/wobot_intern_output',index=False)
opp.head()
opp.replace(0.0,'face_no_mask',inplace=True)
opp.replace(1.0,'face_with_mask',inplace=True)
opp.head()
opp['classname'].value_counts()
images_names[:5]
len(opp)
len(images_names)
names=pd.DataFrame(images_names,columns=['name'])
names.head(10)
coordinates=pd.DataFrame(coordinates,columns=['x1','y1','x2','y2'])
coordinates.head()
new=pd.DataFrame()
new['name']=names['name']
new['x1']=coordinates['x1']
new['y1']=coordinates['y1']
new['x2']=coordinates['x2']
new['y2']=coordinates['y2']
new.head()
new['classname']=opp['classname']
new.head()
new=new.iloc[::-1]
new.head()
new.to_csv('/final_submition.csv',index=False)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
img = mpimg.imread('/content/unzipped/Medical mask/Medical Mask/images/1910.jpg')
imgplot = plt.imshow(img)
