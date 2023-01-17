import pandas as pd
import os
import numpy as np
train=pd.read_csv("../input/global-wheat-detection/train.csv")
def convert_to_yolo(bbox, c=1024.0):
  bbox=np.fromstring(bbox[1:-1],sep=',')
  x=(bbox[0]+bbox[2]/2.0)/c
  y=(bbox[1]+bbox[3]/2.0)/c
  w= bbox[2]/c
  h=bbox[3]/c
  yolo_box=[x,y,w,h]
  return(yolo_box)

train['yolo_box'] = train.bbox.apply(convert_to_yolo)
unique=train.image_id.unique()
if not os.path.exists("../yolo"):
    os.makedirs("../yolo")
for i in unique:
  file= "%s.txt" %i
  a='../yolo'
  b=file
  path=os.path.join(a,b)
  os.mknod(path)
  file_data= train.query('image_id == "%s"' %i)
  boxes = file_data.yolo_box.values
  with open(path, 'a') as file:
    for j in boxes:
      s = "0 %s %s %s %s \n"
      new_line = (s % tuple(j))
      file.write(new_line)

f=open('../yolo/b53afdf5c.txt')
print(f.read())