import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#jpg랑 xml 분리하기
data_jpg=[]
data_xml=[]

for filename in os.listdir('/kaggle/input/fruit-images-for-object-detection/train_zip/train'):
    if filename.split('.')[1] == 'jpg':
        data_jpg.append('/kaggle/input/fruit-images-for-object-detection/train_zip/train/'+filename)
    else :
        data_xml.append('/kaggle/input/fruit-images-for-object-detection/train_zip/train/'+filename)

data_jpg.sort()
data_xml.sort()

# 정렬잘됐나 확인
print(data_jpg[0],data_xml[0])
#데이터 잘 들어갔나 확인
print(len(data_jpg),len(data_xml))
# xml 파싱해서 원하는 오브젝트 저장
# 원래 string으로 저장되지만 int로 형변환
from xml.etree.ElementTree import parse
object_xmin=[]
object_ymin=[]
object_xmax=[]
object_ymax=[]

for i in range(len(data_xml)):
    root_=parse(data_xml[i])
    root=root_.getroot()
    objects = root.findall("object")
    object_xmin.append([int(x.find("bndbox").findtext("xmin")) for x in objects])
    object_ymin.append([int(x.find("bndbox").findtext("ymin")) for x in objects])
    object_xmax.append([int(x.find("bndbox").findtext("xmax")) for x in objects])
    object_ymax.append([int(x.find("bndbox").findtext("ymax")) for x in objects])
# 잘들어갔나 확인
print(object_xmin)
import cv2
import matplotlib.pyplot as plt

for i in range(len(data_jpg)):
    img = cv2.imread(data_jpg[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for j in range(len(object_xmin[i])):
        output=cv2.rectangle(img,(object_xmin[i][j],object_ymin[i][j]),(object_xmax[i][j],object_ymax[i][j]),(255,0,0),2)
    plt.imshow(output)
    plt.show()
