!cd ../input/nvidia-apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

!git clone https://github.com/ultralytics/yolov3
%cd ..

!mkdir tmp

!ls

%cd tmp
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p1BWofDJOKXqCtO0JPT5VyuIPOsuxOuj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p1BWofDJOKXqCtO0JPT5VyuIPOsuxOuj" -O openlogo.tar && rm -rf /tmp/cookies.txt

!tar -xf openlogo.tar

!rm openlogo.tar
%cd openlogo

%cd Annotations

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wRJtZzZguXhpMrl7tVxIE6VkA96D1Y7Y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wRJtZzZguXhpMrl7tVxIE6VkA96D1Y7Y" -O delete.txt && rm -rf /tmp/cookies.txt

!awk '{print substr($1,0,length($1))}' delete.txt | xargs -I % rm %

!rm delete.txt


# hand written code - respectOP

from xml.dom import minidom

import os

import glob





def convert_coordinates(size, box):

    dw = 1.0/size[0]

    dh = 1.0/size[1]

    x = (box[0]+box[1])/2.0

    y = (box[2]+box[3])/2.0

    w = box[1]-box[0]

    h = box[3]-box[2]

    x = x*dw

    w = w*dw

    y = y*dh

    h = h*dh

    return (x,y,w,h)





def convert_xml2yolo():



    for fname in glob.glob("*.xml"):

        

        xmldoc = minidom.parse(fname)

        

        fname_out = (fname[:-4]+'.txt')



        with open(fname_out, "w") as f:



            itemlist = xmldoc.getElementsByTagName('object')

            size = xmldoc.getElementsByTagName('size')[0]

            width = int((size.getElementsByTagName('width')[0]).firstChild.data)

            height = int((size.getElementsByTagName('height')[0]).firstChild.data)



            for item in itemlist:





                # get bbox coordinates

                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data

                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data

                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data

                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data

                b = (float(xmin), float(xmax), float(ymin), float(ymax))

                bb = convert_coordinates((width,height), b)

                #print(bb)



                f.write("0" + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')



        #print ("wrote %s" % fname_out)







def main():

    convert_xml2yolo()



main()



%cd ..

%cd ..
%cd ..



!mkdir working/yolov3/traindata

!mkdir working/yolov3/traindata/images

!mkdir working/yolov3/traindata/labels

!mkdir working/yolov3/validdata

!mkdir working/yolov3/validdata/images

!mkdir working/yolov3/validdata/labels
glob.glob('tmp/openlogo/Annotations/*.txt')
import random

import glob

from shutil import copy

txtfiles = random.sample(glob.glob('tmp/openlogo/Annotations/*.txt'),1600)

imgsrc = 'tmp/openlogo/JPEGImages/'

for x in txtfiles[:-100]:

  name = x.split('/')[-1][:-4]

  copy(imgsrc + name + '.jpg','working/yolov3/traindata/images/' + name + '.jpg')

  copy(x,'working/yolov3/traindata/labels/' + name + '.txt')



for y in txtfiles[-100:]:

  name = y.split('/')[-1][:-4]

  copy(imgsrc + name + '.jpg','working/yolov3/validdata/images/' + name + '.jpg')

  copy(y,'working/yolov3/validdata/labels/' + name + '.txt')

!rm -r tmp/openlogo

%cd working

%cd yolov3
!find ./traindata -name "*.jpg" | awk 'BEGIN{}{print substr($1,3,length($1)-2)}' > data/coco1cls.txt
!find ./validdata -name "*.jpg" | awk 'BEGIN{}{print substr($1,3,length($1)-2)}' > data/coco1cls_valid.txt
!echo "logo" > data/coco.names

!echo "classes=1" > data/coco1cls.data

!echo "train=data/coco1cls.txt" >> data/coco1cls.data

!echo "valid=data/coco1cls_valid.txt" >> data/coco1cls.data

!echo "names=data/coco.names" >> data/coco1cls.data

!cat data/coco1cls.data
#load last training weights

%cd weights

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MXGxpNdIeBaQILlhgPousOzykWOgUmau' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MXGxpNdIeBaQILlhgPousOzykWOgUmau" -O last_1cls_1clscfg.pt && rm -rf /tmp/cookies.txt

%cd ..
!python3 train.py --resume --epochs 200 --data coco1cls.data --cfg yolov3-spp-1cls.cfg --batch 10 --nosave --name 1cls_1clscfg
!ls data/samples

!rm -r data/samples

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VHBpbr3btX3G-G89gGGVC-gqyyNWpjj3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VHBpbr3btX3G-G89gGGVC-gqyyNWpjj3" -O sample.zip && rm -rf /tmp/cookies.txt

!unzip sample.zip -d data/samples
!python detect.py --weights weights/last_1cls_1clscfg.pt --cfg cfg/yolov3-spp-1cls.cfg --names data/coco.names