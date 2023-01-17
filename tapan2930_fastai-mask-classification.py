from fastai import *

from fastai.vision import *

from bs4 import BeautifulSoup as bs
%reload_ext autoreload

%autoreload 2

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
def my_len(*args):

    return [len(x) for x in args]
import shutil



path_cpy = "/kaggle/input/face-mask-detection"

path_pst = "/kaggle/working/data"



shutil.copytree(path_cpy,path_pst)
def gen_box(obj):

#     Getting bounding box coordinates

    xmin = int(obj.find('xmin').text)

    xmax = int(obj.find('xmax').text)

    ymin = int(obj.find('ymin').text)

    ymax = int(obj.find('ymax').text)

    

    return [xmin, ymin, xmax, ymax]



def gen_label(obj):

    if obj.find('name').text == "with_mask":

        return "mask" #     for without_mask label set to 1

    elif obj.find('name').text == "mask_weared_incorrect":

        return "incorrect_mask"     # for mask_weared_incorrect label set to 2

    return "no_mask" #     for without_mask label set to 0
coordinates = []

labels = []

name= []

size = []

def gen_list(path):

    xml_list  = list(sorted(os.listdir(path+"/annotations")))

    img_list = list(sorted(os.listdir(path+"/images")))

    total = len(img_list)

    assert len(img_list) == len(xml_list)

    counter = 0

    for xml in xml_list:

        with open(path+f"/annotations/{xml}") as file:

            data = file.read()

            soup = bs(data, "xml")

            xml_objs = soup.find_all('object')

            counter +=1

            for i in xml_objs:

                coordinates.append(gen_box(i)) 

                labels.append(gen_label(i))

                name.append(soup.find('filename').text)

        print(f"file processed {counter} / {total}...")    

                

gen_list(path_pst)                
my_len(name,labels,coordinates)
labels_set = set(labels)

labels_set
# Creating folder

def gen_folder(label_set, path):

    import shutil

    os.mkdir(path)

    for l in label_set:

            os.mkdir(path+f"/{l}")

            
path = "/kaggle/working/data2"

gen_folder(labels_set,path)
#cropping images

import PIL

for i in range(len(name)):

    im = PIL.Image.open(path_pst+f"/images/{name[i]}") 

    w,h = im.size

#     print(path_pst+f"/images/{name[i]}")

#     print(coordinates[i])

#     [xmin,ymax,xmax ,ymin]

#     coordinates[i] = [coordinates[i][0],coordinates[i][1],w-coordinates[i][2], h-coordinates[i][3] ]

    coordinates[i] = [coordinates[i][0],coordinates[i][1],coordinates[i][2],coordinates[i][3]]

#     print(coordinates[i])

    im = im.crop(box = coordinates[i])

    

    if labels[i] == "mask":

        shutil.copy(path_pst+f"/images/{name[i]}", path+f"/mask/{name[i]}")

        im = im.save(path+f"/mask/{i}.png")

    elif labels[i] == "incorrect_mask":

        im = im.save(path+f"/incorrect_mask/{i}.png")

    else:

        im = im.save(path+f"/no_mask/{i}.png")

        

    print(f"Processed {i}th image...") 
data = ImageDataBunch.from_folder(path = path,size = 64, valid_pct = 0.2).normalize()
data.show_batch(rows = 3, figsize=(7,6))
data.classes
model = create_cnn(data, models.resnet50, metrics = error_rate)
model.fit_one_cycle(5)
model.lr_find()
model.recorder.plot()
model.save('/kaggle/working/stage-1')
model.load('/kaggle/working/stage-1');
model.unfreeze()

model.fit_one_cycle(2, max_lr = slice(1e-3, 1e-02))
model.save('/kaggle/working/stage-2')
model.lr_find()
model.recorder.plot()
model.fit_one_cycle(2, max_lr = slice(1e-3, 1e-2))
model.load("/kaggle/working/stage-2")
# model.export("/kaggle/working/model.pkl")



