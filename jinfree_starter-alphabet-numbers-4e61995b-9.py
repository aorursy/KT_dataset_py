import cv2
import os
import os.path
import csv
import pathlib
import random
import copy
Dir = '../input/alphabetnumbers_trainingdataset/'
imageDir1 = [Dir+'normal_simple/', 15, 'ROI.csv', 'ROI2.csv']
imageDir2 = [Dir+'normal_complex/', 0, 'ROI.csv', 'ROI.csv']
imageDir3 = [Dir+'high_simple/', 0, 'ROI.csv', 'ROI.csv']
imageDir4 = [Dir+'high_complex/', 15, 'ROI.csv', 'ROI2.csv']
imageDir = []
imageDir.append(imageDir1)
imageDir.append(imageDir2)
imageDir.append(imageDir3)
imageDir.append(imageDir4)
#print(imageDir)

def classdir(character):
    dir = "../input/alphabetnumbers_trainingdataset/Image_Classification_Dataset/" + character
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    return dir
def ROI(imagePath):
    split1 = imagePath[0].split('/')
    #print(split1)
    roiDir = split1[0]+'/'
    for i in range(1, len(split1)-1):
        roiDir += split1[i]+'/'
    split2 = split1[-1].split('/')
    name = split2[-1].split('.')
    name = name[0]
    name = name[-2:]
    roimat=[]
    if imagePath[1] == 15:
        if int(name) < 15:
            f = open(roiDir+imagePath[2], 'r', encoding='utf-8')
        else:
            f = open(roiDir+imagePath[3], 'r', encoding='utf-8')
    else:
        f = open(roiDir+imagePath[2], 'r', encoding='utf-8')
    rdr = csv.reader(f)
    ii=0
    for line in rdr:
        for i in range(1, 5):
            line[i] = int(line[i])
        l1 = line[1]
        l2 = line[2]
        l3 = l1+line[3]
        l4 = l2+line[4]
        line[3] = l3
        line[4] = l4
        #print(line)
        roimat.append(line)
        line1 = copy.deepcopy(line)
        line1[1] = l1 + random.randrange(1, 4)
        line1[2] = l2 + random.randrange(1, 4)
        line1[3] = l3 + random.randrange(1, 4)
        line1[4] = l4 + random.randrange(1, 4)
        #print(line1)
        roimat.append(line1)
        line2 = copy.deepcopy(line)
        line2[1] = l1 - random.randrange(1, 4)
        line2[2] = l2 - random.randrange(1, 4)
        line2[3] = l3 - random.randrange(1, 4)
        line2[4] = l4 - random.randrange(1, 4)
        #print(line2)
        roimat.append(line2)
        line3 = copy.deepcopy(line)
        line3[1] = l1 + random.randrange(4, 7)
        line3[2] = l2 + random.randrange(4, 7)
        line3[3] = l3 + random.randrange(4, 7)
        line3[4] = l4 + random.randrange(4, 7)
        #print(line1)
        roimat.append(line1)
        line4 = copy.deepcopy(line)
        line4[1] = l1 - random.randrange(4, 7)
        line4[2] = l2 - random.randrange(4, 7)
        line4[3] = l3 - random.randrange(4, 7)
        line4[4] = l4 - random.randrange(4, 7)
        #print(line2)
        roimat.append(line2)

        ii = ii+1
    #print(roimat)
    #cv2.waitKey()
    return roimat

def saveimage(imagePath):
    image = cv2.imread(imagePath[0])
    roimat = ROI(imagePath)
    for roi in roimat:
        #print(roi)
        savedir = classdir(roi[0])
        #print(savedir)
        x1 = int(roi[1])
        y1 = int(roi[2])
        x2 = int(roi[3])
        y2 = int(roi[4])
        subimage = image[y1:y2, x1:x2]
        file_list = os.listdir(savedir)
        if len(file_list) == 0:
            name = savedir + '/' + "{0:0>3}".format(0) + ".bmp"
            cv2.imwrite(name, subimage)
        else:
            file_list.sort()
            final_list = file_list[-1]
            final_num = int(final_list[:-4])
            num = final_num + 1
            name = savedir + '/' + "{0:0>3}".format(num) + ".bmp"
            cv2.imwrite(name, subimage)

image_path_list = []
valid_image_extensions = [".bmp"]
valid_image_extensions = [item.lower() for item in valid_image_extensions]
for Dir in imageDir:
    #print(Dir)
    for file in os.listdir(Dir[0]):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        appendlist = [os.path.join(Dir[0], file)]
        appendlist.extend(Dir[1:])
        #print(appendlist)
        image_path_list.append(appendlist)

#print(image_path_list)
ii = 0
for imagePath in image_path_list:
    ii = ii+1
    #print(imagePath)
    saveimage(imagePath)