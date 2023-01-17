import matplotlib.pyplot as plt # plotting
from PIL import Image
import matplotlib.patches as patches
import matplotlib.image as IMG
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.io
def mat_to_boundbox(filename):
    input = scipy.io.loadmat(filename)['boxes']
    box_numbers = input.shape[-1]
    bx1_e1 = input[0][0][0][0][0][0]
    bx1_e2 = input[0][0][0][0][1][0]
    bx1_e3 = input[0][0][0][0][2][0]
    bx1_e4 = input[0][0][0][0][3][0]
    bx1 = np.array([bx1_e1,bx1_e2,bx1_e3,bx1_e4])
    output = np.array(bx1)
    output = np.flip(output,1)
    output = np.reshape(output,(1,4,2))
    if box_numbers == 2:
        bx2_e1 = input[0][1][0][0][0][0]
        bx2_e2 = input[0][1][0][0][1][0]
        bx2_e3 = input[0][1][0][0][2][0]
        bx2_e4 = input[0][1][0][0][3][0]
        bx2 = [bx2_e1,bx2_e2,bx2_e3,bx2_e4]
        output = np.array([bx1,bx2])
        output = np.flip(output,2)
   
    return output
    
def drawBoundbox(image,coordinate):
    plt.figure()
    plt.imshow(img)
    if coordinate.shape[0] >= 1:
        plt.scatter(x=[coordinate[0][0][0],coordinate[0][1][0],coordinate[0][2][0],coordinate[0][3][0]], y=[coordinate[0][0][1],coordinate[0][1][1],coordinate[0][2][1],coordinate[0][3][1]], c='r', s=20)
    if coordinate.shape[0] == 2:
        plt.scatter(x=[coordinate[1][0][0],coordinate[1][1][0],coordinate[1][2][0],coordinate[1][3][0]], y=[coordinate[1][0][1],coordinate[1][1][1],coordinate[1][2][1],coordinate[1][3][1]], c='g', s=20)
    plt.show()
def resize_edit_box(imagefile,coordinatefile):
    img = Image.open(imagefile)
    img1 = img.resize((512,512),0)
    img_shape = np.array(img).shape
    img = np.array(img1)
    
    x_scale = 512 / img_shape[1]
    y_scale = 512/ img_shape[0]
    
    coordinate = mat_to_boundbox(coordinatefile)
    
    if coordinate.shape[0] == 1:
        changed_x_coordinate = np.reshape(coordinate[:,:,0]*x_scale,(1,4,1))
        changed_y_coordinate = np.reshape(coordinate[:,:,1]*y_scale,(1,4,1))
    if coordinate.shape[0] == 2:
        changed_x_coordinate = np.reshape(coordinate[:,:,0]*x_scale,(2,4,1))
        changed_y_coordinate = np.reshape(coordinate[:,:,1]*y_scale,(2,4,1))
        
    coordinate = np.concatenate((changed_x_coordinate, changed_y_coordinate), axis=2)
    
    return img , coordinate

testImage_path = '/kaggle/input/hand_dataset/training_dataset/training_data/images/Buffy_110.jpg'
testAnnotation_path = '/kaggle/input/hand_dataset/training_dataset/training_data/annotations/Buffy_110.mat'

img, coordinate = resize_edit_box(testImage_path,testAnnotation_path)
drawBoundbox(img, coordinate)

def distance(point1,point2):
    distance = ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5
    return distance



def coordinate_reformat(coordinate):
    if coordinate.shape[0] >= 1:
        x_center = (coordinate[0,0,0] + coordinate[0,2,0])/2
        y_center = (coordinate[0,0,1] + coordinate[0,2,1])/2
        height = np.amax(coordinate[0,:,1]) - np.amin(coordinate[0,:,1])
        weight = np.amax(coordinate[0,:,0]) - np.amin(coordinate[0,:,0])
        output = np.array([x_center,y_center,height,weight]).reshape(1,4,1)
        
        
    if coordinate.shape[0] == 2:
        x_center_2 = (coordinate[1,0,0] + coordinate[1,2,0])/2
        y_center_2 = (coordinate[1,0,1] + coordinate[1,2,1])/2
        height_2 = np.amax(coordinate[1,:,1]) - np.amin(coordinate[1,:,1])
        weight_2 = np.amax(coordinate[1,:,0]) - np.amin(coordinate[1,:,0])
        temp = np.array([x_center_2,y_center_2,height_2,weight_2]).reshape(1,4,1)
        output = np.concatenate((output, temp), axis=0).reshape(2,4,1)
        
    return output


def coordinate_main(Newcoordinate):
    center_x = Newcoordinate[0,0,0]
    center_y = Newcoordinate[0,1,0]
    height = Newcoordinate[0,2,0]
    weidth = Newcoordinate[0,3,0]
    
    if Newcoordinate.shape[0] >= 1:
        a = np.array([center_x - (height/2 + weidth/2), center_y - (height/2 + weidth/2)])
        b = np.array([center_x - (height/2 - weidth/2), center_y - (height/2 - weidth/2)])
        c = np.array([center_x - (-height/2 - weidth/2), center_y - (-height/2 - weidth/2)])
        d = np.array([center_x - (-height/2 + weidth/2), center_y - (-height/2 + weidth/2)])
        temp1 = np.concatenate((a,b,c,d),axis = 0).reshape(1,4,2)
        
    
    if Newcoordinate.shape[0] == 2:
        center_x = Newcoordinate[1,0,0]
        center_y = Newcoordinate[1,1,0]
        height = Newcoordinate[1,2,0]
        weidth = Newcoordinate[1,3,0]
        a = np.array([center_x - (height/2 + weidth/2), center_y - (height/2 + weidth/2)])
        b = np.array([center_x - (height/2 - weidth/2), center_y - (height/2 - weidth/2)])
        c = np.array([center_x - (-height/2 - weidth/2), center_y - (-height/2 - weidth/2)])
        d = np.array([center_x - (-height/2 + weidth/2), center_y - (-height/2 + weidth/2)])
        temp2 = output = np.concatenate((a,b,c,d),axis = 0).reshape(1,4,2)
        
        output = np.concatenate((temp1, temp2), axis = 0)
        
    return output
all_coordinate = np.empty([2])
for dirname, _, filenames in os.walk('../input/hand_dataset/training_dataset/training_data/annotations'):
    for filename in filenames:
        if 'Buffy' in filename: 
            temp = coordinate_reformat(mat_to_boundbox(os.path.join(dirname, filename))).reshape(1,-1,4)
            if all_coordinate.shape[0] == 2:
                all_coordinate = temp
            else:
                all_coordinate = np.concatenate((temp,all_coordinate),axis = 1)
                
print(all_coordinate.shape)
plt.plot(all_coordinate[0,:,2],all_coordinate[0,:,3],'ro') 
plt.show()
from sklearn.cluster import KMeans
all_coordinate = all_coordinate.reshape(-1,4)
HW = all_coordinate[:,2:4]
kmeans = KMeans(n_clusters=1, random_state=0).fit(HW)
kmeans.cluster_centers_
anchor_DEFUALT = np.array([51.49931633, 49.05936578])
def create_output(imgFile, annoFile, grid = 16):
    img, coordinate = resize_edit_box(imgFile, annoFile)
    coordinate = coordinate_reformat(coordinate)
    step = img.shape[1]/grid
    output = np.zeros((5,grid,grid))
    boundBox_grid_x = int(coordinate[0,0,0]/step)
    if boundBox_grid_x == 16: boundBox_grid_x = 15
    boundBox_grid_y = int(coordinate[0,1,0]/step)
    if boundBox_grid_y == 16: boundBox_grid_y = 15
    boundBox_x = (coordinate[0,0,0]% step)/step
    boundBox_y = (coordinate[0,1,0]% step)/step
    boundBox_height = coordinate[0,2,0]/(anchor_DEFUALT[0] * grid)
    boundBox_weidth = coordinate[0,3,0]/(anchor_DEFUALT[1] * grid)
    temp = np.array([1, boundBox_x, boundBox_y, boundBox_height, boundBox_weidth]).reshape(5)
    output[:,boundBox_grid_x,boundBox_grid_y] = temp
    
    if coordinate.shape[0] == 2:
        boundBox_grid_x = int(coordinate[1,0,0]/step)
        if boundBox_grid_x == 16: boundBox_grid_x = 15
        boundBox_grid_y = int(coordinate[1,1,0]/step)
        if boundBox_grid_y == 16: boundBox_grid_y = 15
        boundBox_x = (coordinate[1,0,0]% step)/step
        boundBox_y = (coordinate[1,1,0]% step)/step
        boundBox_height = coordinate[1,2,0]/(anchor_DEFUALT[0] * grid)
        boundBox_weidth = coordinate[1,3,0]/(anchor_DEFUALT[1] * grid)
        temp1 = np.array([1, boundBox_x, boundBox_y, boundBox_height, boundBox_weidth]).reshape(5)
        output[:,boundBox_grid_x,boundBox_grid_y] = temp1
        
    return img, output
    
    
import csv

csv_head = ["no."]
for i in range(0,262144):
    csv_head.append("pixel " + str(i))
for i in range(0,256):
    csv_head.append("Objectness score(gride"+str(i)+")"), csv_head.append("Center x(gride"+str(i)+")"), csv_head.append("Center y(gride"+str(i)+")"), csv_head.append("Height(gride"+str(i)+")"), csv_head.append("Width(gride"+str(i)+")")
def create_dataset(images,annotations):
    counter = 0
    final_output = []
    temp = []
    for IMGdir, _, IMGfiles in os.walk(images):
        for ANNOdir, _, ANNOfiles in os.walk(annotations):
             for img_file in IMGfiles:
                    for anno_file in ANNOfiles:
                        if counter == 1:break
                        idx = img_file.index('.jpg')
                        if 'Buffy' in img_file and img_file[0:idx]+'.mat' == anno_file:
                            img = os.path.join(IMGdir, img_file)
                            ann = os.path.join(ANNOdir, anno_file)
                            img, out = create_output(img,ann)
                            img = np.reshape(img,(1,-1)).tolist()[0]
                            out = np.reshape(out,(1,-1),order='F').tolist()[0]
                            temp = [counter] + img + out
                            final_output.append(temp)
                            counter+=1
    return final_output
trainimg_dir = '../input/hand_dataset/training_dataset/training_data/images'
trainanno_dir = '../input/hand_dataset/training_dataset/training_data/annotations'
#train = create_dataset(trainimg_dir, trainanno_dir)
testimg_dir = '../input/hand_dataset/test_dataset/test_data/images'
testanno_dir = '../input/hand_dataset/test_dataset/test_data/annotations'
#test = create_dataset(testimg_dir, testanno_dir)

write = []
write.append(csv_head),write.append(train[0])
with open('dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(write)
import pandas as pd
df = pd.read_csv("./dataset.csv")

def generator(images, annotations , batch_size=32):
    """
    Yields the next training batch.
    """
    
   
    while True:
        counter = 0
        X_train = []
        y_train = []
        for IMGdir, _, IMGfiles in os.walk(images):
            for ANNOdir, _, ANNOfiles in os.walk(annotations):
                 for img_file in IMGfiles:
                        for anno_file in ANNOfiles:
                            if counter == batch_size: 
                                X_train = np.array(X_train)
                                y_train = np.array(y_train)
                                yield X_train, y_train
                                X_train = []
                                y_train = []
                                counter = 0
                            idx = img_file.index('.jpg')
                            if 'Buffy' in img_file and img_file[0:idx]+'.mat' == anno_file:
                                img = os.path.join(IMGdir, img_file)
                                ann = os.path.join(ANNOdir, anno_file)
                                img, out = create_output(img,ann)
                                X_train.append(img)
                                y_train.append(out)
                                counter+=1
train_generator = generator(trainimg_dir, trainanno_dir, batch_size=32)
test_generator =  generator(testimg_dir, testanno_dir, batch_size=32)
x, y = next(train_generator)
print(x.shape)