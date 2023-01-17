import numpy as np 

import matplotlib.pyplot as plt 

import os 

import cv2

%matplotlib inline
Data_dir = "E:\PetImages"

categories = ["cat" ,"dog"]
for category in categories:
    
    path = os.path.join(Data_dir,category) #path to the folder dog and cat or vice versa
    for img in os.listdir(path):
        
        img_arr = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE) # read the photos in the folder dog or cat
    
train_data = []

def create_train_data():
     for category in categories:
            path = os.path.join(Data_dir,category)
            class_num = categories.index(category)
            
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
                    new_arr = cv2.resize(img_arr,(50,50))
                    train_data.append([new_arr, class_num])
                except:
                    pass
                    
create_train_data()
len(train_data)  # Note: train_data no is list not an array 
import random 

random.shuffle(train_data)
for sample in train_data[:10]:
    print(sample[1])
X , Y = train_data[0]
print("({}, {})".format(len(X),Y))  # we don't use len(y) because y = 1
x =[]
y= []
for features , labels in train_data:
    x.append(features)
    y.append(labels)
    
x = np.array(x).reshape(-1,50,50,1)
import pickle 

pickle_out = open("x.pickle","wb")

pickle.dump(x,pickle_out) #dump x to pickle_out

pickle_out.close()



pickle_out = open("y.pickle","wb")

pickle.dump(y,pickle_out) #dump y to pickle_out

pickle_out.close()
#read your file 

pickle_in = open("x.pickle","rb")

x = pickle.load(pickle_in) #load pickle file
x.shape