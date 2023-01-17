import numpy as np
from os import listdir
import pandas as pd
import os
os.listdir("../input/plantdisease/plantvillage/PlantVillage")
list_of_dir = os.listdir("../input/plantdisease/plantvillage/PlantVillage")
Req = []
for i in list_of_dir:
    if i.split("_")[0] == 'Tomato' : Req.append(i)
Req
def create(location , var):
    import cv2
    import glob
    X_data = []
    files = glob.glob("../input/plantdisease/plantvillage/PlantVillage/"+location+"/*.JPG")
    for myFile in files:
        #print(myFile)
        image = cv2.imread (myFile)
        image = cv2.resize(image, (64,64)) 
        X_data.append (image)
    print('X_data shape:', np.array(X_data).shape)
    numpy_entry = np.array(X_data).reshape(np.array(X_data).shape[0] , np.array(X_data).shape[1]*np.array(X_data).shape[2]*np.array(X_data).shape[3])
    print('numpy_entry shape:', numpy_entry.shape)
    df=pd.DataFrame(data=numpy_entry[0:,0:],index=[i for i in range(numpy_entry.shape[0])],columns=['Pixel '+str(i) for i in range(numpy_entry.shape[1])])
    df['Category'] = var
    return df
master_df=pd.DataFrame(columns=['Pixel '+str(i) for i in range(64*64*3)])
var = 0
for i in Req:
    df = create(i,var)
    frames=[master_df , df]
    master_df = pd.concat(frames)
    print(" Done For ",i," With category Value ",var)
    print("Master Data Frame   ",master_df.shape)
    var = var + 1
master_df.head()
master_df.shape
master_df.to_csv('Tomato_Pixel_DataSet.csv') 
master_df.columns
master_df['Category'].unique()