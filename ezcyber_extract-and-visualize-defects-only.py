# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np # linear algebra

import pandas as pd

import numpy as np  

import pandas as pd 

import cv2

import matplotlib.pyplot as plt 

from pathlib import Path

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", 101)

plt.rcParams["font.size"] = 9

train_path = Path("//kaggle/input//severstal-steel-defect-detection//train_images//")

train_csv="//kaggle/input//severstal-steel-defect-detection//train.csv"
def name_and_mask(start_idx):

    col = start_idx

    img_names = train['ImageId'][col:col+1].values

    

    labels = train.iloc[col:col+1, 3]

    mask = np.zeros((256, 1600, 4), dtype=np.uint8)



    for idx, label in enumerate(labels.values):

        if label is not np.nan:

            mask_label = np.zeros(1600*256, dtype=np.uint8)

            label = label.split(" ")

            positions = map(int, label[0::2])

            length = map(int, label[1::2])

            for pos, le in zip(positions, length):

                mask_label[pos:(pos+le)] = 1

            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')

    return img_names[0], mask



def show_mask_image(col):

    name, mask = name_and_mask(col)

    img = cv2.imread(str(train_path / name))

    fig, ax = plt.subplots(figsize=(10, 15))



    for ch in range(4):

        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i in range(0, len(contours)):

            cv2.polylines(img, contours[i], True, palet[ch], 2)

    ax.set_title(name)

    ax.imshow(img)

    plt.show()

    return 



def hist(idx_hist,clusterId,hist_st,hist_ed) : 

    plt.rcParams['figure.figsize'] = [15,5] 

    for idx in idx_hist : 

        name, mask=name_and_mask(idx) 

        img = cv2.imread(str(train_path / name))

        hist1 = cv2.calcHist([img],[0],None,[256],[hist_st,hist_ed])

        

        fig = plt.figure()

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        for ch in range(4):

            contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            for i in range(0, len(contours)):

                cv2.polylines(img, contours[i], True, palet[ch], 2) 

        ax1.imshow(img,'gray') ,ax1.set_title(clusterId+":"+str(idx))

        ax2.plot(hist1,color='r'),ax2.set_title(name)

        plt.show() 

    return 





def Defect_Extract(idx_hist,clusterId) :  

    crop_gap=4

    defect = []

    cnt = 0

    for idx in idx_hist : 

        cnt+=1 

        name, mask=name_and_mask(idx)

        img = cv2.imread(str(train_path / name))  

        contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        

        contours_new=[]

        for i in  range(0, len(contours)) : 

            

            contours_01=contours[i].reshape(int(contours[i].size/2),2)

            x1_new = contours_01[:,1].min()-crop_gap 

            y1_new = contours_01[:,0].min()-crop_gap

            x2_new = contours_01[:,1].max()+crop_gap

            y2_new = contours_01[:,0].max()+crop_gap

            

            if (x2_new-x1_new)*(y2_new-y1_new) > 1000 :                                

                contours_new+=[contours_01]  

                 

#        fig, axs = plt.subplots(ncols=len(contours_new),figsize=(20,5)) 

        for i in range(0, len(contours_new)):

            mask_crop = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)

            contour=contours_new[i]

            cv2.fillPoly(mask_crop, [contour]  , (255)) 

            res = cv2.bitwise_and(img,img,mask = mask_crop)

#            print(contour[:,1].min()-crop_gap ,contour[:,1].max()+crop_gap,contour[:,0].min()-crop_gap,contour[:,0].max()+crop_gap)

      

            x1 = contour[:,1].min()-crop_gap 

            y1 = contour[:,0].min()-crop_gap

            x2 = contour[:,1].max()+crop_gap

            y2 = contour[:,0].max()+crop_gap

    

            if x1 <  0 :

                x1 = 0

            if y1  < 0 :

                y1 = 0                                     

                      

            crop=res[x1 :x2,y1:y2]   

            if cnt > 200: 

                crop=[]

            defect.append([clusterId,name[:9],i,x2-x1,y2-y1,(x2-x1)*(y2-y1),x1,x2,y1,y2,crop])

#            cv2.imwrite((save_path+clusterId+"_"+name[:9]+"_"+str(i)+".jpg"),crop)

#

#            if len(contours_new)==1 :

#                axs.imshow(crop)

#                axs.set_title(clusterId+"_"+name[:9]+"_"+str(i))

#            else :

#                axs[i].imshow(crop)

#                axs[i].set_title(clusterId+"_"+name[:9]+"_"+str(i))

#        plt.show()

    return defect











def defect_info() : 

    cluster_1_PF=pd.DataFrame(cluster_1_defect[:,0:10],columns=Defect_columns_str[:-1])

    cluster_2_PF=pd.DataFrame(cluster_2_defect[:,0:10],columns=Defect_columns_str[:-1])

    cluster_3_PF=pd.DataFrame(cluster_3_defect[:,0:10],columns=Defect_columns_str[:-1])

    cluster_4_PF=pd.DataFrame(cluster_4_defect[:,0:10],columns=Defect_columns_str[:-1])

    

    Defect_all=pd.concat([cluster_1_PF,cluster_2_PF,cluster_3_PF,cluster_4_PF]) 

    Defect_all=Defect_all.astype({'Seq': 'int32','Size_X': 'int32','Size_Y': 'int32','Size': 'int32','x1':'int32','x2':'int32','y1':'int32','y2':'int32'})

    Defect_all['X_Y_Ratio']=Defect_all['Size_X']/Defect_all['Size_Y']

    Defect_all_summary= Defect_all.groupby(['ClusterId','ImageId'])['Size','Size_X','Size_Y','X_Y_Ratio'].agg (['sum','count','mean','max'])

        

    Size = pd.DataFrame() 

    Size_X = pd.DataFrame() 

    Size_Y = pd.DataFrame() 

    

    Size = pd.DataFrame()     

    for idx_x in ['class_1','class_2','class_3','class_4'] : 

        DF_Cluser=Defect_all_summary['Size'].loc[idx_x] 

        DF_Cluser['ClusterId']=idx_x 

        DF_Cluser['Type']='Size'

        Size=Size.append(DF_Cluser)  

    

    Size_X = pd.DataFrame()     

    for idx_x in ['class_1','class_2','class_3','class_4'] : 

        DF_Cluser=Defect_all_summary['Size_X'].loc[idx_x] 

        DF_Cluser['ClusterId']=idx_x 

        DF_Cluser['Type']='Size_X'

        Size_X=Size_X.append(DF_Cluser)  

    

    Size_Y = pd.DataFrame()     

    for idx_x in ['class_1','class_2','class_3','class_4'] : 

        DF_Cluser=Defect_all_summary['Size_Y'].loc[idx_x] 

        DF_Cluser['ClusterId']=idx_x 

        DF_Cluser['Type']='Size_Y'

        Size_Y=Size_Y.append(DF_Cluser)  

    

    X_Y_Ratio = pd.DataFrame()     

    for idx_x in ['class_1','class_2','class_3','class_4'] : 

        DF_Cluser=Defect_all_summary['X_Y_Ratio'].loc[idx_x] 

        DF_Cluser['ClusterId']=idx_x 

        DF_Cluser['Type']='X_Y_Ratio'

        X_Y_Ratio=X_Y_Ratio.append(DF_Cluser)  



          

    sns.set(font_scale=1)     

    fig, axes = plt.subplots(nrows=4, ncols=4,figsize=(5*4,15))   

    for idx_z,idx_no,idx_seq in zip([Size,Size_X,Size_Y, X_Y_Ratio],range(4),['Size','Size_X','Size_Y','X_Y_Ratio']) : 

        for idx_x,idx_y in zip(['count','mean','sum','max'],range(4)) :

            sns.boxplot(y=idx_x, x='ClusterId' ,  data=idx_z ,    width=0.5,     palette="colorblind",ax=axes[idx_no][idx_y],showfliers=False).set(title=idx_seq+' : '+ idx_x , xlabel='',ylabel=''  )  

    plt.show()  

    

    Size.columns=['Size_sum','Size_count','Size_mean','Size_max','ClusterId','Type']

    Size_Y.columns=['Size_Y_sum','Size_Y_count','Size_Y_mean','Size_Y_max','ClusterId','Type']

    Size_X.columns=['Size_X_sum','Size_X_count','Size_X_mean','Size_X_max','ClusterId','Type']

    X_Y_Ratio.columns=['X_Y_Ratio_sum','X_Y_Ratio_count','X_Y_Ratio_mean','X_Y_Ratio_max','ClusterId','Type']

    

    Defect_analy_PD=pd.merge(pd.merge(pd.merge(Size, Size_Y, how='inner',on=["ImageId","ClusterId"]),Size_X, how='inner',on=["ImageId","ClusterId"]),X_Y_Ratio, how='inner',on=["ImageId","ClusterId"])

    

    return Defect_analy_PD 



  

def defect_distribution(defect_arr):    

    defect_all_DF = pd.DataFrame(defect_arr,columns=Defect_columns_str)

    defect_all_DF['Seq_01']=defect_all_DF['Seq']+1

    Defect_DF=defect_all_DF

    Defect_CNT_DF = Defect_DF.groupby(['ClusterId','ImageId'])['Seq_01'].count().astype('uint8').to_frame(name='Count').reset_index()

     



    for idx_x, Ncount,mcluster in zip(Defect_CNT_DF['ImageId'], Defect_CNT_DF['Count'],Defect_CNT_DF['ClusterId']) :    

        print(mcluster,idx_x,Ncount)

        if Ncount > 1  : 

            fig, axs = plt.subplots(nrows=2, ncols=Ncount,figsize=(4*Ncount,3)) 

            for idx_y in range(len(defect_all_DF.query("ImageId=='"+idx_x+"'")['Seq'])) : 

                pd_result=defect_all_DF.query("ImageId=='"+idx_x+"' & Seq == "+ str(idx_y) )['image']         

                hist1 = cv2.calcHist(pd_result,[0],None,[256],[hist_st,hist_ed])

                axs[1][idx_y].plot(hist1,color='r') 

                axs[0][idx_y].imshow(pd_result.values[0])

                

        else : 

            fig, axs = plt.subplots(ncols=2,figsize=(4*2,2)) 

            pd_result=defect_all_DF.query("ImageId=='"+idx_x+"'")['image']         

            hist1 = cv2.calcHist(pd_result,[0],None,[256],[hist_st,hist_ed])

            axs[1].plot(hist1,color='r') 

            axs[0].imshow(pd_result.values[0]) 

        plt.show()          



def show_image(img_pd):

    a=0

    plt.subplots(figsize=(10, 15))  

    

    for i in img_pd : 

        ax = plt.subplot(5,2, 1)

        print(str(train_path / img_names))

        img=cv2.imread(str(train_path / img_names))

        

#        ax.imshow(img)

        a +=1

        print(a)

        plt.show()



def train_data() :  

    train_raw = pd.read_csv(train_csv)

    train_raw['idx']=train_raw.index.to_series() 

    

    train_raw['ClassId'] = train_raw['ImageId_ClassId'].str[-1:]

    train_raw['ImageId'] = train_raw['ImageId_ClassId'].str[:-2]

    train_raw['defect'] = train_raw['EncodedPixels'].notnull()

    train_defect_cnt = train_raw.groupby(['ImageId'])['defect'].sum().astype('uint8').to_frame(name='NumDef').reset_index()

    train_raw = train_raw[['ImageId','ClassId','defect','EncodedPixels','idx']]

    

    train_no_defect = pd.DataFrame(train_defect_cnt.query('NumDef ==0').iloc[:,0])

    no_defect_01 = pd.merge(train_raw, train_no_defect, on='ImageId')

    return train_raw,no_defect_01
Defect_columns_str=['ClusterId','ImageId','Seq','Size_X','Size_Y','Size','x1','x2','y1','y2','image']



palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]

       

train, no_defect_01 = train_data()

idx_no_defect  = no_defect_01.query("ClassId == '1'")['idx']

idx_class_1=train.query('ClassId=="1" and defect==True')['idx']

idx_class_2=train.query('ClassId=="2" and defect==True')['idx'] 

idx_class_3=train.query('ClassId=="3" and defect==True')['idx']

idx_class_4=train.query('ClassId=="4" and defect==True')['idx']



print(idx_no_defect.size,idx_class_1.size,idx_class_2.size,idx_class_3.size,idx_class_4.size)
index_partial=-1



cluster_1_defect = np.array(Defect_Extract(idx_class_1[:index_partial],'class_1')) 

cluster_2_defect = np.array(Defect_Extract(idx_class_2[:index_partial],'class_2')) 

cluster_3_defect = np.array(Defect_Extract(idx_class_3[:index_partial],'class_3'))

cluster_4_defect  =np.array(Defect_Extract(idx_class_4[:index_partial],'class_4')) 
print(cluster_1_defect.shape,cluster_2_defect.shape,cluster_3_defect.shape,cluster_4_defect.shape)
defect_all_01=defect_info() 
hist_st=2

hist_ed=254



index_partial=30 



defect_distribution(np.array(Defect_Extract(idx_class_1[:index_partial],'class_1')))  

defect_distribution(np.array(Defect_Extract(idx_class_2[:index_partial],'class_2'))) 

defect_distribution(np.array(Defect_Extract(idx_class_3[:index_partial],'class_3'))) 

defect_distribution(np.array(Defect_Extract(idx_class_4[:index_partial],'class_4'))) 
# Visualize histgram 

index=3

hist(idx_no_defect[:index],'idx_no_defect',24,255)    

hist(idx_class_1[:index],'idx_class_1',24,255)   

hist(idx_class_2[:index],'idx_class_2',24,255)   

hist(idx_class_3[:index],'idx_class_3',24,255)   

hist(idx_class_4[:index],'idx_class_4',24,255)   
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD



# Generate dummy data

import numpy as np

x_train = np.random.random((1000, 20))

y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

x_test = np.random.random((100, 20))

y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)



model = Sequential()

# Dense(64) is a fully-connected layer with 64 hidden units.

# in the first layer, you must specify the expected input data shape:

# here, 20-dimensional vectors.

model.add(Dense(64, activation='relu', input_dim=20))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])



model.fit(x_train, y_train,

          epochs=20,

          batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)



model
model_json = model.to_json()

with open("model.json", "w") as json_file : 

    json_file.write(model_json)
model.save_weights("model.h5")

print("Saved model to disk")
from glob2 import glob

defect_list=glob('*.*')
defect_list