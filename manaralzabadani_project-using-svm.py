import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
os.makedirs('/kaggle/working/results/')
path ="../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/NonDemented/"
dirs = os.listdir(path)
for item in dirs:
                if os.path.isfile(path + item):
                    img = cv2.imread(path + item,cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (190, 340))
                    x,y,w,h = cv2.boundingRect(im)
                    ret,thresh = cv2.threshold(im,127,255,0)
                    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    
                    if int(M['m00'])==0:
                      cx=0
                        
                    else:
                      cx = int(M['m10']/M['m00'])
                        
                    if int(M['m00'])==0:
                      cy=0
                        
                    else:
                      cy = int(M['m01']/M['m00'])
                        
                    corners = cv2.goodFeaturesToTrack(im,35,0.01,10)
                    corners = np.int0(corners)
                    X = []
                    Y = []
                    for i in corners:
                        x,y = i.ravel()
                        x = x-cx
                        y = y-cy
                        X.append(x)
                        Y.append(y)
                     
                    tab = np.array([X,Y]) 
                    tab=tab.T
                    distance=[]
                    distance.append('NonDemented')
                    for i in range(tab.shape[0]):
                        dis = ((tab[i,0])**2+ (tab[i,1])**2)**0.5 
                        distance.append(dis)
            
                    #tab = np.array(distance)
                    
                    data=pd.DataFrame(distance,columns=['NonDemented']) 
                    d=data.to_csv('/kaggle/working/results/'+item+'.csv',index=False)
                    
                    
path ="../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/NonDemented/"
dirs = os.listdir(path)
for item in dirs:
                if os.path.isfile(path + item):
                    img = cv2.imread(path + item,cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (190, 340))
                    x,y,w,h = cv2.boundingRect(im)
                    ret,thresh = cv2.threshold(im,127,255,0)
                    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    
                    if int(M['m00'])==0:
                      cx=0
                        
                    else:
                      cx = int(M['m10']/M['m00'])
                        
                    if int(M['m00'])==0:
                      cy=0
                        
                    else:
                      cy = int(M['m01']/M['m00'])
                        
                    corners = cv2.goodFeaturesToTrack(im,35,0.01,10)
                    corners = np.int0(corners)
                    X = []
                    Y = []
                    for i in corners:
                        x,y = i.ravel()
                        x = x-cx
                        y = y-cy
                        X.append(x)
                        Y.append(y)
                     
                    tab = np.array([X,Y]) 
                    tab=tab.T
                    distance=[]
                    distance.append('NonDemented')
                    for i in range(tab.shape[0]):
                        dis = ((tab[i,0])**2+ (tab[i,1])**2)**0.5 
                        distance.append(dis)
            
                    #tab = np.array(distance)
                    
                    data=pd.DataFrame(distance,columns=['NonDemented']) 
                    d=data.to_csv('/kaggle/working/results/'+item+'.csv',index=False)
                   



path ="../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/VeryMildDemented/"
dirs = os.listdir(path)
for item in dirs:
                if os.path.isfile(path + item):
                    img = cv2.imread(path + item,cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (190, 340))
                    x,y,w,h = cv2.boundingRect(im)
                    ret,thresh = cv2.threshold(im,127,255,0)
                    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    
                    if int(M['m00'])==0:
                      cx=0
                        
                    else:
                      cx = int(M['m10']/M['m00'])
                        
                    if int(M['m00'])==0:
                      cy=0
                        
                    else:
                      cy = int(M['m01']/M['m00'])
                        
                    corners = cv2.goodFeaturesToTrack(im,35,0.01,10)
                    corners = np.int0(corners)
                    X = []
                    Y = []
                    for i in corners:
                        x,y = i.ravel()
                        x = x-cx
                        y = y-cy
                        X.append(x)
                        Y.append(y)
                     
                    tab = np.array([X,Y]) 
                    tab=tab.T
                    distance=[]
                    distance.append('VeryMildDemented')
                    for i in range(tab.shape[0]):
                        dis = ((tab[i,0])**2+ (tab[i,1])**2)**0.5 
                        distance.append(dis)
            
                    #tab = np.array(distance) 
                    
                    data=pd.DataFrame(distance,columns=['VeryMildDemented']) 
                    d=data.to_csv('/kaggle/working/results/'+item+'.csv',index=False)
                    
                    
path ="../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/VeryMildDemented/"
dirs = os.listdir(path)
for item in dirs:
                if os.path.isfile(path + item):
                    img = cv2.imread(path + item,cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (190, 340))
                    x,y,w,h = cv2.boundingRect(im)
                    ret,thresh = cv2.threshold(im,127,255,0)
                    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    
                    if int(M['m00'])==0:
                      cx=0
                        
                    else:
                      cx = int(M['m10']/M['m00'])
                        
                    if int(M['m00'])==0:
                      cy=0
                        
                    else:
                      cy = int(M['m01']/M['m00'])
                        
                    corners = cv2.goodFeaturesToTrack(im,35,0.01,10)
                    corners = np.int0(corners)
                    X = []
                    Y = []
                    for i in corners:
                        x,y = i.ravel()
                        x = x-cx
                        y = y-cy
                        X.append(x)
                        Y.append(y)
                     
                    tab = np.array([X,Y]) 
                    tab=tab.T
                    distance=[]
                    distance.append('VeryMildDemented')
                    for i in range(tab.shape[0]):
                        dis = ((tab[i,0])**2+ (tab[i,1])**2)**0.5 
                        distance.append(dis)
            
                    #tab = np.array(distance) 
                    
                    data=pd.DataFrame(distance,columns=['VeryMildDemented']) 
                    d=data.to_csv('/kaggle/working/results/'+item+'.csv',index=False)                    
                    

path ="../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/ModerateDemented/"
dirs = os.listdir(path)
for item in dirs:
                if os.path.isfile(path + item):
                    img = cv2.imread(path + item,cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (190, 340))
                    x,y,w,h = cv2.boundingRect(im)
                    ret,thresh = cv2.threshold(im,127,255,0)
                    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    
                    if int(M['m00'])==0:
                      cx=0
                        
                    else:
                      cx = int(M['m10']/M['m00'])
                        
                    if int(M['m00'])==0:
                      cy=0
                        
                    else:
                      cy = int(M['m01']/M['m00'])
                        
                    corners = cv2.goodFeaturesToTrack(im,35,0.01,10)
                    corners = np.int0(corners)
                    X = []
                    Y = []
                    for i in corners:
                        x,y = i.ravel()
                        x = x-cx
                        y = y-cy
                        X.append(x)
                        Y.append(y)
                     
                    tab = np.array([X,Y]) 
                    tab=tab.T
                    distance=[]
                    distance.append('ModerateDemented')
                    for i in range(tab.shape[0]):
                        dis = ((tab[i,0])**2+ (tab[i,1])**2)**0.5 
                        distance.append(dis)
            
                    #tab = np.array(distance)
                    
                    data=pd.DataFrame(distance,columns=['ModerateDemented']) 
                    d=data.to_csv('/kaggle/working/results/'+item+'.csv',index=False)
                    
                    
                    
path ="../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/ModerateDemented/"
dirs = os.listdir(path)
for item in dirs:
                if os.path.isfile(path + item):
                    img = cv2.imread(path + item,cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (190, 340))
                    x,y,w,h = cv2.boundingRect(im)
                    ret,thresh = cv2.threshold(im,127,255,0)
                    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    
                    if int(M['m00'])==0:
                      cx=0
                        
                    else:
                      cx = int(M['m10']/M['m00'])
                        
                    if int(M['m00'])==0:
                      cy=0
                        
                    else:
                      cy = int(M['m01']/M['m00'])
                        
                    corners = cv2.goodFeaturesToTrack(im,35,0.01,10)
                    corners = np.int0(corners)
                    X = []
                    Y = []
                    for i in corners:
                        x,y = i.ravel()
                        x = x-cx
                        y = y-cy
                        X.append(x)
                        Y.append(y)
                     
                    tab = np.array([X,Y]) 
                    tab=tab.T
                    distance=[]
                    distance.append('ModerateDemented')
                    for i in range(tab.shape[0]):
                        dis = ((tab[i,0])**2+ (tab[i,1])**2)**0.5 
                        distance.append(dis)
            
                    #tab = np.array(distance)
                    
                    data=pd.DataFrame(distance,columns=['ModerateDemented']) 
                    d=data.to_csv('/kaggle/working/results/'+item+'.csv',index=False)
path ="../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/MildDemented/"
dirs = os.listdir(path)
for item in dirs:
                if os.path.isfile(path + item):
                    img = cv2.imread(path + item,cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (190, 340))
                    x,y,w,h = cv2.boundingRect(im)
                    ret,thresh = cv2.threshold(im,127,255,0)
                    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    
                    if int(M['m00'])==0:
                      cx=0
                        
                    else:
                      cx = int(M['m10']/M['m00'])
                        
                    if int(M['m00'])==0:
                      cy=0
                        
                    else:
                      cy = int(M['m01']/M['m00'])
                        
                    corners = cv2.goodFeaturesToTrack(im,35,0.01,10)
                    corners = np.int0(corners)
                    X = []
                    Y = []
                    for i in corners:
                        x,y = i.ravel()
                        x = x-cx
                        y = y-cy
                        X.append(x)
                        Y.append(y)
                     
                    tab = np.array([X,Y]) 
                    tab=tab.T
                    distance=[]
                    distance.append('MildDemented')
                    for i in range(tab.shape[0]):
                        dis = ((tab[i,0])**2+ (tab[i,1])**2)**0.5 
                        distance.append(dis)
            
                    #tab = np.array(distance) 
                    
                    data=pd.DataFrame(distance,columns=['MildDemented']) 
                    d=data.to_csv('/kaggle/working/results/'+item+'.csv',index=False)
                   
                
path ="../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/MildDemented/"
dirs = os.listdir(path)
for item in dirs:
                if os.path.isfile(path + item):
                    img = cv2.imread(path + item,cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (190, 340))
                    x,y,w,h = cv2.boundingRect(im)
                    ret,thresh = cv2.threshold(im,127,255,0)
                    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    
                    if int(M['m00'])==0:
                      cx=0
                        
                    else:
                      cx = int(M['m10']/M['m00'])
                        
                    if int(M['m00'])==0:
                      cy=0
                        
                    else:
                      cy = int(M['m01']/M['m00'])
                        
                    corners = cv2.goodFeaturesToTrack(im,35,0.01,10)
                    corners = np.int0(corners)
                    X = []
                    Y = []
                    for i in corners:
                        x,y = i.ravel()
                        x = x-cx
                        y = y-cy
                        X.append(x)
                        Y.append(y)
                     
                    tab = np.array([X,Y]) 
                    tab=tab.T
                    distance=[]
                    distance.append('MildDemented')
                    for i in range(tab.shape[0]):
                        dis = ((tab[i,0])**2+ (tab[i,1])**2)**0.5 
                        distance.append(dis)
            
                    #tab = np.array(distance) 
                    
                    data=pd.DataFrame(distance,columns=['MildDemented']) 
                    d=data.to_csv('/kaggle/working/results/'+item+'.csv',index=False)
import glob
import pandas as pd
import os
path='/kaggle/working/results/'

os.chdir(path)
extension = 'csv'
#files= os.listdir(path)
files = [i for i in glob.glob('*.{}'.format(extension))]
result = pd.concat([pd.read_csv(path+f) for f in files],axis=1)
d=result.to_csv('/kaggle/working/Features.csv')

import pandas as pd
d=pd.read_csv('/kaggle/working/Features.csv')
d.head()
d=d.T
d1=d.to_csv('/kaggle/working/FeaturesSVM.csv')
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
data = pd.read_csv('/kaggle/working/FeaturesSVM.csv')
data = data.drop([0], axis=0)
data = data.drop(['Unnamed: 0'], axis=1)
data.head()
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
# convert to cateogry dtype
data['0'] = data['0'].astype('category')
# convert to category codes
data['0'] = data['0'].cat.codes
continuous=['1','2','3','4','5','6','7','8','9',
           '10','11','12','13','14','15','16','17','18','19',
           '20','21','22','23','24','25','26','27','28','29',
           '30','31','32','33','34','35']

scaler = MinMaxScaler(feature_range=(0, 4))
for var in continuous:
    data[var] = data[var].astype('float64')
    data[var] = scaler.fit_transform(data[var].values.reshape(-1,1))
data.head()
X = data.drop('0', axis=1)  
y = data['0'] 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
y_test
X_test=X_test.dropna(axis='rows',how='any')
X_train=X_train.dropna(axis='rows',how='any')
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
y_pred
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print("Accuracy: {}%".format(svclassifier.score(X_test, y_test) * 100 ))