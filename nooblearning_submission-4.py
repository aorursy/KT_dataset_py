import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pydicom

import os

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestRegressor

from skimage import morphology

from skimage import measure

from skimage.transform import resize

import tensorflow as tf

from sklearn.cluster import KMeans

import matplotlib.patches as patches
train_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
len(train_csv)
train_csv.info()
'''unique_ids=train_csv['Patient'].unique()

week_x=[]

fvc_y=[]

percent=[]

for id in unique_ids:

    week=np.array(train_csv[train_csv['Patient']==id]['Weeks'])

    fvc=np.array(train_csv[train_csv['Patient']==id]['FVC'])

    per=np.array(train_csv[train_csv['Patient']==id]['Percent'])

    week_x.append(week)

    fvc_y.append(fvc)

    percent.append(per)

    

unique_train=pd.DataFrame(train_csv['Patient'].unique(),columns=['Patient'])

unique_train['week_x']=week_x

unique_train['fvc_y']=fvc_y

unique_train['percent']=percent

unique_train.head()'''
'''X=unique_train['week_x'][0]

Y=unique_train['fvc_y'][0]'''
'''def using_poly_reg(X,Y,degree=3):

    poly_features=PolynomialFeatures(degree=degree,include_bias=False)

    x_poly=poly_features.fit_transform(X[:,np.newaxis])



    lin_reg=LinearRegression()

    lin_reg.fit(x_poly,Y)



    x_test=np.arange(-12,133)[:,np.newaxis]

    x_test_poly=poly_features.fit_transform(x_test)

    plt.plot(x_test,lin_reg.predict(x_test_poly))

    plt.plot(X,Y)

    #plt.ylim(0,6400)

    #plt.xlim(-12,133)

    plt.grid(True)'''
'''using_poly_reg(X,Y,degree=3)  #here we can customize the degree of the polynomial so it is better

                              #by the way if degree=len(X)-1 then it is same as interpolation'''
'''train_csv['healthy_person_FVC']=(train_csv['FVC']/(train_csv['Percent']/100)).round()'''
'''train_csv'''
'''healthy_fvc_info=train_csv.groupby(['Age','Sex','SmokingStatus'])['healthy_person_FVC'].mean().round()'''
'''plt.plot(healthy_fvc_info[:,'Male','Ex-smoker'],label='male ex smoker')

plt.plot(healthy_fvc_info[:,'Male','Never smoked'],label='male never smoked')

plt.plot(healthy_fvc_info[:,'Male','Currently smokes'],label='male currently smokes')



plt.plot(healthy_fvc_info[:,'Female','Ex-smoker'],label='female ex smoker')

plt.plot(healthy_fvc_info[:,'Female','Never smoked'],label='female never smoked')

plt.plot(healthy_fvc_info[:,'Female','Currently smokes'],label='female currently smokes')



plt.title('healthy fvc related to age,sex and smoking status')

plt.legend()

plt.grid(True)'''
'''def RForestRegressor(x,y):

    reg=RandomForestRegressor(n_estimators=50)

    reg.fit(x[:,np.newaxis],y)

    x_test=np.arange(0,100)

    y_test=reg.predict(x_test[:,np.newaxis])

    plt.plot(x_test,y_test,label='predicted')

    plt.plot(x,y,label='real')

    plt.grid(True)

    plt.legend()'''
'''#x=np.array(healthy_fvc_info[:,'Male','Ex-smoker'].index)

#y=np.array(healthy_fvc_info[:,'Male','Ex-smoker'].values)

x=np.array(healthy_fvc_info[:,'Male','Never smoked'].index)

y=np.array(healthy_fvc_info[:,'Male','Never smoked'].values)



RForestRegressor(x,y)'''

'''X=unique_train['week_x'][0]

Y=unique_train['fvc_y'][0]

RForestRegressor(X,Y)'''
'''age=train_csv.groupby('Patient')['Age'].unique()'''
'''for item in age:

    if len(item)==1:

        continue

    else:

        print(item.index)'''
'''SS=train_csv.groupby('Patient')['SmokingStatus'].unique()

for item in SS:

    if len(item)==1:

        continue

    else:

        print(item)'''
'''sex=[]

for id in unique_train['Patient']:

    sex.append(train_csv[train_csv['Patient']==id]['Sex'].unique()[0])

    

unique_train['sex']=sex'''
'''age=[]

for id in unique_train['Patient']:

    age.append(train_csv[train_csv['Patient']==id]['Age'].unique()[0])

    

unique_train['age']=age'''
'''ss=[]

for id in unique_train['Patient']:

    ss.append(train_csv[train_csv['Patient']==id]['SmokingStatus'].unique()[0])

    

unique_train['smoking-status']=ss'''
'''unique_train'''
'''from sklearn.cluster import KMeans



lung=pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/26.dcm')

image=lung.pixel_array

X = image.reshape(-1,1)

#X=image



#good_init=np.array([[-2048],[-1000],[892],[-177],[190]])

#kmeans = KMeans(n_clusters=8,init=good_init,n_init=1).fit(X)



kmeans = KMeans(n_clusters=6).fit(X)



segmented_img = kmeans.cluster_centers_[kmeans.labels_]

segmented_img = segmented_img.reshape(image.shape)

plt.imshow(segmented_img)'''
def fitter(img):

    

    row_size= img.shape[0]

    col_size = img.shape[1]

    

    mean = np.mean(img)

    std = np.std(img)

    img = img-mean

    img = img/std

    # Find the average pixel value near the lungs

    # to renormalize washed out images

    middle = img[int(col_size/4):int(col_size/4*3),int(row_size/4):int(row_size/4*3)] 

    mean = np.mean(middle)  

    max = np.max(img)

    min = np.min(img)

    # To improve threshold finding, I'm moving the 

    # underflow and overflow on the pixel spectrum

    img[img==max]=mean

    img[img==min]=mean

    #

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    #

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))

    return kmeans



lung=pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/26.dcm')

image=lung.pixel_array*lung.RescaleSlope+lung.RescaleIntercept

kmeans=fitter(image)
def make_lungmask(img,kmeans,display=False):

    image=img

    row_size= img.shape[0]

    col_size = img.shape[1]

    mean = np.mean(img)

    std = np.std(img)

    img = img-mean

    img = img/std

    # Find the average pixel value near the lungs

    # to renormalize washed out images

    middle = img[int(col_size/4):int(col_size/4*3),int(row_size/4):int(row_size/4*3)] 

    mean = np.mean(middle)  

    max = np.max(img)

    min = np.min(img)

    # To improve threshold finding, I'm moving the 

    # underflow and overflow on the pixel spectrum

    img[img==max]=mean

    img[img==min]=mean

    #

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    #

    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)

    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image



    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  

    # We don't want to accidentally clip the lung.



    eroded = morphology.erosion(thresh_img,np.ones([5,5]))

    dilation = morphology.dilation(eroded,np.ones([8,8]))



    labels = measure.label(dilation) # Different labels are displayed in different colors

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)

    good_labels = []

    #for prop in regions:

    #    b = prop.bbox

    #    if (abs((b[2]+b[0])/2-(row_size/2))<100) and ( (abs((b[3]+b[1])/2-(col_size/4))<110) or (abs((b[3]+b[1])/2-(col_size/4)*3)<110) ):

    #        good_labels.append(prop.label)

            

    for prop in regions:

        b = prop.bbox

        lung_row=abs((b[2]+b[0])/2-(row_size/2))

        left_lung_col=abs((b[3]+b[1])/2-(col_size/4))

        right_lung_col=abs((b[3]+b[1])/2-(col_size/4)*3)

        

        if lung_row<100 and (left_lung_col<110 or right_lung_col<110):

            good_labels.append(prop.label)

            

    mask = np.ndarray([row_size,col_size],dtype=np.int8)

    mask[:] = 0



    #

    #  After just the lungs are left, we do another large dilation

    #  in order to fill in and out the lung mask 

    #

    for N in good_labels:

        mask = mask + np.where(labels==N,1,0)

    mask = morphology.dilation(mask,np.ones([8,8])) # one last dilation



    if (display):

        fig, ax = plt.subplots(3, 2, figsize=[12, 12])

        ax[0, 0].set_title("Original")

        ax[0, 0].imshow(img, cmap='gray')

        ax[0, 0].axis('off')

        ax[0, 1].set_title("Threshold")

        ax[0, 1].imshow(thresh_img, cmap='gray')

        ax[0, 1].axis('off')

        ax[1, 0].set_title("After Erosion and Dilation")

        ax[1, 0].imshow(dilation, cmap='gray')

        ax[1, 0].axis('off')

        ax[1, 1].set_title("Color Labels")

        ax[1, 1].imshow(labels)

        ax[1, 1].axis('off')

        ax[2, 0].set_title("Final Mask")

        ax[2, 0].imshow(mask, cmap='gray')

        ax[2, 0].axis('off')

        ax[2, 1].set_title("Apply Mask on Original")

        ax[2, 1].imshow(mask*img, cmap='gray')

        ax[2, 1].axis('off')

        

        plt.show()

        

    air=[]

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            if mask[i][j]==1:

                air.append(image[i][j])

    if len(air)==0 :

        air_percent=0.0

    else:

        air_percent=abs((sum(air)/len(air))/10).round(4)

    return mask,air_percent
#id='ID00011637202177653955184'

#path='../input/osic-pulmonary-fibrosis-progression/train/'+id+'/' 

#filenames=os.listdir(path) 

#fileno=int(len(filenames)/2)

#img=pydicom.dcmread(path+str(fileno)+'.dcm')

#make_lungmask(img,kmeans,display=True)
show_plots=True



if show_plots:

    fig=plt.figure(figsize=(20,20)) 

col=14

row=14 

i=1 

air_percent_dict={}

for id in train_csv['Patient'].unique(): 

    path='../input/osic-pulmonary-fibrosis-progression/train/'+id+'/' 

    filenames=os.listdir(path) 

    fileno=int(len(filenames)/2)

    for item in filenames:

        number=int(item.split('.')[0])

        if number==fileno:

            break

        else:

            continue

    try:

        lung=pydicom.dcmread(path+item) 

        image=lung.pixel_array*lung.RescaleSlope+lung.RescaleIntercept

        mask,air_percent=make_lungmask(image,kmeans,display=False)

        air_percent_dict[id]=air_percent

        if show_plots:

            fig.add_subplot(row,col,i) 

            plt.title(air_percent)

            plt.imshow(mask,cmap='gray')

            plt.grid(False)

            plt.axis(False)

    except: 

        air_percent_dict[id]=np.nan 

    i=i+1 
#here i define a thresold for excluding values with clipped lungs :

for key in air_percent_dict:

    if air_percent_dict[key]<35.0:

        air_percent_dict[key]=np.nan

        print(key)
train=train_csv[['Patient', 'Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']]

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()#sex

train.iloc[:,5]=lb.fit_transform(train.iloc[:,5])

lb2=LabelEncoder()#ss

train.iloc[:,6]=lb2.fit_transform(train.iloc[:,6])
lung_percent=[]

for id in train['Patient']:

    lung_percent.append(float(air_percent_dict[id]))

train['lung percent']=lung_percent

train=train.dropna()

train
train.info()
len(train)
test=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test.iloc[:,5]=lb.transform(test.iloc[:,5])

test.iloc[:,6]=lb2.transform(test.iloc[:,6])
air_percent_dict={}

for id in test['Patient'].unique(): 

    path='../input/osic-pulmonary-fibrosis-progression/test/'+id+'/' 

    filenames=os.listdir(path) 

    fileno=int(len(filenames)/2)

    for item in filenames:

        number=int(item.split('.')[0])

        if number==fileno:

            break

        else:

            continue

    try:

        lung=pydicom.dcmread(path+item) 

        image=lung.pixel_array*lung.RescaleSlope+lung.RescaleIntercept

        mask,air_percent=make_lungmask(image,kmeans,display=False)

        air_percent_dict[id]=air_percent

    except: 

        print(id)
air_percent_dict
lung_percent_test=[]

for id in test['Patient']:

    lung_percent_test.append(float(air_percent_dict[id]))

test['lung percent']=lung_percent_test

test
'''def healthy_fvc_predictor(age,sex,smoking_status):

    x=np.array(healthy_fvc_info[:,sex,smoking_status].index)

    y=np.array(healthy_fvc_info[:,sex,smoking_status].values)

    reg=RandomForestRegressor(n_estimators=50)

    reg.fit(x[:,np.newaxis],y)

    return reg.predict([[age]])'''
def metric(actual_fvc, predicted_fvc, confidence, return_values = False):

    """

    Calculates the modified Laplace Log Likelihood score for this competition.

    """

    sd_clipped = np.maximum(confidence, 70)

    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)

    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)



    if return_values:

        return metric

    else:

        return np.mean(metric)

def RForestRegressor(x,y):

    reg=RandomForestRegressor(n_estimators=400)

    reg.fit(np.array(x),np.array(y))

    return reg



x=train[['Weeks','Percent','lung percent','SmokingStatus','Sex','Age']]

#x=train[['Percent','lung percent','Weeks','Sex']]

y=train['FVC']





from sklearn.model_selection import train_test_split

xtrain,xvalid,ytrain,yvalid=train_test_split(x,y,test_size=0.2)



percent_reg=RForestRegressor(xtrain,ytrain)



preds=percent_reg.predict(np.array(xvalid))

confidence=abs(preds-np.array(yvalid))

print(metric(np.array(yvalid),preds,confidence))
'''def neural(x):

    model=tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(100,activation='relu',input_shape=x.shape[1:]))

    model.add(tf.keras.layers.Dense(100,activation='relu'))

    model.add(tf.keras.layers.Dense(100,activation='relu'))

    model.add(tf.keras.layers.Dense(1))

    model.compile(loss='mse',optimizer='adam')

    return model



x=train[['Weeks','Percent','lung percent','SmokingStatus','Sex','Age']]

#x=train[['Percent','lung percent','Weeks','Sex']]

y=train['FVC']





from sklearn.model_selection import train_test_split

xtrain,xvalid,ytrain,yvalid=train_test_split(x,y,test_size=0.2)



model=neural(x)



class metric_callback(tf.keras.callbacks.Callback):

    def __init__(self,metrics,xvalid,yvalid):

        self.metrics=metrics

        self.xvalid=xvalid

        self.yvalid=yvalid

    def on_epoch_end(self,epoch,logs={}):

        preds=self.model.predict(np.array(self.xvalid))

        confidence=abs(preds-np.array(self.yvalid))

        metric=self.metrics(np.array(self.yvalid),preds,confidence)

        print('\r val metrics score :',metric)



history=model.fit(np.array(xtrain),np.array(ytrain),epochs=10,callbacks=[metric_callback(metric,xvalid,yvalid)])'''

        
def plot_fi(forest,X):

    importances = forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)

    indices = np.argsort(importances)[::-1]



    # Print the feature ranking

    print("Feature ranking:")



    for f in range(X.shape[1]):

        print("%d. feature : %s (%f)" % (f + 1, np.array(X.columns)[indices[f]], importances[indices[f]]))



    # Plot the impurity-based feature importances of the forest

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices],color="g", yerr=std[indices])

    plt.xticks(range(X.shape[1]),np.array(X.columns)[indices])

    plt.xlim([-1, X.shape[1]])

    plt.show()

    

plot_fi(percent_reg,x)
test_csv=test[['Patient', 'Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus','lung percent']]

weeks=np.arange(-12,134)

result={}

for id in test_csv['Patient'].unique():

    percent=np.array(test_csv[test_csv['Patient']==id]['Percent'])

    sex=np.array(test_csv[test_csv['Patient']==id]['Sex'])

    age=np.array(test_csv[test_csv['Patient']==id]['Age'])

    ss=np.array(test_csv[test_csv['Patient']==id]['SmokingStatus'])

    lp=np.array(test_csv[test_csv['Patient']==id]['lung percent'])

    percent=np.repeat(percent,len(weeks))

    sex=np.repeat(sex,len(weeks))

    age=np.repeat(age,len(weeks))

    ss=np.repeat(ss,len(weeks))

    lp=np.repeat(lp,len(weeks))

    x=np.concatenate([weeks[:,np.newaxis],percent[:,np.newaxis],lp[:,np.newaxis],ss[:,np.newaxis],sex[:,np.newaxis],age[:,np.newaxis]],axis=1)

    outcome=percent_reg.predict(x)

    result[id]=outcome
ans_df_list=[]

for id in result:

    ID=np.repeat(id,len(weeks))

    ans=np.concatenate([ID[:,np.newaxis],weeks[:,np.newaxis],result[id][:,np.newaxis]],axis=1)

    ans=pd.DataFrame(ans)

    ans_df_list.append(ans)
submit=pd.concat(ans_df_list,ignore_index=True)

submit.columns=['Patient','Weeks','FVC']



submit['FVC']=submit['FVC'].astype(float)

submit['Weeks']=submit['Weeks'].astype(int)
submit
test_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
'''healthy_fvc_dict={}

for i in range(len(test_csv)):

    hfvc=healthy_fvc_predictor(test_csv.iloc[i,4],test_csv.iloc[i,5],test_csv.iloc[i,6])

    healthy_fvc_dict[test_csv.iloc[i,0]]=hfvc.ravel()[0]'''
'''hfvc_list=[]

for i in range(len(submit)):

    hfvc_list.append(healthy_fvc_dict[submit.iloc[i,0]])

    

submit['healthy_fvc']=hfvc_list

submit['Percent']=submit['Percent'].astype(float)

submit['FVC']=(submit['healthy_fvc']*submit['Percent'])/100'''
confidence_dict={}

for id in submit['Patient'].unique():

    real=float(test_csv[test_csv['Patient']==id]['FVC'])

    

    week=int(test_csv[test_csv['Patient']==id]['Weeks'])

    

    predicted=float(submit[(submit['Patient']==id) & (submit['Weeks']==week) ]['FVC'])

    

    confidence_dict[id]=abs(real-predicted)

    

    #confidence_dict[id]=np.std(np.array(submit[submit['Patient']==id]['FVC']).astype(float))
confidence_dict
confidence=[]

for i in range(len(submit)):

    confidence.append(confidence_dict[submit.iloc[i,0]])

submit['Confidence']=confidence
submit['Patient']=submit['Patient']+'_'+(submit['Weeks'].astype(str))

submit.drop(['Weeks'],axis=1,inplace=True)

submit.columns=['Patient_Week','FVC','Confidence']
submit.to_csv('submission.csv',index=False)