import pandas as pd
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.shape
test.shape
subtrain=train.iloc[:,1:]
subresponse=train.iloc[:,:1]
from sklearn.model_selection import train_test_split
x_train,x_vaild,y_train,y_vaild=train_test_split(subtrain,subresponse,test_size=.2,stratify=subresponse,random_state=42)
import matplotlib.pyplot as plt
import matplotlib.image as mpi
def imageplots(k):
    viewed_image=x_train.iloc[k,:].as_matrix()
    #28*28= 784 which is the number of pixels used per image
    # So like the previous kernal it really is pushing 1D to 2D
    # Although im not sure if thats usually how its done
    im=viewed_image.reshape((28,28))
    plt.title(y_train.iloc[k])
    
    return plt.imshow(im,cmap='binary')
imageplots(0)
imageplots(1)
imageplots(5)
from sklearn.preprocessing import MinMaxScaler
scal=MinMaxScaler()
dtr=scal.fit_transform(x_train)
x_train_s=pd.DataFrame(dtr,columns=x_train.columns)
dvr=scal.transform(x_vaild)
x_vaild_s=pd.DataFrame(dvr,columns=x_vaild.columns)
dtt=scal.transform(test)
test_s=pd.DataFrame(dtt,columns=test.columns)
#from sklearn import svm

#from sklearn.model_selection import GridSearchCV

#parmer_grid={'C':range(1,10),'gamma':[0.01,0.0001,1/785,0.00001]}

#sv=svm.SVC()

#vt=GridSearchCV(sv,parmer_grid,cv=10)

#vt.fit(x_train_s,y_train.values.ravel())

#vt.score(x_vaild_s,y_vaild)