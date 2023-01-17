import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import time
print('Loading Libraries')
ds=pd.read_csv('../input/train.csv')
print("Data shape of training data set is :",ds.shape)
ds.columns
plot5=ds[ds.label==5]#it will pick all rows in data set which represent digit 5
plot5=plot5.drop('label',axis=1)
indx=plot5.index.max()#used this for row selection, i am just going to plot the data from last row and now i know its index value
ax5=plt.imshow(plot5.loc[indx].values.reshape(28,28),cmap='gray')
ds_noLabel=ds.drop('label',axis=1) #Features dataset
ds_label=ds['label']#Target dataset
x_train,x_test,y_train,y_test=train_test_split(ds_noLabel,ds_label,train_size=0.7,random_state=23)
#Using linear kernel
svcLin=svm.SVC(kernel='linear',random_state=42) #initiate the linear kernel
svcLin.fit(x_train[:8000],y_train[:8000])# fit the model on 8000 records
scoreLin=svcLin.score(x_test,y_test) #check the score

#Using RBF kernel 
svcRBF=svm.SVC(random_state=42)#initiate the RBF kernel
svcRBF.fit(x_train[:8000],y_train[:8000])# fit the model on 8000 records
scoreRBF=svcRBF.score(x_test,y_test)#check the score

svcPol=svm.SVC(kernel='poly',random_state=42,degree=2)#initiate the polynomial kernel with degree 2
svcPol.fit(x_train[:8000],y_train[:8000])# fit the model on 8000 records
scorePol=svcPol.score(x_test,y_test)#check the score

score_df=pd.DataFrame({"RBF Kernel score" : scoreRBF,"Linear kernel score":scoreLin,"Polynomial kernel degree 2":scorePol},index=["Score"])
#score_df is data set created to compare the score of each kernel
score_df
test_df=pd.read_csv('../input/test.csv')
test_df.columns
y_pred_final=svcPol.predict(test_df)#Making prediction
sub_pred=pd.DataFrame({"ImageId":test_df.index+1,"Label":y_pred_final})
sub_pred.to_csv('Sub.csv',index=False)
