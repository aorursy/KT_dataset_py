# importing necessary libraries 

import csv

import numpy as np

import pandas as pd

from sklearn import datasets 

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

  

# loading the dataset 

#please enter the correct loaction and file name of the input dataset

df=pd.read_csv("/kaggle/input/career-con-2019/X_train.csv")

#please enter the correct loaction and file name of the output dataset

df2=pd.read_csv("/kaggle/input/career-con-2019/y_train.csv")

print("Raw data dimensions:",df.shape,df2.shape)



#replace poditive and negative infinite values with nan and then drop all nan values present in the data

df = df.replace([np.inf, -np.inf], np.nan)

df2=df2.replace([np.inf, -np.inf], np.nan)

df=df.dropna(how='any',axis=0)

df2=df2.dropna(how='any',axis=0)



print("Data after filtering:",df.shape,df2.shape)
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(df.iloc[:,:].corr(), annot=True, linewidths=.5, fmt= '.1f')
def modify(df):

    dfnew=df[["series_id","measurement_number","orientation_X","orientation_Y","orientation_Z","orientation_W","angular_velocity_X","angular_velocity_Y","angular_velocity_Z","linear_acceleration_X","linear_acceleration_Y","linear_acceleration_Z"]]

    temp1= dfnew.groupby("series_id").mean()

    temp2= dfnew.groupby("series_id").median()

    temp3= dfnew.groupby("series_id").quantile()

    temp4=dfnew.groupby("series_id").std()

    

    df_train_mean=temp1[["orientation_X","orientation_Y","orientation_Z","orientation_W","angular_velocity_X","angular_velocity_Y","angular_velocity_Z","linear_acceleration_X","linear_acceleration_Y","linear_acceleration_Z"]]

    df_train_median=temp2[["orientation_X","orientation_Y","orientation_Z","orientation_W","angular_velocity_X","angular_velocity_Y","angular_velocity_Z","linear_acceleration_X","linear_acceleration_Y","linear_acceleration_Z"]]

    df_train_quantile=temp3[["orientation_X","orientation_Y","orientation_Z","orientation_W","angular_velocity_X","angular_velocity_Y","angular_velocity_Z","linear_acceleration_X","linear_acceleration_Y","linear_acceleration_Z"]]

    df_train_std=temp4[["orientation_X","orientation_Y","orientation_Z","orientation_W","angular_velocity_X","angular_velocity_Y","angular_velocity_Z","linear_acceleration_X","linear_acceleration_Y","linear_acceleration_Z"]]

    

    df_train_mean.rename(columns={"orientation_X":"orientation_X_mean","orientation_Y":"orientation_Y_mean","orientation_Z":"orientation_Z_mean","orientation_W":"orientation_W_mean","angular_velocity_X":"angular_velocity_X_mean","angular_velocity_Y":"angular_velocity_Y_mean","angular_velocity_Z":"angular_velocity_Z_mean","linear_acceleration_X":"linear_acceleration_X_mean","linear_acceleration_Y":"linear_acceleration_Y_mean","linear_acceleration_Z":"linear_acceleration_Z_mean"})

    df_train_median.rename(columns={"orientation_X":"orientation_X_median","orientation_Y":"orientation_Y_median","orientation_Z":"orientation_Z_median","orientation_W":"orientation_W_median","angular_velocity_X":"angular_velocity_X_median","angular_velocity_Y":"angular_velocity_Y_median","angular_velocity_Z":"angular_velocity_Z_median","linear_acceleration_X":"linear_acceleration_X_median","linear_acceleration_Y":"linear_acceleration_Y_median","linear_acceleration_Z":"linear_acceleration_Z_median"})

    df_train_quantile.rename(columns={"orientation_X":"orientation_X_q","orientation_Y":"orientation_Y_q","orientation_Z":"orientation_Z_q","orientation_W":"orientation_W_q","angular_velocity_X":"angular_velocity_X_q","angular_velocity_Y":"angular_velocity_Y_q","angular_velocity_Z":"angular_velocity_Z_q","linear_acceleration_X":"linear_acceleration_X_q","linear_acceleration_Y":"linear_acceleration_Y_q","linear_acceleration_Z":"linear_acceleration_Z_q"})

    df_train_std.rename(columns={"orientation_X":"orientation_X_std","orientation_Y":"orientation_Y_std","orientation_Z":"orientation_Z_std","orientation_W":"orientation_W_std","angular_velocity_X":"angular_velocity_X_std","angular_velocity_Y":"angular_velocity_Y_std","angular_velocity_Z":"angular_velocity_Z_std","linear_acceleration_X":"linear_acceleration_X_std","linear_acceleration_Y":"linear_acceleration_Y_std","linear_acceleration_Z":"linear_acceleration_Z_std"})

    

    X=pd.concat([df_train_mean,df_train_median,df_train_quantile,df_train_std], axis=1)

    

    return X



X=modify(df)



print("Shape of data X with new features:",X.shape)
#eliminate group_id and series_id colunms

y_train=df2[["surface"]].to_numpy()

#convert string vals in surface to interger index using this list of all classes

labels=['carpet','concrete','fine_concrete','hard_tiles','hard_tiles_large_space','soft_pvc','soft_tiles','tiled','wood']

Y= np.zeros(len(y_train))

for i in range(0,len(y_train)):

    Y[i]=labels.index(y_train[i])

Ydf=pd.DataFrame(Y)

print("New shape of Y:",Ydf.shape)
from sklearn import datasets, linear_model

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC

lasso = linear_model.Lasso()

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(max_depth=16)

cv_results = cross_validate(rf, X, Ydf, cv=3)

sorted(cv_results.keys())

scores = cross_validate(rf, X, Ydf, cv=3, scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)

print(scores['test_neg_mean_squared_error'])

print(scores['train_r2'])
import pandas as pd

# split X_train

samples = 20

start_x = X.shape[0] - samples

X_train, X_test = X.iloc[:start_x], X.iloc[start_x:]



# split y_train

start_y = Ydf.shape[0] - samples

y_train, y_test = Ydf.iloc[:start_y], Ydf.iloc[start_y:]



print("Dimensions of the training and testing data:",X_train.shape,X_test.shape,y_train.shape,y_test.shape)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score

import numpy as np

from sklearn import metrics



lin_regression = LogisticRegression()

lin_regression.fit(X_train,y_train)



train_pred_lr = lin_regression.predict(X_train)



print("Logistic Regression training accuracy=",metrics.accuracy_score(y_train,train_pred_lr)) 
test_pred_lr=lin_regression.predict(X_test)

print("Logistic Regression tesing accuracy=",metrics.accuracy_score(y_test,test_pred_lr))
from sklearn.tree import DecisionTreeClassifier



dmodel=DecisionTreeClassifier(splitter="random",max_depth=15).fit(X_train,y_train)

train_pred=dmodel.predict(X_train)

print("Decision tree training accuracy=",metrics.accuracy_score(y_train,train_pred)) 
ypred=dmodel.predict(X_test)

print("Decision tree testing accuracy=",metrics.accuracy_score(ypred,y_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(max_depth=16)

rf.fit(X_train,y_train);

rf_pred=rf.predict(X_train)

print("Random forest training accuracy=",metrics.accuracy_score(rf_pred,y_train))
rf_pred2=rf.predict(X_test)

print("Random forest testing accuracy=",metrics.accuracy_score(rf_pred2,y_test))
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(max_iter=1000)

clf.fit(X_train, y_train)

mlpy_pred=clf.predict(X_train)

print("MLP training accuracy=",metrics.accuracy_score(mlpy_pred,y_train))

mlpy_pred2=clf.predict(X_test)

print(mlpy_pred2.shape)

print("MLP testing accuracy=",metrics.accuracy_score(mlpy_pred2,y_test))
Y = pd.DataFrame(columns=['series_id','surface'])

k=start_y

for i in range(0,rf_pred2.shape[0]):

    j=int(rf_pred2[i])

    Y.loc[i] = k,labels[j]

    k=k+1

print("Final predicted values of surfaces for test data:")

print (Y.to_string(index=False))