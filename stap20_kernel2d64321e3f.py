import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import tensorflow
import os
import cv2
import imageio
import skimage
import skimage.io
import skimage.transform
import itertools
import shutil
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

%matplotlib inline
# define location of dataset
shenzhen_path = '../input/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png'
montgomery_path = '../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png'
shenzhen_image_list = os.listdir(shenzhen_path)
shenzhen_image_list = [s for s in shenzhen_image_list if s != 'Thumbs.db']
montgomery_image_list = os.listdir(montgomery_path)
montgomery_image_list = [s for s in montgomery_image_list if s != 'Thumbs.db']
# plot first few images
for i,img_name in enumerate(shenzhen_image_list):
    if(i==9):
        break
    plt.subplot(330 + 1 + i)
    filename = shenzhen_path+"/"+ img_name
    # load image pixels
    image = cv2.imread(filename)
    # plot raw pixel data
    plt.imshow(image)
# show the figure
plt.show()
# plot first few images
for i,img_name in enumerate(montgomery_image_list):
    if(i==9):
        break
    plt.subplot(330 + 1 + i)
    filename = montgomery_path+"/"+ img_name
    # load image pixels
    image = cv2.imread(filename)
    # plot raw pixel data
    plt.imshow(image)
# show the figure
plt.show()
df_shenzhen = pd.DataFrame(shenzhen_image_list, columns=['image_id'])
df_montgomery = pd.DataFrame(montgomery_image_list, columns=['image_id'])

df_shenzhen.reset_index(inplace=True, drop=True)
df_montgomery.reset_index(inplace=True, drop=True)

print(df_shenzhen.shape)
print(df_montgomery.shape)
df_shenzhen.head()
def extract_y(x):
    target = int(x[-5])
    if target == 0:
        return 'Normal'
    if target == 1:
        return 'Tuberculosis'
df_shenzhen['target'] = df_shenzhen['image_id'].apply(extract_y)

df_montgomery['target'] = df_montgomery['image_id'].apply(extract_y)
df_shenzhen['target'].value_counts()
df_montgomery['target'].value_counts()
df_shenzhen.head()
df_montgomery.head()
median_list = []
mean_list = []
std_list = []
var_list = []
ent_list = []
sobel_list = []
label_list = []
histo_range_list = []
histo_skew_list = []
histo_kurtosis_list = []
for i,img_name in enumerate(shenzhen_image_list):
    if(i%50==0):
        print(i)
    filename = shenzhen_path+"/"+ img_name
    image = cv2.imread(filename,0)
    median = np.median(image)
    mean = np.mean(image)
    std = np.std(image)
    var = np.var(image)
    label = int(img_name[-5])
    
    hist1 = np.histogram(image, bins=50,density=True)
    data = hist1[0]
    
    histo_range = np.ptp(data)
    histo_skew = skew(data)
    histo_kurtosis = kurtosis(data)
    ent = (data*np.log(np.abs(data))).sum()

    gray_image = image
    gray_image = cv2.equalizeHist(gray_image)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

    sobel = cv2.resize(grad_norm,(5,5))
    sobel = sobel.ravel()
    
    median_list.append(median)
    mean_list.append(mean)
    std_list.append(std)
    var_list.append(var)
    histo_range_list.append(histo_range)
    histo_skew_list.append(histo_skew)
    histo_kurtosis_list.append(histo_kurtosis)
    ent_list.append(ent)
    sobel_list.append(sobel)
    label_list.append(label)
for i,img_name in enumerate(montgomery_image_list):
    if(i%50==0):
        print(i)
    filename = montgomery_path+"/"+ img_name
    image = cv2.imread(filename,0)
    median = np.median(image)
    mean = np.mean(image)
    std = np.std(image)
    var = np.var(image)
    label = int(img_name[-5])
    
    hist1 = np.histogram(image, bins=50,density=True)
    data = hist1[0]
    
    histo_range = np.ptp(data)
    histo_skew = skew(data)
    histo_kurtosis = kurtosis(data)
    ent = (data*np.log(np.abs(data))).sum()
    
    gray_image = image
    gray_image = cv2.equalizeHist(gray_image)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

    sobel = cv2.resize(grad_norm,(5,5))
    sobel = sobel.ravel()
    
    median_list.append(median)
    mean_list.append(mean)
    std_list.append(std)
    var_list.append(var)
    histo_range_list.append(histo_range)
    histo_skew_list.append(histo_skew)
    histo_kurtosis_list.append(histo_kurtosis)
    ent_list.append(ent)
    sobel_list.append(sobel)
    label_list.append(label)
feature_list = []
feature_list.append(label_list)
feature_list.append(ent_list)
feature_list.append(histo_kurtosis_list)
feature_list.append(histo_skew_list)
feature_list.append(histo_range_list)
feature_list.append(var_list)
feature_list.append(std_list)
feature_list.append(mean_list)
feature_list.append(median_list)

x = np.asarray(feature_list)
sobel = np.asarray(sobel_list)
x = np.rot90(x,-1)

np.random.shuffle(x)
y= x[:,-1]
x=x[:,:-1]
x=np.concatenate((x,sobel),axis=1)
print(x.shape)
print(y.shape)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=7)
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', svm.SVC()))
models.append(('Linear-SVM', svm.LinearSVC()))
models.append(('NB', GaussianNB()))
results_c = []
names_c = []

for name, model in models:
    # define how to split off validation data
    kfold = KFold(n_splits=10,shuffle=True, random_state=7)    
    # train the model
    cv_results = cross_val_score(model,x_train, y_train, cv=kfold, scoring='accuracy')    
    results_c.append(cv_results)
    names_c.append(name)
    print(name+": "+str(cv_results.mean())+"("+str(cv_results.std())+")")
LR = LogisticRegression()
LR.fit(x_train, y_train)
predictions = LR.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
predictions = RF.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
SVM = svm.SVC()
SVM.fit(x_train, y_train)
predictions = SVM.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
NB = GaussianNB()
NB.fit(x_train, y_train)
predictions = NB.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(32,32,32),alpha=0.5, activation='relu', solver='adam', max_iter=500)
mlp.fit(x_train,y_train)

predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=10, random_state=0)
clf.fit(x_train, y_train)

predict_test=clf.predict(x_test)

print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))

DTC = DecisionTreeClassifier(max_depth=10)
DTC.fit(x_train, y_train)
predictions = DTC.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=7)
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', svm.SVC()))
models.append(('Linear-SVM', svm.LinearSVC()))
models.append(('NB', GaussianNB()))
results_c = []
names_c = []

for name, model in models:
    # define how to split off validation data
    kfold = KFold(n_splits=10,shuffle=True, random_state=7)    
    # train the model
    cv_results = cross_val_score(model,x_train, y_train, cv=kfold, scoring='accuracy')    
    results_c.append(cv_results)
    names_c.append(name)
    print(name+": "+str(cv_results.mean())+"("+str(cv_results.std())+")")
LR = LogisticRegression()
LR.fit(x_train, y_train)
predictions = LR.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
predictions = RF.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
SVM = svm.SVC()
SVM.fit(x_train, y_train)
predictions = SVM.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
NB = GaussianNB()
NB.fit(x_train, y_train)
predictions = NB.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
mlp = MLPClassifier(hidden_layer_sizes=(32,32,32),alpha=0.5, activation='relu', solver='adam', max_iter=500)
mlp.fit(x_train,y_train)

predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)

print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))

clf = AdaBoostClassifier(n_estimators=10, random_state=0)
clf.fit(x_train, y_train)

predict_test=clf.predict(x_test)

print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
DTC = DecisionTreeClassifier(max_depth=10)
DTC.fit(x_train, y_train)
predictions = DTC.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
new_x = np.delete(x, [0,1,2,3], axis=1)
print(new_x.shape)
scaler = StandardScaler()
new_x = scaler.fit_transform(new_x)
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=7)
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', svm.SVC()))
models.append(('Linear-SVM', svm.LinearSVC()))
models.append(('NB', GaussianNB()))
results_c = []
names_c = []

for name, model in models:
    # define how to split off validation data
    kfold = KFold(n_splits=10,shuffle=True, random_state=7)    
    # train the model
    cv_results = cross_val_score(model,x_train, y_train, cv=kfold, scoring='accuracy')    
    results_c.append(cv_results)
    names_c.append(name)
    print(name+": "+str(cv_results.mean())+"("+str(cv_results.std())+")")
LR = LogisticRegression()
LR.fit(x_train, y_train)
predictions = LR.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
predictions = RF.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
SVM = svm.SVC()
SVM.fit(x_train, y_train)
predictions = SVM.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
NB = GaussianNB()
NB.fit(x_train, y_train)
predictions = NB.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))
mlp = MLPClassifier(hidden_layer_sizes=(32,32,32),alpha=0.5, activation='relu', solver='adam', max_iter=500)
mlp.fit(x_train,y_train)

predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)

print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
clf = AdaBoostClassifier(n_estimators=10, random_state=0)
clf.fit(x_train, y_train)

predict_test=clf.predict(x_test)

print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
DTC = DecisionTreeClassifier(max_depth=10)
DTC.fit(x_train, y_train)
predictions = DTC.predict(x_test)
# Accuracy Score 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print('The MSE is', mean_squared_error(y_test, predictions))