import os

import time

import math



import numpy as np

import pandas as pd



import matplotlib.image as mpimg



from sklearn.decomposition.pca import PCA

from sklearn.linear_model.logistic import LogisticRegression

from sklearn.pipeline import Pipeline
#8381 of the images are RGB, so they will need to be converted to greyscale.

def rgb_to_gray(img):

    if len(img.shape) == 2: #already gray

        return(img)

    grayImage = np.zeros(img.shape)

    #These scaling coefficients are evidently standard for RGB->Grayscale conversion

    R = img[:,:,0]*0.299

    G = img[:,:,1]*0.587

    B = img[:,:,2]*0.114

    

    grayImage = R + G + B

    

    return(grayImage)
#We will also need to group together nearby pixels in order to 

#reduce image complexity and uniformize image size (so that every image has

#the same number of features).

    

def img_compress(img, x_bins=100,y_bins=100):

    x_splits = np.linspace(0,img.shape[1]-1,x_bins+1, dtype = int)

    y_splits=np.linspace(0,img.shape[0]-1,y_bins+1, dtype = int)

    

    compressed = np.zeros((y_bins,x_bins))

    

    for i in range(y_bins):

        for j in range(x_bins):

            temp = np.mean(img[y_splits[i]:y_splits[i+1],

                                      x_splits[j]:x_splits[j+1]])

            if math.isnan(temp):

                if y_splits[i]==y_splits[i+1]:

                    compressed[i,j]=compressed[i-1,j]

                else:

                    compressed[i,j] = compressed[i,j-1]

            else:

                compressed[i,j] = int(temp)

    return(compressed)
train_dir = '../input/train/train'

test_dir = '../input/test/test'

min_cols = 138

min_rows = 54
%%time

#Convert .jpg images into pixel arrays

train_file_names = os.listdir(train_dir)[::5]

imgs_train = [rgb_to_gray(mpimg.imread(train_dir + '/' + file, format = 'JPG')) 

              for file in train_file_names]

print(len(imgs_train), 'images loaded')
#Get indices of those pictures which aren't too small

good_pics = [i for i in range(len(imgs_train)) if (imgs_train[i].shape[0]>=min_rows)and(imgs_train[i].shape[1]>=min_cols)]

#Select only pictures that are sufficiently large

imgs_train = [imgs_train[i] for i in good_pics]
#Compress images. Unfortunately we have to distort the aspect ratio, which may

#lose valuable information. But if we do not do this then features will not

#have consistent meaning across pictures, even if we ensure that all pictures

#get compressed to the same resolution (pixel count)



#We need to find the smallest dimension across all images (train and test) in

#order to properly compress; otherwise the compressing algorithm will generate

#NaN-values.



compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]

del imgs_train
#Extract filenames to later associate to whale IDs

filenames = [file for file in train_file_names]

filenames = [filenames[i] for i in good_pics]

#Convert images into pandas dataframe format for learning the model

df1 = pd.DataFrame(data = [img.ravel() for img in compressed_train_imgs])

df2 = pd.DataFrame(data = filenames, columns = ['FileName'])

data1 = pd.concat([df2,df1],axis = 1)

#Obtain filenames, indexed by whale IDs

data2 = pd.read_csv('../input/train.csv',names = ['FileName','WhaleID'])

#Map whale IDs to images via filenames (index of data1 is matched to value of data2)

data = data2.merge(data1, on = 'FileName')

data = data.drop('FileName',axis = 1)

del compressed_train_imgs, df1, df2, data1, data2
data.sample(5)
#Extract training examples and labels, and get test set.

X_train = data.iloc[:,1:]

y_train = data['WhaleID']
%%time

#Perform principal component analysis for dimensionality reduction

#Try playing around with n_components to get better scores

pca = PCA(random_state = 42, n_components = 100, whiten = True)

pca.fit(X_train)
#Try other classifiers for better scores.

logreg = LogisticRegression(C = 1e-2)



#Define our pipeline. First we transform the data into its principal components,

#then learn the logistic regression model

clf = Pipeline([('pca',pca),

                ('logreg',logreg),

                ])



#Fit model

clf.fit(X_train,y_train)
#Score model on training data

print("Score on training set:", clf.score(X_train,y_train))

#We no longer need the training data. Delete to free up memory.

del data, X_train, y_train
#Read and clean test set

all_test_filenames = [file for file in os.listdir(test_dir)]

filenames_test = all_test_filenames[:5000]

print('Predicting for {} of {} total images'.format(len(filenames_test), len(all_test_filenames)))
%%time

imgs_test = [rgb_to_gray(mpimg.imread(test_dir + '/' + file)) for file in filenames_test]

#Smallest number of columns and rows across all test pictures

compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]

del imgs_test

X_test = pd.DataFrame([img.ravel() for img in compressed_test_imgs])

del compressed_test_imgs
%%time

y_preds = clf.predict_proba(X_test)
#Extract top 5 results

results = pd.DataFrame(data = [clf.classes_[np.argsort(y_preds[i,:])[-5:]] for i in range(y_preds.shape[0])],

                       index = filenames_test)

def list_to_str(L):

    string = ""

    for word in L:

        string = string + word + " "

    return(string)

        

results2 = pd.DataFrame(data = [list_to_str(results.iloc[i].values) for i in range(results.shape[0])],

                        index = filenames_test,columns = ['Id'])
# fill in missing results

most_common = results2['Id'].value_counts().index[0]

missing_files = [x for x in all_test_filenames if x not in filenames_test]

full_results_df = pd.concat([

    results2,

    pd.DataFrame([most_common]*len(missing_files), 

                 index=missing_files,

                columns = ['Id'])

])
full_results_df.sample(3)
full_results_df.to_csv('submission.csv', 

                sep = ',', 

                index_label = 'Image', 

                header = True)