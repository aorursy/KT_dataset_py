!pip install colorgram.py



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot data



import os # read file

import colorgram #Extract color from images 



import sklearn #scikit-learn library, where the magic happens!

import cv2 #OpenCV, image processing library
gems_df = pd.DataFrame(columns=['label', 't_set', 'file'])



for root, _, files in os.walk('/kaggle/input/gemstones-images'):

    for file in files:

        path = os.path.join(root, file)

        label = path.split(os.sep)[-2]

        t_set = path.split(os.sep)[-3]

        gems_df.loc[len(gems_df)]=[label,t_set,path]



print(gems_df)
gems_df['label'].value_counts()
def get_pallet(gem):

    n_color = 5

    colors = colorgram.extract(gem['file'], n_color)

    for i in range(len(colors)):

        gem['red'+str(i)] = colors[i].rgb.r

        gem['green'+str(i)] = colors[i].rgb.g

        gem['blue'+str(i)] = colors[i].rgb.b

        gem['proportion'+str(i)] = colors[i].proportion

    return gem



gems_color_df = gems_df.apply(get_pallet, axis=1)

gems_color_df.fillna(0,inplace=True)



gems_color_df
X_train = gems_color_df[gems_color_df['t_set']=='train'].drop(['file', 't_set', 'label'], axis=1)

y_train = gems_color_df[gems_color_df['t_set']=='train'][['label']]



X_test = gems_color_df[gems_color_df['t_set']=='test'].drop(['file', 't_set', 'label'], axis=1)

y_test = gems_color_df[gems_color_df['t_set']=='test'][['label']]



print(X_train)

print(y_train)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



pca = PCA(n_components=20)

pca.fit(X_train)



X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



print("X_train_pca.shape: {}".format(X_train_pca.shape))



tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train_pca, y_train)



print("Accuracy on training set: {:.3f}".format(tree.score(X_train_pca, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test_pca, y_test)))
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[20])

mlp.fit(X_train, y_train.values.ravel())



print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))
def get_pallet(gem):

    n_color = 5

    colors = colorgram.extract(gem['file'], n_color)

    for i in range(len(colors)):

        gem['hue'+str(i)] = colors[i].hsl.h

        gem['sat'+str(i)] = colors[i].hsl.s

        gem['light'+str(i)] = colors[i].hsl.l

        gem['proportion'+str(i)] = colors[i].proportion

    return gem



gems_hsl_df = gems_df.apply(get_pallet, axis=1)

gems_hsl_df.fillna(0,inplace=True)



gems_hsl_df
X_train = gems_hsl_df[gems_hsl_df['t_set']=='train'].drop(['file', 't_set', 'label'], axis=1)

y_train = gems_hsl_df[gems_df['t_set']=='train'][['label']]



X_test = gems_hsl_df[gems_hsl_df['t_set']=='test'].drop(['file', 't_set', 'label'], axis=1)

y_test = gems_hsl_df[gems_hsl_df['t_set']=='test'][['label']]



print(X_train)

print(y_train)
tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
pca = PCA(n_components=20)

pca.fit(X_train)



X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



print("X_train_pca.shape: {}".format(X_train_pca.shape))



tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train_pca, y_train)



print("Accuracy on training set: {:.3f}".format(tree.score(X_train_pca, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test_pca, y_test)))
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[5])

mlp.fit(X_train, y_train.values.ravel())



print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))
img_w, img_h = 200, 200



print(gems_df['file'][0])



image = cv2.imread(gems_df['file'][344])

image = cv2.resize(image,(int(img_w*1.5), int(img_h*1.5)))

image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converts an image from BGR color space to RGB

image = np.array(image_cv)



plt.imshow(image)
blur = cv2.blur(image_cv, (3,3))

blur = cv2.bilateralFilter(blur,5,75,75)



plt.imshow(blur)

canny = cv2.Canny(blur, img_w, img_h)

plt.imshow(canny)
## find the non-zero min-max coords of canny

pts = np.argwhere(canny>0)

if pts.any()>0:

    y1,x1 = pts.min(axis=0)

    y2,x2 = pts.max(axis=0)



    if abs(y1-y2)>50 or abs(x1-x2)>50:

        ## crop the region

        image = image_cv[y1:y2, x1:x2]



#resize

image = cv2.resize(image,(img_w, img_h))



plt.imshow(image)
def get_and_crop_image(series):

    img_w, img_h = 220, 220



    image = cv2.imread(series['file'])

    image = cv2.resize(image,(int(img_w*1.5), int(img_h*1.5)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converts an image from BGR color space to RGB



    #blur

    blur = cv2.blur(image, (3,3))

    blur = cv2.bilateralFilter(blur,5,75,75)

    

    #edge detection - canny

    canny = cv2.Canny(blur, img_w, img_h)



    #crop

    pts = np.argwhere(canny>0)

    if pts.any()>0:

        y1,x1 = pts.min(axis=0)

        y2,x2 = pts.max(axis=0)

        

        

        if abs(y1-y2)>50 or abs(x1-x2)>50:

            ## crop the region

            image = image[y1:y2, x1:x2]

            

    image = cv2.resize(image,(img_w, img_h))

    image = np.array(image).flatten()

    

    series['image'] = image

    return series
def explode(serie):

    serie = serie.join(pd.DataFrame(serie.image.tolist(),index=serie.index).add_prefix('img_'))

    print(serie)

    return serie
gems_image_df = gems_df

gems_image_df = gems_image_df.apply(get_and_crop_image, axis=1)



tags = gems_image_df['image'].apply(pd.Series)

tags = tags.rename(columns = lambda x : 'img_' + str(x))

tags



gems_image_df = gems_image_df.drop(['image'], axis=1)

gems_image_df = pd.concat([gems_image_df[:], tags[:]], axis=1)



gems_image_df
X_train = gems_image_df[gems_image_df['t_set']=='train'].drop(['file', 't_set', 'label'], axis=1)

y_train = gems_image_df[gems_image_df['t_set']=='train'][['label']]



X_test = gems_image_df[gems_image_df['t_set']=='test'].drop(['file', 't_set', 'label'], axis=1)

y_test = gems_image_df[gems_image_df['t_set']=='test'][['label']]



from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



#n_components:20 -> Accuracy: 0.284

#n_components:14 -> Accuracy: 0.292

#n_components:12 -> Accuracy: 0.300

#n_components:11 -> Accuracy: 0.314

#n_components:10 -> Accuracy: 0.320

#n_components:9 -> Accuracy: 0.300

#n_components:8 -> Accuracy: 0.273



pca = PCA(n_components=10)

pca.fit(X_train)



X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



print("X_train_pca.shape: {}".format(X_train_pca.shape))



tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train_pca, y_train)



print("Accuracy on training set: {:.3f}".format(tree.score(X_train_pca, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test_pca, y_test)))
from sklearn.neural_network import MLPClassifier



#solver='lbfgs'

#hidden_layer_sizes:10 -> Accuracy: 0.008

#hidden_layer_sizes:50 -> Accuracy: 0.011

#hidden_layer_sizes:100 -> Accuracy: 0.017

#hidden_layer_sizes:150 -> Accuracy: 0.011



#solver='adam'

#hidden_layer_sizes:10 -> Accuracy: 0.014

#hidden_layer_sizes:50 -> Accuracy: 0.014

#hidden_layer_sizes:100 -> Accuracy: 0.014



mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[150])

mlp.fit(X_train, y_train.values.ravel())



print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))