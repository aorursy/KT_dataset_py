# This is a Notebook for creating an SVM to ID multiple charachters from the simpsons

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os

# Read in image locations
base = '../input/simpsons_dataset/simpsons_dataset/'
img_dict = dict()
# print(os.listdir(base))
for character in os.listdir(base):
    img_dict.update({character: [base + character +"/"+ x for x in os.listdir(base + character)]})

# Basic loading of image to 1d data to a df for manipulating
def load_data(character, im_size=64):
    img_arrays = list()
    print("Loading",len(img_dict[character]),"images for:",character)
    for img_loc in img_dict[character]:
        # Load images
        img = cv2.imread(img_loc)
        # resize to 64 by 64 images (128x128 runs out or ram (could dynamically load images during training to increase resolution))
        res = cv2.resize(img, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC).astype('float32') 
        # Normalize to scale [0,1]
        res = res / 255
        # reshape to single row of data
        res = res.reshape(im_size*im_size*3)
        # place everything into a list to be put into a df later
        img_arrays += [np.append(character, res)]
    print("done.")
    df = pd.DataFrame(img_arrays)
    return(df)

def load_multi_data(list_of_names:list):
    df = pd.concat([load_data(name) for name in list_of_names], ignore_index=True)
    return(df)
# Who do we have and how many?
l = []
for key in img_dict.keys():
    l+= [key]*len(img_dict[key])
pd.Series(l).value_counts()
# Load up a DF with the charachters we want to ID (theres not enough ram for all of them.. ill work on it)
df = load_multi_data(['ned_flanders','charles_montgomery_burns','principal_skinner','comic_book_guy', 'carl_carlson'])
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X = df.loc[:,1:]
Y = df.loc[:,0]
num_classes = len(Y.unique())
categorical_dict = {val:i for i,val in enumerate(Y.unique())}
reverse_dict = {v: k for k, v in categorical_dict.items()}
y2 = [categorical_dict[x] for x in Y]

X_train, X_test, Y_train, Y_test = train_test_split(X, y2, test_size=0.2)
# Lets also look at what this data shows off
print(Y_train[:5])
X_train.head()
# Fit a multiclass classifier using the loaded data
svc = SVC(kernel='rbf', gamma=0.7, C=1, decision_function_shape='ovo')
svc.fit(X_train, Y_train)
# Generate predictions
from sklearn.metrics import confusion_matrix, classification_report
prediction = svc.predict(X_test)
# Classification reports & heatmaps
import seaborn as sns
cm = pd.DataFrame(confusion_matrix(Y_test, prediction),index=categorical_dict.keys(),columns=categorical_dict.keys())
sns.heatmap(cm, annot=True, fmt=".2f")
print(classification_report([reverse_dict[x] for x in Y_test], [reverse_dict[x] for x in prediction]))
# show test predictions if you dont believe the model
def show_img(index):
    data_row = X_test.iloc[index,:]
    plt.imshow(np.reshape(np.array(data_row),(64,64,3)).astype('float32'))
    print("expected:",reverse_dict[Y_test[index]])
    pred = svc.predict([data_row])
    print("predicted:",reverse_dict[pred[0]])
show_img(index=70)