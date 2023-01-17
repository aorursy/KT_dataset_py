# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
#%matplotlib inline

# My import
import random as rd
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)

# This roughly takes 5 seconds.
# now we gonna load the second image in train data, reshape it as matrix than display it
i=1
img=train_images.iloc[i].values   # This puts the data to img in one line of 784-elements array
img=img.reshape((28,28))          # This reshapes the array so the data is now a 28x28 array
plt.imshow(img,cmap='gray')       # Plots the array into image plot
plt.title(train_labels.iloc[i,0]) # Gives the plot a title corresponding with its label (image class)

num_class = np.unique(labeled_images.label) # This returns the array of unique numbers existed in parent data
nums = [[] for i in range(28)] # Prepares appendable 28x280 matrix for the output.
var = rd.randint(0, 50) # This makes it possible to print one of 50 pairs of examples randomly each time it runs.
for i in num_class: # For every unique labels starting from the lowest (0) to the highest (9)...
    current_class = labeled_images[labeled_images.label==i].iloc[0:var+1, 1:]
    img=current_class.iloc[var].values
    img=img.reshape((28,28))
    for j in range(28): # ...Append every row of pixels to its corresponding row in appendable matrix nums.
        for k in img[j]:
            nums[j].append(k)

print(num_class)
plt.imshow(nums, cmap='gray')
plt.title('Random representations\nof each number classes 0-9')

# Please take note i took the data from the parent image instead of the train data like before.

#for i in num_class:
#    current_class = labeled_images[labeled_images.label==i].iloc[0:1, 1:]
#    img=current_class.iloc[0].values
#    img=img.reshape((28,28))
#    plt.imshow(img,cmap='gray')
#    plt.title(i)
dedplctd_train_labels = train_labels.drop_duplicates(keep='first') # drop duplicates but the first occurence
dedplctd_train_labels = dedplctd_train_labels.sort_values('label')
print(dedplctd_train_labels)
dedplctd_train_labels_idx = dedplctd_train_labels.index.values
print("Indices\t\t: {}".format(dedplctd_train_labels_idx))
dedplctd_train_labels_xtr = dedplctd_train_labels.label.values
print("Unique classes\t: {}".format(dedplctd_train_labels_xtr))

dtli = dedplctd_train_labels_idx
dtlx = dedplctd_train_labels_xtr

nums_2 = [[] for i in range(28)] # Prepares appendable 28x280 matrix for the output.
var = rd.randint(0, 50) # This makes it possible to print one of 50 pairs of examples randomly each time it runs.
for i in range(len(dtli)): # For every unique labels starting from the lowest (0) to the highest (9)...
    current_class = train_images[train_images.index==dtli[i]]
    img=current_class.iloc[0].values
    img=img.reshape((28,28))
    for j in range(28): # ...Append every row of pixels to its corresponding row in appendable matrix nums.
        for k in img[j]:
            nums_2[j].append(k)

plt.rcParams["figure.figsize"][0]=12
plt.imshow(nums_2, cmap='gray')
plt.title('Train images first occurence\nrepresentations of each number classes 0-9')
#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
# create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
#data1 = train_images.iloc[1]
#data2 = train_images.iloc[3]
#data1 = np.array(data1)
#data2 = np.array(data2)
#data3 = np.append(data1,data2)
#print(len(data3))
#plt.hist(data3)

k_o = 0
i_o= 1
hist_data = []
while k_o < 10:
    if train_labels.iloc[i_o,0] == k_o:
        curdat = train_images.iloc[i_o].values
        for j in curdat:
            hist_data.append(j)
        k_o+=1
    i_o+=1
print(len(hist_data)) # i there are 10 number classes, so i predict there are 28 x 28 x 10 elements (which is 7840)
plt.hist(hist_data)
one_d_data = []
for i in nums_2:
    for j in i:
        one_d_data.append(j)
        
print(len(one_d_data)) # i there are 10 number classes, so i predict there are 28 x 28 x 10 elements (which is 7840)
plt.hist(one_d_data)
fig_size = plt.rcParams["figure.figsize"]
#print(fig_size) # Prints original matplotlib output size, [6.0, 4.0], which is currently default.
fig_size[0] = 12 # Changes the width
fig_size[1] = 50 # Changes the height
plt.subplots_adjust(hspace=0.45) # Changes the intersubplot height space, default = 0.2

for a in range(len(dtlx)):
    current_class = train_images[train_images.index==dtli[a]]
    
    plt.subplot(10,2,(a*2)+1)
    plt.hist(current_class.iloc[0].values)
    plt.title("Histogram for number class '{}'\nfirst occurence representation".format(dtlx[a]))
    plt.xlabel("Pixel classes")
    plt.ylabel("Frequency")
    
    img=current_class.iloc[0].values
    img=img.reshape((28,28))
    plt.subplot(10,2,(a*2)+2)
    plt.imshow(img, cmap='gray')
    plt.title("number class '{}'\nfirst occurence representation".format(dtlx[a]))
# Let's return customized params back into their default value.
print(fig_size)
fig_size[0] = 6
fig_size[1] = 4
print(fig_size)
plt.subplots_adjust(hspace=0.2)
plt.subplot(1,1,1)
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

# This takes 42 seconds
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
old_model = DecisionTreeRegressor(random_state=0)
old_model.fit(train_images, train_labels)
old_predict = old_model.predict(test_images)
old_mae = mean_absolute_error(test_labels, old_predict)
print (old_mae)
# Put your verification code here
# Todo
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number
i = 1 # I've been messing with the i variable before, so i restore it here :3

test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
# now plot again the histogram
plt.hist(train_images.iloc[i])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

# This takes roughly 14 seconds, 4 times faster than before.
# Test again to data test
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])

# This takes roughly 19 seconds
# separate code section to view the results
print(results)
print(len(results))
# dump the results to 'results.csv'
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
#check if the file created successfully
print(os.listdir("."))
# from https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(df)
#img=train_images.iloc[i].values.reshape((28,28))
#plt.imshow(img,cmap='binary')
#plt.title(train_labels.iloc[i])

#clf = svm.SVC()
#clf.fit(train_images, train_labels.values.ravel())
#clf.score(test_images,test_labels)

#gam_x = np.arange(-8.0, 9, 1.0) 
#tol_x = np.arange(-8.0, -0.0, 1.0) 
gam_x = np.linspace(-8.0, 8.0, 5) # Setting up the constraints of gamma's x
tol_x = np.linspace(-8.0, 0.0, 5) # Setting up the contraints of tol's x
#print (gam_x)
#print (tol_x)
# The documentation says it's more reliable to use numpy.linspace instead of
# numpy.arange, but i think it doesn't matter for now.

gam_arr = np.power(10, gam_x) # Converts it to the result as the power of tens
tol_arr = np.power(10, tol_x) # ditto
#print(gam_arr)
#print(tol_arr)

#These functions below are used to find out the accuracy both for train and test data
#Both gamma and tolerance.

from time import time
tmstart = float(time())

def trainspl_scores_gam(gam_arr):
    out =[]
    tmstart_loc=float(time())
    for gam in gam_arr:
        mdl = svm.SVC(kernel='rbf',gamma=gam, cache_size=12288, random_state=0) # HELL YEAH 12 GB OF RAMS
        mdl.fit(train_images, train_labels.values.ravel())
        out.append(mdl.score(train_images, train_labels))
        print('Produced train score result for (gamma={}) = {}; time elasped = {} seconds'.format(gam, out[-1], int(float(time())-tmstart_loc))) # Debug, lmao
    print('This function runs for {} seconds'.format(int(float(time())-tmstart_loc)))
    return out

def testspl_scores_gam(gam_arr):
    out =[]
    tmstart_loc=float(time())
    for gam in gam_arr:
        mdl = svm.SVC(kernel='rbf',gamma=gam, cache_size=12288, random_state=0)
        mdl.fit(train_images, train_labels.values.ravel())
        out.append(mdl.score(test_images, test_labels))
        print('Produced test score result for (gamma={}) = {}; time elasped = {} seconds'.format(gam, out[-1], int(float(time())-tmstart_loc)))
    print('This function runs for {} seconds'.format(int(float(time())-tmstart_loc)))
    return out

def trainspl_scores_tol(tol_arr):
    out =[]
    tmstart_loc=float(time())
    for tol in tol_arr:
        mdl = svm.SVC(kernel='rbf',tol=tol, cache_size=12288, random_state=0)
        mdl.fit(train_images, train_labels.values.ravel())
        out.append(mdl.score(train_images, train_labels))
        print('Produced train score result for (tolerance={}) = {}; time elasped = {} seconds'.format(tol, out[-1], int(float(time())-tmstart_loc)))
    print('This function runs for {} seconds'.format(int(float(time())-tmstart_loc)))
    return out

def testspl_scores_tol(tol_arr):
    out =[]
    tmstart_loc=float(time())
    for tol in tol_arr:
        mdl = svm.SVC(kernel='rbf',tol=tol, cache_size=12288, random_state=0)
        mdl.fit(train_images, train_labels.values.ravel())
        out.append(mdl.score(test_images, test_labels))
        print('Produced test score result for (tolerance={}) = {}; time elasped = {} seconds'.format(tol, out[-1], int(float(time())-tmstart_loc)))
    print('This function runs for {} seconds'.format(int(float(time())-tmstart_loc)))
    return out

# Each of these takes roughly 3.5 mins, ~210 secs,
# 14 minutes in total, at least.
gam_y_trainspl = trainspl_scores_gam(gam_arr)
gam_y_testspl = testspl_scores_gam(gam_arr)
tol_y_trainspl = trainspl_scores_tol(tol_arr)
tol_y_testspl = testspl_scores_tol(tol_arr)
print("The progress takes {} seconds".format(int(float(time())-tmstart)))

fig_size[0] = 12

plt.subplot(1,2,1)
plt.plot(gam_x, gam_y_trainspl, 'r', gam_x, gam_y_testspl, 'b')
plt.fill_between(gam_x, gam_y_trainspl, gam_y_testspl, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Accuracy Score')
plt.title('Score results from modifying gamma\n red = train, blue = test')

plt.subplot(1,2,2)
plt.plot(tol_x, tol_y_trainspl, 'r', tol_x, tol_y_testspl, 'b')
plt.fill_between(tol_x, tol_y_trainspl, tol_y_testspl, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Accuracy Score')
plt.title('Score results from modifying tolerance\n red = train, blue = test')

# Saved values:
# gam_y_trainspl = [0.1145,  0.8325,  1.0,     1.0,     1.0    ] Each element takes ~60 secs
# gam_y_testspl  = [0.1,     0.804,   0.1,     0.1,     0.1    ] Each element takes ~41 secs
# tol_y_trainspl = [0.93625, 0.93625, 0.93625, 0.93625, 0.93575] Each element takes ~25 secs
# tol_y_testspl  = [0.887,   0.887,   0.887,   0.887,   0.889  ] Each element takes ~13 secs
# 669 secs for one run -> 11 mins 39 secs
gam_x_fix = np.linspace(-2.0, -1.0, 10) # Setting up the constraints of gamma's x
gam_arr = np.power(10, gam_x_fix) # Converts it to the result as the power of tens
gam_y_trainspl_fix = trainspl_scores_gam(gam_arr)
gam_y_testspl_fix = testspl_scores_gam(gam_arr)

best_gam = 0
best_accuracy = 0
best_idx = 0
for i in range(len(gam_y_testspl_fix)):
    if best_accuracy <= gam_y_testspl_fix[i]:
        best_accuracy = gam_y_testspl_fix[i]
        best_gam = gam_arr.item(i)
        best_idx = i

print("Best gamma is {} which produces accuracy of {}".format(best_gam, best_accuracy))

fig_size[0] = 6
plt.subplot(1,1,1)
plt.plot(gam_x_fix, gam_y_trainspl_fix, 'r', gam_x_fix, gam_y_testspl_fix, 'b', gam_x_fix[best_idx], best_accuracy, 'rx')
plt.fill_between(gam_x_fix, gam_y_trainspl_fix, gam_y_testspl_fix, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Accuracy Score')
plt.title('Score results across gamma values\n red = train, blue = test')

# Best gamma is 0.021544346900318832 which produces accuracy of 0.94
# The progress lasts 721 seconds
from sklearn.tree import DecisionTreeClassifier

def find_trainspl_mae_classi_normal(classi_candidates_normal):
    tmstart_loc = float(time())
    out = []
    for i in classi_candidates_normal:
        classi_model = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
        classi_model.fit(train_images, train_labels)
        out.append(mean_absolute_error(train_labels, classi_model.predict(train_images)))
        print("Generated train MAE from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_testspl_mae_classi_normal(classi_candidates_normal):
    tmstart_loc = float(time())
    out = []
    for i in classi_candidates_normal:
        classi_model = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
        classi_model.fit(train_images, train_labels)
        out.append(mean_absolute_error(test_labels, classi_model.predict(test_images)))
        print("Generated test MAE from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_trainspl_score_classi_normal(classi_candidates_normal):
    tmstart_loc = float(time())
    out = []
    for i in classi_candidates_normal:
        classi_model = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
        classi_model.fit(train_images, train_labels)
        out.append(classi_model.score(train_images, train_labels))
        print("Generated train score from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_testspl_score_classi_normal(classi_candidates_normal):
    tmstart_loc = float(time())
    out = []
    for i in classi_candidates_normal:
        classi_model = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
        classi_model.fit(train_images, train_labels)
        out.append(classi_model.score(test_images, test_labels))
        print("Generated test score from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_best_mae_and_idx(maes):
    best_mae = maes[0]
    best_idx = 0
    for i in range(len(maes)):
        if best_mae > maes[i]:
            best_mae = maes[i]
            best_idx = i
    return best_mae, best_idx

def find_best_score_and_idx(scores):
    best_score = scores[0]
    best_idx = 0
    for i in range(len(scores)):
        if best_score < scores[i]:
            best_score = scores[i]
            best_idx = i
    return best_score, best_idx
classi_x_normal_1 = np.linspace(1, 8, 10)
classi_candidates_normal_1 = np.power(10, classi_x_normal_1).astype(int)
classi_trainspl_maes_1 = find_trainspl_mae_classi_normal(classi_candidates_normal_1)
classi_testspl_maes_1 = find_testspl_mae_classi_normal(classi_candidates_normal_1)
classi_best_testspl_mae_normal_1, classi_best_testspl_idx_normal_1 = find_best_mae_and_idx(classi_testspl_maes_1)
classi_best_testspl_x_normal_1 = classi_x_normal_1.item(classi_best_testspl_idx_normal_1)
classi_best_max_leaf_normal_1 = classi_candidates_normal_1.item(classi_best_testspl_idx_normal_1)

plt.plot(classi_x_normal_1, classi_trainspl_maes_1, 'r', classi_x_normal_1, classi_testspl_maes_1, 'b', classi_best_testspl_x_normal_1, classi_best_testspl_mae_normal_1, 'rx')
plt.fill_between(classi_x_normal_1, classi_trainspl_maes_1, classi_testspl_maes_1, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Mean Absolute Error')
plt.title('MAE results across x=[{},{}] values\n red = train, blue = test'.format(classi_x_normal_1.astype(int).item(0), classi_x_normal_1.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(classi_x_normal_1.astype(int).item(0), classi_x_normal_1.astype(int).item(-1)))
print('Best test MAE\t\t\t: {}'.format(classi_best_testspl_mae_normal_1))
print("Best test MAE's x\t\t: {}".format(classi_best_testspl_x_normal_1))
print("Best test MAE's max_tree_nodes\t: {}".format(classi_candidates_normal_1.item(classi_best_testspl_idx_normal_1)))
print('Next narrowed range = [{},{}]'.format(classi_best_testspl_x_normal_1-1,classi_best_testspl_x_normal_1+1))
classi_x_normal_2 = np.linspace(classi_best_testspl_x_normal_1-1,classi_best_testspl_x_normal_1+1, 10)
classi_candidates_normal_2 = np.power(10, classi_x_normal_2).astype(int)
classi_trainspl_maes_2 = find_trainspl_mae_classi_normal(classi_candidates_normal_2)
classi_testspl_maes_2 = find_testspl_mae_classi_normal(classi_candidates_normal_2)
classi_best_testspl_mae_normal_2, classi_best_testspl_idx_normal_2 = find_best_mae_and_idx(classi_testspl_maes_2)
classi_best_testspl_x_normal_2 = classi_x_normal_2.item(classi_best_testspl_idx_normal_2)
classi_best_max_leaf_normal_2 = classi_candidates_normal_2.item(classi_best_testspl_idx_normal_2)

plt.plot(classi_x_normal_2, classi_trainspl_maes_2, 'r', classi_x_normal_2, classi_testspl_maes_2, 'b', classi_best_testspl_x_normal_2, classi_best_testspl_mae_normal_2, 'rx')
plt.fill_between(classi_x_normal_2, classi_trainspl_maes_2, classi_testspl_maes_2, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Mean Absolute Error')
plt.title('MAE results across x=[{},{}] values\n red = train, blue = test'.format(classi_x_normal_2.astype(int).item(0), classi_x_normal_2.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(classi_x_normal_2.astype(int).item(0), classi_x_normal_2.astype(int).item(-1)))
print('Best test MAE\t\t\t: {}'.format(classi_best_testspl_mae_normal_2))
print("Best test MAE's x\t\t: {}".format(classi_best_testspl_x_normal_2))
print("Best test MAE's max_tree_nodes\t: {}".format(classi_candidates_normal_2.item(classi_best_testspl_idx_normal_2)))
#classi_x_normal_3 = np.linspace(classi_best_testspl_x_normal_2-1,classi_best_testspl_x_normal_2+1, 10)
#classi_candidates_normal_3 = np.power(10, classi_x_normal_3).astype(int)
pow_a_3 = ((10**classi_best_testspl_x_normal_2)-50)
pow_b_3 = ((10**classi_best_testspl_x_normal_2)+50)
classi_candidates_normal_3 = np.linspace(pow_a_3, pow_b_3, 100).astype(int)
classi_trainspl_maes_3 = find_trainspl_mae_classi_normal(classi_candidates_normal_3)
classi_testspl_maes_3 = find_testspl_mae_classi_normal(classi_candidates_normal_3)
classi_best_testspl_mae_normal_3, classi_best_testspl_idx_normal_3 = find_best_mae_and_idx(classi_testspl_maes_3)
#classi_best_testspl_x_normal_3 = classi_x_normal_3.item(classi_best_testspl_idx_normal_3)
classi_best_max_leaf_normal_3 = classi_candidates_normal_3.item(classi_best_testspl_idx_normal_3)

plt.plot(classi_candidates_normal_3, classi_trainspl_maes_3, 'r', classi_candidates_normal_3, classi_testspl_maes_3, 'b', classi_best_max_leaf_normal_3, classi_best_testspl_mae_normal_3, 'rx')
plt.fill_between(classi_candidates_normal_3, classi_trainspl_maes_3, classi_testspl_maes_3, facecolor='#AAFF99', interpolate=True)
plt.xlabel('max_leaf_nodes')
plt.ylabel('Mean Absolute Error')
plt.title('MAE results across max_leaf_nodes=[{},{}] values\n red = train, blue = test'.format(classi_candidates_normal_3.astype(int).item(0), classi_candidates_normal_3.astype(int).item(-1)))

print('\nResults for range max_tree_nodes=[{},{}]:'.format(classi_candidates_normal_3.astype(int).item(0), classi_candidates_normal_3.astype(int).item(-1)))
print('Best test MAE\t\t\t: {}'.format(classi_best_testspl_mae_normal_3))
#print("Best test MAE's x\t\t: {}".format(classi_best_testspl_x_normal_3))
print("Best test MAE's max_tree_nodes\t: {}".format(classi_candidates_normal_3.item(classi_best_testspl_idx_normal_3)))
def find_score_from_validation_and_prediction(val_v, val_p):
    correct=0
    for i in range(val_v.size):
        if val_v.item(i)==val_p.item(i): correct+=1
    return correct/val_v.size
tmstart = float(time())
normal_classi_model = DecisionTreeClassifier(max_leaf_nodes=classi_best_max_leaf_normal_3, random_state=0)
normal_classi_model.fit(train_images, train_labels)
normal_classi_predict = normal_classi_model.predict(test_images)
print("Predicting values takes {} seconds".format(int(tmstart-float(time()))))
normal_classi_score = find_score_from_validation_and_prediction(test_labels.label.values,normal_classi_predict)
#print(normal_classi_predict.size)
#print(type(normal_classi_predict))
#print(test_labels.label.values.size)
#print(type(test_labels.label.values))
print("The score of DecisionTreeClassifier using (max_leaf_nodes={}) = {}; manually counted".format(classi_best_max_leaf_normal_3, normal_classi_score))
print("For comparison, SVM.SVC using (gamma={}) has the score = {}".format(best_gam, best_accuracy))

# Oh hey look at lazy me finally dig up to the DecisionTreeClassifier documentation
# and found out if tey have their own .score() function.
print("The score of DecisionTreeClassifier using (max_leaf_noteds={}) = {}".format(classi_best_max_leaf_normal_3, normal_classi_model.score(test_images, test_labels)))
def find_trainspl_mae_regres_normal(regres_candidates_normal):
    tmstart_loc = float(time())
    out = []
    for i in regres_candidates_normal:
        regres_model = DecisionTreeRegressor(max_leaf_nodes=i, random_state=0)
        regres_model.fit(train_images, train_labels)
        out.append(mean_absolute_error(train_labels, regres_model.predict(train_images)))
        print("Generated train MAE from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_testspl_mae_regres_normal(regres_candidates_normal):
    tmstart_loc = float(time())
    out = []
    for i in regres_candidates_normal:
        regres_model = DecisionTreeRegressor(max_leaf_nodes=i, random_state=0)
        regres_model.fit(train_images, train_labels)
        out.append(mean_absolute_error(test_labels, regres_model.predict(test_images)))
        print("Generated test MAE from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_trainspl_score_regres_normal(regres_candidates_normal):
    tmstart_loc = float(time())
    out = []
    for i in regres_candidates_normal:
        regres_model = DecisionTreeRegressor(max_leaf_nodes=i, random_state=0)
        regres_model.fit(train_images, train_labels)
        out.append(regres_model.score(train_images, train_labels))
        print("Generated train score from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_testspl_score_regres_normal(regres_candidates_normal):
    tmstart_loc = float(time())
    out = []
    for i in regres_candidates_normal:
        regres_model = DecisionTreeRegressor(max_leaf_nodes=i, random_state=0)
        regres_model.fit(train_images, train_labels)
        out.append(regres_model.score(test_images, test_labels))
        print("Generated test score from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out
regres_x_normal_1 = np.linspace(1, 8, 10)
regres_candidates_normal_1 = np.power(10, regres_x_normal_1).astype(int)
regres_trainspl_maes_1 = find_trainspl_mae_regres_normal(regres_candidates_normal_1)
regres_testspl_maes_1 = find_testspl_mae_regres_normal(regres_candidates_normal_1)
regres_best_testspl_mae_normal_1, regres_best_testspl_idx_normal_1 = find_best_mae_and_idx(regres_testspl_maes_1)
regres_best_testspl_x_normal_1 = regres_x_normal_1.item(regres_best_testspl_idx_normal_1)
regres_best_max_leaf_normal_1 = regres_candidates_normal_1.item(regres_best_testspl_idx_normal_1)

plt.plot(regres_x_normal_1, regres_trainspl_maes_1, 'r', regres_x_normal_1, regres_testspl_maes_1, 'b', regres_best_testspl_x_normal_1, regres_best_testspl_mae_normal_1, 'rx')
plt.fill_between(regres_x_normal_1, regres_trainspl_maes_1, regres_testspl_maes_1, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Mean Absolute Error')
plt.title('MAE results across x=[{},{}] values\n red = train, blue = test'.format(regres_x_normal_1.astype(int).item(0), regres_x_normal_1.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(regres_x_normal_1.astype(int).item(0), regres_x_normal_1.astype(int).item(-1)))
print('Best test MAE\t\t\t: {}'.format(regres_best_testspl_mae_normal_1))
print("Best test MAE's x\t\t: {}".format(regres_best_testspl_x_normal_1))
print("Best test MAE's max_tree_nodes\t: {}".format(regres_candidates_normal_1.item(regres_best_testspl_idx_normal_1)))
print('Next narrowed range = [{},{}]'.format(regres_best_testspl_x_normal_1-1,regres_best_testspl_x_normal_1+1))
regres_x_normal_2 = np.linspace(regres_best_testspl_x_normal_1-1,regres_best_testspl_x_normal_1+0.25, 10)
regres_candidates_normal_2 = np.power(10, regres_x_normal_2).astype(int)
regres_trainspl_maes_2 = find_trainspl_mae_regres_normal(regres_candidates_normal_2)
regres_testspl_maes_2 = find_testspl_mae_regres_normal(regres_candidates_normal_2)
regres_best_testspl_mae_normal_2, regres_best_testspl_idx_normal_2 = find_best_mae_and_idx(regres_testspl_maes_2)
regres_best_testspl_x_normal_2 = regres_x_normal_2.item(regres_best_testspl_idx_normal_2)
regres_best_max_leaf_normal_2 = regres_candidates_normal_2.item(regres_best_testspl_idx_normal_2)

plt.plot(regres_x_normal_2, regres_trainspl_maes_2, 'r', regres_x_normal_2, regres_testspl_maes_2, 'b', regres_best_testspl_x_normal_2, regres_best_testspl_mae_normal_2, 'rx')
plt.fill_between(regres_x_normal_2, regres_trainspl_maes_2, regres_testspl_maes_2, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Mean Absolute Error')
plt.title('MAE results across x=[{},{}] values\n red = train, blue = test'.format(regres_x_normal_2.astype(int).item(0), regres_x_normal_2.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(regres_x_normal_2.astype(int).item(0), regres_x_normal_2.astype(int).item(-1)))
print('Best test MAE\t\t\t: {}'.format(regres_best_testspl_mae_normal_2))
print("Best test MAE's x\t\t: {}".format(regres_best_testspl_x_normal_2))
print("Best test MAE's max_tree_nodes\t: {}".format(regres_candidates_normal_2.item(regres_best_testspl_idx_normal_2)))
#regres_x_normal_3 = np.linspace(regres_best_testspl_x_normal_2-1,regres_best_testspl_x_normal_2+1, 10)
#regres_candidates_normal_3 = np.power(10, regres_x_normal_3).astype(int)
pow_a_3 = ((10**regres_best_testspl_x_normal_2)-50)
pow_b_3 = ((10**regres_best_testspl_x_normal_2)+50)
regres_candidates_normal_3 = np.linspace(pow_a_3, pow_b_3, 100).astype(int)
regres_trainspl_maes_3 = find_trainspl_mae_regres_normal(regres_candidates_normal_3)
regres_testspl_maes_3 = find_testspl_mae_regres_normal(regres_candidates_normal_3)
regres_best_testspl_mae_normal_3, regres_best_testspl_idx_normal_3 = find_best_mae_and_idx(regres_testspl_maes_3)
#regres_best_testspl_x_normal_3 = regres_x_normal_3.item(regres_best_testspl_idx_normal_3)
regres_best_max_leaf_normal_3 = regres_candidates_normal_3.item(regres_best_testspl_idx_normal_3)

plt.plot(regres_candidates_normal_3, regres_trainspl_maes_3, 'r', regres_candidates_normal_3, regres_testspl_maes_3, 'b', regres_best_max_leaf_normal_3, regres_best_testspl_mae_normal_3, 'rx')
plt.fill_between(regres_candidates_normal_3, regres_trainspl_maes_3, regres_testspl_maes_3, facecolor='#AAFF99', interpolate=True)
plt.xlabel('max_leaf_nodes')
plt.ylabel('Mean Absolute Error')
plt.title('MAE results across max_leaf_nodes=[{},{}] values\n red = train, blue = test'.format(regres_candidates_normal_3.astype(int).item(0), regres_candidates_normal_3.astype(int).item(-1)))

print('\nResults for range max_tree_nodes=[{},{}]:'.format(regres_candidates_normal_3.astype(int).item(0), regres_candidates_normal_3.astype(int).item(-1)))
print('Best test MAE\t\t\t: {}'.format(regres_best_testspl_mae_normal_3))
#print("Best test MAE's x\t\t: {}".format(regres_best_testspl_x_normal_3))
print("Best test MAE's max_tree_nodes\t: {}".format(regres_candidates_normal_3.item(regres_best_testspl_idx_normal_3)))
tmstart = float(time())
normal_regres_model = DecisionTreeRegressor(max_leaf_nodes=regres_best_max_leaf_normal_3, random_state=0)
normal_regres_model.fit(train_images, train_labels)
print("Fitting values takes {} seconds".format(int(tmstart-float(time()))))
print("The score of DecisionTreeRegressor using (max_leaf_nodes={}) = {}".format(regres_best_max_leaf_normal_3, normal_regres_model.score(test_images, test_labels)))
print("For comparison:\nSVM.SVC using (gamma={}) has the score = {}".format(best_gam, best_accuracy))
print("DecisionTreeClassifier using (max_leaf_nodes{}) = {}".format(classi_best_max_leaf_normal_3, normal_classi_model.score(test_images, test_labels)))

# For further comparation
mae_apporach_classi_score = normal_classi_model.score(test_images, test_labels)
mae_approach_regres_score = normal_regres_model.score(test_images, test_labels)
classi_x_normal_1 = np.linspace(1, 8, 10)
classi_candidates_normal_1 = np.power(10, classi_x_normal_1).astype(int)
classi_trainspl_scores_1 = find_trainspl_score_classi_normal(classi_candidates_normal_1)
classi_testspl_scores_1 = find_testspl_score_classi_normal(classi_candidates_normal_1)
classi_best_testspl_score_normal_1, classi_best_testspl_idx_normal_1 = find_best_score_and_idx(classi_testspl_scores_1)
classi_best_testspl_x_normal_1 = classi_x_normal_1.item(classi_best_testspl_idx_normal_1)
classi_best_max_leaf_normal_1 = classi_candidates_normal_1.item(classi_best_testspl_idx_normal_1)

plt.plot(classi_x_normal_1, classi_trainspl_scores_1, 'r', classi_x_normal_1, classi_testspl_scores_1, 'b', classi_best_testspl_x_normal_1, classi_best_testspl_score_normal_1, 'rx')
plt.fill_between(classi_x_normal_1, classi_trainspl_scores_1, classi_testspl_scores_1, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Score')
plt.title('score results across x=[{},{}] values\n red = train, blue = test'.format(classi_x_normal_1.astype(int).item(0), classi_x_normal_1.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(classi_x_normal_1.astype(int).item(0), classi_x_normal_1.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(classi_best_testspl_score_normal_1))
print("Best test score's x\t\t: {}".format(classi_best_testspl_x_normal_1))
print("Best test score's max_tree_nodes\t: {}".format(classi_candidates_normal_1.item(classi_best_testspl_idx_normal_1)))
print('Next narrowed range = [{},{}]'.format(classi_best_testspl_x_normal_1-1,classi_best_testspl_x_normal_1+1))
classi_x_normal_2 = np.linspace(classi_best_testspl_x_normal_1-1,classi_best_testspl_x_normal_1+1, 10)
classi_candidates_normal_2 = np.power(10, classi_x_normal_2).astype(int)
classi_trainspl_scores_2 = find_trainspl_score_classi_normal(classi_candidates_normal_2)
classi_testspl_scores_2 = find_testspl_score_classi_normal(classi_candidates_normal_2)
classi_best_testspl_score_normal_2, classi_best_testspl_idx_normal_2 = find_best_score_and_idx(classi_testspl_scores_2)
classi_best_testspl_x_normal_2 = classi_x_normal_2.item(classi_best_testspl_idx_normal_2)
classi_best_max_leaf_normal_2 = classi_candidates_normal_2.item(classi_best_testspl_idx_normal_2)

plt.plot(classi_x_normal_2, classi_trainspl_scores_2, 'r', classi_x_normal_2, classi_testspl_scores_2, 'b', classi_best_testspl_x_normal_2, classi_best_testspl_score_normal_2, 'rx')
plt.fill_between(classi_x_normal_2, classi_trainspl_scores_2, classi_testspl_scores_2, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Score')
plt.title('score results across x=[{},{}] values\n red = train, blue = test'.format(classi_x_normal_2.astype(int).item(0), classi_x_normal_2.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(classi_x_normal_2.astype(int).item(0), classi_x_normal_2.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(classi_best_testspl_score_normal_2))
print("Best test score's x\t\t: {}".format(classi_best_testspl_x_normal_2))
print("Best test score's max_tree_nodes\t: {}".format(classi_candidates_normal_2.item(classi_best_testspl_idx_normal_2)))
#classi_x_normal_3 = np.linspace(classi_best_testspl_x_normal_2-1,classi_best_testspl_x_normal_2+1, 10)
#classi_candidates_normal_3 = np.power(10, classi_x_normal_3).astype(int)
pow_a_3 = ((10**classi_best_testspl_x_normal_2)-50)
pow_b_3 = ((10**classi_best_testspl_x_normal_2)+50)
classi_candidates_normal_3 = np.linspace(pow_a_3, pow_b_3, 100).astype(int)
classi_trainspl_scores_3 = find_trainspl_score_classi_normal(classi_candidates_normal_3)
classi_testspl_scores_3 = find_testspl_score_classi_normal(classi_candidates_normal_3)
classi_best_testspl_score_normal_3, classi_best_testspl_idx_normal_3 = find_best_score_and_idx(classi_testspl_scores_3)
#classi_best_testspl_x_normal_3 = classi_x_normal_3.item(classi_best_testspl_idx_normal_3)
classi_best_max_leaf_normal_3 = classi_candidates_normal_3.item(classi_best_testspl_idx_normal_3)

plt.plot(classi_candidates_normal_3, classi_trainspl_scores_3, 'r', classi_candidates_normal_3, classi_testspl_scores_3, 'b', classi_best_max_leaf_normal_3, classi_best_testspl_score_normal_3, 'rx')
plt.fill_between(classi_candidates_normal_3, classi_trainspl_scores_3, classi_testspl_scores_3, facecolor='#AAFF99', interpolate=True)
plt.xlabel('max_leaf_nodes')
plt.ylabel('Score')
plt.title('score results across max_leaf_nodes=[{},{}] values\n red = train, blue = test'.format(classi_candidates_normal_3.astype(int).item(0), classi_candidates_normal_3.astype(int).item(-1)))

print('\nResults for range max_tree_nodes=[{},{}]:'.format(classi_candidates_normal_3.astype(int).item(0), classi_candidates_normal_3.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(classi_best_testspl_score_normal_3))
#print("Best test score's x\t\t: {}".format(classi_best_testspl_x_normal_3))
print("Best test score's max_tree_nodes\t: {}".format(classi_candidates_normal_3.item(classi_best_testspl_idx_normal_3)))
regres_x_normal_1 = np.linspace(1, 8, 10)
regres_candidates_normal_1 = np.power(10, regres_x_normal_1).astype(int)
regres_trainspl_scores_1 = find_trainspl_score_regres_normal(regres_candidates_normal_1)
regres_testspl_scores_1 = find_testspl_score_regres_normal(regres_candidates_normal_1)
regres_best_testspl_score_normal_1, regres_best_testspl_idx_normal_1 = find_best_score_and_idx(regres_testspl_scores_1)
regres_best_testspl_x_normal_1 = regres_x_normal_1.item(regres_best_testspl_idx_normal_1)
regres_best_max_leaf_normal_1 = regres_candidates_normal_1.item(regres_best_testspl_idx_normal_1)

plt.plot(regres_x_normal_1, regres_trainspl_scores_1, 'r', regres_x_normal_1, regres_testspl_scores_1, 'b', regres_best_testspl_x_normal_1, regres_best_testspl_score_normal_1, 'rx')
plt.fill_between(regres_x_normal_1, regres_trainspl_scores_1, regres_testspl_scores_1, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Score')
plt.title('score results across x=[{},{}] values\n red = train, blue = test'.format(regres_x_normal_1.astype(int).item(0), regres_x_normal_1.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(regres_x_normal_1.astype(int).item(0), regres_x_normal_1.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(regres_best_testspl_score_normal_1))
print("Best test score's x\t\t: {}".format(regres_best_testspl_x_normal_1))
print("Best test score's max_tree_nodes\t: {}".format(regres_candidates_normal_1.item(regres_best_testspl_idx_normal_1)))
print('Next narrowed range = [{},{}]'.format(regres_best_testspl_x_normal_1-1,regres_best_testspl_x_normal_1+1))
regres_x_normal_2 = np.linspace(regres_best_testspl_x_normal_1-1,regres_best_testspl_x_normal_1+0.25, 10)
regres_candidates_normal_2 = np.power(10, regres_x_normal_2).astype(int)
regres_trainspl_scores_2 = find_trainspl_score_regres_normal(regres_candidates_normal_2)
regres_testspl_scores_2 = find_testspl_score_regres_normal(regres_candidates_normal_2)
regres_best_testspl_score_normal_2, regres_best_testspl_idx_normal_2 = find_best_score_and_idx(regres_testspl_scores_2)
regres_best_testspl_x_normal_2 = regres_x_normal_2.item(regres_best_testspl_idx_normal_2)
regres_best_max_leaf_normal_2 = regres_candidates_normal_2.item(regres_best_testspl_idx_normal_2)

plt.plot(regres_x_normal_2, regres_trainspl_scores_2, 'r', regres_x_normal_2, regres_testspl_scores_2, 'b', regres_best_testspl_x_normal_2, regres_best_testspl_score_normal_2, 'rx')
plt.fill_between(regres_x_normal_2, regres_trainspl_scores_2, regres_testspl_scores_2, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Score')
plt.title('score results across x=[{},{}] values\n red = train, blue = test'.format(regres_x_normal_2.astype(int).item(0), regres_x_normal_2.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(regres_x_normal_2.astype(int).item(0), regres_x_normal_2.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(regres_best_testspl_score_normal_2))
print("Best test score's x\t\t: {}".format(regres_best_testspl_x_normal_2))
print("Best test score's max_tree_nodes\t: {}".format(regres_candidates_normal_2.item(regres_best_testspl_idx_normal_2)))
#regres_x_normal_3 = np.linspace(regres_best_testspl_x_normal_2-1,regres_best_testspl_x_normal_2+1, 10)
#regres_candidates_normal_3 = np.power(10, regres_x_normal_3).astype(int)
pow_a_3 = ((10**regres_best_testspl_x_normal_2)-50)
pow_b_3 = ((10**regres_best_testspl_x_normal_2)+50)
regres_candidates_normal_3 = np.linspace(pow_a_3, pow_b_3, 100).astype(int)
regres_trainspl_scores_3 = find_trainspl_score_regres_normal(regres_candidates_normal_3)
regres_testspl_scores_3 = find_testspl_score_regres_normal(regres_candidates_normal_3)
regres_best_testspl_score_normal_3, regres_best_testspl_idx_normal_3 = find_best_score_and_idx(regres_testspl_scores_3)
#regres_best_testspl_x_normal_3 = regres_x_normal_3.item(regres_best_testspl_idx_normal_3)
regres_best_max_leaf_normal_3 = regres_candidates_normal_3.item(regres_best_testspl_idx_normal_3)

plt.plot(regres_candidates_normal_3, regres_trainspl_scores_3, 'r', regres_candidates_normal_3, regres_testspl_scores_3, 'b', regres_best_max_leaf_normal_3, regres_best_testspl_score_normal_3, 'rx')
plt.fill_between(regres_candidates_normal_3, regres_trainspl_scores_3, regres_testspl_scores_3, facecolor='#AAFF99', interpolate=True)
plt.xlabel('max_leaf_nodes')
plt.ylabel('Score')
plt.title('score results across max_leaf_nodes=[{},{}] values\n red = train, blue = test'.format(regres_candidates_normal_3.astype(int).item(0), regres_candidates_normal_3.astype(int).item(-1)))

print('\nResults for range max_tree_nodes=[{},{}]:'.format(regres_candidates_normal_3.astype(int).item(0), regres_candidates_normal_3.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(regres_best_testspl_score_normal_3))
#print("Best test score's x\t\t: {}".format(regres_best_testspl_x_normal_3))
print("Best test score's max_tree_nodes\t: {}".format(regres_candidates_normal_3.item(regres_best_testspl_idx_normal_3)))
train_imgs, test_imgs,train_lbls, test_lbls = train_test_split(images, labels, test_size=0.2, random_state=0)
# now we gonna load the second image in train data, reshape it as matrix than display it
img=train_imgs.iloc[1].values   # This puts the data to img in one line of 784-elements array
img=img.reshape((28,28))        # This reshapes the array so the data is now a 28x28 array
plt.imshow(img,cmap='gray')     # Plots the array into image plot
plt.title(train_lbls.iloc[1,0]) # Gives the plot a title corresponding with its label (image class)
def find_trainspl_score_classi_unnormal(classi_candidates_unnormal):
    tmstart_loc = float(time())
    out = []
    for i in classi_candidates_unnormal:
        classi_model = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
        classi_model.fit(train_imgs, train_lbls)
        out.append(classi_model.score(train_imgs, train_lbls))
        print("Generated train score from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_testspl_score_classi_unnormal(classi_candidates_unnormal):
    tmstart_loc = float(time())
    out = []
    for i in classi_candidates_unnormal:
        classi_model = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
        classi_model.fit(train_imgs, train_lbls)
        out.append(classi_model.score(test_imgs, test_lbls))
        print("Generated test score from (max_leaf_nodes={}) = {}; Time elasped = {} seconds".format(i, out[-1],int(float(time())-tmstart_loc)))
    print("Time elasped for this function: {} seconds".format(int(float(time())-tmstart_loc)))
    return out

def find_best_score_and_idx(scores):
    best_score = scores[0]
    best_idx = 0
    for i in range(len(scores)):
        if best_score < scores[i]:
            best_score = scores[i]
            best_idx = i
    return best_score, best_idx
classi_x_unnormal_1 = np.linspace(1, 8, 100)
classi_candidates_unnormal_1 = np.power(10, classi_x_unnormal_1).astype(int)
classi_trainspl_scores_1 = find_trainspl_score_classi_unnormal(classi_candidates_unnormal_1)
classi_testspl_scores_1 = find_testspl_score_classi_unnormal(classi_candidates_unnormal_1)
classi_best_testspl_score_unnormal_1, classi_best_testspl_idx_unnormal_1 = find_best_score_and_idx(classi_testspl_scores_1)
classi_best_testspl_x_unnormal_1 = classi_x_unnormal_1.item(classi_best_testspl_idx_unnormal_1)
classi_best_max_leaf_unnormal_1 = classi_candidates_unnormal_1.item(classi_best_testspl_idx_unnormal_1)

plt.plot(classi_x_unnormal_1, classi_trainspl_scores_1, 'r', classi_x_unnormal_1, classi_testspl_scores_1, 'b', classi_best_testspl_x_unnormal_1, classi_best_testspl_score_unnormal_1, 'rx')
plt.fill_between(classi_x_unnormal_1, classi_trainspl_scores_1, classi_testspl_scores_1, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Score')
plt.title('score results across x=[{},{}] values\n red = train, blue = test'.format(classi_x_unnormal_1.astype(int).item(0), classi_x_unnormal_1.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(classi_x_unnormal_1.astype(int).item(0), classi_x_unnormal_1.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(classi_best_testspl_score_unnormal_1))
print("Best test score's x\t\t: {}".format(classi_best_testspl_x_unnormal_1))
print("Best test score's max_tree_nodes\t: {}".format(classi_candidates_unnormal_1.item(classi_best_testspl_idx_unnormal_1)))
print('Next narrowed range = [{},{}]'.format(classi_best_testspl_x_unnormal_1-1,classi_best_testspl_x_unnormal_1+1))
classi_x_unnormal_2 = np.linspace(classi_best_testspl_x_unnormal_1-1,classi_best_testspl_x_unnormal_1+1, 100)
classi_candidates_unnormal_2 = np.power(10, classi_x_unnormal_2).astype(int)
classi_trainspl_scores_2 = find_trainspl_score_classi_unnormal(classi_candidates_unnormal_2)
classi_testspl_scores_2 = find_testspl_score_classi_unnormal(classi_candidates_unnormal_2)
classi_best_testspl_score_unnormal_2, classi_best_testspl_idx_unnormal_2 = find_best_score_and_idx(classi_testspl_scores_2)
classi_best_testspl_x_unnormal_2 = classi_x_unnormal_2.item(classi_best_testspl_idx_unnormal_2)
classi_best_max_leaf_unnormal_2 = classi_candidates_unnormal_2.item(classi_best_testspl_idx_unnormal_2)

plt.plot(classi_x_unnormal_2, classi_trainspl_scores_2, 'r', classi_x_unnormal_2, classi_testspl_scores_2, 'b', classi_best_testspl_x_unnormal_2, classi_best_testspl_score_unnormal_2, 'rx')
plt.fill_between(classi_x_unnormal_2, classi_trainspl_scores_2, classi_testspl_scores_2, facecolor='#AAFF99', interpolate=True)
plt.xlabel('Ten to the power of')
plt.ylabel('Score')
plt.title('score results across x=[{},{}] values\n red = train, blue = test'.format(classi_x_unnormal_2.astype(int).item(0), classi_x_unnormal_2.astype(int).item(-1)))

print('\nResults for range x=[{},{}]:'.format(classi_x_unnormal_2.astype(int).item(0), classi_x_unnormal_2.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(classi_best_testspl_score_unnormal_2))
print("Best test score's x\t\t: {}".format(classi_best_testspl_x_unnormal_2))
print("Best test score's max_tree_nodes\t: {}".format(classi_candidates_unnormal_2.item(classi_best_testspl_idx_unnormal_2)))
#classi_x_unnormal_3 = np.linspace(classi_best_testspl_x_unnormal_2-1,classi_best_testspl_x_unnormal_2+1, 10)
#classi_candidates_unnormal_3 = np.power(10, classi_x_unnormal_3).astype(int)
pow_a_3 = ((10**classi_best_testspl_x_unnormal_2)-50)
pow_b_3 = ((10**classi_best_testspl_x_unnormal_2)+50)
classi_candidates_unnormal_3 = np.linspace(pow_a_3, pow_b_3, 200).astype(int)
classi_trainspl_scores_3 = find_trainspl_score_classi_unnormal(classi_candidates_unnormal_3)
classi_testspl_scores_3 = find_testspl_score_classi_unnormal(classi_candidates_unnormal_3)
classi_best_testspl_score_unnormal_3, classi_best_testspl_idx_unnormal_3 = find_best_score_and_idx(classi_testspl_scores_3)
#classi_best_testspl_x_unnormal_3 = classi_x_unnormal_3.item(classi_best_testspl_idx_unnormal_3)
classi_best_max_leaf_unnormal_3 = classi_candidates_unnormal_3.item(classi_best_testspl_idx_unnormal_3)

plt.plot(classi_candidates_unnormal_3, classi_trainspl_scores_3, 'r', classi_candidates_unnormal_3, classi_testspl_scores_3, 'b', classi_best_max_leaf_unnormal_3, classi_best_testspl_score_unnormal_3, 'rx')
plt.fill_between(classi_candidates_unnormal_3, classi_trainspl_scores_3, classi_testspl_scores_3, facecolor='#AAFF99', interpolate=True)
plt.xlabel('max_leaf_nodes')
plt.ylabel('Score')
plt.title('score results across max_leaf_nodes=[{},{}] values\n red = train, blue = test'.format(classi_candidates_unnormal_3.astype(int).item(0), classi_candidates_unnormal_3.astype(int).item(-1)))

print('\nResults for range max_tree_nodes=[{},{}]:'.format(classi_candidates_unnormal_3.astype(int).item(0), classi_candidates_unnormal_3.astype(int).item(-1)))
print('Best test score\t\t\t: {}'.format(classi_best_testspl_score_unnormal_3))
#print("Best test score's x\t\t: {}".format(classi_best_testspl_x_unnormal_3))
print("Best test score's max_tree_nodes\t: {}".format(classi_candidates_unnormal_3.item(classi_best_testspl_idx_unnormal_3)))