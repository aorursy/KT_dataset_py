import pandas as pd

import numpy as np

import os

import sys

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

from skimage.io import imread, imshow

from skimage import transform, img_as_float, filters

from skimage.color import rgb2gray

from skimage.feature import blob_dog, blob_log, blob_doh, canny

from skimage.transform import hough_line

import glob

import math

import scipy

import warnings

warnings.filterwarnings('ignore')
# some functions

def load_sample_training_data(ftype='jpg', n=100):

    """Returns (train_imgs, labels, im_names, tagged_df)

    train_imgs is the raw image data in the sample folder

    n is number of images to read in

    """

    #open data from current directory. Should work with any direcotry path

    tagged_df = pd.read_csv('../input/train_v2.csv')



    #split the tags into new rows

    tagged_df = pd.DataFrame(tagged_df.tags.str.split(' ').tolist(), index=tagged_df.image_name).stack()

    tagged_df = tagged_df.reset_index()[[0, 'image_name']] # dataframe with two columns

    tagged_df.columns = ['tags', 'image_name'] # rename columns

    tagged_df.set_index('image_name', inplace=True) # rest index to image_name again



    #create dummy variables for each tag

    tagged_df = pd.get_dummies(tagged_df['tags']) # creates dummy rows

    tagged_df = tagged_df.groupby(tagged_df.index).sum() # adds dummy rows together by image_name index



    train_imgs = []

    labels = []

    im_names = []

    print('Loading {} image dataset'.format(ftype))

    path = os.path.join('..', 'input','train-{}'.format(ftype),'*.'+ftype)



    files = glob.glob(path)

    for fs in files[1:n]:

        img = imread(fs)

        # img = transform.resize(img, output_shape=(h,w,d), preserve_range=True)  if needed

        train_imgs.append(img)

        

        imname = os.path.basename(fs).split('.')[0]

        im_names.append(imname)

        

        labels_temp = tagged_df.loc[imname]

        labels.append(labels_temp)

    train_imgs = img_as_float(np.asarray(train_imgs))

    return train_imgs, labels, im_names, tagged_df
X_sample, labels, names_train, tagged_df = load_sample_training_data(ftype='jpg', n=100)
def get_labels(fname, tagged_df):

    """return list of labels for a given filename"""

    return ", ".join(tagged_df.loc[fname][tagged_df.loc[fname]==1].index.tolist())  



def plot_samples(X_train, names_train, tagged_df, nrow, ncol):

    """Plots random sample images with their titles and tag names"""

    sampling = np.random.randint(low=0, high=X_train.shape[0]-1, size = nrow*ncol)

    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 12))

    for i in range(0,len(sampling)):

        name = names_train[sampling[i]]

        tags = get_labels(name, tagged_df)



        row = math.floor(i/ncol)

        col = i - math.floor(i/ncol)*ncol

        if (nrow == 1 or ncol == 1):

            ind = (max(row,col))

        else:

            ind = (row,col)

        axes[ind].imshow(X_train[sampling[i]])

        axes[ind].set_title(name+'\n'+tags)

        axes[ind].tick_params(left=False, right=False)

        axes[ind].set_yticklabels([])

        axes[ind].set_xticklabels([])

        axes[ind].axis('off')

    plt.tight_layout()



plot_samples(X_sample, names_train, tagged_df, nrow=4, ncol=4)
#Barplot of tag counts

def plot_sample_size(tagged_df):

    plt.rcParams['figure.figsize'] = (12, 5)

    print('There are {} unique tags in this data'.format(len(tagged_df.columns)))

    colors = cm.rainbow(np.linspace(0, 1, len(tagged_df.columns)))

    tagged_df.sum().sort_values(ascending=False).plot(title="Counts of Tags", color=colors, kind='bar')

    plt.show()



plot_sample_size(tagged_df)
fig, axes = plt.subplots(1, 3, figsize=(10, 6))

axes[0].imshow(X_sample[1,:,:,0], cmap='Reds')

axes[1].imshow(X_sample[1,:,:,1], cmap='Greens')

axes[2].imshow(X_sample[1,:,:,2], cmap='Blues')
#Binned mode differences feature creation to detect bimodal patterns

def binned_mode_features_with_diagnostics(img, steps):

    ## red ##

    #split on mean

    m=img[:,:,0].flatten().mean()

    left = img[:,:,0].flatten()[img[:,:,0].flatten()<m]

    right = img[:,:,0].flatten()[img[:,:,0].flatten()>=m]

    #find mode in left and right

    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]

    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]

    #calc bimodal metric

    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]

    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]

    mods_diff_r=abs(mo1-mo2)

    print("The mean of the red distribution is {}".format(m.round(2)))

    print("After splitting on the mean, the two modes are found at {} and {}".format(mo2, mo1))

    plt.hist(img[:,:,0].flatten(), color='red', bins=steps)

    plt.axvline(img[:,:,0].mean(), color='black', linestyle='dashed', linewidth=2)

    plt.axvline(mo1, color='yellow', linestyle='dashed', linewidth=2)

    plt.axvline(mo2, color='yellow', linestyle='dashed', linewidth=2)

    plt.show()

    

    ## green ##

    m=img[:,:,1].flatten().mean()

    left = img[:,:,1].flatten()[img[:,:,1].flatten()<m]

    right = img[:,:,1].flatten()[img[:,:,1].flatten()>=m]

    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]

    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]

    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]

    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]

    mods_diff_g=abs(mo1-mo2)

    print("The mean of the green distribution is {}".format(m.round(2)))

    print("After splitting on the mean, the two modes are found at {} and {}".format(mo2, mo1))

    plt.hist(img[:,:,1].flatten(), color='green', bins=steps)

    plt.axvline(img[:,:,1].mean(), color='black', linestyle='dashed', linewidth=2)

    plt.axvline(mo1, color='yellow', linestyle='dashed', linewidth=2)

    plt.axvline(mo2, color='yellow', linestyle='dashed', linewidth=2)

    plt.show()

    

    ## blue ##

    m=img[:,:,2].flatten().mean()

    left = img[:,:,2].flatten()[img[:,:,2].flatten()<m]

    right = img[:,:,2].flatten()[img[:,:,2].flatten()>=m]

    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]

    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]

    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]

    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]

    mods_diff_b=abs(mo1-mo2)

    print("The mean of the blue distribution is {}".format(m.round(2)))

    print("After splitting on the mean, the two modes are found at {} and {}".format(mo2, mo1))

    plt.hist(img[:,:,2].flatten(), color='blue', bins=steps)

    plt.axvline(img[:,:,2].mean(), color='black', linestyle='dashed', linewidth=2)

    plt.axvline(mo1, color='yellow', linestyle='dashed', linewidth=2)

    plt.axvline(mo2, color='yellow', linestyle='dashed', linewidth=2)

    plt.show()

    

    return mods_diff_r[0].round(2), mods_diff_g[0].round(2), mods_diff_b[0].round(2)



img=X_sample[2]

steps=np.arange(start=0,stop=1, step=.01)

binned_mode_features_with_diagnostics(img, steps)
from skimage.color import rgb2gray

from skimage import transform, img_as_float, filters

X_train_g = rgb2gray(X_sample)



X_train_sobel = []

for i in range(X_train_g.shape[0]):

    X_train_sobel.append(filters.sobel(X_train_g[i]))

X_train_sobel = np.asarray(X_train_sobel)



plot_samples(X_train_sobel, names_train, tagged_df, 4,4)
def xform_to_gray(imgs):

    return rgb2gray(imgs)



def xform_to_sobel(imgs):

    imgs = xform_to_gray(imgs)

    sobels = []

    if imgs.ndim == 2:

        sobels.append(filters.sobel(imgs))

    else:

        for i in range(imgs.shape[0]):

            sobels.append(filters.sobel(imgs[i]))

    return np.asarray(sobels)



def xform_to_canny(imgs, sigma):

    imgs = xform_to_gray(imgs)

    cannys = []

    if imgs.ndim == 2:

        cannys.append(canny(imgs, sigma))

    else:

        for i in range(imgs.shape[0]):

            cannys.append(canny(imgs[i], sigma))

    return np.asarray(cannys)



def get_num_blobs(img):

    return len(blob_log(rgb2gray(img)))



def binned_mode_features(img, nbins=100):

                                          

    steps=np.arange(start=0,stop=1, step=1/nbins)

                                                                            

    ## red ##

    #split on mean

    m=img[:,:,0].flatten().mean()

    left = img[:,:,0].flatten()[img[:,:,0].flatten()<m]

    right = img[:,:,0].flatten()[img[:,:,0].flatten()>=m]

    #find mode in left and right

    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]

    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]

    #calc bimodal metric

    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]

    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]

    mods_diff_r=abs(mo1-mo2)



    ## green ##

    m=img[:,:,1].flatten().mean()

    left = img[:,:,1].flatten()[img[:,:,1].flatten()<m]

    right = img[:,:,1].flatten()[img[:,:,1].flatten()>=m]

    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]

    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]

    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]

    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]

    mods_diff_g=abs(mo1-mo2)



    ## blue ##

    m=img[:,:,2].flatten().mean()

    left = img[:,:,2].flatten()[img[:,:,2].flatten()<m]

    right = img[:,:,2].flatten()[img[:,:,2].flatten()>=m]

    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]

    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]

    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]

    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]

    mods_diff_b=abs(mo1-mo2)



    return mods_diff_r[0], mods_diff_g[0], mods_diff_b[0]

def get_features(img):

    """Input is a Nx256x256x3 numpy array of images, where N is number of images"""

        

    # METRIC FOR BIMODALITY

    # bin each color intensity (histogram)

    # find 2 most populated bins

    # subtract and abs() to quantify bimodality

        

    r = img[:,:,0].ravel()

    g = img[:,:,1].ravel()

    b = img[:,:,2].ravel()

                

    s = xform_to_sobel(img)

    

    can = xform_to_canny(img, 0.5)

    

    #hough, _, _ = hough_line(rgb2gray(img))

    

    r_mean = np.mean(r)

    g_mean = np.mean(g)

    b_mean = np.mean(b)

    

    r_std = np.std(r)

    g_std = np.std(g)

    b_std = np.std(b)

    

    r_max = np.max(r)

    b_max = np.max(b)

    g_max = np.max(g)

    

    r_min = np.min(r)

    b_min = np.min(b)

    g_min = np.min(g)

    

    r_kurtosis = scipy.stats.kurtosis(r)

    b_kurtosis = scipy.stats.kurtosis(b)

    g_kurtosis = scipy.stats.kurtosis(g)

    

    r_skew = scipy.stats.skew(r)

    b_skew = scipy.stats.skew(b)

    g_skew = scipy.stats.skew(g)

    

    sobel_mean = np.mean(s.ravel())

    sobel_std = np.std(s.ravel())

    sobel_max = np.max(s.ravel())

    sobel_min = np.min(s.ravel())

    sobel_kurtosis = scipy.stats.kurtosis(s.ravel())

    sobel_skew = scipy.stats.skew(s.ravel())

    sobel_rowmean_std = np.std(np.mean(s,axis=1))

    sobel_colmean_std = np.std(np.mean(s,axis=0))

    

    canny_mean = np.mean(can.ravel())

    canny_std = np.std(can.ravel())

    canny_max = np.max(can.ravel())

    canny_min = np.min(can.ravel())

    canny_kurtosis = scipy.stats.kurtosis(can.ravel())

    canny_skew = scipy.stats.skew(can.ravel())

    canny_rowmean_std = np.std(np.mean(can,axis=1))

    canny_colmean_std = np.std(np.mean(can,axis=0))

    

    r_bimodal, g_bimodal, b_bimodal = binned_mode_features(img)

    

    #n_blobs = get_num_blobs(img)

    

    #hough_mean = np.mean(hough)

    #hough_std = np.std(hough)

    #hough_max = np.max(hough)

    #hough_min = np.max(hough)

    #hough_kurtosis = scipy.stats.kurtosis(hough.ravel())

    #hough_skew = scipy.stats.skew(hough.ravel())

                  

    return pd.Series(

        {'r_mean':r_mean, 'g_mean':g_mean, 'b_mean':b_mean,

         'r_std':r_std, 'g_std':g_std, 'b_std':b_std,

         'r_max':r_max, 'g_max':g_max, 'b_max':b_max,

         'r_min':r_min, 'g_min':g_min, 'b_min':b_min,

         'r_kurtosis':r_kurtosis, 'g_kurtosis':g_kurtosis, 'b_kurtosis':b_kurtosis,

         'r_skew':r_skew, 'g_skew':g_skew, 'b_skew':b_skew,

         'sobel_mean':sobel_mean, 'sobel_std':sobel_std, 

         'sobel_max':sobel_max, 'sobel_min':sobel_min,

         'sobel_kurtosis':sobel_kurtosis, 'sobel_skew':sobel_skew,

         'sobel_rowmean_std':sobel_rowmean_std, 'sobel_colmean_std':sobel_colmean_std,

         'canny_mean':canny_mean, 'canny_std':canny_std, 

         'canny_max':canny_max, 'canny_min':canny_min,

         'canny_kurtosis':canny_kurtosis, 'canny_skew':canny_skew,

         'canny_rowmean_std':canny_rowmean_std, 'canny_colmean_std':canny_colmean_std,

         'r_bimodal':r_bimodal, 'g_bimodal':g_bimodal, 'b_bimodal':b_bimodal

         #'n_blobs':n_blobs

         #'hough_mean':hough_mean, 'hough_std':hough_std, 'hough_max':hough_max,

         #'hough_min':hough_min, 'hough_kurtosis':hough_kurtosis, 'hough_skew':hough_skew

        })
y = tagged_df



X = pd.DataFrame([])

for i in np.arange(0,99):

    x  = get_features(X_sample[i,])

    X =  X.append(x, ignore_index=True)

X
#create table of each feature histograms for each label

X.set_index(y.index[0:99], inplace=True)

print(X.columns) #possible features to plot

plt.rcParams['figure.figsize'] = (10, 20)

#function to plot distributions of a features by class label

def plot_a_feature_by_labels(feature):

    colors = cm.rainbow(np.linspace(0, 1, len(y.columns))) #pick colors for plots by labels

    for i in np.arange(0, len(y.columns)-1):

        col=y.columns[i]

        ind_list = y[y[col]==1].index.tolist()

        X.ix[ind_list][feature].hist(bins=25, color=colors[i])

        plt.title(col)

        plt.grid(True)

        plt.subplot(6,3,i+1) 



print("Blue bimodal feauture")

plot_a_feature_by_labels('b_bimodal')   

plt.show()

print("sobel column mean standard deviation")

plot_a_feature_by_labels('sobel_colmean_std')

plt.show()
#do a test/train split

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y[0:99], test_size=0.40, random_state=14113) 



print('X_train is a {} object'.format(type(X_train)))

print('it has shape {}'.format(X_train.shape))



print('y_train is a {} object'.format(type(y_train)))

print('it has {} elements'.format(len(y_train)))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100, 

                            max_features = 'sqrt',

                            bootstrap = True, 

                            oob_score = True,

                            n_jobs = -1,

                            random_state = 14113,

                            class_weight = 'balanced_subsample')



rf.fit(X_train, y_train)

print('The oob error for this random forest is {}'.format(rf.oob_score_.round(2)))
#features ranking of features. 

Feature_importance = pd.DataFrame(rf.feature_importances_, X_train.columns)

def plot_feature_importance(Feature_importance, n):

    '''

    plot top n features

    '''

    plt.rcParams['figure.figsize'] = (12, 5)

    Feature_importance = pd.DataFrame(rf.feature_importances_, X_train.columns)

    Feature_importance.columns = ['features']

    Feature_importance = Feature_importance.sort_values(by='features', axis=0, ascending=False)

    colors = cm.gist_heat(np.linspace(0, 1, len(tagged_df.columns)))

    Feature_importance.head(n).plot(title="Counts of Tags", color=colors, kind='bar')

    plt.show()



plot_feature_importance(Feature_importance, 15)
from sklearn.metrics import fbeta_score

np.asarray(y_validation)

predictions = rf.predict(X_validation)

fbeta_score(np.asarray(y_validation), predictions, beta=2, average='samples')
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_validation, predictions)

Metrics = pd.DataFrame([precision, recall, support], index=['precision', 'recall', 'support'])

Metrics.columns = y_validation.columns

Metrics
probs = rf.predict_proba(X_validation)



from sklearn import metrics



def plot_ROC(tag):

    '''

    plot ROC curve for a specific tag

    '''

    plt.rcParams['figure.figsize'] = (6,6)

    n = np.where(y_validation.columns==tag)[0][0]

    fpr, tpr, threshs = metrics.roc_curve(y_validation[tag], probs[n][:,1],

                                          pos_label=None, sample_weight=None, drop_intermediate=False)

    plt.figure()

    lw = 2

    plt.plot(fpr, tpr, color='darkorange',

             lw=lw)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(tag+'\nReceiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()

    

plot_ROC('agriculture')

plot_ROC('bare_ground')
def plot_decision_hist(tag):

    '''

    plots decision histograms with thresholds

    '''

    plt.rcParams['figure.figsize'] = (6,6)

    #Less than .5 is 0. greater is 1

    n = np.where(y_validation.columns==tag)[0][0]

    probs_df = pd.DataFrame(probs[n][:,1]).set_index(y_validation[tag])

    class0 =  np.array(probs_df.ix[0][0]) #0 does not have true tag

    class1 =  np.array(probs_df.ix[1][0]) #1 does have true tag



    S = class0

    # Histogram:

    # Bin it

    n, bin_edges = np.histogram(S, 30)

    # Normalize it, so that every bins value gives the probability of that bin

    bin_probability = n/float(n.sum())

    # Get the mid points of every bin

    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.

    # Compute the bin-width

    bin_width = bin_edges[1]-bin_edges[0]

    # Plot the histogram as a bar plot

    plt.bar(bin_middles, bin_probability, width=bin_width, color='red', alpha=.4)



    S = class1

    n, bin_edges = np.histogram(S, 30)

    bin_probability = n/float(n.sum())

    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.

    bin_width = bin_edges[1]-bin_edges[0]

    plt.bar(bin_middles, bin_probability, width=bin_width, color='green', alpha=.8)



    plt.axvline(x=0.5, color='k', linestyle='--')

    plt.title(tag+'\nScore distributions with splitting on a 0.5 threshold')

    plt.xlabel('Classification model score')

    plt.ylabel('Frequency')

    plt.show()

    

plot_decision_hist('agriculture')

plot_decision_hist('bare_ground')    