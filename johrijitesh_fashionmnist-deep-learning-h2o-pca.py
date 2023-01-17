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
# Objectives

# Image data multiclass classification and object identification - apparel and clothing

# Using H2O Deep Learning

# Self Organizing Maps where we will perform clustering on Fashion MNIST using a neural network.

# A self-organizing map (SOM) or self-organizing feature map (SOFM) is a type of artificial neural network (ANN) 

# that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), 

# discretized representation of the input space of the training samples, called a map, 

# and is therefore a method to do dimensionality reduction. 

# Self-organizing maps differ from other artificial neural networks as they apply competitive learning 

# as opposed to error-correction learning (such as backpropagation with gradient descent), 

# and in the sense that they use a neighborhood function to preserve the topological properties of the input space.

# 2 Parts - Part I without reducing dimensionality; 

#           Part II reducing dimensionality using random projection

# Dataset : Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. 

# Each example is a 28x28 grayscale image, associated with a label from 10 classes. 

# 1 Loading libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

import concurrent.futures

import time

from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig



# 1.1 For measuring time elapsed

from time import time

from imblearn.over_sampling import SMOTE, ADASYN



# 1.2 Processing data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler 

from sklearn.compose import ColumnTransformer as ct

from sklearn.random_projection import SparseRandomProjection as sr  # Projection features



# 1.3 Data imputation

from sklearn.impute import SimpleImputer



# 1.4 Model building

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator



# 1.5 for ROC graphs & metrics

import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics
# 1.6 Change ipython options to display all data columns

pd.options.display.max_columns = 300
# 2.0 Read data from Local directory



# Loading training and test set

train = pd.read_csv('../input/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashion-mnist_test.csv')
# 3.0 # Combining training and test set to get over 70k samples

new_train = train.drop(columns=['label'])

new_test = test.drop(columns=['label'])

som_data = pd.concat([new_train, new_test], ignore_index=True).values

labels = pd.concat([train['label'], test['label']], ignore_index=True).values
# 3.1 Sample image



f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5):

    ax[i].imshow(som_data[i].reshape(28, 28))

plt.show()
# 4.0 Plotting the image - not necessary for the modelling exercise

for i in range(5000,5005): 

    sample = np.reshape(train[train.columns[1:]].iloc[i].values, (28,28))

    plt.figure()

    #plt.title("labeled class {}".format(get_label_cls(train["label"]/255.iloc[i])))

    plt.imshow(sample)
# 5.0 Data exploration

train.shape     # 60000 X 785

test.shape      # 10000 X 785
train.label.value_counts()
test.label.value_counts()
# 6.0 Combine test & train  

tmp = pd.concat([train,test],

               axis = 0,            # Stack one upon another (rbind)

               ignore_index = True

              )
tmp.shape    #(70000, 785)
# 7.0 Separation into target/predictors

y = tmp.iloc[:,0]

X = tmp.iloc[:,1:]

X.shape              # (70000,784)

y.shape              # (70000,)
# 8.0 Transform to numpy array



fash_tmp = X.values

fash_tmp.shape       # (70000 X 784)

target_tmp = y.values
# 9.1 Create a StandardScaler instance

ss = StandardScaler()
# 9.2 fit() and transform() in one step

fash_tmp = ss.fit_transform(fash_tmp)
# 9.3

fash_tmp.shape               # 70000 X 784 (an ndarray)
# 10.1 Separate train and test

X = fash_tmp[: train.shape[0], : ]

X.shape                             # 60000 X 784
# 10.2

test = fash_tmp[train.shape[0] :, : ]

test.shape                         # 10000X 784
target_train = target_tmp[: train.shape[0]]

target_test  = target_tmp[train.shape[0]: ]

target_train.shape       #(60000,)

target_test.shape        #(10000,)
################## Model building #####################

# 11.0 Split train into training and validation dataset

X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    target_train,

                                                    test_size = 0.3)
# 11.1

X_train.shape    # 42000 X 784

X_test.shape     # 18000 X 784
# 11.2 composite data with both predictors and target

y_train = y_train.reshape(len(y_train),1)

y_train
X = np.hstack((X_train,y_train))

X.shape            # 42000 X 785
# 12.0 Modelling using H2O Deep Learning

# 12.1 Start h2o

h2o.init()
# 12.2 Transform data to h2o dataframe

df = h2o.H2OFrame(X)

len(df.columns)    # 785
df.shape           # 42000 X 785
X_columns = df.columns[0:784]        # Only column names. No data

X_columns       # C1 to C784
y_columns = df.columns[784]

y_columns
df['C785'].head()
# 13 As required by H2O , For classification, target column must be factor

df['C785'] = df['C785'].asfactor()
# 13.1. Build a deeplearning model on balanced data

#     http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html

dl_model = H2ODeepLearningEstimator(epochs=500,

                                    distribution = 'multinomial',                 # Response has two levels

                                    missing_values_handling = "skip", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 3,                           # CV folds

                                    fold_assignment = "auto",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [100,100],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')
# 13.2 Train model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df)





end = time()

(end - start)/60
# 14. Predictions on actual  'test' data

#     Create a composite X_test data before transformation to

#     H2o dataframe.

y_test = y_test.reshape(len(y_test), 1)     # Needed to hstack

y_test.shape     # 18000 X 1

X_test.shape     # 18000 X 784
# 15 Column-wise stack 

X_test = np.hstack((X_test,y_test))         # cbind data

X_test.shape     # 18000 X 785



X_test = h2o.H2OFrame(X_test)

X_test['C785'] = X_test['C785'].asfactor()
# 16. Make prediction on X_test

result = dl_model.predict(X_test[: , 0:784])

result.shape       # 18000 X 11

result.as_data_frame().head()   # Class-wise predictions
result.shape       #(18000, 11)

fash_pred = X_test['C785'].as_data_frame()

fash_pred['result'] = result[0].as_data_frame()

fash_pred.head()

fash_pred.columns    #Index(['C785', 'result'], dtype='object')

                     # 2 columns 'C785', 'result'

fash_pred.head(5)
# 17. So compare ground truth with predicted --accuracy of the model

out = (fash_pred['result'] == fash_pred['C785'])

np.sum(out)/out.size
# 18  create confusion matrix using pandas dataframe

g  = confusion_matrix(fash_pred['C785'], fash_pred['result'] )

g
fash_tmp.shape  # Numpy ndarray  70000 X 784
###Now We will understand PCA

#Dimensionality Reduction and PCA for Fashion MNIST

#Principal Components Analysis is the simplest example of dimensionality reduction. 

#Dimensionality reduction is a the problem of taking a matrix with many observations, and "compressing it" 

#to a matrix with fewer observations which preserves as much of the information in the full matrix as possible.



#Principal components is the most straightforward of the methodologies for doing so. 

#It relies on finding an orthonormal basis (a set of perpendicular vectors) within the dimensional space 

#of the dataset which explain the largest possible amount of variance in the dataset. 

#For example, here is PCA applied to a small two-dimensional problem:

#PCA then remaps the values of the points in the dataset to their projection onto the newly minted bases, 

#spitting out a matrix of observations with as few variables as you desire!

#we pick just two principal components, and try to map the result
import pandas as pd

test = pd.read_csv("../input/fashion-mnist_test.csv")

X = test.iloc[:, 1:]

y = test.iloc[:, :1]

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X_r = pca.fit(X).transform(X)
#To visualize the result of applying PCA to our datasets, 

#we'll plot how much weight each pixel in the clothing picture gets in the resulting vector, 

#using a heatmap. 

#This creates a nice, potentially interpretable picture of what each vector is "finding".

pca.explained_variance_ratio_
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



fig, axarr = plt.subplots(1, 2, figsize=(12, 4))



sns.heatmap(pca.components_[0, :].reshape(28, 28), ax=axarr[0], cmap='gray_r')

sns.heatmap(pca.components_[1, :].reshape(28, 28), ax=axarr[1], cmap='gray_r')

axarr[0].set_title(

    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[0]*100),

    fontsize=12

)

axarr[1].set_title(

    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[1]*100),

    fontsize=12

)

axarr[0].set_aspect('equal')

axarr[1].set_aspect('equal')



plt.suptitle('2-Component PCA')
#The first component looks like...some kind of large clothing object 

#(e.g. not a shoe or accessor). 

#The second component looks like negative space around a pair of pants.
pca = PCA(n_components=4)

X_r = pca.fit(X).transform(X)



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



fig, axarr = plt.subplots(2, 2, figsize=(12, 8))



sns.heatmap(pca.components_[0, :].reshape(28, 28), ax=axarr[0][0], cmap='gray_r')

sns.heatmap(pca.components_[1, :].reshape(28, 28), ax=axarr[0][1], cmap='gray_r')

sns.heatmap(pca.components_[2, :].reshape(28, 28), ax=axarr[1][0], cmap='gray_r')

sns.heatmap(pca.components_[3, :].reshape(28, 28), ax=axarr[1][1], cmap='gray_r')



axarr[0][0].set_title(

    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[0]*100),

    fontsize=12

)

axarr[0][1].set_title(

    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[1]*100),

    fontsize=12

)

axarr[1][0].set_title(

    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[2]*100),

    fontsize=12

)

axarr[1][1].set_title(

    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[3]*100),

    fontsize=12

)

axarr[0][0].set_aspect('equal')

axarr[0][1].set_aspect('equal')

axarr[1][0].set_aspect('equal')

axarr[1][1].set_aspect('equal')



plt.suptitle('4-Component PCA')

pass
#Both of these components look like they have some sort of shoe-related thing going on.

#Each additional dimension we add to the PCA captures less and less of the variance in the model. 

#The first component is the most important one, followed by the second, then the third, and so on.

#When using PCA, it's a good idea to pick a decomposition with a reasonably small number 

#of variables by looking for a "cutoff" in the effectiveness of the model. 

#To do that, you can plot the explained variance ratio (out of all variance) 

#for each of the components of the PCA. 



import numpy as np



pca = PCA(n_components=10)

X_r = pca.fit(X).transform(X)



plt.plot(range(10), pca.explained_variance_ratio_)

plt.plot(range(10), np.cumsum(pca.explained_variance_ratio_))

plt.title("Component-wise and Cumulative Explained Variance")

pass
#For the diagnoses that follow, we will use a standardly normalized 120-component PCA.

from sklearn.preprocessing import normalize 

X_norm = normalize(X.values)

# X_norm = X.values / 255



from sklearn.decomposition import PCA



pca = PCA(n_components=120)

X_norm_r = pca.fit(X_norm).transform(X_norm)
sns.heatmap(pd.DataFrame(X_norm).mean().values.reshape(28, 28), cmap='gray_r')
sns.heatmap(pd.DataFrame(X).std().values.reshape(28, 28), cmap='gray_r')
#Assessing fit by reconstructing individual sample images

#PCA compresses the 28x28 pixel values in the dataset to n vectors across those pixels. 

#A way of diagnosing how well we did this is to compare original images with ones reconstructed from those vectors.

def reconstruction(X, n, trans):

    """

    Creates a reconstruction of an input record, X, using the topmost (n) vectors from the

    given transformation (trans)

    

    Note 1: In this dataset each record is the set of pixels in the image (flattened to 

    one row).

    Note 2: X should be normalized before input.

    """

    vectors = [trans.components_[n] * X[n] for n in range(0, n)]

    

    # Invert the PCA transformation.

    ret = trans.inverse_transform(X)

    

    # This process results in non-normal noise on the margins of the data.

    # We clip the results to fit in the [0, 1] interval.

    ret[ret < 0] = 0

    ret[ret > 1] = 1

    return ret
#For example, here is how well a 120-variable reconstruction 

#(15% as many variables as in the root dataset) does for the first image in the dataset,a T-shirt:

fig, axarr = plt.subplots(1, 2, figsize=(12, 4))



sns.heatmap(X_norm[0, :].reshape(28, 28), cmap='gray_r',

            ax=axarr[0])

sns.heatmap(reconstruction(X_norm_r[0, :], 120, pca).reshape(28, 28), cmap='gray_r',

            ax=axarr[1])

axarr[0].set_aspect('equal')

axarr[0].axis('off')

axarr[1].set_aspect('equal')

axarr[1].axis('off')
def n_sample_reconstructions(X, n_samples=5, trans_n=120, trans=None):

    """

    Returns a tuple with `n_samples` reconstructions of records from the feature matrix X,

    as well as the indices sampled from X.

    """

    sample_indices = np.round(np.random.random(n_samples)*len(X))

    return (sample_indices, 

            np.vstack([reconstruction(X[int(ind)], trans_n, trans) for ind in sample_indices]))





def plot_reconstructions(X, n_samples=5, trans_n=120, trans=None):

    """

    Plots `n_samples` reconstructions.

    """

    fig, axarr = plt.subplots(n_samples, 3, figsize=(12, n_samples*4))

    ind, reconstructions = n_sample_reconstructions(X, n_samples, trans_n, trans)

    for (i, (ind, reconstruction)) in enumerate(zip(ind, reconstructions)):

        ax0, ax1, ax2 = axarr[i][0], axarr[i][1], axarr[i][2]

        sns.heatmap(X_norm[int(ind), :].reshape(28, 28), cmap='gray_r', ax=ax0)

        sns.heatmap(reconstruction.reshape(28, 28), cmap='gray_r', ax=ax1)

        sns.heatmap(np.abs(X_norm[int(ind), :] - reconstruction).reshape(28, 28), 

                    cmap='gray_r', ax=ax2)

        ax0.axis('off')

        ax0.set_aspect('equal')

        ax0.set_title("Original Image", fontsize=12)

        ax1.axis('off')

        ax1.set_aspect('equal')

        ax1.set_title("120-Vector Reconstruction", fontsize=12)

        ax2.axis('off')

        ax2.set_title("Original-Reconstruction Difference", fontsize=12)

        ax2.set_aspect('equal')
plot_reconstructions(X_norm_r, n_samples=10, trans_n=120, trans=pca)
#Assessing how to interpret individual components using record similarity

from sklearn.metrics import mean_squared_error



def quartile_record(X, vector, q=0.5):

    """

    Returns the data which is the q-quartile fit for the given vector.

    """

    errors = [mean_squared_error(X_norm[i, :], vector) for i in range(len(X_norm))]

    errors = pd.Series(errors)

    

    e_value = errors.quantile(q, interpolation='lower')

    return X[errors[errors == e_value].index[0], :]
sns.heatmap(quartile_record(X_norm, pca.components_[0], q=0.98).reshape(28, 28), 

            cmap='gray_r')
#Extending the idea to the first eight components in the dataset.

def plot_quartiles(X, trans, n):



    fig, axarr = plt.subplots(n, 7, figsize=(12, n*2))

    for i in range(n):

        vector = trans.components_[i, :]

        sns.heatmap(quartile_record(X, vector, q=0.02).reshape(28, 28), 

            cmap='gray_r', ax=axarr[i][0], cbar=False)

        axarr[i][0].set_aspect('equal')

        axarr[i][0].axis('off')

        

        sns.heatmap(quartile_record(X, vector, q=0.1).reshape(28, 28), 

            cmap='gray_r', ax=axarr[i][1], cbar=False)

        axarr[i][1].set_aspect('equal')

        axarr[i][1].axis('off')

        

        sns.heatmap(quartile_record(X, vector, q=0.25).reshape(28, 28), 

            cmap='gray_r', ax=axarr[i][2], cbar=False)

        axarr[i][2].set_aspect('equal')

        axarr[i][2].axis('off')

        

        sns.heatmap(quartile_record(X, vector, q=0.5).reshape(28, 28), 

            cmap='gray_r', ax=axarr[i][3], cbar=False)

        axarr[i][3].set_aspect('equal')

        axarr[i][3].axis('off')



        sns.heatmap(quartile_record(X, vector, q=0.75).reshape(28, 28), 

            cmap='gray_r', ax=axarr[i][4], cbar=False)

        axarr[i][4].set_aspect('equal')

        axarr[i][4].axis('off')



        sns.heatmap(quartile_record(X, vector, q=0.9).reshape(28, 28), 

            cmap='gray_r', ax=axarr[i][5], cbar=False)

        axarr[i][5].set_aspect('equal')

        axarr[i][5].axis('off')        

        

        sns.heatmap(quartile_record(X, vector, q=0.98).reshape(28, 28), 

            cmap='gray_r', ax=axarr[i][6], cbar=False)        

        axarr[i][6].set_aspect('equal')

        axarr[i][6].axis('off')

        

    axarr[0][0].set_title('2nd Percentile', fontsize=12)

    axarr[0][1].set_title('10th Percentile', fontsize=12)

    axarr[0][2].set_title('25th Percentile', fontsize=12)

    axarr[0][3].set_title('50th Percentile', fontsize=12)

    axarr[0][4].set_title('75th Percentile', fontsize=12)

    axarr[0][5].set_title('90th Percentile', fontsize=12)

    axarr[0][6].set_title('98th Percentile', fontsize=12)
plot_quartiles(X_norm, pca, 8)
#Assessing component fit using record similarity

def record_similarity(X, vector, metric=mean_squared_error):

    """

    Returns the record similarity to the vector, using MSE is the measurement metric.

    """

    return pd.Series([mean_squared_error(X_norm[i, :], vector) for i in range(len(X_norm))])
fig, axarr = plt.subplots(2, 4, figsize=(12, 6))

axarr = np.array(axarr).flatten()



for i in range(0, 8):

    record_similarity(X_norm, pca.components_[i]).plot.hist(bins=50, ax=axarr[i])

    axarr[i].set_title("Component {0} Errors".format(i + 1), fontsize=12)

    axarr[i].set_xlabel("")

    axarr[i].set_ylabel("")
# Conclusion

# The advantage of PCA (and dimensionality reduction in general) is that it compresses your data

#down to something that is more effectively modeled. This means that it will, 

#for example, compress away highly correlated and colinear variables, 

#a useful thing to do when trying to run models that would otherwise be sensitive to these data problems.


