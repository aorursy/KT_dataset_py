import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
train = pd.read_csv('../input/digit-recognizer/train.csv')
print(train.head())
print(train.info())
print("\n SHape of the dataset:", train.shape)
#NaN values in the dataset ?
nan = train.isnull().sum()
print(nan[nan != 0])
#Displays 4 handwritten digit images
def display_digits(N):
    """Picks-up randomly N images within the 
    train dataset between 0 and 41999 and displays the images
    with 4 images/row"""
    
    train = pd.read_csv('../input/digit-recognizer/train.csv')
    images = np.random.randint(low=0, high=42001, size=N).tolist()
    
    subset_images = train.iloc[images,:]
    subset_images.index = range(1, N+1)
    print("Handwritten picked-up digits: ", subset_images['label'].values)
    subset_images.drop(columns=['label'], inplace=True)

    for i, row in subset_images.iterrows():
        plt.subplot((N//8)+1, 8, i)
        pixels = row.values.reshape((28,28))
        plt.imshow(pixels, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.title('Randomly picked-up images from the training dataset')
    plt.show()

    return ""
display_digits(40)
#Analyse the pixels intensity values
subset_pixels = train.iloc[:, 1:]
subset_pixels.describe()
#Distribution of the digits in the dataset
_ = train['label'].value_counts().plot(kind='bar')
plt.show()
def remove_constant_pixels(pixels_df):
    """Removes from the images the pixels that have a constant intensity value,
    either always black (0) or white (255)
    Returns the cleared dataset & the list of the removed pixels (columns)"""

    #Remove the pixels that are always black to compute faster
    changing_pixels_df = pixels_df.loc[:]
    dropped_pixels_b = []

    #Pixels with max value =0 are pixels that never change
    for col in pixels_df:
        if changing_pixels_df[col].max() == 0:
            changing_pixels_df.drop(columns=[col], inplace=True)
            dropped_pixels_b.append(col)
    print("Constantly black pixels that have been dropped: {}".format(dropped_pixels_b))


    #Same with pixels with min=255 (white pixels)
    dropped_pixels_w = []
    for col in changing_pixels_df:
        if changing_pixels_df[col].min() == 255:
            changing_pixels_df.drop(columns=[col], inplace=True)
            dropped_pixel_w.append(col)
    print("\n Constantly white pixels that have been dropped: {}".format(dropped_pixels_b))

    print(changing_pixels_df.head())
    print("Remaining pixels: {}".format(len(changing_pixels_df.columns)))
    print("Pixels removed: {}".format(784-len(changing_pixels_df.columns)))
    
    return changing_pixels_df, dropped_pixels_b + dropped_pixels_w
train_pixels_df = pd.read_csv('../input/digit-recognizer/train.csv').drop(columns=['label'])
train_changing_pixels_df, dropped_pixels = remove_constant_pixels(train_pixels_df)
#To save time and not have to run the entire function
DROPPED_PIX = ['pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7', 'pixel8', 'pixel9', 
               'pixel10', 'pixel11', 'pixel16', 'pixel17', 'pixel18', 'pixel19', 'pixel20', 'pixel21', 'pixel22', 
               'pixel23', 'pixel24', 'pixel25', 'pixel26', 'pixel27', 'pixel28', 'pixel29', 'pixel30', 'pixel31', 
               'pixel52', 'pixel53', 'pixel54', 'pixel55', 'pixel56', 'pixel57', 'pixel82', 'pixel83', 'pixel84', 
               'pixel85', 'pixel111', 'pixel112', 'pixel139', 'pixel140', 'pixel141', 'pixel168', 'pixel196', 
               'pixel392', 'pixel420', 'pixel421', 'pixel448', 'pixel476', 'pixel532', 'pixel560', 'pixel644', 
               'pixel645', 'pixel671', 'pixel672', 'pixel673', 'pixel699', 'pixel700', 'pixel701', 'pixel727', 
               'pixel728', 'pixel729', 'pixel730', 'pixel731', 'pixel754', 'pixel755', 'pixel756', 'pixel757', 
               'pixel758', 'pixel759', 'pixel760', 'pixel780', 'pixel781', 'pixel782', 'pixel783']
train_changing_pixels_df = pd.read_csv('../input/changing-pixels/train_changing_pixels_DB.csv', index_col=0)
print(train_changing_pixels_df.head())
train = pd.read_csv('../input/digit-recognizer/train.csv')
print(train.head())
#Pick-up one random image from original training set
i = np.random.randint(low=0, high=42001, size=1).tolist()[0]
pixels = train.iloc[i, 1:]
image = train.iloc[i, 1:].values.reshape((28,28))

#Pixel intensity hstogram
plt.hist(pixels, bins=256, range=(0,256), normed=True)
plt.title('original image - pixel intensity distribution')
plt.show()

#Rescaling the intensity
pmin, pmax = image.min(), image.max()
rescaled_image = 255*(image-pmin) / (pmax - pmin)
rescaled_pixels = rescaled_image.flatten()

#Only black or white pixels
bw_pixels = pixels.apply(lambda x: 0 if x<128 else 255)
bw_image = bw_pixels.values.reshape((28,28))


#Visual comparison of images
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('original image')
plt.subplot(1, 3, 2)
plt.imshow(rescaled_image, cmap='gray')
plt.title('rescaled image')
plt.subplot(1, 3, 3)
plt.imshow(bw_image, cmap='gray')
plt.title('black&wite only image')
plt.show()


#Visual Histogram comparison
plt.subplot(1, 2, 1)
plt.hist(pixels, bins=256, range=(0,256), normed=True)
plt.title('original image')
plt.subplot(1, 2, 2)
plt.hist(rescaled_pixels, bins=256, range=(0,256), normed=True)
plt.title('rescaled image')
plt.show()
#Dummy target ?

#Preparing samples and labels arrays
#s = np.random.randint(low=0, high=42001, size=1050).tolist()
samples = train_changing_pixels_df.values  #.iloc[s, :]
digits = train['label'].tolist()  #.iloc[s, :]
print(samples.shape)
#PCA model
pca = PCA()
pca.fit(samples)

#PCA features variance visualization
pca_features = range(pca.n_components_)
_ = plt.figure(figsize=(30,20))
_ = plt.bar(pca_features, pca.explained_variance_)
_ = plt.xticks(pca_features)
_ = plt.title('Principal Components Analysis for Dimension Reduction')
_ = plt.xlabel('PCA features')
_ = plt.ylabel('Variance of the PCA feature')
_ = plt.savefig('visualizations/PCA features variance.png')
plt.show()

#PCA features variance visualization - ZOOM in
l= 100
x = range(l)
_ = plt.figure(figsize=(30,20))
_ = plt.bar(x, pca.explained_variance_[:l])
_ = plt.xticks(x)
_ = plt.title('Principal Components Analysis for Dimension Reduction - Zoom In {} first features'.format(l))
_ = plt.xlabel('PCA features')
_ = plt.ylabel('Variance of the PCA feature')
_ = plt.savefig('visualizations/PCA features variance_zoom.png')
plt.show()
#Visualization of the variance of the data carried by the number of PCA features
n_components = np.array([1,2,3,4,5,6, 10, 30, 60, 80, 100, 200, 400, 700])
cumul_variance = np.empty(len(n_components))
for i, n in enumerate(n_components):
    pca = PCA(n_components=n)
    pca.fit(samples)
    cumul_variance[i] = np.sum(pca.explained_variance_ratio_)

print(cumul_variance)

_ = plt.figure(figsize=(30,20))
_ = plt.grid(which='both')
_ = plt.plot(n_components, cumul_variance, color='red')
_ = plt.xscale('log')
_ = plt.xlabel('Number of PCA features', size=20)
_ = plt.ylabel('Cumulated variance of data (%)', size=20)
_ = plt.title('Data variance cumulated vs number of PCA features', size=20)
plt.savefig('visualizations/cumulated variance_pca features.png')
plt.show()
#Preparing samples and labels arrays
s = np.random.randint(low=0, high=42001, size=8200).tolist()
samples = train.drop(columns='label').values  #.iloc[s, :]
digitsa = train['label'].tolist()  #.iloc[s, :]
print(samples.shape)
#Creating the NMF model and the features & components
nmf = NMF(n_components=16)
nmf_features = nmf.fit_transform(samples)
nmf_components = nmf.components_
print("Shape of NMF features: {}, shape of NMF components: {}".format(nmf_features.shape, nmf_components.shape))

#Visualization of the features
for i, component in enumerate(nmf_components):
    N = nmf_components.shape[0]
    ax = plt.subplot((N//3)+1, 3, i+1)
    bitmap = component.reshape((28,28))
    plt.imshow(bitmap, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.title('NMF Components from the original images')
plt.show()
%%time

#Sample randomly the dataset using discrete_uniform pick-up to reduce the amount of data
#sample = np.random.randint(low=0, high=42001, size=8400).tolist()
X = train_changing_pixels_df.values  #.iloc[sample, :]
X = X / 255.0
y = train['label'].values  #.iloc[sample, :]
print("Shape of X and Y arrays: {}".format((X.shape, y.shape)))

#Split the training set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)

#Yielding the scores according to number of NMF components
components = np.arange(1, 100)
scores = np.empty(len(components))
for n in components:
    pipeline = make_pipeline(NMF(n_components=n), SVC(kernel='rbf', cache_size=1000))
    pipeline.fit(X_train, y_train)
    scores[n-1] = pipeline.score(X_test, y_test)

#Plotting of the scores evlution
_ = plt.figure(figsize=(30,20))
_ = plt.grid(which='both')
_ = plt.plot(components, scores)
_ = plt.xlabel('Number of NMF components', size=20)
_ = plt.ylabel('Score obtained', size=20)
_ = plt.title('Evolution of SVC classification score (samples={})'.format(len(y)), 
              size=30)
plt.savefig('visualizations/Score vs components NMF.png')
plt.show()

print("Best score {} obtained for {} components".format(scores.max(), scores.argmax()+1))
#Sparsity visuaization & sparse matrix creation
samples = train_changing_pixels_df.values
_ = plt.figure(figsize=(10,100))
_ = plt.spy(samples)
plt.show()

sparse_samples = csr_matrix(samples)

#Memory Size comparison
dense_size = samples.nbytes/1e6
sparse_size = (sparse_samples.data.nbytes + 
               sparse_samples.indptr.nbytes + sparse_samples.indices.nbytes)/1e6
print("From {} to {} Mo in memory usage with the sparse matrix".format(dense_size, sparse_size))

#Dimension reduction using PCA equivalent for sparse matrix
model = TruncatedSVD(n_components=10)
model.fit(sparse_samples)
reduced_sparse_samples = model.transform(sparse_samples)
print(reduced_sparse_samples.shape)
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler
from sklearn.decomposition import PCA, NMF, TruncatedSVD

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
#Sample randomly the dataset using discrete_uniform pick-up
sample = np.random.randint(low=0, high=42001, size=2100).tolist()

#Prepare the X (features) and y (label) arrays for the sampled images
X = train_changing_pixels_df.iloc[sample, :].values
y = train.loc[sample, 'label'].values#.reshape(-1,1)
print("Shape of X and Y arrays: {}".format((X.shape, y.shape)))

#Split the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)

#Fine tune the k value
param_grid = {'n_neighbors': np.arange(1,10)}
knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_cv.fit(X_train, y_train)

#Best k parameter
best_k = knn_cv.best_params_
best_accuracy = knn_cv.best_score_
print("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))
%%time

#Sample randomly the dataset using discrete_uniform pick-up
#sample = np.random.randint(low=0, high=42001, size=4200).tolist()

#Prepare the X (features) and y (label) arrays for 4000 images
X = train_simplified_pixels_df.values #.iloc[sample, 1:]
y = train.loc[:, 'label'].values#.reshape(-1,1)
print("Shape of X and Y arrays: {}".format((X.shape, y.shape)))

#Split the training set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)

#Fit the model (no hyperparameter tuning for this model)
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') #best results with these arguments
#lda = QuadraticDiscriminantAnalysis()#reg_parameter=?
lda.fit(X_train, y_train)
score = lda.score(X_test, y_test)
print("Accuracy on test set: {}".format(score))

#Best cv scores
lda_cv_scores = cross_val_score(lda, X_train, y_train, cv=5)
best_accuracy = lda_cv_scores.max()
print("Best accuracy during CV is {}".format(best_accuracy))
%%time

#Sample randomly the dataset using discrete_uniform pick-up
sample = np.random.randint(low=0, high=42001, size=2100).tolist()

#Prepare the X (features) and y (label) arrays for the sampled images
X = train_simplified_pixels_df.iloc[sample, :].values
y = train.loc[sample, 'label'].values#.reshape(-1,1)
print("Shape of X and Y arrays: {}".format((X.shape, y.shape)))

#Split the training set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)

#Fine tune the hyperparameters using RandomizeSearchCV rather than GridSearchCV (too expensive with more than 1 hyperparameter)
param_grid = {'C': np.logspace(0, 3, 20),
              'gamma':np.logspace(0, -4, 20)#, not for linear kernel
              #'degree': [2,3,4,5]  #only for poly kernel
              #'coef0': []  #only for poly & sigmoid kernels
             }
svm_cv = RandomizedSearchCV(SVC(kernel='rbf', cache_size=3000), 
                            param_grid, cv=5)
svm_cv.fit(X_train, y_train)

#Best k parameter
best_k = svm_cv.best_params_
best_accuracy = svm_cv.best_score_
print("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))
%%time

#Sample randomly the dataset using discrete_uniform pick-up
sample = np.random.randint(low=0, high=42001, size=4100).tolist()

#Prepare the X (features) and y (label) arrays for the sampled images
X = train_simplified_pixels_df.iloc[sample, :].values
y = train.loc[sample, 'label'].values#.reshape(-1,1)
print("Shape of X and Y arrays: {}".format((X.shape, y.shape)))

#Split the training set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)

#Fine tune the hyperparameters using RandomizeSearchCV rather than GridSearchCV (too expensive with more than 1 hyperparameter)
param_grid = {'multi_class': ['ovr', 'crammer_singer'],
              'penalty': ['l1', 'l2'],
              'C': np.logspace(0, 4, 50)}

linsvc_cv = GridSearchCV(LinearSVC(dual=False), param_grid, cv=5)
#linsvc_cv = RandomizedSearchCV(LinearSVC(dual=False), param_grid, cv=5)
linsvc_cv.fit(X_train, y_train)

#Best k parameter
best_k = linsvc_cv.best_params_
best_accuracy = linsvc_cv.best_score_
print("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))
%%time

#Sample randomly the dataset using discrete_uniform pick-up
#sample = np.random.randint(low=0, high=42001, size=4200).tolist()

#Prepare the X (features) and y (label) arrays for 4000 images
X = train_changing_pixels_df.values  #.iloc[sample, :]
y = train.loc[:, 'label'].values#.reshape(-1,1)
print("Shape of X and Y arrays: {}".format((X.shape, y.shape)))

#Split the training set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)

#Fine tune the hyperparameters
param_grid = {'max_depth': np.arange(3, 50),
              'min_samples_leaf': np.arange(5, 50, 1),
              'min_samples_split': np.arange(2,50, 1)
             }
tree_cv = RandomizedSearchCV(DecisionTreeClassifier(),
                       param_grid, cv=5)
tree_cv.fit(X_train, y_train)

#Best k parameter
best_k = tree_cv.best_params_
best_accuracy = tree_cv.best_score_
print("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))
#To save time and not have to run the entire function
DROPPED_PIX = ['pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7', 'pixel8', 'pixel9', 
               'pixel10', 'pixel11', 'pixel16', 'pixel17', 'pixel18', 'pixel19', 'pixel20', 'pixel21', 'pixel22', 
               'pixel23', 'pixel24', 'pixel25', 'pixel26', 'pixel27', 'pixel28', 'pixel29', 'pixel30', 'pixel31', 
               'pixel52', 'pixel53', 'pixel54', 'pixel55', 'pixel56', 'pixel57', 'pixel82', 'pixel83', 'pixel84', 
               'pixel85', 'pixel111', 'pixel112', 'pixel139', 'pixel140', 'pixel141', 'pixel168', 'pixel196', 
               'pixel392', 'pixel420', 'pixel421', 'pixel448', 'pixel476', 'pixel532', 'pixel560', 'pixel644', 
               'pixel645', 'pixel671', 'pixel672', 'pixel673', 'pixel699', 'pixel700', 'pixel701', 'pixel727', 
               'pixel728', 'pixel729', 'pixel730', 'pixel731', 'pixel754', 'pixel755', 'pixel756', 'pixel757', 
               'pixel758', 'pixel759', 'pixel760', 'pixel780', 'pixel781', 'pixel782', 'pixel783']
train_changing_pixels_df = pd.read_csv('../input/changing-pixels/train_changing_pixels_DB.csv', index_col=0)
print(train_changing_pixels_df.head())
train = pd.read_csv('../input/digit-recognizer/train.csv')
print(train.head())
%%time

#Sample randomly the dataset using discrete_uniform pick-up to reduce the amount of data
#sample = np.random.randint(low=0, high=42001, size=21000).tolist()

#Prepare the X (features) and y (label) arrays for the images
X = csr_matrix(train_changing_pixels_df.values)  #use .iloc[sample, :] for reduced sample
#X = X / 255.0  #intensities recaled between 0 and 1, NMF don't take negative values
y = train['label'].values#.reshape(-1,1)   #idem if using sample
print("Shape of X and Y arrays: {}".format((X.shape, y.shape)))

#Split the training set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)

#Pipeline with fine tune
pipeline = Pipeline([#('scaler', StandardScaler()), 
                     ('pca', TruncatedSVD()),
                     ('svm', SVC(kernel='rbf', cache_size=3000))
                    ])
param_grid = {'pca__n_components': np.arange(5, 80),
              'svm__C': np.logspace(0, 4, 50),
              'svm__gamma':np.logspace(0, -4, 50)}
pipeline_cv = RandomizedSearchCV(pipeline, param_grid, cv=5)

#fitting
pipeline_cv.fit(X_train, y_train)

#Best k parameter
best_k = pipeline_cv.best_params_
best_accuracy = pipeline_cv.best_score_
print("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))
%%time

#Predict on the test dataset (holdout) that MUST contain as many columns (ie pixels) than in the training set
holdout = pd.read_csv('test.csv').drop(columns=DROPPED_PIX)
X_holdout = holdout.values
print(X_holdout.shape)

predictions = pipeline_cv.predict(X_holdout)
submission_df = pd.DataFrame({'ImageId': range(1,28001), 'Label': predictions})
print("Overview of the obtained predictions :\n", submission_df.head())

#Save as submission file for competition
submission_df.to_csv('submission_pca_svc_DB.csv', index=False)
