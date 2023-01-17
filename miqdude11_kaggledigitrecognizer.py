! pip install mglearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mglearn
# importing dataset
digits = pd.read_csv('../input/digit-recognizer/train.csv')
digits_test = pd.read_csv('../input/digit-recognizer/test.csv')
digits.head()
# separate the label and the pixel data
targets = digits.label
data = digits.drop('label', axis=1)
data.head()
# visualizing the label counts
barplot = targets.value_counts()
barplot.plot(kind='bar', rot=0)
plt.xlabel("Label")
plt.ylabel("Frequency")
# showing the data as image
plt.imshow(data.iloc[3].values.reshape(28, 28), cmap='gray', vmin=0, vmax=255)
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler((0, 255))
digits_sum = digits.groupby('label').agg(np.sum)
digits_sum = minmax.fit_transform(digits_sum.values)

fig, axes = plt.subplots(2, 5, figsize=(20,8))
indexes = [np.arange(0,5), np.arange(5,10)]
for i, axx in zip([0, 1], axes):
    for j, ax in enumerate(axx):
        ax.imshow(digits_sum[indexes[i][j]].reshape(28,28), interpolation='nearest',
                 aspect='auto', cmap="gray")
            
# before max pooling
fig, axes= plt.subplots(2,5, figsize=(20,8))
labels = [np.arange(0,5), np.arange(5,10)]
for i, axx in enumerate(axes):
    for j, ax in enumerate(axx):
        sig = digits[digits['label'] == labels[i][j]].drop('label', axis=1).iloc[0].values
        ax.imshow(sig.reshape(28, 28), cmap='gray', vmin=0, vmax=255)
# after max pooling
from skimage.measure import block_reduce
fig, axes= plt.subplots(2,5, figsize=(20,8))
labels = [np.arange(0,5), np.arange(5,10)]
for i, axx in enumerate(axes):
    for j, ax in enumerate(axx):
        sig = digits[digits['label'] == labels[i][j]].drop('label', axis=1).iloc[0].values.reshape(28, 28)
        
        # apply max pooling
        digits_max_pool = block_reduce(sig, (2,2), np.max) # reduced shape is 14x14 pixels
        ax.imshow(digits_max_pool, cmap='gray', vmin=0, vmax=255)
# test applying highpass filter
from scipy import signal

fig, axes= plt.subplots(2,5, figsize=(20,8))
labels = [np.arange(0,5), np.arange(5,10)]
for i, axx in enumerate(axes):
    for j, ax in enumerate(axx):
        sig = digits[digits['label'] == labels[i][j]].drop('label', axis=1).iloc[0].values.reshape(28, 28)
        # filter using high pass signal
        sos_filter = signal.butter(10, .03, 'hp', output='sos')
        filtered_signal = signal.sosfilt(sos_filter, sig)
        # visualization
        ax.imshow(filtered_signal.reshape(28, 28), cmap='gray', vmin=0, vmax=255)
fig, axes= plt.subplots(2,5, figsize=(20,8))
labels = [np.arange(0,5), np.arange(5,10)]
for i, axx in enumerate(axes):
    for j, ax in enumerate(axx):
        sig = digits[digits['label'] == labels[i][j]].drop('label', axis=1).iloc[0].values.reshape(28, 28)
        
        # apply max pooling
        digits_max_pool = block_reduce(sig, (2,2), np.max)
        
        # filter using high pass signal
        sos_filter = signal.butter(10, .03, 'hp', output='sos')
        filtered_signal = signal.sosfilt(sos_filter, digits_max_pool)
        
        # second max pooling
        # output shape is 7x7
        digits_max_pool_again = block_reduce(filtered_signal, (2,2), np.max)
        
        # apply 
        ax.imshow(digits_max_pool_again, cmap='gray', vmin=0, vmax=255)
# Since the data can be see as a vector and each
# of the features could be strongly correlated
# these two classifer are feasible for this data
# Due to my hardware limitation the number of sample
# used would be 60% from total
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

digits_sampled = digits.sample(1797, random_state=42)

pixels = digits_sampled.drop('label', axis=1)
labels = digits_sampled['label']

X_train, X_test, y_train, y_test = train_test_split(
    pixels, labels, random_state=0
)
from sklearn.metrics import SCORERS

print(SCORERS.keys())
# create a pipeline for the benchmark classifier
pipeline = Pipeline([('classifier', SVC())])
param_grid = [
    {
        'classifier': [SVC()],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__random_state': [42]
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 2, 1], # since the image size is 28 x 28 pixel 5 is too big
        'classifier__weights': ['uniform', 'distance'] 
    }
]

def find_best_model(X_train, X_test, y_train, y_test):
    grid_model = GridSearchCV(pipeline, param_grid= param_grid,
        scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
    grid_model.fit(X_train, y_train)
    
    y_pred = grid_model.predict(X_test)

    print("Best params:\n{}\n".format(grid_model.best_params_))
    print("Best cross-validation score: {:.3f}".format(grid_model.best_score_))
    print("Test-set score: {:.3f}".format(grid_model.score(X_test, y_test)))
    print("Classification report\n", classification_report(y_test, y_pred))
    plt.figure(figsize=(20,8))
    conf_matrix = mglearn.tools.heatmap(confusion_matrix(y_test, y_pred), cmap="gray", xlabel="Predicted Labels",
            ylabel="True Label", xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.gca().invert_yaxis()
%%time
# run the best model search
find_best_model(X_train, X_test, y_train, y_test)
# create a transformation class
# to transform the features using
# max pooling and high pass filter
from sklearn.base import BaseEstimator, TransformerMixin

class ImagePoolingFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, pooling_size=(2, 2), wn=.03, filter_type="hp"):
        """
            pooling_size: tuple, define kernel size
                          of the filter
            wn: number, critical value for the filter
            filter_type: string, specify the type of filter
                        highpass, lowpass, bandpass, bandstop
        """
        self.wn = wn
        self.pooling_size = pooling_size
        self.filter_type = filter_type
    
    def fit(self, X, y=None):
        return self
    
    def max_pooling_filter(self, X):
        # apply max pooling
        img = X.values.reshape(28, 28)
        img_pooled = block_reduce(img, (2,2), np.max)
        
        # filter using high pass signal
        sos_filter = signal.butter(10, .03, 'hp', output='sos')
        filtered_signal = signal.sosfilt(sos_filter, img_pooled)
        
        returned_series = pd.Series(filtered_signal.flatten())
        return returned_series
        
    def transform(self, X, y=None):
        """
            Transform our data using max pooling
            and filter
            
            returns the flatten image array
        """
        
        X_ = X.copy()
        X_ = X_.apply(lambda image_features: self.max_pooling_filter(image_features), axis=1)
    
        return X_
imagePoolingFilter = ImagePoolingFilter()
X_train_transformed = imagePoolingFilter.transform(X_train)
X_test_transformed = imagePoolingFilter.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train_transformed, y_train)

y_pred = knn.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)

print("accuracy ", accuracy)
print("classification report", classification_report(y_test, y_pred))
plt.figure(figsize=(20,8))
conf_matrix = mglearn.tools.heatmap(confusion_matrix(y_test, y_pred), cmap="gray", xlabel="Predicted Labels",
        ylabel="True Label", xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.gca().invert_yaxis()
from sklearn.preprocessing import FunctionTransformer

def Max_Pooling(X):
    # apply max pooling
    img = X.values.reshape(28, 28)
    img_pooled = block_reduce(img, (2,2), np.max)
    
    # flatten the output
    returned_series = pd.Series(img_pooled.flatten())
    return returned_series

def pooling(X):
    X_ = X.copy()
    return X_.apply(lambda image_features: Max_Pooling(image_features), axis=1)


X_train_transformed = FunctionTransformer(pooling).transform(X_train)
X_test_transformed = FunctionTransformer(pooling).transform(X_test)
%%time
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train_transformed, y_train)

y_pred = knn.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy ", accuracy)
print("classification report\n", classification_report(y_test, y_pred))
plt.figure(figsize=(20,8))
conf_matrix = mglearn.tools.heatmap(confusion_matrix(y_test, y_pred), cmap="gray", xlabel="Predicted Labels",
        ylabel="True Label", xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.gca().invert_yaxis()
def filtering(X):
    # filter using high pass signal
    img = X.values.reshape(28, 28)
    sos_filter = signal.butter(10, .03, 'hp', output='sos')
    filtered_signal = signal.sosfilt(sos_filter, img)
    
    # flatten the output
    returned_series = pd.Series(filtered_signal.flatten())
    return returned_series

def highpass_filter(X):
    X_ = X.copy()
    return X_.apply(lambda image_features: filtering(image_features), axis=1)


X_train_transformed = FunctionTransformer(highpass_filter).transform(X_train)
X_test_transformed = FunctionTransformer(highpass_filter).transform(X_test)
%%time
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train_transformed, y_train)

y_pred = knn.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy ", accuracy)
print("classification report\n", classification_report(y_test, y_pred))
plt.figure(figsize=(20,8))
conf_matrix = mglearn.tools.heatmap(confusion_matrix(y_test, y_pred), cmap="gray", xlabel="Predicted Labels",
        ylabel="True Label", xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.gca().invert_yaxis()
from sklearn.decomposition import PCA

pca = PCA(n_components=33, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train_pca, y_train)

score_train = accuracy_score(y_train, knn.predict(X_train_pca))
print("Train score ", score_train)

score_valid = accuracy_score(y_test, knn.predict(X_test_pca))
print("Validation score ", score_valid)
find_best_model(X_train_pca, X_test_pca, y_train, y_test)
from sklearn.decomposition import NMF

nmf = NMF(n_components=33, random_state=0).fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train_nmf, y_train)

score_train = accuracy_score(y_train, knn.predict(X_train_nmf))
print("Train score ", score_train)

score_valid = accuracy_score(y_test, knn.predict(X_test_nmf))
print("Validation score ", score_valid)
find_best_model(X_train_nmf, X_test_nmf, y_train, y_test)
X_test_data = digits_test.values
X_test_data_nmf = nmf.transform(X_test_data)

X_test_data_pca = pca.transform(X_test_data)
svc = SVC(C=10, gamma=0.01, random_state=42)
svc.fit(X_train_nmf, y_train)

y_pred = svc.predict(X_test_data_nmf)

predictions = pd.DataFrame({'ImageID': np.arange(1,28001),'Label': y_pred})
# predictions.to_csv('predictions.csv', index=False)

# got about ~.92
y_pred = knn.predict(X_test_data_pca)

predictions = pd.DataFrame({'ImageID': np.arange(1,28001),
                           'Label': y_pred})
# predictions.to_csv('predictions_knn.csv', index=False)

# got about ~.91