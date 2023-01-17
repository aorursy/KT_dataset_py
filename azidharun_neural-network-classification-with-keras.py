# import required package
import math
import pandas as pd
import numpy as np
from scipy import stats

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from IPython.display import Image
from mpl_toolkits.mplot3d import Axes3D

import itertools
import seaborn as sns

%matplotlib inline 
#!/usr/bin/python

# preprocess country data
def gendata(features, path):

    # read data
    c = pd.read_csv(path, decimal=",")

    # set output and classes
    output = ['Birthrate']
    classes = ['low', 'medium', 'high']

    # shorten feature names
    c.columns = ['Country', 'Region', 'Population', 'Area', 'Density', 'Coastline', 'Migration', 'InfantMortality', \
                 'GDP', 'Literacy', 'Phones', 'Arable', 'Crops', 'OtherLand', 'Climate', 'Birthrate', 'Deathrate', \
                 'Agriculture', 'Industry', 'Service']

    # strip all whitespace from all columns
    c = c.applymap(lambda x: x.strip() if type(x) is str else x)

#     # set index to country
#     c.set_index('Country', inplace=True)

    # reduce to feature and type columns
    dataset = c[features + output]

    # drop duplicates and null values
    dataset = dataset.drop_duplicates().dropna()
    # make new birthrate class column
    btype = []
    for b in dataset.Birthrate:
        if (b < 15):
            btype.append('low')
        elif ((b >= 15) and (b < 30)):
            btype.append('medium')
        elif (b >= 30):
            btype.append('high')

    # remove original birth rate column
    dataset = dataset.drop(columns=['Birthrate'])

    # NOTE - alternative if using pandas < 0.20
    #del dataset['Birthrate']

    # append to dataset
    dataset['BRClass'] = pd.Series(btype, index=dataset.index)

    # return values
    return dataset.values

# clearer and simpler version of featureplot
# keeping featureplot in for backwards compatibility
# restrict to 6 features
def featuresplot(data, target, features=None, classes=None):

    if (features is None):
        print("Please provide a list of feature names")
        return
    if (classes is None):
        print("Please provide a list of class names")
        return

    plt_colors = "rybgcm"
    n_classes = len(classes)

    if (len(features) > 6):
        print("Number of features is too high to plot")
        return

    # get pair list of permutations and get unique set
    n_features = data.shape[1]
    x = [sorted(i) for i in itertools.permutations(np.arange(n_features), r=2)]
    x.sort()
    pairs = list(k for k,_ in itertools.groupby(x))

    # set subplot layout
    sub_y = math.ceil(len(pairs)/4.)
    full_y = sub_y * 3.5

    # set figure size
    plt.figure(1, figsize=(15, full_y))

    # enumerate over combinations
    for pairidx, pair in enumerate(pairs):

        # extract data for pair
        datapair = data[:, pair]

        # define new plot
        plt.subplot(sub_y, 4, pairidx + 1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        # add labels
        plt.xlabel(features[pair[0]])
        plt.ylabel(features[pair[1]])

        # Plot the points
        for i, color in zip(range(n_classes), plt_colors):

            idx = np.where(target == classes[i])

            plt.scatter(datapair[idx, 0], datapair[idx, 1], c=color, label=classes[i],
                            cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")

    return

# heat map for confusion matrices and parameter scans
# adapted from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def heatmap(d, labels=None, classes=None, title=None,
            palette="Green",
            normalize=False,
            annot=True):

    if normalize:
        d = d.astype('float') / d.sum(axis=1)[:, np.newaxis]

    ax = plt.subplot()

    # define colour map
    my_cmap = sns.light_palette(palette, as_cmap=True)

    # plot heatmap
    sns.heatmap(d, annot=True, ax=ax, cmap=my_cmap)

    # labels, title and ticks
    if (labels is not None):
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if (title is not None):
        ax.set_title('Confusion Matrix')
    if (classes is not None):
        ax.xaxis.set_ticklabels(classes[0])
        ax.yaxis.set_ticklabels(classes[1])

    return

# Note - expect convergence warning at small training sizes
def compare_traintest(data, target, model, split=0, scale='linear', **params):
    
    #preprocess target data
    le = preprocessing.LabelEncoder()
    target_label = le.fit_transform(target)

    # convert integers to dummy variables (i.e. one hot encoded)
    target_label = np_utils.to_categorical(target_label).astype(float)
    
    # define 0.01 - 0.1, 0.1 - 0.9, 0.91 - 0.99 sample if split array not defined
    if (split == 0):
        split = np.concatenate((np.linspace(0.01,0.09,9), np.linspace(0.1,0.9,9), np.linspace(0.91,0.99,9)), axis=None)

    print("Parameters")
    print(params)
        
    print("Split sample:")
    print(split)

    train_scores = []
    test_scores = []

    for s in split:

        print("Running with test size of: %0.2f" % s)

        # get train/test for this split
        d = model_selection.train_test_split(data, target_label,
                                             test_size=s, random_state=0)

        # get training and test data and targets
        train_data, test_data, train_target, test_target = d

        # Data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale both the training inputs and outputs
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        
        # Train the model
        model.fit(
            train_data,
            train_target,
            **params,
            validation_data=(test_data, test_target)
        )

        train_error_rate = model.evaluate(train_data, train_target, verbose=0)
        test_error_rate = model.evaluate(test_data, test_target, verbose=0)

        # get test scores for fit and prediction
        train_scores.append(train_error_rate[1])
        test_scores.append(test_error_rate[1])

    # plot results
    plt.figure(figsize=(15.0, 5.0))
    if (scale == 'log'):
        plt.yscale('log')
    else:
        plt.yscale('linear')
    plt.plot(split, train_scores, label='Training accuracy', marker='o')
    plt.plot(split, test_scores, label='Testing accuracy', marker='o')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Test sample proportion')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, 1.0, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim([min(split),max(split)])
    plt.ylim([0,1.01])
    plt.grid()
    plt.legend()

    return

def evaluate_model_accuracy(model, data, target_label_1d, **params):
    print("k-Fold Cross Validation")
    print("Parameters")
    print(params)

    kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    cvscores = []
    for train, test in kfold.split(data, target_label_1d):
        # convert integers to dummy variables (i.e. one hot encoded)
        target_label = np_utils.to_categorical(target_label_1d).astype(float)
        
        # Data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale both the training inputs and outputs
        data[train] = scaler.fit_transform(data[train])
        data[test] = scaler.fit_transform(data[test])
        
        # Fit the model
        model.fit(data[train], target_label[train], epochs=800, verbose=0)

        # evaluate the model
        scores = model.evaluate(data[test], target_label[test], verbose=0)
        print('{0:} : {1:0.2f}%'.format(model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print('Model Accuracy : {0:0.2f}% (+/- {1:0.2f}%)'.format(np.mean(cvscores), np.std(cvscores)))
    
    return np.mean(cvscores)

# Load country dataset 
bclass = ['low', 'medium', 'high'] # birth rate class (provided)
features_all = ['Country', 'Population', 'Area', 'Density', 'Coastline', 'Migration', 'InfantMortality', 'GDP', 'Literacy', 'Phones', 'Arable', 'Crops', 'OtherLand', 'Climate', 'Deathrate', 'Agriculture', 'Industry', 'Service'] # fdefine features here 
path = '../input/countries of the world.csv'
data_all = gendata(features_all, path)
# Choose only 4 features
feature_idx = [6, 7, 8, 12]

# Show feature names
features = []
for i in feature_idx:
    features.append(features_all[i])
features
data, target = data_all[:, feature_idx].astype(float), data_all[:, [-1]].ravel()

#preprocess target data
le = preprocessing.LabelEncoder()
target_label_1d = le.fit_transform(target)

# convert integers to dummy variables (i.e. one hot encoded)
target_label = np_utils.to_categorical(target_label_1d).astype(float)

# # Show dataset dimensions
data.shape, target.shape, np.unique(target)
# generate 2D plots 
featuresplot(data, target, features, bclass)
plt.show()
# split data into training and test dataset
d = model_selection.train_test_split(
    data, target_label, test_size=0.2, random_state=0)

# get training and test data and targets
train_data, test_data, train_target, test_target = d

print(train_data.shape, train_target.shape, test_data.shape, test_target.shape)
# Data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
def baseline_model():
    # Define the model
    model = Sequential()
    model.add(Dense(350, kernel_initializer='uniform', input_dim=4, activation='relu'))
    model.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model
model = baseline_model()
RUN_NAME = 'run 6 epoch 800 nodes 350 relu,3 sigmoid, adagrad'

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(RUN_NAME),
    histogram_freq=5,
    write_graph=True
)
# Train the model
model.fit(
    train_data,
    train_target,
    epochs=800,
    shuffle=True,
    verbose=0,
#     callbacks=[logger],
    validation_data=(test_data, test_target)
)
test_error_rate = model.evaluate(test_data, test_target, verbose=0)
print("The categorical cross-entropy (CCE) for the test data set is: {0:0.4f}".format(test_error_rate[0]))
print("The accuracy of the test data set evaluation is: {0:0.2f}%".format(test_error_rate[1]*100))
model_acc = evaluate_model_accuracy(model, data, target_label_1d, epochs=800, shuffle=True, verbose=0)
# Save the model to disk
# model.save("{0:} - modelACC{1:0.2f}.h5".format(RUN_NAME, model_acc))
# print('Model saved to disk.')