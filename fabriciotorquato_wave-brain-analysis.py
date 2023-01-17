from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def get_correlated_pairs_sample(threshold, correlation_df, group):

    ## create dictionary wheke keys are the pairs and values are the amount of high correlation pair

    corr_pairs_dict = {}

    for i in range(len(list_of_pairs)):

        temp_corr_pair = dict(zip(list_of_pairs[i], [0]))

        corr_pairs_dict.update(temp_corr_pair)



    j = 0

    for column in correlation_df.columns:

        j += 1

        for i in range(j, len(correlation_df)):

            if ((correlation_df[column][i] >= threshold) & (column != correlation_df.index[i])):

                corr_pairs_dict[str(column) + '-' + str(correlation_df.index[i])] += 1



    corr_count = pd.DataFrame(corr_pairs_dict, index=['count']).T.reset_index(drop=False).rename(columns={'index': 'channel_pair'})

    print('Channel pairs that have correlation value >= ' + str(threshold) + ' (' + group + '):')

    print(corr_count['channel_pair'][corr_count['count'] > 0].tolist())
def plot_sensors_correlation(df, threshold_value):

    """Funtion plots the the correlation plots between sensor positions for each group"""

    correlations = df.corr()





    fig = plt.figure(figsize=(17,10))

    ax = fig.add_subplot(121)

    ax.set_title('User A', fontsize=14)

    mask = np.zeros_like(correlations, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(correlations, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5})



    plt.suptitle('Correlation between Sensor Positions for stimulus', fontsize=16)

    plt.show()

    

    get_correlated_pairs_sample(threshold=threshold_value, correlation_df=correlations, group='User A')
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
df = pd.read_csv('/kaggle/input/eeg-data-from-hands-movement/Dataset/user_a.csv', delimiter=',', index_col=False)

df.dataframeName = 'dataset.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
df.describe()
df.info()
df.corr(method ='pearson')



list_of_pairs = []

X = df.iloc[:,1:]

Y = df.iloc[:,0]

j = 0

for column in X.columns:

    j += 1

    for i in range(j, len(X.columns)):

        if column != X.columns[i]:

            temp_pair = [column + '-' + str(X.columns[i])]

            list_of_pairs.append(temp_pair)
plot_sensors_correlation(df=X, threshold_value=.97)
import os

import numpy as np

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

import pandas as pd

from keras.metrics import top_k_categorical_accuracy

def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y, 3)

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from glob import glob

import gc

gc.enable()
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
def get_model():

    model = Sequential()

    model.add(LSTM(16,input_shape=(11,11), return_sequences=True))    

    model.add(LSTM(16))  

    model.add(Dense(3, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# Fit the model

X = df.iloc[:,1:]

Y = df.iloc[:,0]

l = ['complement'] * (121 - X.shape[1]) 

for index,col in enumerate(l):

    X[col+str(index)] = 0



X = X.values

Y = Y.values
# encode class values as integers

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)
get_model().summary()
from sklearn.model_selection import KFold

import numpy

# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)



# define 10-fold cross validation test harness

kfold = KFold(n_splits=3, shuffle=True, random_state=seed)

cvscores = []

best = -1



for train, test in kfold.split(X, dummy_y):

    model = get_model()

    standard = StandardScaler().fit(X[train])

    x_train_standard = standard.transform(X[train]).reshape(1920,11,11)

    x_test_standard = standard.transform(X[test]).reshape(960,11,11)

    model_history = model.fit(x_train_standard, dummy_y[train], epochs=25, batch_size=16, verbose=1)

    scores = model.evaluate(x_test_standard, dummy_y[test], verbose=1)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)

    if scores[1] > best:

        best = scores[1]

        history = model_history

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
from keras.utils import Sequence

class SeqGen(Sequence):



    def __init__(self, x_set, y_set, batch_size):

        self.x, self.y = x_set, y_set

        self.batch_size = batch_size



    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]



        return batch_x, batch_y
# Fit the model

model = get_model()



x_train, x_test, y_train, y_test = train_test_split(

            X, dummy_y, test_size=0.3, random_state=42, shuffle=True

        )

standard = StandardScaler().fit(x_train)

x_train_standard = standard.transform(x_train).reshape(2016,11,11)

x_test_standard = standard.transform(x_test).reshape(864,11,11)

history = model.fit_generator(SeqGen(x_train_standard,y_train,batch_size=16), validation_data=(x_test_standard,y_test), epochs=25, verbose=1)
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()