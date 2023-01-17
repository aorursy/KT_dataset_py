from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# !rm -rf train_test_split

# print(os.listdir('../working'))
import os

from scipy import misc

import numpy as np

import sys

import keras

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.layers.wrappers import TimeDistributed



class MasterReader(object):

    def __init__(self, nc, ne, bs, lr):

        self.num_classes = nc

        self.num_epochs = ne

        self.batch_size = bs

        self.learning_rate = lr

        self.MAX_WIDTH = 90

        self.MAX_HEIGHT = 90

        self.max_seq_length = 20





    def load_data(self):

        data_dir = 'train_test_split'

        max_seq_length = self.max_seq_length



        X_train = []

        y_train = []

        X_val = []

        y_val = []

        X_test = []

        y_test = []



        if os.path.exists(data_dir):

            print("Loading saved data ...")

            X_train = np.load(f"{data_dir}/X_train.npy")

            y_train = np.load(f"{data_dir}/y_train.np'Fy")



            X_val = np.load(f"{data_dir}/X_val.npy")

            y_val = np.load(f"{data_dir}/y_val.npy")



            X_test = np.load(f"{data_dir}/X_test.npy")

            y_test = np.load(f"{data_dir}/y_test.npy")



            return X_train, y_train, X_val, y_val, X_test, y_test

#            print('Read data arrays from disk.')

        else:



            people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']

            data_types = ['phrases', 'words']

            folder_enum = ['01','02','03','04','05','06','07','08','09','10']



            UNSEEN_VALIDATION_SPLIT = ['F05']

            UNSEEN_TEST_SPLIT = ['F06']



            directory = '../input/cropped/cropped/cropped'

            for person_id in people:

                instance_index = 0

                for data_type in data_types:

                    for word_index, word in enumerate(folder_enum):

                        print(f"Instance #{instance_index}")

                        for iteration in folder_enum:

                            path = os.path.join(directory, person_id, data_type, word, iteration)

                            filelist = sorted(os.listdir(path + '/'))

                            sequence = []

                            for img_name in filelist:

                                if img_name.startswith('color'):

                                    image = misc.imread(path + '/' + img_name)

                                    image = misc.imresize(image, (self.MAX_WIDTH, self.MAX_HEIGHT))

                                    sequence.append(image)                                         

                            pad_array = [np.zeros((self.MAX_WIDTH, self.MAX_HEIGHT))]

                            sequence.extend(pad_array * (max_seq_length - len(sequence)))

                            sequence = np.stack(sequence, axis=0)



                            if person_id in UNSEEN_TEST_SPLIT:

                                X_test.append(sequence)

                                y_test.append(instance_index)

                            elif person_id in UNSEEN_VALIDATION_SPLIT:

                                X_val.append(sequence)

                                y_val.append(instance_index)

                            else:

                                X_train.append(sequence)

                                y_train.append(instance_index)

                        instance_index += 1

                print("......")

                print('Finished reading images for person ' + person_id)



            print('Finished reading images.')

#             print(np.shape(X_train))



            X_train = np.array(X_train[:])

            X_val = np.array(X_val)

            X_test = np.array(X_test)



            print('Finished stacking the data into the right dimensions. About to start saving to disk...')



            if not os.path.isdir(data_dir):

                os.mkdir(data_dir)

            np.save(data_dir+'/X_train', X_train)

            np.save(data_dir+'/y_train', np.array(y_train))

            np.save(data_dir+'/X_val', X_val)

            np.save(data_dir+'/y_val', np.array(y_val))

            np.save(data_dir+'/X_test', X_test)

            np.save(data_dir+'/y_test', np.array(y_test))

            print('Finished saving all data to disk. Returing.')



            return X_train, y_train, X_val, y_val, X_test, y_test





    def training_generator(self):



        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()



        while True:

            for i in range(int(np.shape(X_train)[0] / self.batch_size)):

                x = X_train[i * self.batch_size : (i+1) * self.batch_size]

                y = y_train[i * self.batch_size : (i+1) * self.batch_size]

                one_hot_labels = keras.utils.to_categorical(y, num_classes=self.num_classes)

                yield (x, one_hot_labels)





    def create_model(self):



        np.random.seed(0)



        bottleneck_train_path = 'bottleneck_features_train.npy'

        bottleneck_val_path = 'bottleneck_features_val.npy'

        top_model_weights = 'bottleneck_TOP_LAYER.h5'



        input_layer = keras.layers.Input(shape=(self.max_seq_length, self.MAX_WIDTH, self.MAX_HEIGHT))



        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(self.MAX_WIDTH, self.MAX_HEIGHT, 3))



        vgg = Model(input=vgg_base.input, output=vgg_base.output)



        for layer in vgg.layers[:15]:

            layer.trainable = False



        x = TimeDistributed(vgg)(input_layer)



        model = Model(input=input_layer, output=x)



        input_layer_2 = keras.layers.Input(shape=model.output_shape[1:])



        x = TimeDistributed(keras.layers.core.Flatten())(input_layer_2)



        lstm = keras.layers.Recurrent.LSTM(256)

        x = keras.layers.wrappers.Bidirectional(lstm, merge_mode='concat', weights=None)(x)

        x = keras.layers.core.Dropout(rate=0.2)(x)

        x = keras.layers.core.Dense(10)(x)



        preds = keras.layers.core.Activation('softmax')(x)



        model_top = Model(input=input_layer_2, output=preds)



        x = model(input_layer)

        preds = model_top(x)



        final_model = Model(input=input_layer, output=preds)



        adam = keras.optimizers.SGD(lr=self.learning_rate)

        

        final_model.compile(optimizer=adam, loss='categorical_crossentropy',

                            metrics=['accuracy'])



        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()

        

        one_hot_labels_val = keras.utils.to_categorical(y_val, num_classes=self.num_classes)



        history = final_model.fit_generator(self.training_generator(),

                                            steps_per_epoch=np.shape(X_train)[0] / self.batch_size,

                                            epochs=self.num_epochs,

                                            validation_data=(X_val, one_hot_labels_val))



#if __name__ == '__main__':

# lp.create_model()

lp = MasterReader(20, 10, 50, 0.001)

# X_train, y_train, X_val, y_val, X_test, y_test = lp.load_data()
# X_train.shape
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
