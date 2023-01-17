from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt # plotting

import matplotlib.image as mpimg



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os # accessing directory structure
DATASET_PATH = "/kaggle/input/myntradataset/"

print(os.listdir(DATASET_PATH))
df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=5000, error_bad_lines=False)

df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)

df = df.sample(frac=1).reset_index(drop=True)

df.head(10)
batch_size = 32
from keras_preprocessing.image import ImageDataGenerator



image_generator = ImageDataGenerator(

    validation_split=0.2

)



training_generator = image_generator.flow_from_dataframe(

    dataframe=df,

    directory=DATASET_PATH + "images",

    x_col="image",

    y_col="subCategory",

    target_size=(96,96),

    batch_size=batch_size,

    subset="training"

)



validation_generator = image_generator.flow_from_dataframe(

    dataframe=df,

    directory=DATASET_PATH + "images",

    x_col="image",

    y_col="subCategory",

    target_size=(96,96),

    batch_size=batch_size,

    subset="validation"

)



classes = len(training_generator.class_indices)
from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

from keras.applications.mobilenet_v2 import MobileNetV2



# create the base pre-trained model

base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(classes, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
from math import ceil



model.fit_generator(

    generator=training_generator,

    steps_per_epoch=ceil(0.8 * (df.size / batch_size)),



    validation_data=validation_generator,

    validation_steps=ceil(0.2 * (df.size / batch_size)),



    epochs=1,

    verbose=1

)



model.save('/kaggle/working/model.h5')
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

plotPerColumnDistribution(df)

plotCorrelationMatrix(df)

plotScatterMatrix(df)
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

plotPerColumnDistribution(df)

plotCorrelationMatrix(df)

plotScatterMatrix(df)