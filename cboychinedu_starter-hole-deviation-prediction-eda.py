#!/usr/bin/env python3 
# Author: Mbonu Chinedum Endurance 

# Project Name: Hole Deviation Prediction 

# Universty: Nnamdi Azikiwe University 

# Faculty: Engineering 

# Department: Chemical Engineering 

# Date Created: 18/01/2020 
# Importing the necessary packages

import os 

import pandas as pd 

import numpy as np 

import seaborn as sns 

import matplotlib.pyplot as plt 

import plotly.express as px 

import matplotlib.patches as pathes 

import plotly.graph_objects as go 

from matplotlib.patches import Circle, Wedge, Polygon, Rectangle 

from IPython.display import Image 

from matplotlib.collections import PathCollection 

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

from matplotlib.colors import ListedColormap 

from sklearn.preprocessing import StandardScaler 

from scipy.cluster.hierarchy import fcluster 

from sklearn.pipeline import make_pipeline 

from sklearn.cluster import KMeans 

from sklearn import preprocessing 

from plotly.subplots import make_subplots 

from imblearn.over_sampling import SMOTE 

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense 

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler



# Setting some basic parameters for the analysis 

smote = SMOTE(random_state = 33)

%matplotlib inline 
# Loading the dataset into memory 

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Specifying the path to the well log dataset 

dataset_path = '/kaggle/input/well_log.csv'



# Reading the well log dataset into memroy 

df = pd.read_csv(dataset_path)

df = df.drop('Unnamed: 0', axis = 1)

# viewing the head of the dataset 

df.head() 
# # Distribution graphs (histogram/bar graph) of column data

# def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

#     nunique = df.nunique()

#     df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

#     nRow, nCol = df.shape

#     columnNames = list(df)

#     nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

#     plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

#     for i in range(min(nCol, nGraphShown)):

#         plt.subplot(nGraphRow, nGraphPerRow, i + 1)

#         columnDf = df.iloc[:, i]

#         if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

#             valueCounts = columnDf.value_counts()

#             valueCounts.plot.bar()

#         else:

#             columnDf.hist()

#         plt.ylabel('counts')

#         plt.xticks(rotation = 90)

#         plt.title(f'{columnNames[i]} (column {i})')

#     plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

#     plt.show()



# plotPerColumnDistribution(df, 100, 100)
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.columns

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



plotCorrelationMatrix(df, 20)
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



plotScatterMatrix(df, 20, 7)
# Displaying the unique values for the classification class  

# Here, 0 stands for Non-deviated class, while 1 stands for Deviated class 

classified_label = df['Classification'].unique()

print('The labels are: {}'.format(classified_label))
# Converting the columns into numpy arrays 

depth = df['Depth'].values 

gamma_ray = df['Gamma-ray'].values 

shale_volume = df['Shale_Volume'].values

restivity = df['Restivity'].values 

Delta_t = df['Delta T'].values 

vp = df['Vp'].values 

vs = df['Vs'].values 

density = df['Density'].values 

density_calculated = df['Density_Calculated'].values 

neuron_porosity = df['Neuron_Porosity'].values 

density_porosity = df['Density_Porosity'].values 

possion_ratio = df['Possions_Ratio'].values 

classification = df['Classification'].values 



# making 11 column and only 1 row 

fig = make_subplots(rows=1, cols=9)



# plotting the graph of Gamma ray against depth 

fig.append_trace(go.Scatter(

    x=gamma_ray,

    y=depth,

    name='Gamma-Ray',

), row=1, col=1)

# plotting the graph of shale volume against depth 

fig.append_trace(go.Scatter(

    x=shale_volume,

    y=depth,

    name='Shale_Volume',

), row=1, col=2)

# plotting the graph of resitivity against depth 

fig.append_trace(go.Scatter(

    x=restivity,

    y=depth, 

    name='Restivity', 

), row=1, col=3)

# plotting the graph of temperature against depth 

fig.append_trace(go.Scatter(

    x=Delta_t, 

    y=depth,

    name='Temperature', 

), row=1, col=4)

# plotting the graph of velocity against depth 

fig.append_trace(go.Scatter(

    x=vp, 

    y=depth,

    name='Velocity', 

), row=1, col=5)

# plotting the graph of density against depth 

fig.append_trace(go.Scatter(

    x=density, 

    y=depth,

    name='Density', 

), row=1, col=6)

# plotting the graph of neuron_porosity against depth 

fig.append_trace(go.Scatter(

    x=neuron_porosity, 

    y=depth, 

    name='Neuron Porosity',

),  row=1, col=7)

# plotting the graph of possion_ratio against depth 

fig.append_trace(go.Scatter(

    x=possion_ratio, 

    y=depth, 

    name='Possion Ratio',

),  row=1, col=8)

# plotting the graph of classification against depth 

fig.append_trace(go.Scatter(

    x=classification, 

    y=depth, 

    name='Hole Deviation',

),  row=1, col=9)



# showing the plots in a horizontal order

fig.update_layout(height=1000, width=1300, title_text="Well Log Exploratory Data Analysis")

fig.show()
# Displaying the count for the Deviated class 

majority_class = df.loc[df['Classification'] == 1].count()[0]



# Showing the count for Non Hole Deviation 

minority_class = df.loc[df['Classification'] == 0].count()[0]



# Printing the classes for the deviated and non-deviated class 

print('Deviated Class: {}'.format(majority_class))

print('Non Deviated Class: {}'.format(minority_class))



# Plotting a graph of the total count for the deviated and Non deviated class 

degree_count = df['Classification'].value_counts() 

plt.figure(figsize = (20, 8)) 

degree_count.plot(kind='bar') 

plt.xlabel('0 : Non-Deviated,  1 : Deviated')

plt.ylabel('Counts')

plt.title('Imbalanced Classification of Hole Deviation')

plt.grid(True) 

plt.show() 
# specifying the input variable as the input features X and converting 

# it into a numpy array 

X = df[['Depth', 'Gamma-ray', 'Shale_Volume', 'Restivity', 'Delta T', 'Vs', 'Density', 'Density_Porosity', 'Possions_Ratio']]

X = X.values 

# specifying the output variable as an output feature y and converting  

# it into a numpy array 

y = df.iloc[:, -1]



# standardize the data 

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 

X = scaler.fit_transform(X)



# Using SMOTE to Balance the imbalanced data 

X, y = smote.fit_sample(X, y.ravel())



from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y)



# showing a plot of the Balanced dataset 

plt.figure(figsize = (20, 8)) 

pd.Series(y).value_counts().plot.bar() 

plt.grid(True)

plt.title('Balanced Dataset')

plt.xlabel('0 = Non-deviated, 1 = Deviated')

plt.ylabel('Counts')

plt.show() 



# Prining the counts of the Balanced Data 

print('Deviated Class: {}'.format(pd.Series(y).value_counts()[0]))

print('Non-Deviated Class: {}'.format(pd.Series(y).value_counts()[1]))
# Displaying the shape of the input and output data   

print('Input Shape: {}'.format(X.shape))

print('Output Shape: {}'.format(y.shape))
# Building An MLP 

model = Sequential() 

model.add(Dense(32, input_dim = X_train.shape[1], activation = 'relu'))

model.add(Dense(8, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



# Model summary 

model.summary() 
# Training the model 

hist = model.fit(X_train, y_train, epochs = 100, batch_size = 4, validation_data = (X_test, y_test))
# Evaulating the model accuracy  

accuracy = model.evaluate(X_test, y_test)[1] * 100 

accuracy = str(accuracy)[:5]

!echo 

print('The model is {}% Accurate.'.format(accuracy))
# showing a plot of how Accurate the model is against the 

# test dataset

plt.figure(figsize=(20, 8))

plt.grid(True)

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
# showing a plot of the loss with respect to the number of epochs 

plt.figure(figsize=(20, 8))

plt.grid(True)

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show() 
# Viewing the tail of the well log datast 

df.tail() 
# Specifying the input parameters to be fed into the neural net trained model 

# for accurate predictions. 

['Depth', 'Gamma-ray', 'Shale_Volume', 'Restivity', 'Delta T',

'Vs', 'Density', 'Density_Porosity', 'Possions_Ratio']



# Creating the input features for predictions 

new_input = ['4300', '38.885', '0.2750', '0.70', '111.28', '4320.6076', 

            '2.1866', '0.2687', '0.3497']



# Reshaping and converting into a numpy array 

new_input = np.array(new_input).reshape(1, -1)



# Using the min max scalar to fit the new input feature and printing its input shape 

new_input = scaler.transform(new_input)

print('Shape of Input Data: {}'.format(new_input.shape))



# Performing prediction on the new input dataset and displaying the predicted target variable 

predicted = model.predict(new_input)[0][0]

print('The Predicted Classification is : {}'.format(predicted))