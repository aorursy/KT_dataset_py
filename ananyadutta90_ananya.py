#Ananya Dutta

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#more imports

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import math
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

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

nRowsRead = 1000 # specify 'None' if want to read whole file

# continuous_factory_process.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/continuous_factory_process.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'continuous_factory_process.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
## First stage inputs and outputs

dataset = df1.values

# split into input (X) and output (Y) variables 

X = dataset[:,1:41]

Y1 = dataset[:,42]#output for first location

Y2 = dataset[:,44]#output for second location

Y3 = dataset[:,46]#output for third location

Y4 = dataset[:,48]#output for fourth location

Y5 = dataset[:,50]#output for fifth location

Y6 = dataset[:,52]#output for sixth location

Y7 = dataset[:,54]#output for seventh location

Y8 = dataset[:,56]#output for eighth location

Y9 = dataset[:,58]#output for ninth location

Y10 = dataset[:,60]#output for tenth location

Y11 = dataset[:,62]#output for eleventh location

Y12 = dataset[:,64]#output for twelfth location

Y13 = dataset[:,66]#output for thirteenth location

Y14 = dataset[:,68]#output for fourteenth location

Y15 = dataset[:,70]#output for fifteenth location

##training and test dataset

X_train, X_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size=.2)

X_train, X_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size=.2)

X_train, X_test, Y3_train, Y3_test = train_test_split(X, Y3, test_size=.2)

X_train, X_test, Y4_train, Y4_test = train_test_split(X, Y4, test_size=.2)

X_train, X_test, Y5_train, Y5_test = train_test_split(X, Y5, test_size=.2)

X_train, X_test, Y6_train, Y6_test = train_test_split(X, Y6, test_size=.2)

X_train, X_test, Y7_train, Y7_test = train_test_split(X, Y7, test_size=.2)

X_train, X_test, Y8_train, Y8_test = train_test_split(X, Y8, test_size=.2)

X_train, X_test, Y9_train, Y9_test = train_test_split(X, Y9, test_size=.2)

X_train, X_test, Y10_train, Y10_test = train_test_split(X, Y10, test_size=.2)

X_train, X_test, Y11_train, Y11_test = train_test_split(X, Y11, test_size=.2)

X_train, X_test, Y12_train, Y12_test = train_test_split(X, Y12, test_size=.2)

X_train, X_test, Y13_train, Y13_test = train_test_split(X, Y13, test_size=.2)

X_train, X_test, Y14_train, Y14_test = train_test_split(X, Y14, test_size=.2)

X_train, X_test, Y15_train, Y15_test = train_test_split(X, Y15, test_size=.2)

# define base model

def b_model():

  # create model

  model = Sequential()#ANN model

    #rectified linear unit(relu) activation

  model.add(Dense(13, input_dim=40, kernel_initializer='normal', activation='relu'))#weights are initialized to small Gaussian random values

  model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

  model.compile(loss='mean_squared_error', optimizer='adam')#Adaptive Moment Estimation optimizer

  return model
# evaluate model

estimator = KerasRegressor(build_fn=b_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10)



#location 1

results1 = cross_val_score(estimator, X_train, Y1_train, cv=kfold)#training

print("Baseline: %.2f (%.2f) MSE for location 1 output" % (results1.mean(), results1.std()))
estimator.fit(X,Y1)

prediction = estimator.predict(X_test)#prediction on test dataset

mse=mean_squared_error(Y1_test, prediction.round())

print ("RMSE for location 1 output is "+str(math.sqrt(mse)))

plt.plot(Y1_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 2



results2 = cross_val_score(estimator, X, Y2, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 2 output" % (results2.mean(), results2.std()))
estimator.fit(X,Y2)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y2_test, prediction.round())

print ("RMSE for location 2 output is "+str(math.sqrt(mse)))
plt.plot(Y2_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 3

results3 = cross_val_score(estimator, X, Y3, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 3 output" % (results3.mean(), results3.std()))
estimator.fit(X,Y3)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y3_test, prediction.round())

print ("RMSE for location 3 output is "+str(math.sqrt(mse)))
plt.plot(Y3_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 4

results4 = cross_val_score(estimator, X, Y4, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 4 output" % (results4.mean(), results4.std()))
estimator.fit(X,Y4)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y4_test, prediction.round())

print ("RMSE for location 4 output is "+str(math.sqrt(mse)))
plt.plot(Y4_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 5

results5 = cross_val_score(estimator, X, Y5, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 5 output" % (results5.mean(), results5.std()))
estimator.fit(X,Y5)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y5_test, prediction.round())

print ("RMSE for location 5 output is "+str(math.sqrt(mse)))
plt.plot(Y5_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 6

results6 = cross_val_score(estimator, X, Y6, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 6 output" % (results6.mean(), results6.std()))
estimator.fit(X,Y6)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y6_test, prediction.round())

print ("RMSE for location 6 output is "+str(math.sqrt(mse)))

plt.plot(Y6_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 7

results7 = cross_val_score(estimator, X, Y7, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 7 output" % (results7.mean(), results7.std()))
estimator.fit(X,Y7)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y7_test, prediction.round())

print ("RMSE for location 7 output is "+str(math.sqrt(mse)))
plt.plot(Y7_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 8

results8 = cross_val_score(estimator, X, Y8, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 8 output" % (results8.mean(), results8.std()))
estimator.fit(X,Y8)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y8_test, prediction.round())

print ("RMSE for location 8 output is "+str(math.sqrt(mse)))
plt.plot(Y8_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 9

results9 = cross_val_score(estimator, X, Y9, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 9 output" % (results9.mean(), results9.std()))
estimator.fit(X,Y9)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y9_test, prediction.round())

print ("RMSE for location 9 output is "+str(math.sqrt(mse)))
plt.plot(Y9_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 10

results10 = cross_val_score(estimator, X, Y10, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 10 output" % (results10.mean(), results10.std()))
plt.plot(Y10_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
estimator.fit(X,Y10)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y10_test, prediction.round())

print ("RMSE for location 10 output is "+str(math.sqrt(mse)))
#location 11

results11 = cross_val_score(estimator, X, Y11, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 11 output" % (results11.mean(), results11.std()))
plt.plot(Y11_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
estimator.fit(X,Y12)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y11_test, prediction.round())

print ("RMSE for location 11 output is "+str(math.sqrt(mse)))
#location 12

results12 = cross_val_score(estimator, X, Y12, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 12 output" % (results12.mean(), results12.std()))
plt.plot(Y12_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
estimator.fit(X,Y12)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y12_test, prediction.round())

print ("RMSE for location 12 output is "+str(math.sqrt(mse)))
#location 13

results13 = cross_val_score(estimator, X, Y13, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 13 output" % (results13.mean(), results13.std()))
estimator.fit(X,Y13)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y13_test, prediction.round())

print ("RMSE for location 13 output is "+str(math.sqrt(mse)))
plt.plot(Y13_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 14

results14 = cross_val_score(estimator, X, Y14, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 14 output" % (results14.mean(), results14.std()))
estimator.fit(X,Y14)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y14_test, prediction.round())

print ("RMSE for location 14 output is "+str(math.sqrt(mse)))
plt.plot(Y14_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 15

results15 = cross_val_score(estimator, X, Y2, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 15 output" % (results.mean(), results.std()))
estimator.fit(X,Y15)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y15_test, prediction.round())

print ("RMSE for location 15 output is "+str(math.sqrt(mse)))
plt.plot(Y15_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
## Second stage inputs and outputs

dataset = df1.values

# split into input (X) and output (Y) variables 

X2 = dataset[:,72:85]

Y1 = dataset[:,86]#output for first location

Y2 = dataset[:,88]#output for second location

Y3 = dataset[:,90]#output for third location

Y4 = dataset[:,92]#output for fourth location

Y5 = dataset[:,94]#output for fifth location

Y6 = dataset[:,96]#output for sixth location

Y7 = dataset[:,98]#output for seventh location

Y8 = dataset[:,100]#output for eighth location

Y9 = dataset[:,102]#output for ninth location

Y10 = dataset[:,104]#output for tenth location

Y11 = dataset[:,106]#output for eleventh location

Y12 = dataset[:,108]#output for twelfth location

Y13 = dataset[:,110]#output for thirteenth location

Y14 = dataset[:,112]#output for fourteenth location

Y15 = dataset[:,114]#output for fifteenth location
##training and test dataset

X_train, X_test, Y1_train, Y1_test = train_test_split(X2, Y1, test_size=.2)

X_train, X_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=.2)

X_train, X_test, Y3_train, Y3_test = train_test_split(X2, Y3, test_size=.2)

X_train, X_test, Y4_train, Y4_test = train_test_split(X2, Y4, test_size=.2)

X_train, X_test, Y5_train, Y5_test = train_test_split(X2, Y5, test_size=.2)

X_train, X_test, Y6_train, Y6_test = train_test_split(X2, Y6, test_size=.2)

X_train, X_test, Y7_train, Y7_test = train_test_split(X2, Y7, test_size=.2)

X_train, X_test, Y8_train, Y8_test = train_test_split(X2, Y8, test_size=.2)

X_train, X_test, Y9_train, Y9_test = train_test_split(X2, Y9, test_size=.2)

X_train, X_test, Y10_train, Y10_test = train_test_split(X2, Y10, test_size=.2)

X_train, X_test, Y11_train, Y11_test = train_test_split(X2, Y11, test_size=.2)

X_train, X_test, Y12_train, Y12_test = train_test_split(X2, Y12, test_size=.2)

X_train, X_test, Y13_train, Y13_test = train_test_split(X2, Y13, test_size=.2)

X_train, X_test, Y14_train, Y14_test = train_test_split(X2, Y14, test_size=.2)

X_train, X_test, Y15_train, Y15_test = train_test_split(X2, Y15, test_size=.2)
# define base model for stage 2

def b_model():

  # create model

  model = Sequential()#ANN model

    #rectified linear unit(relu) activation

  model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))#weights are initialized to small Gaussian random values

  model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

  model.compile(loss='mean_squared_error', optimizer='adam')#Adaptive Moment Estimation optimizer

  return model
# evaluate model

estimator = KerasRegressor(build_fn=b_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10)
#location 1

results1 = cross_val_score(estimator, X_train, Y1_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 1 output" % (results1.mean(), results1.std()))
estimator.fit(X2,Y1)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y1_test, prediction.round())

print ("RMSE for location 1 output is "+str(math.sqrt(mse)))
plt.plot(Y1_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 2

results2 = cross_val_score(estimator, X_train, Y2_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 2 output" % (results2.mean(), results2.std()))
estimator.fit(X2,Y2)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y2_test, prediction.round())

print ("RMSE for location 2 output is "+str(math.sqrt(mse)))
plt.plot(Y2_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 3

results3 = cross_val_score(estimator, X_train, Y3_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 3 output" % (results3.mean(), results3.std()))
estimator.fit(X2,Y3)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y3_test, prediction.round())

print ("RMSE for location 3 output is "+str(math.sqrt(mse)))
plt.plot(Y3_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 4

results4 = cross_val_score(estimator, X_train, Y4_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 4 output" % (results4.mean(), results4.std()))
estimator.fit(X2,Y4)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y4_test, prediction.round())

print ("RMSE for location 4 output is "+str(math.sqrt(mse)))
plt.plot(Y4_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 5

results5 = cross_val_score(estimator, X_train, Y5_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 5 output" % (results5.mean(), results5.std()))
estimator.fit(X2,Y5)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y5_test, prediction.round())

print ("RMSE for location 5 output is "+str(math.sqrt(mse)))
plt.plot(Y5_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 6

results6 = cross_val_score(estimator, X_train, Y6_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 6 output" % (results6.mean(), results6.std()))
estimator.fit(X2,Y6)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y6_test, prediction.round())

print ("RMSE for location 6 output is "+str(math.sqrt(mse)))
plt.plot(Y6_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 7

results7 = cross_val_score(estimator, X_train, Y7_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 7 output" % (results7.mean(), results7.std()))
estimator.fit(X2,Y7)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y7_test, prediction.round())

print ("RMSE for location 7 output is "+str(math.sqrt(mse)))
plt.plot(Y7_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 8

results8 = cross_val_score(estimator, X_train, Y8_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 8 output" % (results8.mean(), results8.std()))
estimator.fit(X2,Y8)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y8_test, prediction.round())

print ("RMSE for location 8 output is "+str(math.sqrt(mse)))
plt.plot(Y8_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 9

results9 = cross_val_score(estimator, X_train, Y9_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 9 output" % (results9.mean(), results9.std()))
estimator.fit(X2,Y9)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y9_test, prediction.round())

print ("RMSE for location 9 output is "+str(math.sqrt(mse)))
plt.plot(Y9_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 10

results10 = cross_val_score(estimator, X_train, Y10_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 10 output" % (results10.mean(), results10.std()))
estimator.fit(X2,Y10)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y10_test, prediction.round())

print ("RMSE for location 10 output is "+str(math.sqrt(mse)))
plt.plot(Y10_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 11

results11 = cross_val_score(estimator, X_train, Y11_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 11 output" % (results11.mean(), results11.std()))
estimator.fit(X2,Y11)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y11_test, prediction.round())

print ("RMSE for location 11 output is "+str(math.sqrt(mse)))
plt.plot(Y11_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 12

results12 = cross_val_score(estimator, X_train, Y12_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 12 output" % (results12.mean(), results12.std()))
estimator.fit(X2,Y12)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y12_test, prediction.round())

print ("RMSE for location 12 output is "+str(math.sqrt(mse)))
plt.plot(Y12_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 13

results13 = cross_val_score(estimator, X_train, Y13_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 13 output" % (results13.mean(), results13.std()))
estimator.fit(X2,Y13)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y13_test, prediction.round())

print ("RMSE for location 13 output is "+str(math.sqrt(mse)))
plt.plot(Y13_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 14

results14 = cross_val_score(estimator, X_train, Y14_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 14 output" % (results14.mean(), results14.std()))
estimator.fit(X2,Y14)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y14_test, prediction.round())

print ("RMSE for location 14 output is "+str(math.sqrt(mse)))
plt.plot(Y14_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
#location 15

results15 = cross_val_score(estimator, X_train, Y15_train, cv=kfold)

print("Baseline: %.2f (%.2f) MSE for location 15 output" % (results15.mean(), results15.std()))
estimator.fit(X2,Y15)

prediction = estimator.predict(X_test)

mse=mean_squared_error(Y15_test, prediction.round())

print ("RMSE for location 15 output is "+str(math.sqrt(mse)))
plt.plot(Y15_test, label="y-original")

plt.plot(prediction , label="y-predicted")

plt.legend()

plt.show()
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 24)
plotScatterMatrix(df1, 20, 10)