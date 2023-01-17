### import necessary packages
import numpy as np 
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import tensorflow.keras.callbacks as cb
from keras.models import load_model
from math import exp, log
### define some global parameters
# "look back" days. This is the number of historical days we use to predict tomorrow. Personally, I don't like the term "look back" because
# it is quite confusing. However, it seems that this term has become the convention in the field of CNN/RNN stock prediction, so I will use it
# here as well. In LSTM, this is the number of time steps.
LB = 10
# number of epochs. setting it too high might cause overfitting, setting it too low the model won't get enough training.
# so setting epochs to an adequate number is very important. How big this number should be is more an art. some people examine the
# test set performance metrics (such as loss, accuracy) during iterations of training and then decide how many epochs to go to. However,
# this approach peeks into the test set, and thus invalidates the purpose of the test set. Maybe RNN/CNN should internally divide its 
# training data into two parts, use one part for internal training, the other part for internal testing, or use something like Leave-one-out
# and find an appropriate number of epochs.
EPOCHS = 50

### load data file "DIA.csv"
data = np.array(pd.read_csv('../input/DIA.csv')).reshape(-1,7)
print("data shape = " + str(data.shape))
print(data[0:5]) # print the top 5 data lines
pv = data[:,(1,2,3,4,6)] # Prive & Volume. Only use Open, High, Low, Close, Volume columns for prediction
print(pv[0:5,]) # print the top 5 data lines
### split the data into training set and testing (validating) set
splitIndex = int(pv.shape[0]*0.80)
print("total pv.length = " + str(pv.shape[0]) + ", splitIndex = " + str(splitIndex) + ", diff len = " + str(pv.shape[0]- splitIndex))
pv[0:5,]
#Create a function to process the data into LB day look back slices
def processData(data,lb):
    # X_orig: orignal X values
    # Y_orig: orignal Y values
    # X     : transformed X values as input
    # Y     : transformed Y values as predicted outcome (output)
    X_orig, Y_orig, X, Y = [], [], [], []
    
    for i in range(len(data) - lb - 1):
        xo1, x1 = [], []
        
        pmax = max(data[i:(i+lb),1]) # get the maximum price using the "High" column
        pmin = min(data[i:(i+lb),2]) # get the minimum price using the "Low" column
        pdiff = pmax - pmin
        vmax = max(data[i:(i+lb), 4]) # maximum volume
        vmin = min(data[i:(i+lb), 4]) # minimum volume
        vdiff = vmax - vmin

        for j in range(lb):
            # original, no scaling
            xo1.append(data[i + j,]) 
            # minmax scaling for prices (Open, High, Low, Close) and volume respectively
            x1.append(np.append((data[i + j, 0:4] - pmin) / pdiff, (data[i + j, 4] - vmin) / vdiff))

        # store the original data
        X_orig.append(xo1) 
        Y_orig.append(data[(i+lb),])
        # add a transformed model input entry, which is an array of "LB" elements of [Open, High, Low, Close, Volume] arrays.
        X.append(x1) 
        # add a transformed model output entry. The prices are normalized by the previous day close column (indexed 3)
        # and volume by the previous day volume
        y1 = []
        
        for j in range(4):
            y1.append(log(data[i+lb, j] / data[i+lb-1, 3])) # normalize by previous Close and log transform
        
        y1.append(log(data[i+lb, 4] / data[i+lb-1, 4])) # normalize by previous Volume and log transform
        Y.append(y1) 
        
    return np.array(X_orig), np.array(Y_orig), np.array(X), np.array(Y)

X_orig, Y_orig, X, Y = processData(pv,LB)
print("X rows: " + str(X.shape[0]))
print("Y rows: " + str(Y.shape[0]))
X_train_orig,X_test_orig = X_orig[:splitIndex],X_orig[splitIndex:]
Y_train_orig,Y_test_orig = Y_orig[:splitIndex],Y_orig[splitIndex:]
X_train,X_test = X[:splitIndex],X[splitIndex:]
Y_train,Y_test = Y[:splitIndex],Y[splitIndex:]

### do some sanity check
print("X_train len = " + str(len(X_train)) + ", Y_train len = " + str(len(Y_train)))
print("X_test len = " + str(len(X_test)) + ", Y_test len = " + str(len(Y_test)))

print("X_train_orig: " + str(X_train_orig[0:2]))
print("Y_train_orig: " + str(Y_train_orig[0:2]))
print("X_test_orig: " + str(X_test_orig[0:2]))
print("Y_test_orig: " + str(Y_test_orig[0:2]))

print("X_train:" + str(X_train[0:2]))
print("Y_train:" + str(Y_train[0:2]))
print("X_test:" + str(X_test[0:2]))
print("Y_test:" + str(Y_test[0:2]))
### Construct the model
model = Sequential()
model.add(LSTM(256,input_shape=(LB,5))) # note the values for "input_shape". 5 is the nubmer of features: Open, High, Low, Close, Volume
model.add(Dense(5))
model.compile(optimizer='adam',loss='mse', metrics=['mae', 'acc'])

### Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],LB,5))
X_test = X_test.reshape((X_test.shape[0],LB,5))

#Fit model with history to check for overfitting
bestModelPath="DIA.LB" + str(LB) + "EPOCH{epoch:02d}.VAL_ACC{val_acc:.2f}.hdf5"
checkpoint = cb.ModelCheckpoint(bestModelPath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]
history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=64, validation_data=(X_test,Y_test) ,callbacks=callbacks_list, shuffle=False)

### plot the performance of the traing process
plt.plot(history.history['loss'], color='red')
plt.plot(history.history['val_loss'], color='green')
### make predictions on the training and testing
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
### define a function to reverse transform the predicted outcomes
def reverse(Xorig, Ypred):
    rst = [] # result

    for i in (range(len(Ypred))):
        pall, vall = [], []

        for j in range(LB):
            for k in range(4):
                pall.append(Xorig[i][j, k])    

            vall.append(Xorig[i][j, 4])

        pmax = max(pall)
        pmin = min(pall)
        pdiff = pmax - pmin
        vmax = max(vall)
        vmin = min(vall)
        vdiff = vmax - vmin

        orig1 = []

        for j in range(4):
            orig1.append(exp(Ypred[i][j]) * Xorig[i][LB-1,3])
            
        orig1.append(exp(Ypred[i][4]) * Xorig[i][LB-1, 4]) # volume
        rst.append(orig1)

    return np.array(rst)

### reverse the transforms
Y_train_pred_orig = reverse(X_train_orig, Y_train_pred)
Y_test_pred_orig = reverse(X_test_orig, Y_test_pred)
### plot out the predictions for Open, High, Low, Close, Volume respectively for the training & testing (validating) data sets
for i in range(5):
    plt.figure(i + 1)
    
    if i == 0:
        plt.title("Open")
    elif i == 1:
        plt.title("High")
        
    elif i == 2:
        plt.title("Low")
    elif i == 3:
        plt.title("Close")
    else: # i == 4
        plt.title("Volume")
        
    plt.plot(np.append(Y_train_orig[:,i], Y_test_orig[:,i]), color='red')
    plt.plot(np.append(Y_train_pred_orig[:,i], Y_test_pred_orig[:,i]), color='green')
    plt.axvline(x=splitIndex) # separate the training & testing time points
### plot just the predicted part for a closer look
NPOINTS = 100 # number of points to plot

### plot the comparision in price
print("Y_test: " + str(Y_test_orig[0:2]))
print("Y_test_pred: " + str(Y_test_pred_orig[0:2]))

for i in range(5):
    plt.figure(i+1)
    plt.figure(figsize=(20, 5))
    
    if i == 0:
        plt.title("Pred Open")
    elif i == 1:
        plt.title("Pred High")
        
    elif i == 2:
        plt.title("Pred Low")
    elif i == 3:
        plt.title("Pred Close")
    else: # i == 4
        plt.title("Pred Volume")
        
    #plt.figure(figsize=(15,10))
    plt.plot(Y_test_orig[0:NPOINTS, i], 'o', color='red')
    plt.plot(Y_test_pred_orig[0:NPOINTS, i], 'x', color='green')
### compare the prediction in relative changes
# print("Y_test: " + str(Y_test[0:2]))
# print("Y_test_pred: " + str(Y_test_pred[0:2]))

for i in range(5):
    plt.figure(i+1)
    plt.figure(figsize=(16,4))
    
    if i == 0:
        msg = "Pred Open Change"
    elif i == 1:
        msg = "Pred High Change"
        
    elif i == 2:
        msg = "Pred Low Change"
    elif i == 3:
        msg = "Pred Close Change"
    else: # i == 4
        msg = "Pred Volume Change"
        
    plt.title(msg)
    plt.plot(Y_test[0:NPOINTS, i], 'o', color='red')
    plt.plot(Y_test_pred[0:NPOINTS, i], 'x', color='green')    
    plt.axhline(y=0) # separate the training & testing time points

    ### do some statistics
    topSame, topDiff, allSame, allDiff = 0, 0, 0, 0 # number of pairs in the same or different direction for the top NPOINTS points or all
    
    for j in range(NPOINTS):
        if Y_test[j, i] * Y_test_pred[j, i] >= 0:
            topSame += 1
        else:
            topDiff += 1
            
    for j in range(len(Y_test[:,i])):
        if Y_test[j, i] * Y_test_pred[j, i] >= 0:
            allSame += 1
        else:
            allDiff += 1
    
    print("\n" + msg)
    print("Number of points go in the same direction for the first " + str(NPOINTS) + " points: %d (%.2f%%)" % (topSame, 100 * topSame / (topSame + topDiff)))
    print("Number of points go in the diff direction for the first " + str(NPOINTS) + " points: %d (%.2f%%)" % (topDiff, 100 * topDiff / (topSame + topDiff)))
    print("Number of points go in the same direction for all " + str(len(Y_test[:,i])) + " points: %d (%.2f%%)" % (allSame, 100 * allSame / (allSame + allDiff)))
    print("Number of points go in the diff direction for all " + str(len(Y_test[:,i])) + " points: %d (%.2f%%)" % (allDiff, 100 * allDiff / (allSame + allDiff)))
### scatter plot prediction performance
from numpy.polynomial.polynomial import polyfit

for i in range(5):
    plt.figure(i+1)
    
    if i == 0:
        plt.title("Open")
    elif i == 1:
        plt.title("High")
        
    elif i == 2:
        plt.title("Low")
    elif i == 3:
        plt.title("Close")
    else: # i == 4
        plt.title("Volume")
        

    ### get the linear regression parameters
    b, m = polyfit(np.array(Y_test[:,i], dtype=float), np.array(Y_test_pred[:,i], dtype=float), 1)
    predReg = [0] * len(Y_test_pred[:,i]) # predicted based on regression

    predReg = b + m * Y_test[:,i]

    # draw the scatter plot
    plt.plot(Y_test[:,i], Y_test_pred[:,i], '.')
    plt.xlabel("actual change")
    plt.ylabel("predicted change")
    # draw the regression line
    plt.plot(np.array(Y_test[:,i], dtype=float), predReg, '-')
# get the Rsq
from scipy import stats

### return the R-value (correlation coefficient) & p-value of the two input variables
def getRvalPval(var1, var2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(var1, var2)

    return r_value, p_value
    
for i in range(5):
    if i == 0:
        title = "Open"
    elif i == 1:
        title = "High"
        
    elif i == 2:
        title = "Low"
    elif i == 3:
        title = "Close"
    else: # i == 4
        title = "Volume"
        
    print("\n" + title)
    r_value, p_value = getRvalPval(np.array(Y_test_orig[:,i], dtype=float), np.array(Y_test_pred_orig[:,i], dtype=float))
    print("Original Value R: %.3f\tp-value: %.2e" %(r_value, p_value))

    r_value, p_value = getRvalPval(np.array(Y_train[:,i], dtype=float), np.array(Y_train_pred[:,i], dtype=float))
    print("Training R: %.3f\tp-value: %.2e" %(r_value, p_value))

    r_value, p_value = getRvalPval(np.array(Y_test[:,i], dtype=float), np.array(Y_test_pred[:,i], dtype=float))
    print("Testing R: %.3f\tp-value: %.2e" %(r_value, p_value))