"""Rather than use a CNN, let's use a very basic NN, and just feed it
Hue, Saturation, and Brightness for each image.

Reference: https://pdfs.semanticscholar.org/1f0d/0add993944b5230ff3548c6cb7b2e9954535.pdf
"""
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import os

import tensorflow as tf
import random as rn
# Seeding
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(102)
rn.seed(123)
tf.set_random_seed(142)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

print('Imports complete')
# Input data files are available in the "../input/" directory.
IN_DIR = os.path.join('..', 'input')
IMAGES_DIR = os.path.join(IN_DIR, 'archive')

# Load the training and testing files.
train_df = pd.read_csv(os.path.join(IN_DIR, "train.csv"), index_col=0)
test_df = pd.read_csv(os.path.join(IN_DIR, "sample.csv"), index_col=0)

print('Training DGCI data:')
print(train_df['DGCI'].describe())

# Extract training labels.
y = train_df['DGCI'].values

# Function to load an image, then compute it's hue, saturation, and brightness.
# Calculations from https://pdfs.semanticscholar.org/1f0d/0add993944b5230ff3548c6cb7b2e9954535.pdf
def load_dgci_hsb(im_id, divisions=1):
    # NOTE: Not here to claim this is efficient.
    # Load the image.
    im = cv2.imread(os.path.join(IMAGES_DIR, str(im_id) + '.jpg')) / 255.0
    # plt.imshow(im)
    # NOTE: Image should be width x height x 3
    # Extract blue, green, red
    # https://stackoverflow.com/questions/41500637/how-to-extract-r-g-b-values-with-numpy-into-seperate-arrays
    b, g, r = cv2.split(im)
    
#     print('Image shape:')
#     print(b.shape)
#     print('Height increment: {}, Width increment: {}'.format(h_incr, w_incr))

    # Initialize list for tracking hsb
    area_hsb = []
    
    # Compute hsb for the entire area, always.
    overall_hsb = hsb_from_bgr(b, g, r)
    overall_dgci = get_dgci(*overall_hsb)
    overall_dhsb = [overall_dgci, *overall_hsb]
    area_hsb.extend(overall_dhsb)

    # Break early if we aren't dividing the image up.
    if divisions <= 1:
        return area_hsb
        
    # Divide image into areas.
    h_incr = round(b.shape[0] / divisions)
    w_incr = round(b.shape[1] / divisions)
    
    # Initialize height index
    h0 = 0

    # Loop over the number of divisions
    for j in range(divisions):
        # Initialize width index
        w0 = 0
        
        # Increment the height index.
        h1 = h0 + h_incr

        # Deal with rounding error
        if h1 > b.shape[0]:
            h1 = b.shape[0]

        for k in range(divisions):
            # Increment the width index
            w1 = w0 + w_incr

            # Deal with rounding error
            if w1 > b.shape[1]:
                w1 = b.shape[1]

            # Grab area
            b1 = b[h0:h1, w0:w1]
            g1 = g[h0:h1, w0:w1]
            r1 = r[h0:h1, w0:w1]
            
#             print('Rows: {}:{}'.format(h0, h1))
#             print('Columns: {}:{}'.format(w0, w1))
            
            # Compute hsb for this area
            this_hsb = hsb_from_bgr(b1, g1, r1)
            this_dgci = get_dgci(*this_hsb)
            this_dhsb = [this_dgci, *this_hsb]
            area_hsb.extend(this_dhsb)

            # Set w0 to w1 for the next iteration
            w0 = w1

        # Set h0 to h1 for the next iteration
        h0 = h1
    
    # Return.
    return area_hsb
    
def hsb_from_bgr(b, g, r):
    """Compute hue, saturation, and brightness given matrices for blue, red, and green pixels."""
    # Put means in a list.
#     print(b.shape)
#     print(g.shape)
#     print(r.shape)
    b_m = np.mean(b)
    g_m = np.mean(g)
    r_m = np.mean(r)
    m = [b_m, g_m, r_m]
    
    # Compute brightness
    brightness = max(m)
    
    # Compute the range.
    rng = brightness - min(m)
    
    # Compute saturation
    sat = rng / brightness
    
    # Compute the hue.
    # Alternate computation from paper: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
    if brightness == r_m:
        # hue = 60 * ((g_m - b_m)/rng)
        hue = (60 * (g_m - b_m)/rng + 360) % 360 
    elif brightness == g_m:
        # hue = 60 * (2 + ((b_m - r_m)/rng))
        hue = (60 * (b_m - r_m)/rng + 120) % 360
    elif brightness == b_m:
        # hue = 60 * (4 + ((r_m - g_m)/rng))
        hue = (60 * (r_m - g_m)/rng + 240) % 360
    else:
        raise UserWarning('Something is going wrong.')   
        
    # If hue is < 0, add 360
#     if hue < 0:
#         hue = hue + 360
    
    return hue, sat, brightness

def get_dgci(h, s, b):
    # https://pdfs.semanticscholar.org/1f0d/0add993944b5230ff3548c6cb7b2e9954535.pdf
    return ((h - 60)/60 + 2 - s - b)/3
    
# Initialize training data.
# Determine how many divisions we're using
DIVISIONS = 4
# (dgci, hue, saturation, brightness (entire image)), (dgci, h, s, b (area)), ...
COLUMNS = (1 * 4) + (4 * DIVISIONS**2)

def get_data(df):
    """Helper to get data from a given DataFrame"""
    # Initialize.
    x = np.zeros(shape=(df.shape[0], COLUMNS))
    
    # Loop to load images.
    x_idx = 0
    for row in df.itertuples():
        # NOTE: THIS IS NOT VECTORIZED OR EFFICIENT.
        dgci_hsb = load_dgci_hsb(row.Index, divisions=DIVISIONS)
        x[x_idx, :] = dgci_hsb
        x_idx += 1
    
    return x

x = get_data(train_df)

print('\nDGCI, Hue, Saturation, and Brightness data:')
print(pd.DataFrame(x).describe())
print('Data loaded.')
# Order is Hue, Saturation, Brightness
for k in range(3):
    ax = plt.subplot(3, 1, k+1)
    ax.hist(x[:, k+1])
    
plt.tight_layout()
plt.plot(x[:, 0], y, linestyle='None', marker='.')
plt.xlabel('Computed DGCI')
plt.ylabel('Given DGCI')
# The StandardScaler() led to smallest MSE on training data
x_scaler = StandardScaler()
x_norm = x_scaler.fit_transform(x)

y_scaler = StandardScaler()
y_norm = y_scaler.fit_transform(y.reshape(-1, 1))
print('Normalization complete.')
x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, test_size=0.25)
print('Data split for training/validation')
###################################################################
# ROBUST SCALER
# For the following, network fixed at 64, 64, 1, and patience @ 10
# RMSPropOptimizer, epochs: 28, mse: 0.010
# AdaDelta(), epochs: 46, mse: 0.011

##################################################################
# MIN MAX SCALER
# For the following, network fixed at 64, 64, 1, and patience @ 10
# RMSPropOptimizer: epochs 64, mse: 0.0076
# AdaDelta(), epochs 64, mse: 0.0081

#################################################################
# STANDARD SCALER
# For the following, network fixed at 64, 64, 1, and patience @ 10
# RMSPropOptimizer: epochs 74, mse: 0.0074
# AdaDelta(), epochs 33, mse: 0.0077

################################################################
# STANDARD SCALER
# RMSPROPOPTMIZER
# PATIENCE: 10
# 0.25 dropout: epochs 71, mse 0.0098

################################################################
# STANDARD SCALER
# RMSPROPOPTIMIZER
# PATIENCE: 10
# 0.25 DROPOUT BETWEEN LAYERS
#
# 64, 64: mse: 0.0105, val_mse: 0.0099
# 128, 128: mse: 0.0094, val_mse: 0.0103
# 128, 64: mse: 0.0095, val_mse: 0.0072

model = keras.Sequential()
model.add(Dense(256, activation='relu', input_shape=(x_norm.shape[1],)))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1))

early_stop = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10)

optimizer = tf.train.RMSPropOptimizer(0.001)
# optimizer = keras.optimizers.Adadelta()
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

model.fit(x_train, y_train, epochs=200, callbacks=[early_stop], validation_data=(x_test, y_test))
# Perform predictions, then rescale.

y_tr_pred = model.predict(x_train)
y_te_pred = model.predict(x_test)
y_tr_mse = mean_squared_error(y_scaler.inverse_transform(y_train.reshape(-1, 1)), y_scaler.inverse_transform(y_tr_pred.reshape(-1, 1)))
y_te_mse = mean_squared_error(y_scaler.inverse_transform(y_test.reshape(-1, 1)), y_scaler.inverse_transform(y_te_pred.reshape(-1, 1)))
print('\nFinal Training MSE: {:.4f}'.format(y_tr_mse))
print('Final Testing MSE: {:.4f}'.format(y_te_mse))
# Use reduced patience to avoid over-fitting
early_stop = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=5)
model.fit(x_norm, y_norm, epochs=200, callbacks=[early_stop])
# Initialize testing data.
x_t = get_data(test_df)

# Normalize
x_norm_test = x_scaler.transform(x_t)

print('Testing data loaded and normalized.')
y_test = model.predict(x_norm_test)
test_df['DGCI'] = y_scaler.inverse_transform(y_test.reshape(-1, 1))
print(test_df.head())
print(test_df.describe())
#test_df.to_csv('output.csv')
# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(x_norm, y_norm)
y_pred = lin_reg.predict(x_norm)
print('Linear Regression:')
print('Training MSE {:.4f}'.format(mean_squared_error(y_scaler.inverse_transform(y_norm.reshape(-1, 1)),
                                                      y_scaler.inverse_transform(y_pred.reshape(-1, 1)))))

y_test = lin_reg.predict(x_norm_test)
test_df['DGCI'] = y_scaler.inverse_transform(y_test.reshape(-1, 1))
print(test_df.head())
print(test_df.describe())
test_df.to_csv('output_lin.csv')