%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

with pd.HDFStore('../input/madrid.h5') as data:
    df = data['28079016']


df = df.sort_index()
print(df.columns.values)
x_label = ['CO','NO_2','PM10','SO_2','NOx']
y_label = ['O_3']
df = df[y_label + x_label] 
df = df.dropna() # There are quite a few nans so lets just remove them. We have enough data for our purposes
df.describe()

# Pair plot
sns.pairplot(df)
# Split the data into test and training sets.
np.random.seed(100)
X_train, X_test, y_train, y_test = train_test_split(df[x_label],df[y_label],test_size=0.1)
# Print the dimensions
print('Training set dimensions X, y: ' + str(X_train.shape) + ' ' +str(y_train.shape))
print('Test set dimensions X, y: ' + str(X_test.shape) + ' '+ str(y_test.shape))
# Define regression model in Keras
def regression_model():
    # Define model
    model = Sequential()
    model.add(Dense(5, input_dim=5, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # Compile model
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam,metrics=['accuracy'])
    
    return model

# Use KerasRegressor wrapper (from Keras to sklearn)
# The packages we use are meant to be run with sklearn models
estimator = KerasRegressor(build_fn=regression_model, validation_split = 0.2, batch_size=100, epochs=100, verbose=0)
history = estimator.fit(X_train, y_train)
# summarize history loss
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()
fitted = estimator.predict(X_train)
residuals = y_train['O_3'] - fitted
# Two plots
fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(12,6))

# 1. Histogram of residuals
sns.distplot(residuals, ax=ax1)
ax1.set_title('Histogram of residuals')

# Fitted vs residuals
x1 = pd.Series(fitted, name='Fitted O_3')
x2 = pd.Series(y_train['O_3'], name="O_3 values")
sns.kdeplot(x1, x2, n_levels=40,ax = ax2)
sns.regplot(x=x1,y=x2, scatter=False, ax = ax2)
ax2.set_title('Fitted vs actual values')
ax2.set_xlim([0,120])
ax2.set_ylim([0,120])
ax2.set_aspect('equal')

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(estimator, random_state=1).fit(X_train,y_train)
eli5.show_weights(perm, feature_names = X_train.columns.tolist())
from pdpbox import pdp, get_dataset, info_plots

# Gather pdp data
pdp_goals_NOx = pdp.pdp_isolate(model = estimator, 
                                dataset = X_train, 
                                model_features = x_label,
                                feature='NOx')
# plot NOX pdp
pdp.pdp_plot(pdp_goals_NOx, 'NOx', 
             x_quantile=False, 
            plot_pts_dist=False)
plt.show()
# Gather pdp data
pdp_goals_NO2 = pdp.pdp_isolate(model = estimator, 
                                dataset = X_train, 
                                model_features = x_label,
                                feature='NO_2')
# plot NO_2 pdp
pdp.pdp_plot(pdp_goals_NO2, 'NO_2',
            x_quantile=False, 
            plot_pts_dist=False)
plt.show()
import shap

# SHAP expects model functions to take a 2D numpy array as input, so we define a wrapper function around the original Keras predict function.
def f_wrapper(X):
    return estimator.predict(X).flatten()

# Too many input data - use a random slice
# rather than use the whole training set to estimate expected values, we summarize with
# a set of weighted kmeans, each weighted by the number of points they represent.
X_train_summary = shap.kmeans(X_train, 20)

# Compute Shap values
explainer = shap.KernelExplainer(f_wrapper,X_train_summary)

# Make plot with combined shap values
# The training set is too big so let's sample it. We get enough point to draw conclusions
X_train_sample = X_train.sample(400)
shap_values  = explainer.shap_values(X_train_sample)
shap.summary_plot(shap_values, X_train_sample)
