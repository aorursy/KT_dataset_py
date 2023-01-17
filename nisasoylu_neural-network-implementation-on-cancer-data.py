# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as visual
import matplotlib.pyplot as plt    # for plotting the data.
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

from sklearn.preprocessing import StandardScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
whole_data = pd.read_csv("/kaggle/input/breast-cancer/data5.csv")
whole_data
whole_data.drop(["id"], axis = 1, inplace =True)
# whole_data = whole_data.replace({"M": 1, "B": 0})
whole_data.diagnosis = [1.00 if i =="M" else 0.00 for i in whole_data.diagnosis]
whole_data_float = np.array(whole_data, dtype = float)
y = whole_data.diagnosis.values.reshape(-1,1)
x = whole_data.drop(["diagnosis"], axis = 1)
x_inputs = np.vstack((whole_data.radius_mean, whole_data.texture_mean,
                      whole_data.perimeter_mean, whole_data.area_mean, 
                      whole_data.smoothness_mean, whole_data.compactness_mean,
                      whole_data.concavity_mean, whole_data.concave_points_mean,
                      whole_data.symmetry_mean, whole_data.fractal_dimension_mean,
                      whole_data.radius_se, whole_data.texture_se, whole_data.perimeter_se,
                      whole_data.area_se, whole_data.smoothness_se, whole_data.compactness_se, 
                      whole_data.concavity_se, whole_data.concave_points_se, 
                      whole_data.symmetry_se, whole_data.fractal_dimension_se,
                      whole_data.radius_worst, whole_data.texture_worst, 
                      whole_data.perimeter_worst, whole_data.area_worst,
                      whole_data.smoothness_worst, whole_data.compactness_worst,
                      whole_data.concavity_worst, whole_data.concave_points_worst,
                      whole_data.symmetry_worst, whole_data.fractal_dimension_worst))
x_inputs = np.transpose(x_inputs)
y_outputs = y
scaler1 = MinMaxScaler(feature_range = (0,1))
normalized_x_inputs = scaler1.fit_transform(x_inputs)

scaler2 = MinMaxScaler(feature_range = (0,1))
normalized_y_outputs = scaler2.fit_transform(y_outputs)
xtrain, xtest, ytrain, ytest = train_test_split(normalized_x_inputs, normalized_y_outputs, test_size = 0.3, random_state = 42)
model = Sequential()
model.add(Dense(70, input_dim = 30, activation = "relu"))
model.add(Dense(1, activation = "linear"))
model.summary()
model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])
model.fit(xtrain, ytrain, epochs = 100, batch_size = 30)
test_loss, test_accuracy = model.evaluate(xtest, ytest)
print("Test accuracy: %.2f " %(test_accuracy * 100))
predictions = model.predict(normalized_x_inputs)
correct_values = scaler2.inverse_transform(predictions)
print("Mean squared error : ", mean_squared_error(y_outputs[:,0], correct_values[:,0]))
plt.figure(1)
#ax1 = plt.subplot(121)
plt.title("Cancer Data Error Graph")
plt.plot(y_outputs[:100,0], color = "blue", label = "Real Output")   
plt.plot(correct_values[:100,0], color = "red", linestyle= "--", label = "Predicted Output")
plt.xlabel("Time (s)")
plt.ylabel("X (m)")
plt.legend()
plt.show()
print("Mean squared error : ", mean_squared_error(y_outputs[:,0], correct_values[:,0]))
plt.hist(y_outputs)
plt.title("Number of People Having Malignant or \n Bening Tumor as a Result of Test ")
plt.xlabel("Bening / Malignant")
plt.ylabel("Number of People")
plt.show()
features_mean=list(x)
dfM=whole_data[whole_data['diagnosis'] == 1]
dfB=whole_data[whole_data['diagnosis'] == 0]
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()

for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(whole_data[features_mean[idx]]) - min(whole_data[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(whole_data[features_mean[idx]]),
            max(whole_data[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5, stacked=True, density = True,
            label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx] + "(mm)")
plt.tight_layout()
plt.show()