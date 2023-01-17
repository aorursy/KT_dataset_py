import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
eeg_data_dir = "../input/eeg-brainwave-dataset-feeling-emotions/emotions.csv"

eeg_data = pd.read_csv(eeg_data_dir)

#eeg_data.head()

#eeg_data.describe()

eeg_data_columns = eeg_data.columns.values.tolist()

print (eeg_data_columns[1:6])
#checks the names of categories

print(eeg_data["label"].value_counts())



#rules for replace

cleanup_rules = ({'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2} )

#new dataset with replaced values

eeg_data_encoded = eeg_data.replace(cleanup_rules, inplace=False)



##print(eeg_data_encoded.head())

print(eeg_data_encoded['label'].value_counts())
y = eeg_data_encoded.label



##eeg_data_features = ['mean_1_a', 'mean_2_a', 'mean_3_a', 'mean_4_a','mean_d_0_a', 'mean_d_1_a', 'mean_d_2_a', 'mean_d_3_a', 'mean_d_4_a']

eeg_data_features = eeg_data_columns[1:10]

X=eeg_data_encoded[eeg_data_features]



print("Predictions:")

print(y.head())

print("Features:")

print(X.head())
from pandas.plotting import parallel_coordinates

from sklearn.preprocessing import StandardScaler 

#from sklearn import preprocessing

from matplotlib import pyplot as plt

import plotly.express as px

eeg_data_selected_columns = eeg_data_columns[1:10]

eeg_data_selected_columns.append("label")

print(eeg_data_selected_columns)

for_fig = eeg_data_encoded[eeg_data_selected_columns]



#scaler

##sc = StandardScaler()

##for_fig_scaled = sc.fit_transform(for_fig)

##ffc = pd.DataFrame(for_fig_scaled)

##print(ffc.head())



fig = px.parallel_coordinates(for_fig, color='label' ,color_continuous_scale=px.colors.diverging.Portland,color_continuous_midpoint=1)

fig.show()
eeg_data_neutral = eeg_data_encoded[eeg_data_encoded.label==0]

cols_neutral = ['mean_1_a', 'mean_2_a', 'mean_3_a', 'mean_4_a','mean_d_0_a', 'mean_d_1_a', 'mean_d_2_a', 'mean_d_3_a', 'mean_d_4_a']

eeg_data_positive = eeg_data_encoded[eeg_data_encoded.label==1]

eeg_data_negative = eeg_data_encoded[eeg_data_encoded.label==2]



boxplot = eeg_data.boxplot(column=['mean_1_a', 'mean_2_a', 'mean_3_a', 'mean_4_a','mean_d_0_a', 'mean_d_1_a', 'mean_d_2_a', 'mean_d_3_a', 'mean_d_4_a'], by='label',rot=0, fontsize=12, figsize=(20, 15))
from sklearn.tree import DecisionTreeRegressor 



eeg_model = DecisionTreeRegressor(random_state=1)



eeg_model.fit(X,y)
eeg_model_1_result = eeg_model.predict(X.head())

print("Predictions:")

print(eeg_model_1_result)

print("Actuals:")

print(y.head())

print(cleanup_rules)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
eeg_model_2 = DecisionTreeRegressor(random_state=1)

eeg_model_2.fit(X_train,y_train)

eeg_model_2_result = eeg_model_2.predict(X_test.head())

print("Predictions:")

print(eeg_model_2_result)

print("Actuals:")

print(y_test.head())

print(cleanup_rules)
import matplotlib.pyplot as plt



import tensorflow as tf

import sklearn



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD



from sklearn.metrics import confusion_matrix
eeg_model_3 = Sequential()

#First Hidden Layer 

###>>9 is the number of features

eeg_model_3.add(Dense(12, activation='relu', kernel_initializer='random_normal', input_dim=9))

#Second  Hidden Layer

eeg_model_3.add(Dense(8, activation='relu', kernel_initializer='random_normal'))

#Output Layer

eeg_model_3.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network

eeg_model_3.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset

eeg_model_3.fit(X_train,y_train, batch_size=10, epochs=150)

eval_eeg_model_3=eeg_model_3.evaluate(X_train, y_train)



y_pred=eeg_model_3.predict(X_test)

y_pred =(y_pred>0.5)



model_3_con_mat = confusion_matrix(y_test, y_pred)

print("Confusion Matrix is:")

print(model_3_con_mat)

print("Evaluation is:")

print(eval_eeg_model_3)



plt.show()



print("Predictions:")

eeg_model_3_result = eeg_model_3.predict(X_test.head())

print(eeg_model_3_result)

print("Actuals:")

print(y_test.head())

print(cleanup_rules)
from sklearn import svm
eeg_model_4 = svm.LinearSVC()

eeg_model_4.fit(X_train, y_train)

print("Predictions:")

eeg_model_4_result = eeg_model_4.predict(X_test.head())

print(eeg_model_4_result)

print("Actuals:")

print(y_test.head())

print(cleanup_rules)