import os

import datetime
import pandas as pd

import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks

from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')
DATA_PATH = '../input/data-for-activity-recognition/data/data/'
running_folder = 'running'

idle_folder = 'idle'

walking_folder = 'walking'

stairs_folder = 'stairs'



activity_list = [running_folder, idle_folder, walking_folder, stairs_folder]
# checking



for activity in activity_list:

    file_names_list = os.listdir(os.path.join(DATA_PATH, activity))

    print(activity, ': ', len(file_names_list))
def plot_3d_trajectory(x, y, z):

    """ 

    Plot 3D Trajectory

    Next we will calculate the phone’s motion 

    by integrating the linear-accelerations, 

    and plot the results.

    """

    x = cumtrapz(x)

    y = cumtrapz(y)

    z = cumtrapz(z)

    

    fig3,ax = plt.subplots()

    fig3.suptitle('3D Trajectory of phone',fontsize=20)

    ax = plt.axes(projection='3d')

    ax.plot3D(x,y,z,c='red',lw=1,label='phone trajectory')

    ax.set_xlabel('X position')

    ax.set_ylabel('Y position')

    ax.set_zlabel('Z position')

    plt.show()
def plot_frequency_spectrum(x, y, z):

    """ Plot Frequency spectrum """

    fig4,[ax1,ax2,ax3] = plt.subplots(3,1,sharex=True,sharey=True)

    fig4.suptitle('Spectrum',fontsize=20)

    ax1.plot(x,c='r',label='x')

    ax1.legend()

    ax2.plot(y,c='b',label='y')

    ax2.legend()

    ax3.plot(z,c='g',label='z')

    ax3.legend()

    ax3.set_xlabel('Freqeuncy (Hz)')

    plt.show()
def select_random_df(folder_name):

    custom_path = os.path.join(DATA_PATH, folder_name)

    data = pd.read_csv(os.path.join(custom_path, os.listdir(custom_path)[0]))

    x = data.accelerometer_X.values

    y = data.accelerometer_Y.values

    z = data.accelerometer_Z.values

    return x, y, z
# running

x,y,z = select_random_df(running_folder)

plot_3d_trajectory(x, y, z)
plot_frequency_spectrum(x, y, z)
# idle

x,y,z = select_random_df(idle_folder)

plot_3d_trajectory(x, y, z)
plot_frequency_spectrum(x, y, z)
# walking

x,y,z = select_random_df(walking_folder)

plot_3d_trajectory(x, y, z)
plot_frequency_spectrum(x, y, z)
# stairs

x,y,z = select_random_df(stairs_folder)

plot_3d_trajectory(x, y, z)
plot_frequency_spectrum(x, y, z)
def mean_calculator(three_axis):

    """ Return mean of each vectors """

    three_axis = np.array(three_axis)

    vector_x = three_axis[:, 0]

    vector_y = three_axis[:, 1]

    vector_z = three_axis[:, 2]

    x_mean = np.mean(vector_x)

    y_mean = np.mean(vector_y)

    z_mean = np.mean(vector_z)

    return x_mean, y_mean, z_mean
def std_calculator(three_axis):

    """ Return standart deviation of each vectors """

    three_axis = np.array(three_axis)

    vector_x = three_axis[:, 0]

    vector_y = three_axis[:, 1]

    vector_z = three_axis[:, 2]

    x_std = np.std(vector_x)

    y_std = np.std(vector_y)

    z_std = np.std(vector_z)

    return x_std, y_std, z_std
def peaks_calculator(three_axis):

    """ Return number of peaks of each vectors """

    three_axis = np.array(three_axis)

    vector_x = three_axis[:, 0]

    vector_y = three_axis[:, 1]

    vector_z = three_axis[:, 2]

    x_peaks = len(find_peaks(vector_x)[0])

    y_peaks = len(find_peaks(vector_y)[0])

    z_peaks = len(find_peaks(vector_z)[0])

    return x_peaks, y_peaks, z_peaks
def feature_engineer(action, target, df):

    try:

        x_mean, y_mean, z_mean = mean_calculator(action)

        x_std, y_std, z_std = std_calculator(action)

        x_peaks, y_peaks, z_peaks = peaks_calculator(action)

    except:

        print(action.shape, target)

    dictionary = {

        'x_mean': x_mean,

        'y_mean': y_mean, 

        'z_mean': z_mean,

        'x_std': x_std, 

        'y_std': y_std,

        'z_std': z_std,

        'x_peaks': x_peaks, 

        'y_peaks': y_peaks, 

        'z_peaks': z_peaks,

        'target': target

    }

    df = df.append(

        dictionary, 

        ignore_index=True

    )

    return df
columns = [

    'x_mean', 'y_mean', 'z_mean', 

    'x_std', 'y_std', 'z_std', 

    'x_peaks', 'y_peaks', 'z_peaks',

    'target'

]

dataframe = pd.DataFrame(columns=columns)
for activity in activity_list:

    activity_files = os.listdir(os.path.join(DATA_PATH, activity))

    for file in activity_files:

        try:

            df = pd.read_csv(os.path.join(DATA_PATH, activity, file))

            array = df.to_numpy()

            dataframe = feature_engineer(

                action=array, 

                target=activity, 

                df=dataframe

            )

        except:

            print('some error')
print(dataframe.shape)

dataframe.head()
dataframe.target.unique()
dataframe['target'].value_counts()
dataframe['target'].value_counts().plot(kind='barh')
# data frame to csv

# dataframe.to_csv('data/final_data.csv', index=False)
import sys
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns

import matplotlib.pyplot as plt
df = dataframe
df.shape
df.head()
df = df.sample(frac=1).reset_index(drop=True)
df.shape
df.head()
x_columns = [

    'x_mean', 'y_mean', 'z_mean', 

    'x_std', 'y_std', 'z_std', 

    'x_peaks', 'y_peaks', 'z_peaks'

]

X = df[x_columns]

y = df.target
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42

)
print('X tran shape:', X_train.shape)

print('X test shape:', X_test.shape)

print('y tran shape:', y_train.shape)

print('y test shape:', y_test.shape)
labels = df.target.unique()
def train_model(model):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    return confusion_matrix(y_test, y_pred)
def visualize_confusion_matrix(cm, labels=labels):

    df_cm = pd.DataFrame(cm, columns=labels, index=labels)

    df_cm.index.name = 'Actual'

    df_cm.columns.name = 'Predicted'

    plt.figure(figsize = (10,7))

    sns.set(font_scale=1.4)#for label size

    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
lr = LogisticRegression()

lr_cm = train_model(lr)
visualize_confusion_matrix(lr_cm)
rf = RandomForestClassifier()

rf_cm = train_model(rf)
visualize_confusion_matrix(rf_cm)
svc = SVC()

svc_cm = train_model(svc)
visualize_confusion_matrix(svc_cm)
dt = DecisionTreeClassifier()

dt_cm = train_model(dt)
visualize_confusion_matrix(dt_cm)
gb = GradientBoostingClassifier()

gb_cm = train_model(gb)
visualize_confusion_matrix(gb_cm)