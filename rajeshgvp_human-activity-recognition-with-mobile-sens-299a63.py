import os

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline
def read_data(path, filename):

    return pd.read_csv(os.path.join(path, filename), index_col=0)



df = read_data('../input/motionsense-dataset/A_DeviceMotion_data/A_DeviceMotion_data/dws_1/', 'sub_1.csv')

df.head()
def produce_magnitude(df, column):

    df[column+'.mag'] = np.sqrt(df[column+'.x']**2 + df[column+'.y']**2 + df[column+'.z']**2)



produce_magnitude(df, 'userAcceleration')

produce_magnitude(df, 'rotationRate')

df.head()
import numpy as np

import pandas as pd



def get_ds_infos():

    """

    Read the file includes data subject information.

    

    Data Columns:

    0: code [1-24]

    1: weight [kg]

    2: height [cm]

    3: age [years]

    4: gender [0:Female, 1:Male]

    

    Returns:

        A pandas DataFrame that contains inforamtion about data subjects' attributes 

    """ 



    dss = pd.read_csv("../input/motionsense-dataset/data_subjects_info.csv")

    print("[INFO] -- Data subjects' information is imported.")

    

    return dss



def set_data_types(data_types=["userAcceleration"]):

    """

    Select the sensors and the mode to shape the final dataset.

    

    Args:

        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 



    Returns:

        It returns a list of columns to use for creating time-series from files.

    """

    dt_list = []

    for t in data_types:

        if t != "attitude":

            dt_list.append([t+".x",t+".y",t+".z"])

        else:

            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    print(dt_list)

    return dt_list





def creat_time_series(folder_name, dt_list, act_labels, trial_codes, mode="mag", labeled=True):

    """

    Args:

        folder_name: one of 'A_DeviceMotion_data', 'B_Accelerometer_data', or C_Gyroscope_data

        dt_list: A list of columns that shows the type of data we want.

        act_labels: list of activites

        trial_codes: list of trials

        mode: It can be 'raw' which means you want raw data

        for every dimention of each data type,

        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].

        or it can be 'mag' which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)

        labeled: True, if we want a labeld dataset. False, if we only want sensor values.



    Returns:

        It returns a time-series of sensor data.

    

    """

    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)



    if labeled:

        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 

    else:

        dataset = np.zeros((0,num_data_cols))

        

    ds_list = get_ds_infos()

    

    print("[INFO] -- Creating Time-Series")

    for sub_id in ds_list["code"]:

        for act_id, act in enumerate(act_labels):

            for trial in trial_codes[act_id]:

                fname = folder_name+'/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'

                raw_data = pd.read_csv(fname)

                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)

                vals = np.zeros((len(raw_data), num_data_cols))

                for x_id, axes in enumerate(dt_list):

                    if mode == "mag":

                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        

                    else:

                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values

                    vals = vals[:,:num_data_cols]

                if labeled:

                    lbls = np.array([[act_id,

                            sub_id-1,

                            ds_list["weight"][sub_id-1],

                            ds_list["height"][sub_id-1],

                            ds_list["age"][sub_id-1],

                            ds_list["gender"][sub_id-1],

                            trial          

                           ]]*len(raw_data), dtype=int)

                    vals = np.concatenate((vals, lbls), axis=1)

                dataset = np.append(dataset,vals, axis=0)

    cols = []

    for axes in dt_list:

        if mode == "raw":

            cols += axes

        else:

            cols += [str(axes[0][:-2])]

            

    if labeled:

        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]

    

    dataset = pd.DataFrame(data=dataset, columns=cols)

    return dataset

#________________________________





ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]

TRIAL_CODES = {

    ACT_LABELS[0]:[1,2,11],

    ACT_LABELS[1]:[3,4,12],

    ACT_LABELS[2]:[7,8,15],

    ACT_LABELS[3]:[9,16],

    ACT_LABELS[4]:[6,14],

    ACT_LABELS[5]:[5,13]

}
sdt = ["rotationRate", "userAcceleration"]

print("Selected sensor data types:\n" + str(sdt))

dt_list = set_data_types(sdt)

print("\nSelected columns from dataset:\n" + str(dt_list))
ACT_LABELS = ["sit", "std", "dws", "ups", "wlk", "jog"]

act_labels = ACT_LABELS [0:6]  # all activities

print("Selected activites: " + str(act_labels))    
TRIAL_CODES = {

    ACT_LABELS[0]:[5],

    ACT_LABELS[1]:[6],

    ACT_LABELS[2]:[1],

    ACT_LABELS[3]:[3],

    ACT_LABELS[4]:[7],

    ACT_LABELS[5]:[9],

}

trial_codes = [TRIAL_CODES[act] for act in act_labels]

print("[INFO] -- Selected trials: " + str(trial_codes))
print("Loading...")

dataset = creat_time_series("../input/motionsense-dataset/A_DeviceMotion_data/A_DeviceMotion_data", dt_list, act_labels, trial_codes, mode="mag", labeled=True)

print("Finished!")

dataset.head()
plt.rcParams['figure.figsize'] = (30,8)

plt.rcParams['font.size'] = 32

plt.rcParams['image.cmap'] = 'plasma'

plt.rcParams['axes.linewidth'] = 2

clr1 = ["rs-","r*-","ro-","rv-","rp-","r^-"]

clr2 = ["bs-","b*-","bo-","bv-","bp-","b^-"]

act_lbl = ["Sat", "Stand-Up", "Downstairs", "Upstairs", "Walking", "Jogging"]

lbl = ["rotation", "acceleration"]
period = 2.5 # Seconds

sample_rate = 50 # Hz

points = int(period*sample_rate)

x_ticks = np.arange(0.,points/sample_rate,1./sample_rate)

print("Data points per time-series: " + str(points))
act_data = np.zeros((6,points))

fig, ax = plt.subplots(1, 6, sharex='col', sharey='row')

uid = 12 # We have 24 users in the dataset, uid can be selected from {0,1,...23}

for i in np.unique(dataset["act"]):

    i =int(i)

    data = dataset[(dataset["id"] == uid) & (dataset["act"] == i)]

    acc = data["userAcceleration"].values

    rot = data["rotationRate"].values

    acc = acc[:points]

    rot = rot[:points]

    

    if i!=0:

        ax[i].plot(x_ticks, rot, "ro-", linewidth=2, markersize=8)

        ax[i].plot(x_ticks, acc, "b^-", linewidth=2, markersize=8)

    else:

        ax[i].plot(x_ticks, rot, "ro-", linewidth=2, markersize=12, label=lbl[0])

        ax[i].plot(x_ticks, acc, "b^-", linewidth=2, markersize=12, label=lbl[1])



    ax[i].set_title(act_lbl[i])

plt.setp(ax, yticks=np.arange(0, 11, 2))

fig.text(0.5, 0.004, 'second', ha='center')

fig.text(0.075, 0.5, 'magnitude value', va='center', rotation='vertical', )

ax[0].legend(loc="upper center", fontsize = 20)
def read_data(path, filename):

    return pd.read_csv(os.path.join(path, filename))



df = read_data('../input/human-activity-recognition-with-smartphones', 'train.csv')

df.head()
df[['Activity']].head()
def load_dataset(label_dict):

    train_X = read_data('../input/human-activity-recognition-with-smartphones', 'train.csv').values[:,:-2]

    train_y = read_data('../input/human-activity-recognition-with-smartphones', 'train.csv')['Activity']

    train_y = train_y.map(label_dict).values

    test_X = read_data('../input/human-activity-recognition-with-smartphones', 'test.csv').values[:,:-2]

    test_y = read_data('../input/human-activity-recognition-with-smartphones', 'test.csv')

    test_y = test_y['Activity'].map(label_dict).values

    return(train_X, train_y, test_X, test_y)

label_dict = {'WALKING':0, 'WALKING_UPSTAIRS':1, 'WALKING_DOWNSTAIRS':2, 'SITTING':3, 'STANDING':4, 'LAYING':5}

train_X, train_y, test_X, test_y = load_dataset(label_dict)
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100)
train_X.shape
model.fit(train_X, train_y)
yhat = model.predict(test_X)

yhat
from sklearn.metrics import classification_report

target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']



print(classification_report(test_y, yhat, target_names=target_names))