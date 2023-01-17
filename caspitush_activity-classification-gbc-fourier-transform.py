import os

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, plot_confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier as gbc
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0) # I like big figures!
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



    dss = pd.read_csv("data_subjects_info.csv")

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



    return dt_list





def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):

    """

    Args:

        dt_list: A list of columns that shows the type of data we want.

        act_labels: list of activites

        trial_codes: list of trials

        mode: It can be "raw" which means you want raw data

        for every dimention of each data type,

        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].

        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)

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

                fname = 'A_DeviceMotion_data/A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'

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

                           ]]*len(raw_data))

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

ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]

TRIAL_CODES = {

    ACT_LABELS[0]:[1,2,11],

    ACT_LABELS[1]:[3,4,12],

    ACT_LABELS[2]:[7,8,15],

    ACT_LABELS[3]:[9,16],

    ACT_LABELS[4]:[6,14],

    ACT_LABELS[5]:[5,13]

}



## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"

## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)

sdt = ["attitude", "userAcceleration"]

print("[INFO] -- Selected sensor data types: "+str(sdt))    

act_labels = ACT_LABELS [0:6]

print("[INFO] -- Selected activites: "+str(act_labels))    

trial_codes = [TRIAL_CODES[act] for act in act_labels]

dt_list = set_data_types(sdt)
os.chdir('/kaggle/input/motionsense-dataset/')

dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)

print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))    

dataset.head()
act_dict = {0: 'dws',

            1: 'ups',

            2: 'wlk',

            3: 'jog',

            4: 'std',

            5: 'sit'}

dataset['label'] =  dataset.act.apply(lambda act: act_dict[act])
dataset.isna().sum()
dataset.label.value_counts()
metrics = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'userAcceleration.x', 'userAcceleration.y', 

            'userAcceleration.z']
def plot_data(start_position, number_of_frames, metrics=metrics, fourier=False):

    print(f"Looking at {metrics} for {number_of_frames} from {start_position}")

    for label in dataset.label.unique():

        mini_df = dataset[dataset['label']==label].iloc[start_position:start_position+number_of_frames].reset_index()

        for metric in metrics:

            mini_df[metric].plot(title=label, legend=True)

        plt.show()
plot_data(np.random.randint(1,130000), 300)
plot_data(np.random.randint(1,130000), 300)
acceleration_metrics = [metric for metric in metrics if metric.startswith('userAcceler')]

acceleration_metrics
for metric in acceleration_metrics:

    dataset[f'{metric}_ra'] = dataset[metric].rolling(20).median()
avg_acc_metrics = [metric for metric in dataset.columns if metric.endswith('_ra')]

avg_acc_metrics
dataset.columns
df = dataset.fillna(0, axis=0) # the rolling window made a bunch of nans and it makes our fft unhappy.
metrics = ['userAcceleration.x_ra', 'userAcceleration.y_ra', 'userAcceleration.z_ra', 

           'attitude.roll', 'attitude.pitch', 'attitude.yaw']



def build_features(data=df, number_of_frames=300, metrics=metrics):

    instances = pd.DataFrame() # this is where our features and labels will end up.

    instance = 0

    data.fillna(method='bfill')

    for label in data.label.unique():

        print(f"Building features for {label}...")

        start_position=0

        label_df = data[data['label']==label]

        while len(label_df) > start_position+number_of_frames:

            for metric in metrics:

                instance_df = label_df.iloc[start_position:start_position+number_of_frames].reset_index()

                instances.loc[instance, 'label'] = label

                instances.loc[instance, f'median_{metric}'] = instance_df[metric].median()

                instances.loc[instance, f'std_{metric}'] = instance_df[metric].std()

                fourier = np.fft.rfft(instance_df[metric])[1:]

                amplitude = max(np.abs(fourier))

                frequency = np.where(np.abs(fourier)==amplitude)[0][0]

                instances.loc[instance, f'amplitude_{metric}'] = amplitude

                instances.loc[instance, f'frequency_{metric}'] = frequency

                instances.loc[instance, f'phase_{metric}'] = np.angle(fourier)[frequency]

            instance = instance + 1

            start_position = start_position + number_of_frames

    return instances
instances = build_features()
instances.head()
len(instances)
instances.label.value_counts()
for label in instances.label.unique():

    for feature in ['median_attitude.pitch']:

        instances[instances.label==label][feature].hist(alpha=0.3, label=label, bins=20)

plt.legend()

plt.show()
frequencies = [feature for feature in instances.columns if feature.startswith('frequency')]

frequencies
instances['med_freq'] = instances[frequencies].median(axis=1)
for label in instances.label.unique():

    instances[instances.label==label]['med_freq'].hist(alpha=0.3, label=label, bins=20)

plt.legend()

plt.show()
def plot_feature_importance(model, feature_names):

    feature_importances = model.feature_importances_

    idxSorted = np.argsort(feature_importances)[-10:]

    barPos = np.arange(idxSorted.shape[0]) + .5

    plt.barh(barPos, feature_importances[idxSorted], align='center')

    plt.yticks(barPos, feature_names[idxSorted])

    plt.xlabel('Feature Importance')

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    plt.show()
y = instances.label

X = instances.drop('label', axis=1)

kf = KFold(n_splits=5, shuffle=True)

gbc_model = gbc()

MMC = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y.loc[train_index], y.loc[test_index]

    gbc_model.fit(X_train, y_train)

    y_pred = gbc_model.predict(X_test)

    print(f"MMC: {matthews_corrcoef(y_test, y_pred):.3f}")

    MMC.append(matthews_corrcoef(y_test, y_pred))

print(f"Mean MMC: {np.mean(MMC):.3f}")

print(f"Std of MMCs: {np.std(MMC):.4f}")

print("These are the plots of the last test so we could get an idea of what it looks like:")

plot_confusion_matrix(gbc_model, X=X_test, y_true=y_test, labels=gbc_model.classes_, cmap='Blues')

plt.show()

plot_feature_importance(gbc_model, X_test.columns)
no_fft_features = [feature for feature in instances.columns  if feature.startswith('median') or feature.startswith('std')]

no_fft_features
y = instances.label

features = no_fft_features

X = instances[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

gbc_model = gbc()

gbc_model.fit(X_train, y_train)

y_pred = gbc_model.predict(X_test)

print(f"MMC: {matthews_corrcoef(y_test, y_pred)}")

plot_confusion_matrix(gbc_model, X=X_test, y_true=y_test, labels=gbc_model.classes_, cmap='Blues')

plt.show()

plot_feature_importance(gbc_model, X_test.columns)