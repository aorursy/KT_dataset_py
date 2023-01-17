!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index



import efficientnet.tfkeras as efn
import os

import random

import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold, train_test_split 

import tensorflow as tf

from tensorflow.keras import Model, backend

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.utils import Sequence

from keras.utils.vis_utils import plot_model



import pydicom

import cv2



def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)



config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)



palette_ro = ["#ee2f35", "#fa7211", "#fbd600", "#75c731", "#1fb86e", "#0488cf", "#7b44ab"]



ROOT = "../input/osic-pulmonary-fibrosis-progression/"
train = pd.read_csv(ROOT + "train.csv")

test = pd.read_csv(ROOT + "test.csv")

sub = pd.read_csv(ROOT + "sample_submission.csv")



print("Training data shape: ", train.shape)

print("Test data shape: ", test.shape)



train.head(10)
train.isnull().sum()
test.isnull().sum()
dupRows_train = train[train.duplicated(subset=['Patient', 'Weeks'], keep=False)]



print("There are {} duplicate rows here ({:.2f} percent of the total).".format(len(dupRows_train), len(dupRows_train)/len(train)*100))

dupRows_train
train.drop_duplicates(subset=['Patient', 'Weeks'], keep=False, inplace=True)
stats = []

for col in train.columns:

    stats.append((col,

                  train[col].nunique(),

                  train[col].value_counts().index[0],

                  train[col].value_counts().values[0],

                  train[col].isnull().sum() * 100 / train.shape[0],

                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100,

                  train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique values', 'Most frequent item', 'Freuquence of most frequent item', 'Percentage of missing values', 'Percentage of values in the biggest category', 'Type'])

stats_df.sort_values('Percentage of missing values', ascending=False)
data = train.groupby("Patient").first().reset_index(drop=True)

data.head()
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 12))



sns.distplot(data["Age"], ax=ax1, bins=data["Age"].max()-data["Age"].min()+1, color=palette_ro[1])

ax1.annotate("Min: {:,}".format(data["Age"].min()), xy=(data["Age"].min(), 0.005), 

             xytext=(data["Age"].min()-8, 0.02),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:,}".format(data["Age"].max()), xy=(data["Age"].max(), 0.005), 

             xytext=(data["Age"].max()-2, 0.02),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                             connectionstyle="arc3, rad=-0.2"))

ax1.axvline(x=data["Age"].median(), color=palette_ro[0], linestyle="--", alpha=0.5)

ax1.annotate("Med: {:.0f}".format(data["Age"].median()), xy=(data["Age"].median(), 0.056), 

             xytext=(data["Age"].median()-15, 0.065),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                             connectionstyle="arc3, rad=0.25"))



sns.countplot(x="Sex", ax=ax2, data=data, palette=palette_ro[-2::-4])

sns.countplot(x="SmokingStatus", ax=ax3, data=data,

              order=["Never smoked", "Ex-smoker", "Currently smokes"], palette=palette_ro[-3::-2])



sns.distplot(data[data["Sex"]=="Male"].Age, label="Male", ax=ax4, hist=False, color=palette_ro[5])

sns.distplot(data[data["Sex"]=="Female"].Age, label="Female", ax=ax4, hist=False, color=palette_ro[1])



sns.distplot(data[data["SmokingStatus"]=="Never smoked"].Age, label="Never smoked", ax=ax5, hist=False, color=palette_ro[4])

sns.distplot(data[data["SmokingStatus"]=="Ex-smoker"].Age, label="Ex-smoker", ax=ax5, hist=False, color=palette_ro[2])

sns.distplot(data[data["SmokingStatus"]=="Currently smokes"].Age, label="Currently smokes", ax=ax5, hist=False, color=palette_ro[0])



sns.countplot(x="SmokingStatus", ax=ax6, data=data, hue="Sex",

              order=["Never smoked", "Ex-smoker", "Currently smokes"], palette=palette_ro[-2::-4])



fig.suptitle("Distribution of unique patients data", fontsize=18);
fig, ax = plt.subplots(figsize=(16, 6))



sns.distplot(train["Weeks"], ax=ax, color=palette_ro[1], bins=train["Weeks"].max()-train["Weeks"].min()+1)

ax.annotate("Min: {:,}".format(train["Weeks"].min()), xy=(train["Weeks"].min(), 0.005), 

            xytext=(train["Weeks"].min()-8, 0.008),

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            arrowprops=dict(arrowstyle="->",

                            connectionstyle="arc3, rad=0.2"))

ax.annotate("Max: {:,}".format(train["Weeks"].max()), xy=(train["Weeks"].max(), 0.005), 

            xytext=(train["Weeks"].max()-2, 0.008),

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            arrowprops=dict(arrowstyle="->",

                            connectionstyle="arc3, rad=-0.2"))

ax.axvline(x=0, color=palette_ro[5], linestyle="--", alpha=0.5)

ax.annotate("CT Scan", xy=(0, 0.013), 

            xytext=(-12, 0.016),

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            arrowprops=dict(arrowstyle="->",

                            connectionstyle="arc3, rad=0.2"))

ax.axvline(x=train["Weeks"].median(), color=palette_ro[0], linestyle="--", alpha=0.5)

ax.annotate("Med: {:.0f}".format(train["Weeks"].median()), xy=(train["Weeks"].median(), 0.020), 

            xytext=(train["Weeks"].median()+2, 0.024),

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=-0.2"))



ax.set_title("Weeks Distribution", fontsize=18);
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



sns.distplot(train["FVC"], ax=ax1, color=palette_ro[5], hist=False)

ax1.annotate("Min: {:,}".format(train["FVC"].min()), xy=(train["FVC"].min(), 0.00005), 

             xytext=(train["FVC"].min()-300, 0.0001),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                            connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:,}".format(train["FVC"].max()), xy=(train["FVC"].max(), 0.00005), 

             xytext=(train["FVC"].max()-200, 0.0001),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                             connectionstyle="arc3, rad=-0.2"))

ax1.axvline(x=train["FVC"].median(), color=palette_ro[0], linestyle="--", alpha=0.5)

ax1.annotate("Med: {:,.0f}".format(train["FVC"].median()), xy=(train["FVC"].median(), 0.00005), 

             xytext=(train["FVC"].median()-750, 0.0001),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=0.25"))



ax1.set_title("FVC Distribution", fontsize=16);



sns.distplot(train["Percent"], ax=ax2, color=palette_ro[3], hist=False)

ax2.annotate("Min: {:.2f}".format(train["Percent"].min()), xy=(train["Percent"].min(), 0.0015), 

             xytext=(train["Percent"].min()-8, 0.0040),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                            connectionstyle="arc3, rad=0.2"))

ax2.annotate("Max: {:.2f}".format(train["Percent"].max()), xy=(train["Percent"].max(), 0.0015), 

             xytext=(train["Percent"].max()-4, 0.0040),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                             connectionstyle="arc3, rad=-0.2"))

ax2.axvline(x=train["Percent"].median(), color=palette_ro[0], linestyle="--", alpha=0.5)

ax2.annotate("Med: {:.2f}".format(train["Percent"].median()), xy=(train["Percent"].median(), 0.0015), 

             xytext=(train["Percent"].median()-17, 0.0040),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=0.25"))



ax2.set_title("Percent Distribution", fontsize=16);
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



sns.distplot(train[train["Sex"]=="Male"].FVC, label="Male", ax=ax1, hist=False, color=palette_ro[5])

ax1.axvline(x=train[train["Sex"]=="Male"].FVC.median(), color=palette_ro[5], linestyle="--", alpha=0.5)

ax1.annotate("Male\nMed: {:,.0f}".format(train[train["Sex"]=="Male"].FVC.median()), xy=(train[train["Sex"]=="Male"].FVC.median(), 0.0006), 

             xytext=(train[train["Sex"]=="Male"].FVC.median()+100, 0.00065),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=-0.2"))

sns.distplot(train[train["Sex"]=="Female"].FVC, label="Female", ax=ax1, hist=False, color=palette_ro[1])

ax1.axvline(x=train[train["Sex"]=="Female"].FVC.median(), color=palette_ro[1], linestyle="--", alpha=0.5)

ax1.annotate("Female\nMed: {:,.0f}".format(train[train["Sex"]=="Female"].FVC.median()), xy=(train[train["Sex"]=="Female"].FVC.median(), 0.0008), 

             xytext=(train[train["Sex"]=="Female"].FVC.median()+100, 0.00085),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=-0.2"))



sns.distplot(train[train["SmokingStatus"]=="Never smoked"].FVC, label="Never smoked", ax=ax2, hist=False, color=palette_ro[4])

ax2.axvline(x=train[train["SmokingStatus"]=="Never smoked"].FVC.median(), color=palette_ro[4], linestyle="--", alpha=0.5)

ax2.annotate("Never smoked\nMed: {:.0f}".format(train[train["SmokingStatus"]=="Never smoked"].FVC.median()), xy=(train[train["SmokingStatus"]=="Never smoked"].FVC.median(), 0.0005), 

             xytext=(train[train["SmokingStatus"]=="Never smoked"].FVC.median()-1000, 0.00055),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=0.25"))

sns.distplot(train[train["SmokingStatus"]=="Ex-smoker"].FVC, label="Ex-smoker", ax=ax2, hist=False, color=palette_ro[2])

ax2.axvline(x=train[train["SmokingStatus"]=="Ex-smoker"].FVC.median(), color=palette_ro[2], linestyle="--", alpha=0.75)

ax2.annotate("Ex-smoker\nMed: {:.0f}".format(train[train["SmokingStatus"]=="Ex-smoker"].FVC.median()), xy=(train[train["SmokingStatus"]=="Ex-smoker"].FVC.median(), 0.00058), 

             xytext=(train[train["SmokingStatus"]=="Ex-smoker"].FVC.median()-1200, 0.0007),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=-0.25"))

sns.distplot(train[train["SmokingStatus"]=="Currently smokes"].FVC, label="Currently smokes", ax=ax2, hist=False, color=palette_ro[0])

ax2.axvline(x=train[train["SmokingStatus"]=="Currently smokes"].FVC.median(), color=palette_ro[0], linestyle="--", alpha=0.5)

ax2.annotate("Currently smokes\nMed: {:.0f}".format(train[train["SmokingStatus"]=="Currently smokes"].FVC.median()), xy=(train[train["SmokingStatus"]=="Currently smokes"].FVC.median(), 0.0009), 

             xytext=(train[train["SmokingStatus"]=="Currently smokes"].FVC.median()+400, 0.00095),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=0.25"))



ax1.set_title("Relationship between FVC and Sex", fontsize=16)

ax2.set_title("Relationship between FVC and SmokingStatus", fontsize=16);
train_m = train[train["Sex"]=="Male"].reset_index(drop=True)

train_f = train[train["Sex"]=="Female"].reset_index(drop=True)



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



sns.distplot(train_m[train_m["SmokingStatus"]=="Never smoked"].FVC, label="Never smoked", ax=ax1, hist=False, color=palette_ro[4])

ax1.axvline(x=train_m[train_m["SmokingStatus"]=="Never smoked"].FVC.median(), color=palette_ro[4], linestyle="--", alpha=0.5)

ax1.annotate("Never smoked\nMed: {:.0f}".format(train_m[train_m["SmokingStatus"]=="Never smoked"].FVC.median()), xy=(train_m[train_m["SmokingStatus"]=="Never smoked"].FVC.median(), 0.0005), 

             xytext=(train_m[train_m["SmokingStatus"]=="Never smoked"].FVC.median()-1400, 0.0006),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=0.25"))

sns.distplot(train_m[train_m["SmokingStatus"]=="Ex-smoker"].FVC, label="Ex-smoker", ax=ax1, hist=False, color=palette_ro[2])

ax1.axvline(x=train_m[train_m["SmokingStatus"]=="Ex-smoker"].FVC.median(), color=palette_ro[2], linestyle="--", alpha=0.75)

ax1.annotate("Ex-smoker\nMed: {:.0f}".format(train_m[train_m["SmokingStatus"]=="Ex-smoker"].FVC.median()), xy=(train_m[train_m["SmokingStatus"]=="Ex-smoker"].FVC.median(), 0.00063), 

             xytext=(train_m[train_m["SmokingStatus"]=="Ex-smoker"].FVC.median()-1400, 0.00045),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=-0.2"))

sns.distplot(train_m[train_m["SmokingStatus"]=="Currently smokes"].FVC, label="Currently smokes", ax=ax1, hist=False, color=palette_ro[0])

ax1.axvline(x=train_m[train_m["SmokingStatus"]=="Currently smokes"].FVC.median(), color=palette_ro[0], linestyle="--", alpha=0.5)

ax1.annotate("Currently smokes\nMed: {:.0f}".format(train_m[train_m["SmokingStatus"]=="Currently smokes"].FVC.median()), xy=(train_m[train_m["SmokingStatus"]=="Currently smokes"].FVC.median(), 0.00066), 

             xytext=(train_m[train_m["SmokingStatus"]=="Currently smokes"].FVC.median()+400, 0.00055),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=0.25"))



sns.distplot(train_f[train_f["SmokingStatus"]=="Never smoked"].FVC, label="Never smoked", ax=ax2, hist=False, color=palette_ro[4])

ax2.axvline(x=train_f[train_f["SmokingStatus"]=="Never smoked"].FVC.median(), color=palette_ro[4], linestyle="--", alpha=0.5)

ax2.annotate("Never smoked\nMed: {:.0f}".format(train_f[train_f["SmokingStatus"]=="Never smoked"].FVC.median()), xy=(train_f[train_f["SmokingStatus"]=="Never smoked"].FVC.median(), 0.001), 

             xytext=(train_f[train_f["SmokingStatus"]=="Never smoked"].FVC.median()-600, 0.0015),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=-0.2"))

sns.distplot(train_f[train_f["SmokingStatus"]=="Ex-smoker"].FVC, label="Ex-smoker", ax=ax2, hist=False, color=palette_ro[2])

ax2.axvline(x=train_f[train_f["SmokingStatus"]=="Ex-smoker"].FVC.median(), color=palette_ro[2], linestyle="--", alpha=0.75)

ax2.annotate("Ex-smoker\nMed: {:.0f}".format(train_f[train_f["SmokingStatus"]=="Ex-smoker"].FVC.median()), xy=(train_f[train_f["SmokingStatus"]=="Ex-smoker"].FVC.median(), 0.0013), 

             xytext=(train_f[train_f["SmokingStatus"]=="Ex-smoker"].FVC.median()+100, 0.0018),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=-0.25"))

sns.distplot(train_f[train_f["SmokingStatus"]=="Currently smokes"].FVC, label="Currently smokes", ax=ax2, hist=False, color=palette_ro[0])

ax2.axvline(x=train_f[train_f["SmokingStatus"]=="Currently smokes"].FVC.median(), color=palette_ro[0], linestyle="--", alpha=0.5)

ax2.annotate("Currently smokes\nMed: {:.0f}".format(train_f[train_f["SmokingStatus"]=="Currently smokes"].FVC.median()), xy=(train_f[train_f["SmokingStatus"]=="Currently smokes"].FVC.median(), 0.0035), 

             xytext=(train_f[train_f["SmokingStatus"]=="Currently smokes"].FVC.median()+200, 0.004),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->",

                           connectionstyle="arc3, rad=0.25"))



ax1.set_title("Relationship between FVC and SmokingStatus in Male", fontsize=16)

ax2.set_title("Relationship between FVC and SmokingStatus in Female", fontsize=16);
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



sns.scatterplot(x=train["FVC"], y=train["Age"], ax=ax1,

                palette=[palette_ro[5], palette_ro[1]], hue=train["Sex"], style=train["Sex"])

sns.scatterplot(x=train["FVC"], y=train["Age"], ax=ax2,

                palette=[palette_ro[2], palette_ro[4], palette_ro[0]], hue=train["SmokingStatus"], style=train["SmokingStatus"])



fig.suptitle("Correlation between FVC and Age (Pearson Corr: {:.4f})".format(train["FVC"].corr(train["Age"])), fontsize=16);
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



sns.scatterplot(x=train["FVC"], y=train["Weeks"], ax=ax1,

                palette=[palette_ro[5], palette_ro[1]], hue=train["Sex"], style=train["Sex"])

sns.scatterplot(x=train["FVC"], y=train["Weeks"], ax=ax2,

                palette=[palette_ro[2], palette_ro[4], palette_ro[0]], hue=train["SmokingStatus"], style=train["SmokingStatus"])



fig.suptitle("Correlation between FVC and Weeks (Pearson Corr: {:.4f})".format(train["FVC"].corr(train["Weeks"])), fontsize=16);
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



sns.scatterplot(x=train["FVC"], y=train["Age"], ax=ax1,

                palette=[palette_ro[5], palette_ro[1]], hue=train["Sex"], style=train["Sex"])

sns.scatterplot(x=train["FVC"], y=train["Age"], ax=ax2,

                palette=[palette_ro[2], palette_ro[4], palette_ro[0]], hue=train["SmokingStatus"], style=train["SmokingStatus"])



fig.suptitle("Correlation between FVC and Age (Pearson Corr: {:.4f})".format(train["FVC"].corr(train["Age"])), fontsize=16);
def get_tab(df):

    vector = [(df.Age.values[0] - 30) / 30]

    

    if df.Sex.values[0] == 'male':

       vector.append(0)

    else:

       vector.append(1)

    

    if df.SmokingStatus.values[0] == 'Never smoked':

        vector.extend([0,0])

    elif df.SmokingStatus.values[0] == 'Ex-smoker':

        vector.extend([1,1])

    elif df.SmokingStatus.values[0] == 'Currently smokes':

        vector.extend([0,1])

    else:

        vector.extend([1,0])

    return np.array(vector)
A = {} 

TAB = {} 

P = [] 

for i, p in tqdm(enumerate(train.Patient.unique())):

    sub = train.loc[train.Patient == p, :] 

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc)[0]

    

    A[p] = a

    TAB[p] = get_tab(sub)

    P.append(p)



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.distplot(list(A.values()), ax=ax, color=palette_ro[1]);
# def get_img(path):

#     d = pydicom.dcmread(path)

#     return cv2.resize(d.pixel_array / 2**11, (528, 528))    # changed from 512



# https://www.kaggle.com/allunia/pulmonary-dicom-preprocessing

def get_img(path, new_shape=(528, 528)):

    d = pydicom.dcmread(path)

    scan = d.pixel_array / 2**11

    

    left = int((scan.shape[0]-512)/2)

    right = int((scan.shape[0]+512)/2)

    top = int((scan.shape[1]-512)/2)

    bottom = int((scan.shape[1]+512)/2)

    

    img = scan[top:bottom, left:right]

    cropped_resized_scan = cv2.resize(img, new_shape, interpolation=cv2.INTER_LANCZOS4)

    return cropped_resized_scan



# get_img("../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm")
class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab, batch_size=32):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.a = a

        self.tab = tab

        self.batch_size = batch_size

        

        self.train_data = {}

        for p in train.Patient.values:

            self.train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

    

    def __len__(self):

        return 1000

    

    def __getitem__(self, idx):

        x = []

        a, tab = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)

        for k in keys:

            try:

                i = np.random.choice(self.train_data[k], size=1)[0]

                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

                x.append(img)

                a.append(self.a[k])

                tab.append(self.tab[k])

            except:

                print(k, i)

       

        x,a,tab = np.array(x), np.array(a), np.array(tab)

        x = np.expand_dims(x, axis=-1)

        return [x, tab] , a
%%time

def get_efficientnet(model, shape):

    models_dict = {

        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),

        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),

        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),

        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),

        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),

        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),

        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),

        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)

    }

    return models_dict[model]



def build_model(shape=(528, 528, 1), model_class=None):    # changed from 512

    inp = L.Input(shape=shape)

    base = get_efficientnet(model_class, shape)

    x = base(inp)

    x = L.GlobalAveragePooling2D()(x)

    inp2 = L.Input(shape=(4,))

    x2 = L.GaussianNoise(0.2)(inp2)

    x = L.Concatenate()([x, x2]) 

    x = L.Dropout(0.32)(x)    # changed from 0.4

    x = L.Dense(1)(x)

    model = Model([inp, inp2] , x)

    

    weights = [w for w in os.listdir('../input/osic-model-weights') if model_class in w][0]

    model.load_weights('../input/osic-model-weights/' + weights)

    return model



model_classes = ["b6"] #['b0','b1','b2','b3',b4','b5','b6','b7']    # changed from b5

models = [build_model(model_class=m, shape=(528, 528, 1)) for m in model_classes]    # changed from 512

print('Number of models: ' + str(len(models)))
plot_model(models[0])
def score(fvc_true, fvc_pred, sigma):

    sigma_clip = np.maximum(sigma, 70)

    delta = np.abs(fvc_true - fvc_pred)

    delta = np.minimum(delta, 1000)

    sq2 = np.sqrt(2)

    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip * sq2)

    return np.mean(metric)
tr_p, vl_p = train_test_split(P, 

                              shuffle=True, 

                              train_size=0.8)



subs = []

for model in models:

    metric = []

    for q in tqdm(range(1, 10)):

        m = []

        for p in vl_p:

            x = [] 

            tab = [] 



            if p in ['ID00011637202177653955184', 'ID00052637202186188008618']:

                continue



            ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

            for i in ldir:

                if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                    x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}')) 

                    tab.append(get_tab(train.loc[train.Patient == p, :])) 

            if len(x) < 1:

                continue

            tab = np.array(tab) 



            x = np.expand_dims(x, axis=-1) 

            _a = model.predict([x, tab]) 

            a = np.quantile(_a, q / 10)



            percent_true = train.Percent.values[train.Patient == p]

            fvc_true = train.FVC.values[train.Patient == p]

            weeks_true = train.Weeks.values[train.Patient == p]



            fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]

            percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])

            m.append(score(fvc_true, fvc, percent))

        print(np.mean(m))

        metric.append(np.mean(m))



    q = (np.argmin(metric) + 1)/ 10



    sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv') 

    test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv') 

    A_test, B_test, P_test, W, FVC = {}, {}, {}, {}, {} 

    STD, WEEK = {}, {} 

    for p in test.Patient.unique():

        x = [] 

        tab = [] 

        ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')

        for i in ldir:

            if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/{i}')) 

                tab.append(get_tab(test.loc[test.Patient == p, :])) 

        if len(x) <= 1:

            continue

        tab = np.array(tab) 



        x = np.expand_dims(x, axis=-1) 

        _a = model.predict([x, tab]) 

        a = np.quantile(_a, q)

        A_test[p] = a

        B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]

        P_test[p] = test.Percent.values[test.Patient == p] 

        WEEK[p] = test.Weeks.values[test.Patient == p]



    for k in sub.Patient_Week.values:

        p, w = k.split('_')

        w = int(w) 



        fvc = A_test[p] * w + B_test[p]

        sub.loc[sub.Patient_Week == k, "FVC"] = fvc

        sub.loc[sub.Patient_Week == k, "Confidence"] = (

            P_test[p] - A_test[p] * abs(WEEK[p] - w) 

    ) 



    _sub = sub[["Patient_Week","FVC","Confidence"]].copy()

    subs.append(_sub)
N = len(subs)

sub = subs[0].copy() # ref

sub["FVC"] = 0

sub["Confidence"] = 0

for i in range(N):

    sub["FVC"] += subs[0]["FVC"] * (1/N)

    sub["Confidence"] += subs[0]["Confidence"] * (1/N)
sub.head()
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("submission_img.csv", index=False)
img_sub = sub[["Patient_Week","FVC","Confidence"]].copy()
sub = pd.read_csv(ROOT + "sample_submission.csv")

sub.head()
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]

sub = sub.merge(test.drop('Weeks', axis=1), on="Patient")

sub.head()
train['WHERE'] = 'train'

test['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = train.append([test, sub])



print(train.shape, test.shape, sub.shape, data.shape)

print(train.Patient.nunique(), test.Patient.nunique(), sub.Patient.nunique(), data.Patient.nunique())



data.head(10)
data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test', 'min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')



data.head(10)
base = data.loc[data.Weeks == data.min_week]

base = base[['Patient', 'FVC']].copy()

base.columns = ['Patient', 'base_FVC']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)



base.head()
data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base



data.head(10)
categorical_features = ['Sex', 'SmokingStatus']

features_nn = []

for col in categorical_features:

    for mod in data[col].unique():

        features_nn.append(mod)

        data[mod] = (data[col] == mod).astype(int)



data.head(10)
data['Percent_n'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

data['Age_n'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['base_FVC_n'] = (data['base_FVC'] - data['base_FVC'].min() ) / ( data['base_FVC'].max() - data['base_FVC'].min() )

data['base_week_n'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

features_nn += ['Age_n', 'Percent_n', 'base_week_n', 'base_FVC_n']



print(features_nn)

data.head(10)
train = data.loc[data.WHERE=='train']

test = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']

del data



train.shape, test.shape, sub.shape
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")

#=============================#

def score(y_true, y_pred):

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]



    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip * sq2)

    return backend.mean(metric)

#============================#

def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.2, 0.5, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return backend.mean(v)

#=============================#

def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss
def make_model():

    inp = L.Input(len(features_nn), name="Patient")

    x = L.Dense(100, activation="relu", name="d1")(inp)

    x = L.Dense(100, activation="relu", name="d2")(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)

    p2 = L.Dense(3, activation="relu", name="p2")(x)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                     name="preds")([p1, p2])

    

    model = M.Model(inp, preds, name="NeuralNet")

    model.compile(loss=mloss(0.64),    # changed from 0.8

                  optimizer=tf.keras.optimizers.Adam(lr=0.1, decay=0.01),

                  metrics=[score])

    return model



model = make_model()

model.summary()
plot_model(model)
X_train = train[features_nn].values

X_test = sub[features_nn].values



y_train = train['FVC'].values



oof_train = np.zeros((X_train.shape[0], 3))

y_preds = np.zeros((X_test.shape[0], 3))
BATCH_SIZE = 128

EPOCHS = 804    # changed from 800

NFOLD = 5



kf = KFold(n_splits=NFOLD)
%%time

for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train)):

    print(f"FOLD {fold_id+1}")

    model = make_model()

    model.fit(X_train[tr_idx], y_train[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, 

              validation_data=(X_train[va_idx], y_train[va_idx]), verbose=0)

    print("train", model.evaluate(X_train[tr_idx], y_train[tr_idx], verbose=0, batch_size=BATCH_SIZE))

    print("val", model.evaluate(X_train[va_idx], y_train[va_idx], verbose=0, batch_size=BATCH_SIZE))

    oof_train[va_idx] = model.predict(X_train[va_idx], batch_size=BATCH_SIZE, verbose=0)

    y_preds += model.predict(X_test, batch_size=BATCH_SIZE, verbose=0) / NFOLD
fig, ax = plt.subplots(figsize=(12, 12))



idxs = np.random.randint(0, y_train.shape[0], 100)

ax.plot(y_train[idxs], label="ground truth", color=palette_ro[0])

ax.plot(oof_train[idxs, 0], label="q20", color=palette_ro[3], ls=':', alpha=0.5)

ax.plot(oof_train[idxs, 1], label="q50", color=palette_ro[4], ls=':', alpha=0.5)

ax.plot(oof_train[idxs, 2], label="q80", color=palette_ro[5], ls=':', alpha=0.5)

ax.legend(loc="best");
sigma_opt = mean_absolute_error(y_train, oof_train[:, 1])

sigma_unc = oof_train[:, 2] - oof_train[:, 0]

sigma_mean = np.mean(sigma_unc)

print(sigma_opt, sigma_mean)
print(sigma_unc.min(), sigma_unc.mean(), sigma_unc.max(), (sigma_unc>=0).mean())
print(np.mean(y_train / oof_train[:, 1]))
fig, ax = plt.subplots(figsize=(16, 6))



sns.distplot(sigma_unc, ax=ax, color=palette_ro[1])

ax.set_title("uncertainty in prediction", fontsize=18);
sub.head(10)
sub['FVC1'] = y_preds[:, 1]

sub['Confidence1'] = y_preds[:, 2] - y_preds[:, 0]



sub.head(10)
subm = sub[['Patient_Week', 'FVC', 'Confidence', 'FVC1', 'Confidence1']].copy()

subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']



subm.head(10)
subm.describe().T
org_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(org_test)):

    subm.loc[subm['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'FVC'] = org_test.FVC[i]

    subm.loc[subm['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'Confidence'] = 70



subm[["Patient_Week","FVC","Confidence"]].to_csv("submission_regression.csv", index=False)

reg_sub = subm[["Patient_Week","FVC","Confidence"]].copy()
df1 = img_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)

df2 = reg_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)



df = df1[['Patient_Week']].copy()

df['FVC'] = 0.2*df1['FVC'] + 0.8*df2['FVC']    # changed from 0.25, 0.75

df['Confidence'] = 0.0*df1['Confidence'] + 1.0*df2['Confidence']    # changed from 0.26, 0.74

df.head()
df.to_csv('submission.csv', index=False)