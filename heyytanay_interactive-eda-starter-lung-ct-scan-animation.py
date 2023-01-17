! pip -q install chart_studio
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import pydicom

from tqdm.notebook import tqdm

import glob

import random

import os



import matplotlib.animation as animation

from matplotlib.widgets import Slider

from IPython.display import HTML, Image



import plotly.express as px

import plotly.figure_factory as ff

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot



from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.ndimage

from skimage import morphology

from skimage import measure



from colorama import Fore, Back, Style



DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"

plt.style.use("fivethirtyeight")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

train_df.head()
print(Fore.BLUE+f"In Total there are {train_df['Patient'].count()} patients ids in train set and {test_df['Patient'].count()} patient ids in test set"+Style.RESET_ALL)

print(Fore.GREEN+f"Of which there are {train_df['Patient'].nunique()} unique patients in the train set"+Style.RESET_ALL)

print(Fore.YELLOW+f"And {test_df['Patient'].nunique()} unique patients in the test set"+Style.RESET_ALL)
print(Fore.GREEN+f"There are {train_df.any().isna().sum()} null values in the train set"+Style.RESET_ALL)

print(Fore.YELLOW+f"There are {test_df.any().isna().sum()} null values in the test set"+Style.RESET_ALL)
train_set = set(train_df['Patient'])

test_set = set(test_df['Patient'])

inter = train_set.intersection(test_set)



print(Fore.CYAN + f"There are {len(inter)} Same Samples between both datasets" + Style.RESET_ALL)
nb_train_imgs = glob.glob(os.path.join(DATA_DIR, "train/**/*.dcm"))

nb_test_imgs = glob.glob(os.path.join(DATA_DIR, "test/**/*.dcm"))



print(Fore.BLUE + f"There are {len(nb_train_imgs)+len(nb_test_imgs)} total DICOM files in this dataset."+Style.RESET_ALL)

print(Fore.GREEN+f"In training set, there are: {len(nb_train_imgs)} DICOM files"+Style.RESET_ALL)

print(Fore.YELLOW+f"In testing set, there are {len(nb_test_imgs)} DICOM files"+Style.RESET_ALL)
avg_train_imgs = len(nb_train_imgs) // train_df['Patient'].count()

avg_test_imgs = len(nb_test_imgs) // test_df['Patient'].count()



print(Fore.GREEN+f"In training set, an average patient has: {avg_train_imgs} images"+Style.RESET_ALL)

print(Fore.YELLOW+f"In testing set, an average patient has: {avg_test_imgs} images"+Style.RESET_ALL)
new_df = train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates()

new_df = new_df.sample(frac=1).reset_index(drop=True)

new_df.head()
vals = new_df['SmokingStatus'].value_counts().tolist()

idx = ['Ex-Smoker', 'Never Smoked', 'Currently Smokes']

fig = px.pie(

    values=vals,

    names=idx,

    title='Smoking Status of Patients',

    color_discrete_sequence=['cyan', 'blue', 'darkblue']

)

iplot(fig)
vals = new_df['Sex'].value_counts().tolist()

idx = new_df['Sex'].value_counts().keys().tolist()

fig = px.pie(

    values=vals,

    names=idx,

    title='Gender Distribution of Patients',

    color_discrete_sequence=['blue', 'magenta']

)

iplot(fig)
fig = px.histogram(

    new_df, x="Age",

    marginal="violin",

    hover_data=new_df.columns,

    color='Sex',

    color_discrete_sequence=['blue', 'magenta'],

    title=f"Unique Patients Age Distribution [\u03BC : ~{int(new_df.mean())} years | \u03C3 : ~{int(new_df.std())} years]",

)



iplot(fig)
fig = px.histogram(

    train_df, x="Weeks",

    marginal="box",

    hover_data=train_df.columns,

    color='Sex',

    title=f"Weeks Distribution [\u03BC : ~{int(train_df['Weeks'].mean())} weeks | \u03C3 : ~{int(train_df['Weeks'].std())} weeks]",

)



iplot(fig)
print(Fore.BLUE + f"Maximum Weeks for a Male Patient are: {train_df.loc[train_df['Sex']=='Male', 'Weeks'].max()}, average are: {int(train_df.loc[train_df['Sex']=='Male', 'Weeks'].mean())} and minimum are: {train_df.loc[train_df['Sex']=='Male', 'Weeks'].min()}" + Style.RESET_ALL)

print(Fore.MAGENTA + f"Maximum Weeks for a Female Patient are: {train_df.loc[train_df['Sex']=='Female', 'Weeks'].max()}, average are: {int(train_df.loc[train_df['Sex']=='Female', 'Weeks'].mean())} and minimum are: {train_df.loc[train_df['Sex']=='Female', 'Weeks'].min()}" + Style.RESET_ALL)
plt.figure(figsize=(16, 6))

sns.kdeplot(new_df.loc[new_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)

sns.kdeplot(new_df.loc[new_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)



# Labeling of plot

plt.xlabel('Age')

plt.ylabel('Density')

plt.title('Distribution of Ages for Male and Female Patients')

plt.show()
plt.figure(figsize=(16, 6))

sns.kdeplot(new_df.loc[new_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently Smokes',shade=True)

sns.kdeplot(new_df.loc[new_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never Smoked',shade=True)

sns.kdeplot(new_df.loc[new_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-Smoker',shade=True)



plt.xlabel('Age')

plt.ylabel('Density')

plt.title('Distribution of Ages vs SmokingStatus Category')

plt.show()
fig = px.histogram(

    train_df,

    x='FVC',

    marginal='violin',

    hover_data=train_df.columns,

    color_discrete_sequence=['maroon'],

    title=f"FVC Count Distribution [ \u03BC : {int(train_df['FVC'].mean())} ml. | \u03C3 : {int(train_df['FVC'].std())} ml. ]"

)

iplot(fig)
smoker = random.choice(train_df.query("SmokingStatus == 'Currently smokes'")['Patient'].unique())

non_smoker = random.choice(train_df.query("SmokingStatus == 'Never smoked'")['Patient'].unique())

exsmoker = random.choice(train_df.query("SmokingStatus == 'Ex-smoker'")['Patient'].unique())



fig = go.Figure()

fig.add_trace(go.Scatter(x=train_df[train_df.Patient==smoker]['Weeks'], y=train_df[train_df.Patient==smoker]['FVC'],

                    mode='lines+markers',

                    name='Current smoker'))

fig.add_trace(go.Scatter(x=train_df[train_df.Patient==non_smoker]['Weeks'], y=train_df[train_df.Patient==non_smoker]['FVC'],

                    mode='lines+markers',

                    name='Non-smoker'))

fig.add_trace(go.Scatter(x=train_df[train_df.Patient==exsmoker]['Weeks'], y=train_df[train_df.Patient==exsmoker]['FVC'],

                    mode='lines+markers', name='Ex-smoker'))



fig.update_layout(

    title="Patient Lung Capacity over weeks",

    xaxis_title="Weeks",

    yaxis_title="Lung Capacity (in ml)",

    legend_title="Smoker Status",

)



fig.show()
def plot_dicom(patient_id="ID00019637202178323708467", cmap='jet'):

    image_dir = os.path.join("../input/osic-pulmonary-fibrosis-progression/train/", patient_id)

    fig = plt.figure(figsize=(12, 12))

    cols = 4

    row = 5

    img_list = os.listdir(image_dir)

    plt.title(f"DICOM Images of Patient: {patient_id}")

    for i in range(1, row*cols+1):

        filename = os.path.join(image_dir, str(i)+".dcm")

        image = pydicom.dcmread(filename)

        fig.add_subplot(row, cols, i)

        plt.grid(False)

        plt.imshow(image.pixel_array, cmap=cmap)
plot_dicom()
%%capture

# Choose a patient id and then get all the dicom images of that patient

patient_id = "ID00012637202177665765362"

dicom_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"



# Just Sort all the dicom files based only on their first names and then put back all the sorted files with their .dcm extensions

files = np.array([f.replace(".dcm","") for f in os.listdir(f"{dicom_path}/{patient_id}/")])

files = np.sort(files.astype("int"))

dicoms = [f"{dicom_path}/{patient_id}/{f}.dcm" for f in files]



# Iterate through all the dicom images, read every image and reshape it

# Then save the plt.imshow(img) in img_ and append it to an array 

# At the end, pass it to the ArtistAnimation function with 0.1s interval and 1s delay to make an animation out of them

ims = []

fig = plt.figure()

for img in dicoms:

    img_ = pydicom.dcmread(img).pixel_array.reshape(512, 512)

    img_ = plt.imshow(img_, animated=True, cmap='gray')

    plt.axis("off")

    ims.append([img_])



ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=1000)
# Now just show the animations as an HTML component.

HTML(ani.to_jshtml())