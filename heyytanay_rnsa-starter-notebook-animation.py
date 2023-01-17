! pip install -q dabl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dcm
import glob
import random
import warnings
from tqdm.notebook import tqdm
from colorama import Fore, Style
import os

import dabl

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import iplot

import matplotlib.animation as animation
from matplotlib.widgets import Slider
from IPython.display import HTML, Image

warnings.simplefilter("ignore")
def cout(string: str, color: str) -> str:
    """
    Prints a string in the required color
    """
    print(color+string+Style.RESET_ALL)
    
def read_image(filename: str) -> np.ndarray:
    """
    Read a DICOM Image File and return it (as a numpy array)
    """
    img = dcm.dcmread(filename).pixel_array
    img[img == -2000] = 0
    return img

def plot_dicom(image_list, rows=5, cols=4, cmap='jet', is_train=True):
    fig = plt.figure(figsize=(12, 12))
    if is_train:
        plt.title(f"DICOM Images from Training Set")
    else:
        plt.title(f"DICOM Images from Testing Set")
    img_count = 0
    for i in range(1, rows*cols+1):
        filename = image_list[img_count]
        image = read_image(filename)
        fig.add_subplot(rows, cols, i)
        plt.grid(False)
        plt.imshow(image, cmap=cmap)
        img_count += 1
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")
sub = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")
%%time
train_files = glob.glob("../input/rsna-str-pulmonary-embolism-detection/train/*/*/*.dcm")
test_files = glob.glob("../input/rsna-str-pulmonary-embolism-detection/test/*/*/*.dcm")
cout(f"Total Number of DICOM Images in Training Set: {len(train_files)}", Fore.GREEN)
cout(f"Total Number of DICOM Images in Testing Set:  {len(test_files)}", Fore.YELLOW)
train.head()   
test.head()
sub.head()
train.describe()
features = train[['qa_motion', 'qa_contrast', 'true_filling_defect_not_pe', 'flow_artifact']]
features.head()
vals = features['qa_motion'].value_counts().tolist()
idx = ['No Issue', 'Issue']
fig = px.pie(
    values=vals,
    names=idx,
    title='Issue with Motion in Studies',
    color_discrete_sequence=['blue', 'cyan']
)
iplot(fig)
vals = features['qa_contrast'].value_counts().tolist()
idx = ['No Issue', 'Issue']
fig = px.pie(
    values=vals,
    names=idx,
    title='Issue with Contrast in Studies',
    color_discrete_sequence=['gold', 'yellow']
)
iplot(fig)
vals = features['true_filling_defect_not_pe'].value_counts().tolist()
idx = ['Defect not PE', 'Defect is PE']
fig = px.pie(
    values=vals,
    names=idx,
    title='Is defect PE or Not',
    color_discrete_sequence=['black', 'gray']
)
iplot(fig)
vals = features['flow_artifact'].value_counts().tolist()
idx = ['Is an Artifact', 'Is not an Artifact']
fig = px.pie(
    values=vals,
    names=idx,
    title='Flow Artifact',
)
iplot(fig)
targets = train[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe', 'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']]
targets.head()
dabl.plot(targets, target_col='negative_exam_for_pe')
dabl.plot(targets, target_col='rv_lv_ratio_gte_1')
dabl.plot(targets, 'rightsided_pe')
# Plot any-20 training images
train_imgs = train_files[:20]
plot_dicom(image_list=train_imgs, is_train=True, cmap='gray')
# Plot any-20 testing images
test_imgs = test_files[:20]
plot_dicom(image_list=test_imgs, is_train=False, cmap='gray')
%%capture

ids = "../input/rsna-str-pulmonary-embolism-detection/train/6897fa9de148/2bfbb7fd2e8b/"

img_datas = []
for im in os.listdir(ids):
    meta = dcm.dcmread(os.path.join(ids, im))
    srl = meta.InstanceNumber
    data = meta.pixel_array
    data[data == -2000] = 0
    img_datas.append((srl, data))
    
img_datas.sort()
ims = []
fig = plt.figure()
for gg in img_datas:
    img_ = plt.imshow(gg[1], cmap='jet', animated=True)
    plt.axis("off")
    ims.append([img_])

ani = animation.ArtistAnimation(fig, ims, interval=1000//24, blit=False, repeat_delay=1000)
HTML(ani.to_jshtml())
