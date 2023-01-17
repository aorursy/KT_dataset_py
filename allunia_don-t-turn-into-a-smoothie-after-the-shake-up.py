!pip install efficientnet_pytorch torchtoolbox
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pylab as pl

from IPython import display

import seaborn as sns

sns.set()



import re



import pydicom

import random



import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from efficientnet_pytorch import EfficientNet



from scipy.special import softmax



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import roc_auc_score, auc



from skimage.io import imread

from PIL import Image



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.subplots import make_subplots



import os

import copy



from albumentations import Compose, RandomCrop, Normalize,HorizontalFlip, Resize

from albumentations import VerticalFlip, RGBShift, RandomBrightness

from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.pytorch import ToTensor



from tqdm.notebook import tqdm



os.listdir("../input/")
basepath = "../input/siim-isic-melanoma-classification/"

modelspath = "../input/pytorch-pretrained-image-models/"

imagestatspath = "../input/siimisic-melanoma-classification-image-stats/"
os.listdir(basepath)
train_info = pd.read_csv(basepath + "train.csv")

train_info.head()
test_info = pd.read_csv(basepath + "test.csv")

test_info.head()
train_info.shape[0] / test_info.shape[0]
missing_vals_train = train_info.isnull().sum() / train_info.shape[0]

missing_vals_train[missing_vals_train > 0].sort_values(ascending=False)
missing_vals_test = test_info.isnull().sum() / test_info.shape[0]

missing_vals_test[missing_vals_test > 0].sort_values(ascending=False)
train_info.image_name.value_counts().max()
test_info.image_name.value_counts().max()
train_info.patient_id.value_counts().max()
test_info.patient_id.value_counts().max()
patient_counts_train = train_info.patient_id.value_counts()

patient_counts_test = test_info.patient_id.value_counts()



fig, ax = plt.subplots(2,2,figsize=(20,12))



sns.distplot(patient_counts_train, ax=ax[0,0], color="orangered", kde=True);

ax[0,0].set_xlabel("Counts")

ax[0,0].set_ylabel("Frequency")

ax[0,0].set_title("Patient id value counts in train");



sns.distplot(patient_counts_test, ax=ax[0,1], color="lightseagreen", kde=True);

ax[0,1].set_xlabel("Counts")

ax[0,1].set_ylabel("Frequency")

ax[0,1].set_title("Patient id value counts in test");



sns.boxplot(patient_counts_train, ax=ax[1,0], color="orangered");

ax[1,0].set_xlim(0, 250)

sns.boxplot(patient_counts_test, ax=ax[1,1], color="lightseagreen");

ax[1,1].set_xlim(0, 250);
np.quantile(patient_counts_train, 0.75) - np.quantile(patient_counts_train, 0.25)
np.quantile(patient_counts_train, 0.5)
print(np.quantile(patient_counts_train, 0.95))

print(np.quantile(patient_counts_test, 0.95))
200/test_info.shape[0] * 100
train_patient_ids = set(train_info.patient_id.unique())

test_patient_ids = set(test_info.patient_id.unique())



train_patient_ids.intersection(test_patient_ids)
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.countplot(train_info.sex, palette="Reds_r", ax=ax[0]);

ax[0].set_xlabel("")

ax[0].set_title("Gender counts");



sns.countplot(test_info.sex, palette="Blues_r", ax=ax[1]);

ax[1].set_xlabel("")

ax[1].set_title("Gender counts");
fig, ax = plt.subplots(1,2,figsize=(20,5))



sns.countplot(train_info.age_approx, color="orangered", ax=ax[0]);

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_xlabel("");

ax[0].set_title("Age distribution in train");



sns.countplot(test_info.age_approx, color="lightseagreen", ax=ax[1]);

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_xlabel("");

ax[1].set_title("Age distribution in test");
fig, ax = plt.subplots(1,2,figsize=(20,5))



image_locations_train = train_info.anatom_site_general_challenge.value_counts().sort_values(ascending=False)

image_locations_test = test_info.anatom_site_general_challenge.value_counts().sort_values(ascending=False)



sns.barplot(x=image_locations_train.index.values, y=image_locations_train.values, ax=ax[0], color="orangered");

ax[0].set_xlabel("");

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_title("Image locations in train");



sns.barplot(x=image_locations_test.index.values, y=image_locations_test.values, ax=ax[1], color="lightseagreen");

ax[1].set_xlabel("");

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_title("Image locations in test");
fig, ax = plt.subplots(1,2, figsize=(20,5))



sns.countplot(x=train_info.diagnosis, orient="v", ax=ax[0], color="Orangered")

ax[0].set_xlabel("")

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_title("Diagnosis");



sns.countplot(train_info.benign_malignant, ax=ax[1], palette="Reds_r");

ax[1].set_xlabel("")

ax[1].set_title("Type");
train_info.groupby("benign_malignant").target.nunique()
patient_ages_table_train = train_info.groupby(["patient_id", "age_approx"]).size() / train_info.groupby("patient_id").size()

patient_ages_table_train = patient_ages_table_train.unstack().transpose()

patient_ages_table_test = test_info.groupby(["patient_id", "age_approx"]).size() / test_info.groupby("patient_id").size()

patient_ages_table_test = patient_ages_table_test.unstack().transpose()



patient_with_known_ages_train = train_info[train_info.patient_id.isin(patient_ages_table_train.columns.values)]



sorted_patients_train = patient_with_known_ages_train.patient_id.value_counts().index.values

patient_with_known_ages_test = test_info[test_info.patient_id.isin(patient_ages_table_test.columns.values)]

sorted_patients_test = patient_with_known_ages_test.patient_id.value_counts().index.values



fig, ax = plt.subplots(2,1, figsize=(20,20))

sns.heatmap(patient_ages_table_train[sorted_patients_train], cmap="Reds", ax=ax[0], cbar=False);

ax[0].set_title("Image coverage in % per patient and age in train data");

sns.heatmap(patient_ages_table_test[sorted_patients_test], cmap="Blues", ax=ax[1], cbar=False);

ax[1].set_title("Image coverage in % per patient and age in test data");

ax[0].set_xlabel("")

ax[1].set_xlabel("");
fig, ax = plt.subplots(2,2,figsize=(20,15))



sns.boxplot(train_info.sex, train_info.age_approx, ax=ax[0,0], palette="Reds_r");

ax[0,0].set_title("Age per gender in train");



sns.boxplot(test_info.sex, test_info.age_approx, ax=ax[0,1], palette="Blues_r");

ax[0,1].set_title("Age per gender in test");



sns.countplot(train_info.age_approx, hue=train_info.sex, ax=ax[1,0], palette="Reds_r");

sns.countplot(test_info.age_approx, hue=test_info.sex, ax=ax[1,1], palette="Blues_r");
sex_and_cancer_map = train_info.groupby(

    ["benign_malignant", "sex"]

).size().unstack(level=0) / train_info.groupby("benign_malignant").size() * 100



cancer_sex_map = train_info.groupby(

    ["benign_malignant", "sex"]

).size().unstack(level=1) / train_info.groupby("sex").size() * 100





fig, ax = plt.subplots(1,3,figsize=(20,5))



sns.boxplot(train_info.benign_malignant, train_info.age_approx, ax=ax[0], palette="Greens");

ax[0].set_title("Age and cancer");

ax[0].set_xlabel("");



sns.heatmap(sex_and_cancer_map, annot=True, cmap="Greens", cbar=False, ax=ax[1])

ax[1].set_xlabel("")

ax[1].set_ylabel("");



sns.heatmap(cancer_sex_map, annot=True, cmap="Greens", cbar=False, ax=ax[2])

ax[2].set_xlabel("")

ax[2].set_ylabel("");
fig, ax = plt.subplots(2,2,figsize=(20,15))



sns.countplot(train_info[train_info.benign_malignant=="benign"].age_approx, hue=train_info.sex, palette="Purples_r", ax=ax[0,0])

ax[0,0].set_title("Benign cases in train");



sns.countplot(train_info[train_info.benign_malignant=="malignant"].age_approx, hue=train_info.sex, palette="Oranges_r", ax=ax[0,1])

ax[0,1].set_title("Malignant cases in train");



sns.violinplot(train_info.sex, train_info.age_approx, hue=train_info.benign_malignant, split=True, ax=ax[1,0], palette="Greens_r");

sns.violinplot(train_info.benign_malignant, train_info.age_approx, hue=train_info.sex, split=True, ax=ax[1,1], palette="RdPu");
patient_gender_train = train_info.groupby("patient_id").sex.unique().apply(lambda l: l[0])

patient_gender_test = test_info.groupby("patient_id").sex.unique().apply(lambda l: l[0])



train_patients = pd.DataFrame(index=patient_gender_train.index.values, data=patient_gender_train.values, columns=["sex"])

test_patients = pd.DataFrame(index=patient_gender_test.index.values, data=patient_gender_test.values, columns=["sex"])



train_patients.loc[:, "num_images"] = train_info.groupby("patient_id").size()

test_patients.loc[:, "num_images"] = test_info.groupby("patient_id").size()



train_patients.loc[:, "min_age"] = train_info.groupby("patient_id").age_approx.min()

train_patients.loc[:, "max_age"] = train_info.groupby("patient_id").age_approx.max()

test_patients.loc[:, "min_age"] = test_info.groupby("patient_id").age_approx.min()

test_patients.loc[:, "max_age"] = test_info.groupby("patient_id").age_approx.max()



train_patients.loc[:, "age_span"] = train_patients["max_age"] - train_patients["min_age"]

test_patients.loc[:, "age_span"] = test_patients["max_age"] - test_patients["min_age"]



train_patients.loc[:, "benign_cases"] = train_info.groupby(["patient_id", "benign_malignant"]).size().loc[:, "benign"]

train_patients.loc[:, "malignant_cases"] = train_info.groupby(["patient_id", "benign_malignant"]).size().loc[:, "malignant"]

train_patients["min_age_malignant"] = train_info.groupby(["patient_id", "benign_malignant"]).age_approx.min().loc[:, "malignant"]

train_patients["max_age_malignant"] = train_info.groupby(["patient_id", "benign_malignant"]).age_approx.max().loc[:, "malignant"]
train_patients.sort_values(by="malignant_cases", ascending=False).head()
fig, ax = plt.subplots(2,2,figsize=(20,12))

sns.countplot(train_patients.sex, ax=ax[0,0], palette="Reds")

ax[0,0].set_title("Gender counts with unique patient ids in train")

sns.countplot(test_patients.sex, ax=ax[0,1], palette="Blues");

ax[0,1].set_title("Gender counts with unique patient ids in test");



train_age_span_perc = train_patients.age_span.value_counts() / train_patients.shape[0] * 100

test_age_span_perc = test_patients.age_span.value_counts() / test_patients.shape[0] * 100



sns.barplot(train_age_span_perc.index, train_age_span_perc.values, ax=ax[1,0], color="Orangered");

sns.barplot(test_age_span_perc.index, test_age_span_perc.values, ax=ax[1,1], color="Lightseagreen");

ax[1,0].set_title("Patients age span in train")

ax[1,1].set_title("Patients age span in test")

for n in range(2):

    ax[1,n].set_ylabel("% in data")

    ax[1,n].set_xlabel("age span");
example_files = os.listdir(basepath + "train/")[0:2]

example_files
train_info.head(2)
train_info["dcm_path"] = basepath + "train/" + train_info.image_name + ".dcm"

test_info["dcm_path"] = basepath + "test/" + test_info.image_name + ".dcm"
print(train_info.dcm_path[0])

print(test_info.dcm_path[0])
example_dcm = pydicom.dcmread(train_info.dcm_path[2])

example_dcm
image = example_dcm.pixel_array

print(image.shape)
train_info["image_path"] = basepath + "jpeg/train/" + train_info.image_name + ".jpg"

test_info["image_path"] = basepath + "jpeg/test/" + test_info.image_name + ".jpg"
os.listdir(imagestatspath)
test_image_stats = pd.read_csv(imagestatspath +  "test_image_stats.csv")

test_image_stats.head(1)
train_image_stats_1 = pd.read_csv(imagestatspath + "train_image_stats_10000.csv")

train_image_stats_2 = pd.read_csv(imagestatspath + "train_image_stats_20000.csv")

train_image_stats_3 = pd.read_csv(imagestatspath + "train_image_stats_toend.csv")

train_image_stats_4 = train_image_stats_1.append(train_image_stats_2)

train_image_stats = train_image_stats_4.append(train_image_stats_3)

train_image_stats.shape
plot_test = True
if plot_test:

    N = test_image_stats.shape[0]

    selected_data = test_image_stats

    my_title = "Test image statistics"

else:

    N = train_image_stats.shape[0]

    selected_data = train_image_stats

    my_title = "Train image statistics"



trace1 = go.Scatter3d(

    x=selected_data.img_mean.values[0:N], 

    y=selected_data.img_std.values[0:N],

    z=selected_data.img_skew.values[0:N],

    mode='markers',

    text=selected_data["rows"].values[0:N],

    marker=dict(

        color=selected_data["columns"].values[0:N],

        colorscale = "Jet",

        colorbar=dict(thickness=10, title="image columns", len=0.8),

        opacity=0.4,

        size=2

    )

)



figure_data = [trace1]

layout = go.Layout(

    title = my_title,

    scene = dict(

        xaxis = dict(title="Image mean"),

        yaxis = dict(title="Image standard deviation"),

        zaxis = dict(title="Image skewness"),

    ),

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

    showlegend=True

)



fig = go.Figure(data=figure_data, layout=layout)

py.iplot(fig, filename='simple-3d-scatter')
test_image_stats.groupby(["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10] / test_image_stats.shape[0]
train_image_stats.groupby(["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10] / train_image_stats.shape[0]
examples1 = {"rows": 1080, "columns": 1920}

examples2 = {"rows": 4000, "columns": 6000}
selection1 = np.random.choice(test_image_stats[

    (test_image_stats["rows"]==examples1["rows"]) & (test_image_stats["columns"]==examples1["columns"])

].path.values, size=8, replace=False)



fig, ax = plt.subplots(2,4,figsize=(20,8))



for n in range(2):

    for m in range(4):

        path = selection1[m + n*4]

        dcm_file = pydicom.dcmread(path)

        image = dcm_file.pixel_array

        ax[n,m].imshow(image)

        ax[n,m].grid(False)
selection2 = np.random.choice(test_image_stats[

    (test_image_stats["rows"]==examples2["rows"]) & (test_image_stats["columns"]==examples2["columns"])

].path.values, size=8, replace=False)



fig, ax = plt.subplots(2,4,figsize=(20,6))



for n in range(2):

    for m in range(4):

        path = selection2[m + n*4]

        dcm_file = pydicom.dcmread(path)

        image = dcm_file.pixel_array

        ax[n,m].imshow(image)

        ax[n,m].grid(False)
class MelanomaDataset(Dataset):

    

    def __init__(self, df, transform=None):

        self.transform = transform

        self.df = df

    

    def __getitem__(self, idx):

        path = self.df.iloc[idx]["image_path"]

        image = Image.open(path)

        

        if self.transform:

            image = self.transform(image)

        

        if "target" in self.df.columns.values:

            target = self.df.iloc[idx]["target"]

            return {"image": image,

                    "target": target}

        else:

            return {"image": image}

    

    def __len__(self):

        return len(self.df)
class ResizedNpyMelanomaDataset(Dataset):

    

    def __init__(self, npy_file, indices_to_select, df=None, transform=None):

        self.transform = transform

        self.npy_file = npy_file

        self.df = df

        self.indices_to_select = indices_to_select

    

    def __getitem__(self, n):

        idx = self.indices_to_select[n]

        

        image = Image.fromarray(self.npy_file[idx])

        if self.transform:

            image = self.transform(image)

        

        target = self.df.loc[idx].target

        

        return {"image": image,

                "target": target}

    

    def __len__(self):

        return len(self.indices_to_select)
class AlbuMelanomaDataset(Dataset):

    

    def __init__(self, df, transform=None):

        self.transform = transform

        self.df = df

    

    def __getitem__(self, idx):

        path = self.df.iloc[idx]["image_path"]

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented['image']

        

        if "target" in self.df.columns.values:

            target = self.df.iloc[idx]["target"]

            return {"image": image,

                    "target": target}

        else:

            return {"image": image}

    

    def __len__(self):

        return len(self.df)
def random_microscope(img):

    circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8), # image placeholder

                        (img.shape[0]//2, img.shape[1]//2), # center point of circle

                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15), # radius

                        (0, 0, 0), # color

                        -1)



    mask = circle - 255

    img = np.multiply(img, mask)

    return img
import cv2



class Microscope:

    """

    Cutting out the edges around the center circle of the image

    Imitating a picture, taken through the microscope



    Args:

        p (float): probability of applying an augmentation

    """



    def __init__(self, p: float = 0.5):

        self.p = p

    

    def __call__(self, img):

        """

        Args:

            img (PIL Image): Image to apply transformation to.



        Returns:

            PIL Image: Image with transformation.

        """

        img = np.asarray(img)

        if random.random() < p:

            img = random_microscope(img)

        img = Image.fromarray(np.uint8(img))

        return img

    

    def __repr__(self):

        return f'{self.__class__.__name__}(p={self.p})'



    

class AlbuMicroscope(ImageOnlyTransform):

    

    def __init__(self, always_apply=False, p=0.5):

        super(AlbuMicroscope, self).__init__(always_apply, p)

    

    def apply(self, img, **params):

        return random_microscope(img)
def transform_fun(resize_shape, key="train", plot=False):

    train_sequence = [transforms.Resize((resize_shape, resize_shape)),

                      transforms.RandomHorizontalFlip(),

                      transforms.RandomVerticalFlip(),

                      Microscope(p=0.6)]

    dev_sequence = [transforms.Resize((resize_shape, resize_shape))]

    if plot==False:

        train_sequence.extend([

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        dev_sequence.extend([

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        

    data_transforms = {'train': transforms.Compose(train_sequence),

                       'dev': transforms.Compose(dev_sequence),

                       'test_tta': transforms.Compose(train_sequence),

                       'test': transforms.Compose(dev_sequence)}

    return data_transforms[key]
def albu_transform_fun(resize_shape=None, key="train", plot=False):

    train_sequence = [

        Resize(resize_shape, resize_shape),

        RandomCrop(224,224),

        VerticalFlip(),

        HorizontalFlip(),

        RGBShift(r_shift_limit=40),

        RandomBrightness(0.1),

        AlbuMicroscope(p=0.6)]

    dev_sequence = [Resize(224, 224)]

    

    if plot==False:

        train_sequence.extend([

            Normalize(mean=[0.485, 0.456, 0.406],

                      std=[0.229, 0.224, 0.225],),

            ToTensor()])

        dev_sequence.extend([Normalize(mean=[0.485, 0.456, 0.406],

                                       std=[0.229, 0.224, 0.225],),

                             ToTensor()])

    

    data_transforms = {'train': Compose(train_sequence),

                       'dev': Compose(dev_sequence),

                       'test_tta': Compose(train_sequence),

                       'test': Compose(dev_sequence)}

    return data_transforms[key]
N = 10



fig, ax = plt.subplots(2,N,figsize=(20,5))



selection = np.random.choice(train_info.index.values, size=N, replace=False)



for n in range(N):

    

    org_image = cv2.imread(train_info.loc[selection[n]].image_path)

    org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

    label = train_info.loc[selection[n]].target

    augmented = albu_transform_fun(resize_shape=256, key="train", plot=True)(**{"image":org_image, "label": label})

    ax[0,n].imshow(org_image)

    ax[1,n].imshow(augmented["image"])

    ax[1,n].axis("off")

    ax[0,n].axis("off")

    ax[0,n].set_title("Original")

    ax[1,n].set_title("Augmented");
def get_ce_loss():   

    criterion = torch.nn.CrossEntropyLoss()

    return criterion
def get_wce_loss(train_targets):

    weights = compute_class_weight(y=train_targets,

                                   class_weight="balanced",

                                   classes=np.unique(train_targets))    

    class_weights = torch.FloatTensor(weights)

    if device.type=="cuda":

        class_weights = class_weights.cuda()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    return criterion
class MulticlassFocalLoss(torch.nn.Module):

    

    def __init__(self, train_targets=None, gamma=2):

        super(MulticlassFocalLoss, self).__init__()

        self.gamma = gamma

        if train_targets is None:

            self.class_weights = None

        else:

            weights = compute_class_weight(y=train_targets,

                                   class_weight="balanced",

                                   classes=np.unique(train_targets))    

            self.class_weights = torch.FloatTensor(weights)

            if device.type=="cuda":

                self.class_weights = self.class_weights.cuda()

    

    def forward(self, input, target):

        if self.class_weights is None:

            ce_loss = F.cross_entropy(input, target, reduction='none')

        else:

            ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.class_weights)

        pt = torch.exp(-ce_loss)

        loss = (1-pt)**self.gamma * ce_loss

        return torch.mean(loss)
os.listdir(modelspath)
def get_model(kind="resnet34"):

    if kind == "resnet34":

        model = models.resnet34(pretrained=False)

        model.load_state_dict(torch.load(modelspath + "resnet34.pth"))

    elif kind == "resnet50":

        model = models.resnet50(pretrained=False)

        model.load_state_dict(torch.load(modelspath + "resnet50.pth"))

    elif kind == "densenet121":

        model = models.densenet121(pretrained=False)

        model.load_state_dict(torch.load(modelspath + "densenet121.pth"))

    elif kind == "densenet201":

        model = models.densenet201(pretrained=False)

        model.load_state_dict(torch.load(modelspath + "densenet201.pth"))

    elif kind == "efficientnet_b1":

        model = EfficientNet.from_pretrained('efficientnet-b1')

    else:

        model = models.resnet34(pretrained=False)

        model.load_state_dict(torch.load(modelspath + "resnet34.pth"))

    return model        
def init_weights(m):

    if type(m) == torch.nn.Linear:

        torch.nn.init.xavier_uniform_(m.weight)

        m.bias.data.fill_(0.01)



def build_model(the_model):

    model = get_model(the_model)

    

    if "efficientnet" in the_model:

        num_features = model._fc.in_features

    else:

        num_features = model.fc.in_features

    

    basic_modules = torch.nn.Sequential(torch.nn.Linear(num_features, 128),

                                        torch.nn.ReLU(),

                                        torch.nn.BatchNorm1d(128),

                                        torch.nn.Dropout(0.2),



                                        torch.nn.Linear(128, num_classes))

    

    if "efficientnet" in the_model:

        model._fc = basic_modules

    else:

        model.fc = basic_modules

        

    

    return model
# make sure that counter*batch_size is the same as len(dataset)

def predict(fold_results, dataloader, TTA=1):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    avg_preds = np.zeros(len(dataloader.dataset))

    avg_probas = np.zeros((len(dataloader.dataset),2))

        

    for fold_num in fold_results.keys():

        

        results = fold_results[fold_num]

        model = results.model

        

        dataloader_iterator = tqdm(dataloader, total=int(len(dataloader)))

        

        for t in range(TTA):

            print("TTA phase {}".format(t))

            for counter, data in enumerate(dataloader_iterator):    

                image_input = data["image"]

                image_input = image_input.to(device, dtype=torch.float)



                pred_probas = model(image_input)

                _, preds = torch.max(pred_probas, 1)



                avg_preds[

                    (counter*dataloader.batch_size):(dataloader.batch_size*(counter+1))

                ] += preds.cpu().detach().numpy()/(len(fold_results)*TTA)

                avg_probas[

                    (counter*dataloader.batch_size):(dataloader.batch_size*(counter+1))

                ] += softmax(pred_probas.cpu().detach().numpy(), axis=1)/(len(fold_results)*TTA)

        

    return avg_preds, avg_probas
def get_scheduler(optimiser, min_lr, max_lr, stepsize):

    # suggested_stepsize = 2*num_iterations_within_epoch

    stepsize_up = np.int(stepsize/2)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimiser,

                                               base_lr=min_lr,

                                               max_lr=max_lr,

                                               step_size_up=stepsize_up,

                                               step_size_down=stepsize_up,

                                               mode="triangular")

    return scheduler

    
def get_lr_search_scheduler(optimiser, min_lr, max_lr, max_iterations):

    # max_iterations should be the number of steps within num_epochs_*epoch_iterations

    # this way the learning rate increases linearily within the period num_epochs*epoch_iterations 

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimiser, 

                                               base_lr=min_lr,

                                               max_lr=max_lr,

                                               step_size_up=max_iterations,

                                               step_size_down=max_iterations,

                                               mode="triangular")

    

    return scheduler
from scipy.special import expit



def run_training(criterion,

                 num_epochs,

                 dataloaders_dict,

                 fold_num,

                 patience,

                 results,

                 find_lr):

    

    if find_lr:

        phases = ["train"]

    else:

        phases = ["train", "dev"]

        

    best_auc = 0

    patience_counter = 0

    epsilon = 1e-7

    

    for epoch in range(num_epochs):

        

        for phase in phases:

            

            dataloader = dataloaders_dict[phase]

            dataloader_iterator = tqdm(dataloader, total=int(len(dataloader)))

            

            if phase=="train":

                results.model.train()

            else:

                results.model.eval()

                

            all_preds = np.zeros(len(dataloader)*dataloader.batch_size)

            all_targets = np.zeros(len(dataloader)*dataloader.batch_size)   

            running_loss = 0.0

            running_true_positives = 0

            running_false_positives = 0

            running_false_negatives = 0

            

                      

            for counter, data in enumerate(dataloader_iterator):

                image_input = data["image"]

                target_input = data["target"]

                

                image_input = image_input.to(device, dtype=torch.float)

                target_input = target_input.to(device, dtype=torch.long)

    

                results.optimiser.zero_grad()

                

                raw_output = results.model(image_input) 

                

                _, preds = torch.max(raw_output,1)

                

                running_true_positives += (preds*target_input).sum().cpu().detach().numpy()

                running_false_positives += ((1-target_input)*preds).sum().cpu().detach().numpy()

                running_false_negatives += (target_input*(1-preds)).sum().cpu().detach().numpy()



                precision = running_true_positives / (running_true_positives + running_false_positives + epsilon)

                recall = running_true_positives / (running_true_positives + running_false_negatives + epsilon)

                

                f1_score = 2*precision*recall / (precision+recall+epsilon) 

                

                

                results.results[phase].learning_rates.append(optimiser.state_dict()["param_groups"][0]["lr"])

                results.results[phase].precision.append(precision)

                results.results[phase].recall.append(recall)

                results.results[phase].f1_scores.append(f1_score)

                        

                batch_size = dataloader.batch_size

                all_targets[(counter*batch_size):((counter+1)*batch_size)] = target_input.cpu().detach().numpy()

                all_preds[(counter*batch_size):((counter+1)*batch_size)] = preds.cpu().detach().numpy()

                

                loss = criterion(raw_output, target_input)

                # redo the average over mini_batch

                running_loss += (loss.item() * batch_size)

    

                # save averaged loss over processed number of batches:

                processed_loss = running_loss / ((counter+1) * batch_size)

                results.results[phase].losses.append(processed_loss)

                

                if phase == 'train':

                    loss.backward()

                    results.optimiser.step()

                    if results.scheduler is not None:

                        results.scheduler.step()

                        

            epoch_auc_score = roc_auc_score(all_targets, all_preds)

            results.results[phase].epoch_scores.append(epoch_auc_score)

                

            

            # average over all samples to obtain the epoch loss

            epoch_loss = running_loss / len(dataloader.dataset)

            results.results[phase].epoch_losses.append(epoch_loss)

            

            print("fold: {}, epoch: {}, phase: {}, e-loss: {}, e-auc: {}".format(

                fold_num, epoch, phase, epoch_loss, epoch_auc_score))

            

            if not find_lr:

                if phase == "dev":

                    if epoch_auc_score >= best_auc:

                        best_auc = epoch_auc_score

                        best_model_wts = copy.deepcopy(results.model.state_dict())

                        best_model_optimiser = copy.deepcopy(results.optimiser.state_dict())

                        best_scheduler = copy.deepcopy(results.scheduler.state_dict())

                        best_epoch = epoch

                        best_loss = processed_loss

                    else:

                        patience_counter += 1

                        if patience_counter == patience:

                            print("Model hasn't improved for {} epochs. Training finished.".format(patience))

                            break

               

    # load best model weights

    if not find_lr:

        results.model.load_state_dict(best_model_wts)

        results.optimiser.load_state_dict(best_model_optimiser)

        results.scheduler.load_state_dict(best_scheduler)

        results.best_epoch = best_epoch

        results.best_loss = best_loss

    return results
class ResultsBean:

    

    def __init__(self):

        

        self.precision = []

        self.recall = []

        self.f1_scores = []

        self.losses = []

        self.learning_rates = []

        self.epoch_losses = []

        self.epoch_scores = []



class Results:

    

    def __init__(self, fold_num, model=None, optimiser=None, scheduler=None, model_kind="resnet34"):

        self.model = model

        self.model_kind = model_kind

        self.optimiser = optimiser

        self.scheduler = scheduler

        self.best_epoch = 0

        self.best_loss = 0

        

        self.fold_num = fold_num

        self.train_results = ResultsBean()

        self.dev_results = ResultsBean()

        self.results = {"train": self.train_results,

                        "dev": self.dev_results}
def train(model,

          model_kind,

          criterion,

          optimiser,

          num_epochs,

          dataloaders_dict,

          fold_num,

          scheduler,

          patience,

          find_lr=False):

    

    single_results = Results(fold_num=fold_num,

                             model=model,

                             optimiser=optimiser,

                             scheduler=scheduler,

                             model_kind=model_kind)

    

    

    single_results = run_training(criterion,

                                  num_epochs,

                                  dataloaders_dict,

                                  fold_num,

                                  patience,

                                  single_results, 

                                  find_lr=find_lr)

       

    return single_results
def save_as_csv(series, name, path):

    df = pd.DataFrame(index=np.arange(len(series)), data=series, columns=[name])

    output_path = path + name + ".csv"

    df.to_csv(output_path, index=False)



def save_results(results, foldername):

    for fold in results.keys():

        

        base_dir = foldername + "/fold_" + str(fold) + "/"

        if not os.path.exists(base_dir):

            os.makedirs(base_dir)

        

        # save the model for inference

        model = results[fold].model

        model_kind = results[fold].model_kind

        #model_path = base_dir + model_kind + ".pth"

        #torch.save(model.state_dict(), model_path)

        

        # save checkpoint for inference and retraining:

        checkpoint_path = base_dir + model_kind + ".tar"

        torch.save({

            'epoch': results[fold].best_epoch,

            'loss': results[fold].best_loss,

            'model_state_dict': results[fold].model.state_dict(),

            'optimizer_state_dict': results[fold].optimiser.state_dict(),

            'scheduler_state_dict': results[fold].scheduler.state_dict()}, checkpoint_path)

        

        for phase in ["train", "dev"]:

            losses = results[fold].results[phase].losses

            epoch_losses = results[fold].results[phase].epoch_losses

            epoch_scores = results[fold].results[phase].epoch_scores

            lr_rates = results[fold].results[phase].learning_rates

            f1_scores = results[fold].results[phase].f1_scores

            precision = results[fold].results[phase].precision

            recall = results[fold].results[phase].recall

            

            save_as_csv(losses, phase + "_losses", base_dir)

            save_as_csv(epoch_losses, phase + "_epoch_losses", base_dir)

            save_as_csv(epoch_scores, phase + "_epoch_scores", base_dir)

            save_as_csv(lr_rates, phase + "_lr_rates", base_dir)

            save_as_csv(f1_scores, phase + "_f1_scores", base_dir)

            save_as_csv(precision, phase + "_precision", base_dir)

            save_as_csv(recall, phase + "_recall", base_dir)
def load_checkpoint(model_kind,

                    checkpoint_path,

                    for_inference,

                    single_results,

                    lr,

                    num_epochs,

                    min_lr, max_lr, len_train):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    checkpoint = torch.load(checkpoint_path)

    

    single_results.model = build_model(model_kind)

    single_results.model.load_state_dict(checkpoint["model_state_dict"])

    single_results.model.to(device)

    

    if "efficientnet" in model_kind:

        single_results.optimiser = torch.optim.SGD(single_results.model._fc.parameters(), lr=lr)

    else:

        single_results.optimiser = torch.optim.SGD(single_results.model.fc.parameters(), lr=lr)

    single_results.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

    

    max_iterations = num_epochs * len_train

    single_results.scheduler = get_lr_search_scheduler(single_results.optimiser, min_lr, max_lr, max_iterations)

    single_results.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    

    single_results.best_epoch = checkpoint["epoch"]

    single_results.best_loss = checkpoint["loss"]

    

    # set into inference state

    if for_inference:

        single_results.model.eval()

    else:

        single_results.model.train()

    

    return single_results
def load_results(save_folder, total_folds, model_kind, lr, num_epochs, len_train, min_lr, max_lr, for_inference=True):

    results = {}

    

    for fold in range(total_folds): 

        single_results = Results(fold)

        

        base_path = save_folder + "/fold_" + str(fold) + "/"

        checkpoint_path = base_path + model_kind + ".tar"

        single_results = load_checkpoint(model_kind,

                                         checkpoint_path,

                                         for_inference,

                                         single_results,

                                         lr,

                                         num_epochs,

                                         min_lr, max_lr, len_train)

        

        for phase in ["train", "dev"]:

            single_results.results[phase].losses = pd.read_csv(base_path + phase + "_losses.csv")

            single_results.results[phase].epoch_losses = pd.read_csv(base_path + phase + "_epoch_losses.csv")

            single_results.results[phase].epoch_scores = pd.read_csv(base_path + phase + "_epoch_scores.csv")

            single_results.results[phase].learning_rates = pd.read_csv(base_path + phase + "_lr_rates.csv")

            single_results.results[phase].f1_scores = pd.read_csv(base_path + phase + "_f1_scores.csv")

            single_results.results[phase].precision = pd.read_csv(base_path + phase + "_precision.csv")

            single_results.results[phase].recall = pd.read_csv(base_path + phase + "_recall.csv")

        

        results[fold] = single_results

    return results
train_image_stats.head(1)
train_image_stats.groupby(

    ["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10] / train_image_stats.shape[0]
relevant_groups = train_image_stats.groupby(

    ["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10].index.values
group_stats = pd.DataFrame(train_image_stats.groupby(["rows", "columns"]).img_mean.median().loc[relevant_groups])

group_stats["img_std"] = train_image_stats.groupby(["rows", "columns"]).img_std.median().loc[relevant_groups]
plt.figure(figsize=(20,5))

plt.scatter(group_stats.img_mean, group_stats.img_std, label="remaining train groups");

plt.scatter(group_stats.loc[(480, 640)].img_mean, group_stats.loc[(480, 640)].img_std,

            c="lime", label="candidate 480, 640")

plt.scatter(group_stats.loc[(3456, 5184)].img_mean, group_stats.loc[(3456, 5184)].img_std,

            c="deeppink", label="candidate 3456, 5184");

plt.title("Image statistics groups");

plt.legend()

plt.xlabel("Median of group image means")

plt.ylabel("Std of group image means");
selected_hold_out_group = train_image_stats.loc[

    (train_image_stats["rows"]==3456) & (train_image_stats["columns"]==5184)

].path.values



hold_out_df = train_info.loc[train_info.dcm_path.isin(selected_hold_out_group)].copy()

reduced_train_df = train_info.loc[train_info.dcm_path.isin(selected_hold_out_group)==False].copy()
hold_out_df.shape[0] / reduced_train_df.shape[0]
test_info.shape[0] / train_info.shape[0]
reduced_train_df, add_to_hold_out_df = train_test_split(

    reduced_train_df, test_size=0.163, stratify=reduced_train_df.target.values)
hold_out_df = hold_out_df.append(add_to_hold_out_df)

print(hold_out_df.shape[0] / reduced_train_df.shape[0])

print(hold_out_df.shape[0], reduced_train_df.shape[0])
fig, ax = plt.subplots(1,2,figsize=(20,5))



h_target_perc = hold_out_df.target.value_counts() / hold_out_df.shape[0] * 100

rt_target_perc = reduced_train_df.target.value_counts() / reduced_train_df.shape[0] * 100 



sns.barplot(h_target_perc.index, h_target_perc.values, ax=ax[0], palette="Oranges_r")

sns.barplot(rt_target_perc.index, rt_target_perc.values, ax=ax[1], palette="Purples_r");



ax[0].set_title("Target distribution of \n hold-out");

ax[1].set_title("Target distribution of \n reduced train");

for n in range(2):

    ax[n].set_ylabel("% in data")

    ax[n].set_xlabel("Target")
h_target_perc
rt_target_perc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
torch.manual_seed(0)

np.random.seed(0)
num_classes = 2



my_model = "resnet50"

TRAIN_BATCH_SIZE = 64

LR = 0.01
os.listdir("../input/siimisic-melanoma-resized-images")
RESIZE_SHAPE = 256
#train_npy = np.load("../input/siimisic-melanoma-resized-images/x_train_" + str(RESIZE_SHAPE) + ".npy", mmap_mode="r")

#train_npy.shape
#x_test = np.load("../input/siimisic-melanoma-resized-images/x_test_" + str(RESIZE_SHAPE) + ".npy", mmap_mode="r")

#x_test.shape
hold_out_indices = hold_out_df.index.values

reduced_train_indices = reduced_train_df.index.values
#hold_out_dataset_1 = ResizedNpyMelanomaDataset(train_npy, hold_out_indices, df=hold_out_df,

#                                              transform=transform_fun(RESIZE_SHAPE, key="dev", plot=True))

#hold_out_dataset_2 = MelanomaDataset(hold_out_df, transform=transform_fun(RESIZE_SHAPE, key="dev", plot=True))



#idx = 10



#hold_out_example_1 = hold_out_dataset_1.__getitem__(idx)

#hold_out_example_2 = hold_out_dataset_2.__getitem__(idx)



#fig, ax = plt.subplots(1,4,figsize=(20,5))

#ax[0].imshow(hold_out_example_1["image"])

#ax[0].axis("off")

#ax[0].set_title(hold_out_example_1["target"]);

#sns.distplot(hold_out_example_1["image"], ax=ax[1])

#ax[2].imshow(hold_out_example_2["image"])

#ax[2].axis("off")

#ax[2].set_title(hold_out_example_2["target"]);

#sns.distplot(hold_out_example_1["image"], ax=ax[3])
train_indices = train_info.index.values
find_lr = True

min_lr = 0.001

max_lr = 1

NUM_EPOCHS = 3

save_folder = "learning_rate_search"

load_folder = "../input/melanomaclassificationsmoothiestarter/learning_rate_search"
external_data_path = "../input/melanoma-external-malignant-256/"

external_train = pd.read_csv(external_data_path + "/train_concat.csv")

external_train["image_path"] = external_data_path + "train/train/" + external_train.image_name + ".jpg"
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.countplot(train_info.target, ax=ax[0], palette="Reds_r")

sns.countplot(external_train.target, ax=ax[1], palette="Reds_r")

ax[1].set_title("Target imbalance in external train")

ax[0].set_title("Target imbalance in original train");
if find_lr:

    

    results = {}

    

    #train_idx, dev_idx = train_test_split(train_indices,

                                          #stratify=train_info.target.values,

                                          #test_size=0.3,

                                          #random_state=0)

    

    #train_dataset = ResizedNpyMelanomaDataset(npy_file=train_npy,

    #                                          indices_to_select=train_idx, 

    #                                          df=train_info,

    #                                          transform=transform_fun(RESIZE_SHAPE, "train"))

    #dev_dataset = ResizedNpyMelanomaDataset(npy_file=train_npy,

    #                                        indices_to_select=dev_idx, 

    #                                        df=train_info,

    #                                        transform=transform_fun(RESIZE_SHAPE, "dev"))

    

    

    train_df, dev_df = train_test_split(external_train,

                                        stratify=external_train.target.values,

                                        test_size=0.3,

                                        random_state=0)

    

    train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))

    dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))

    

    

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)

    dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}

    

    model = build_model(my_model)

    model.apply(init_weights)

    model = model.to(device)

    

    # if you are using the external data

    criterion = MulticlassFocalLoss(gamma=2)

    # if you are using the resized data:

    #criterion = MulticlassFocalLoss(train_info.iloc[train_idx].target.values)

    #criterion = get_wce_loss(train_df.target.values)

    

    if "efficientnet" in my_model:

        optimiser = torch.optim.SGD(model._fc.parameters(), lr=LR)

    else:

        optimiser = torch.optim.SGD(model.fc.parameters(), lr=LR)

    

    max_iterations = NUM_EPOCHS * len(train_dataloader)

    scheduler = get_lr_search_scheduler(optimiser, min_lr, max_lr, max_iterations)

    

    single_results = train(model=model,

                           model_kind=my_model,

                           criterion=criterion,

                           optimiser=optimiser,

                           num_epochs=NUM_EPOCHS,

                           dataloaders_dict=dataloaders_dict,

                           fold_num=0,

                           scheduler=scheduler, 

                           patience=1,

                           find_lr=find_lr)

    

    results = {0: single_results}

    save_results(results, save_folder)

    

# prepare for retraining and/or inference:

else:

    train_df, dev_df = train_test_split(external_train,

                                        stratify=external_train.target.values,

                                        test_size=0.3,

                                        random_state=0)

    

    train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))

    dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))

    

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)

    dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}

    

    criterion = MulticlassFocalLoss(gamma=2)

    

    results = load_results(load_folder,

                           total_folds=1,

                           model_kind=my_model,

                           lr=LR,

                           num_epochs=NUM_EPOCHS,

                           len_train=len(train_dataloader),

                           min_lr=min_lr,

                           max_lr=max_lr,

                           for_inference=True)
fig, ax = plt.subplots(3,2,figsize=(20,18))



rates = results[0].results["train"].learning_rates

f1_score = results[0].results["train"].f1_scores

precision = results[0].results["train"].precision

recall = results[0].results["train"].recall

losses = results[0].results["train"].losses

epoch_losses = results[0].results["train"].epoch_losses



ax[0,0].plot(rates, f1_score, '.-', c="maroon", label="f1-score")

ax[0,0].plot(rates, precision, '.-', c="salmon", label="precision")

ax[0,0].plot(rates, recall, '.-', c="lightsalmon", label="recall")





ax[0,0].legend();

ax[0,0].set_xlabel("Learning rate")

ax[0,0].set_ylabel("Score values")

ax[0,0].set_title("Evaluation scores for learning rate search within {} epochs".format(NUM_EPOCHS));



ax[1,1].plot(rates, precision, '.-', c="salmon", label="precision")

ax[1,1].set_title("Precision")

ax[1,1].set_xlabel("learning rates")

ax[1,1].set_ylabel("precision")



ax[1,0].plot(rates, recall, '.-', c="lightsalmon", label="recall")

ax[1,0].set_title("Recall")

ax[1,0].set_xlabel("learning rates")

ax[1,0].set_ylabel("recall")



ax[0,1].plot(rates, f1_score, '.-', c="maroon", label="f1-score")

ax[0,1].set_title("F1-score")

ax[0,1].set_xlabel("learning rates")

ax[0,1].set_ylabel("f1-score")



ax[2,0].plot(rates, losses, 'o-', c="deepskyblue")

ax[2,0].set_title("Loss change with rates")

ax[2,0].set_ylabel("loss")

ax[2,0].set_xlabel("Learning rates")



ax[2,1].set_title("Learning rate increase")

ax[2,1].plot(rates, 'o', c="mediumseagreen");

ax[2,1].set_ylabel("learning rate")

ax[2,1].set_xlabel("Iteration step");
check_workflow = False

save_folder = "check_workflow"

load_folder = "../input/melanomaclassificationsmoothiestarter/check_workflow"

NUM_EPOCHS = 10

LR = 0.01

min_lr = 0.0001

max_lr = 0.25

find_lr=False
if check_workflow:

    

    results = {}

    

    train_df, dev_df = train_test_split(external_train,

                                        stratify=external_train.target.values,

                                        test_size=0.3,

                                        random_state=0)

    

    train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))

    dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))

    

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)

    dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}

    

    model = build_model(my_model)

    model.apply(init_weights)

    model = model.to(device)

    

    criterion = MulticlassFocalLoss(gamma=2)

    #criterion = get_wce_loss(train_df.target.values)

    if "efficientnet" in my_model:

        optimiser = torch.optim.SGD(model._fc.parameters(), lr=LR)

    else:

        optimiser = torch.optim.SGD(model.fc.parameters(), lr=LR)

    

    stepsize = 2*len(train_dataloader)

    scheduler = get_scheduler(optimiser, min_lr, max_lr, stepsize)

    

    single_results = train(model=model,

                           model_kind=my_model,

                           criterion=criterion,

                           optimiser=optimiser,

                           num_epochs=NUM_EPOCHS,

                           dataloaders_dict=dataloaders_dict,

                           fold_num=0,

                           scheduler=scheduler, 

                           patience=1,

                           find_lr=find_lr)

    

    results = {0: single_results}

    save_results(results, save_folder)



else:

    

    train_df, dev_df = train_test_split(external_train,

                                        stratify=external_train.target.values,

                                        test_size=0.3,

                                        random_state=0)

    

    train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))

    dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))

    

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)

    dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}

    

    criterion = MulticlassFocalLoss(gamma=2)

    

    results = load_results(load_folder,

                           total_folds=1,

                           model_kind=my_model,

                           lr=LR,

                           num_epochs=NUM_EPOCHS,

                           len_train=len(train_dataloader),

                           min_lr=min_lr,

                           max_lr=max_lr,

                           for_inference=True)
save_results(results, save_folder)
fig, ax = plt.subplots(3,2,figsize=(20,15))



rates = results[0].results["train"].learning_rates

f1_score = results[0].results["train"].f1_scores

precision = results[0].results["train"].precision

recall = results[0].results["train"].recall



train_losses = results[0].results["train"].losses

dev_losses = results[0].results["dev"].losses



train_epoch_losses = results[0].results["train"].epoch_losses

dev_epoch_losses = results[0].results["dev"].epoch_losses

train_epoch_auc = results[0].results["train"].epoch_scores

dev_epoch_auc = results[0].results["dev"].epoch_scores



ax[0,0].plot(f1_score, '.-', c="maroon", label="f1-score")

ax[0,0].plot(precision, '.-', c="salmon", label="precision")

ax[0,0].plot(recall, '.-', c="lightsalmon", label="recall")



ax[0,0].legend();

ax[0,0].set_xlabel("Learning rate")

ax[0,0].set_ylabel("Score values")

ax[0,0].set_title("Evaluation scores for learning rate search within {} epochs".format(NUM_EPOCHS));



ax[0,1].plot(rates)

ax[0,1].set_title("Learning rates")



ax[1,0].plot(train_losses, label="train")



ax[1,1].plot(dev_losses, label="dev");

ax[1,1].legend()

ax[1,1].set_title("Losses")



ax[2,0].plot(train_epoch_losses, label="train")

ax[2,0].plot(dev_epoch_losses, label="dev")

ax[2,0].set_title("Epoch losses")



ax[2,1].plot(train_epoch_auc)

ax[2,1].plot(dev_epoch_auc)

ax[2,1].set_title("Epoch AUC");
run_kfold = False

n_splits = 3

save_folder = "kfold_workflow"

load_folder = "../input/melanomaclassificationsmoothiestarter/kfold_workflow"

NUM_EPOCHS = 10

LR = 0.01

min_lr = 0.0001

max_lr = 0.25

find_lr=False
skf = StratifiedKFold(n_splits=5, random_state=0)
if run_kfold:

    

    results = {}

    

    n_fold = 0

    for train_idx, dev_idx in skf.split(external_train, external_train.target.values):

        train_df = external_train.iloc[train_idx]

        dev_df = external_train.iloc[dev_idx]

        

    

        train_dataset = AlbuMelanomaDataset(train_df, albu_transform_fun(RESIZE_SHAPE, key="train"))

        dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))

    

        train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

        dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)

        dataloaders_dict = {"train": train_dataloader, "dev": dev_dataloader}



        model = build_model(my_model)

        model.apply(init_weights)

        model = model.to(device)

        

        criterion = MulticlassFocalLoss(gamma=2)

        #criterion = get_wce_loss(train_df.target.values)

        if "efficientnet" in my_model:

            optimiser = torch.optim.SGD(model._fc.parameters(), lr=LR)

        else:

            optimiser = torch.optim.SGD(model.fc.parameters(), lr=LR)

    

        stepsize = 2*len(train_dataloader)

        scheduler = get_scheduler(optimiser, min_lr, max_lr, stepsize)

    

        single_results = train(model=model,

                               model_kind=my_model,

                               criterion=criterion,

                               optimiser=optimiser,

                               num_epochs=NUM_EPOCHS,

                               dataloaders_dict=dataloaders_dict,

                               fold_num=0,

                               scheduler=scheduler, 

                               patience=1,

                               find_lr=find_lr)

    

        results = {n_fold: single_results}

        n_fold += 1

    

    save_results(results, save_folder)
max_size = 120



for m in range(max_size+1):

    to_try = max_size - m

    if dev_df.shape[0] % to_try == 0:

        break

        

DEV_BATCH_SIZE = to_try

to_try
from sklearn.metrics import confusion_matrix



def get_confusion_matrix(y_true, y_pred):

    transdict = {1: "malignant", 0: "benign"}

    y_t = np.array([transdict[x] for x in y_true])

    y_p = np.array([transdict[x] for x in y_pred])

    

    labels = ["benign", "malignant"]

    index_labels = ["actual benign", "actual malignant"]

    col_labels = ["predicted benign", "predicted malignant"]

    confusion = confusion_matrix(y_t, y_p, labels=labels)

    confusion_df = pd.DataFrame(confusion, index=index_labels, columns=col_labels)

    for n in range(2):

        confusion_df.iloc[n] = confusion_df.iloc[n] / confusion_df.sum(axis=1).iloc[n]

    return confusion_df
dev_dataset = AlbuMelanomaDataset(dev_df, albu_transform_fun(key="dev"))

dev_dataloader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False, drop_last=False)

preds, probas = predict(results, dev_dataloader)
confusion = get_confusion_matrix(dev_df.target.values, preds)

plt.figure(figsize=(6,6))

sns.heatmap(confusion, cbar=False, annot=True, fmt="g", square=True, cmap="Reds");
external_test_path = "../input/melanoma-external-malignant-256/test/test/"

test_info["image_path"] = external_test_path + test_info.image_name +".jpg"
TEST_BATCH_SIZE=68

test_dataset = AlbuMelanomaDataset(test_info, albu_transform_fun(RESIZE_SHAPE, "test"))

test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)

preds, probas = predict(results, test_dataloader)
submission = pd.read_csv(basepath + "sample_submission.csv")

submission.target = probas[:,1]
submission.head()
sns.distplot(submission.target)
submission.to_csv("submission.csv", index=False)