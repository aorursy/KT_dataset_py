from IPython.display import HTML

HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/e6h7BxOZuCU?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
import os

import gc

import time

import math

import datetime

from math import log, floor

from sklearn.neighbors import KDTree



import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.utils import shuffle

from tqdm.notebook import tqdm as tqdm



import seaborn as sns

from matplotlib import colors

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

import matplotlib



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import sklearn



import skimage.io

import openslide

import glob

import cv2



import warnings

warnings.filterwarnings("ignore")
INPUT_DIR = "../input/prostate-cancer-grade-assessment/"

TRAIN_IMG_DIR = "../input/prostate-cancer-grade-assessment/train_images/"

TRAIN_MASK_DIR = "../input/prostate-cancer-grade-assessment/train_label_masks/"

TEST_IMG_DIR = "../input/prostate-cancer-grade-assessment/test_images)"



train = pd.read_csv(INPUT_DIR+"train.csv").set_index("image_id")

test = pd.read_csv(INPUT_DIR+"test.csv")

sample_submission = pd.read_csv(INPUT_DIR+"sample_submission.csv")
train.head()
test.head()
isup_grade_count = train.isup_grade.value_counts().reset_index()

isup_grade_count.columns = ["isup_grade", "count"]





fig = make_subplots(1,2, specs=[[{"type": "bar"}, {"type": "pie"}]])



colors=px.colors.sequential.Plasma[:6]



fig.add_trace(go.Bar(

        x=isup_grade_count["isup_grade"].values, 

        y=isup_grade_count["count"].values,

        marker=dict(color=colors)

          

), row=1, col=1)



fig.add_trace(go.Pie(

        labels = isup_grade_count["isup_grade"].values,

        values = isup_grade_count["count"].values,

        marker=dict(colors=colors)

), row=1, col=2)



fig.update_layout(title="Isup_grade - Count plots")

fig.show()
karolinska = train.groupby(["data_provider", "isup_grade"])["data_provider"].count().loc["karolinska"].reset_index()

radboud = train.groupby(["data_provider", "isup_grade"])["data_provider"].count().loc["radboud"].reset_index()



fig = go.Figure()



fig.add_trace(go.Bar(

    x=karolinska.isup_grade,

    y=karolinska.data_provider,

    name='karolinska',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=radboud.isup_grade,

    y=radboud.data_provider,

    name='rodboud',

    marker_color='lightsalmon'

))



fig.update_layout(title="targets count based on data provider")

fig.show()
gleason_score_count = train.gleason_score.value_counts().reset_index()

gleason_score_count.columns = ["gleason_score", "count"]





fig = make_subplots(1,2, specs=[[{"type": "bar"}, {"type": "pie"}]])



colors=px.colors.sequential.Plotly3



fig.add_trace(go.Bar(

        x=gleason_score_count["gleason_score"].values, 

        y=gleason_score_count["count"].values,

        marker=dict(color=colors)

          

), row=1, col=1)



fig.add_trace(go.Pie(

        labels = gleason_score_count["gleason_score"].values,

        values = gleason_score_count["count"].values,

        marker=dict(colors=colors)

), row=1, col=2)



fig.update_layout(title="Gleason_score - Count plots")

fig.show()
karolinska = train.groupby(["data_provider", "gleason_score"])["data_provider"].count().loc["karolinska"].reset_index()

radboud = train.groupby(["data_provider", "gleason_score"])["data_provider"].count().loc["radboud"].reset_index()



fig = go.Figure()



fig.add_trace(go.Bar(

    x=karolinska.gleason_score,

    y=karolinska.data_provider,

    name='karolinska',

    marker_color=px.colors.sequential.Blackbody[1]

))

fig.add_trace(go.Bar(

    x=radboud.gleason_score,

    y=radboud.data_provider,

    name='rodboud',

    marker_color=px.colors.sequential.Blackbody[2]

))



fig.update_layout(title="gleason_score count based on data provider")

fig.show()
path = f"{TRAIN_IMG_DIR}005e66f06bce9c2e49142536caf2f6ee.tiff"

biopsy = openslide.OpenSlide(path)

# do somethiing with the slide hear

biopsy.close()
def print_slide_details(slide, show_thumbnail=True, max_size=(600, 400)):

    """Print some basic information about a slide"""

    # Generate a small image thumbnail

    if show_thumbnail:

        #fig = px.imshow(slide.get_thumbnail(size=max_size))

        #fig.show()

        display(slide.get_thumbnail(size=max_size))

    

    

    # Here we compute the "Pixel spacing" : the physical size of a pixel in the image.

    # OpenSlide gives the resolution in centimeters so we convert this to microns.

    

    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

    

    print(f"File id: {slide}")

    print(f"Dimensions: {slide.dimensions}")

    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")

    print(f"Number of levels in the image: {slide.level_count}")

    print(f"Downsample factor per level: {slide.level_downsamples}")

    print(f"Dimensions of levels: {slide.level_dimensions}")
example_slides = train.index.values[-4:]



for case_id in example_slides:

    biopsy = openslide.OpenSlide(os.path.join(TRAIN_IMG_DIR, f'{case_id}.tiff'))

    print_slide_details(biopsy)

    biopsy.close()

    

    # Print the case-level label

    print(f"ISUP grade: {train.loc[case_id, 'isup_grade']}")

    print(f"Gleason score: {train.loc[case_id, 'gleason_score']}\n\n")
biopsy = openslide.OpenSlide(os.path.join(TRAIN_IMG_DIR, '00928370e2dfeb8a507667ef1d4efcbb.tiff'))



x = 5150

y = 21000

level = 0

width = 512

height = 512



region = biopsy.read_region((x,y), level, (width, height))

#fig = px.imshow(region)

#fig.show()

display(region)
x = 5140

y = 21000

level = 1

width = 512

height = 512



region = biopsy.read_region((x,y), level, (width, height))

#fig = px.imshow(region)

#fig.show()

display(region)
biopsy.close()
def print_mask_details(slide, center='radboud', show_thumbnail=True, max_size=(400,400)):

    """Print some basic information about a slide"""

    

    if center not in ['radboud', 'karolinska']:

        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

        

    

    # Generate a small image thumbnail

    if show_thumbnail:

        # Read in the mask data from the highest level

        # We cannot use thumbnail() here because we need to load the raw label data.

        mask_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

        # Mask data is present in the R channel

        mask_data = mask_data.split()[0]

        

        # To show the masks we map the raw label values to RGB values

        

        preview_palette = np.zeros(shape=768, dtype=int)

        if center == "radboud":

            # Mapping : {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

            

            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

            

        elif center == "karolinska":

            

            # Mapping: {0: background, 1: benign, 2: cancer}

            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)

            

        mask_data.putpalette(data=preview_palette.tolist())

        mask_data = mask_data.convert(mode='RGB')

        mask_data.thumbnail(size=max_size, resample=0)

        

        #fig = px.imshow(mask_data)

        #fig.show()

        display(mask_data)

        

        # Compute microns per pixel (openslide gives resolution in centimeters)

        spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)



        print(f"File id: {slide}")

        print(f"Dimensions: {slide.dimensions}")

        print(f"Microns per pixel / pixel spacing: {spacing:.3f}")

        print(f"Number of levels in the image: {slide.level_count}")

        print(f"Downsample factor per level: {slide.level_downsamples}")

        print(f"Dimensions of levels: {slide.level_dimensions}")
mask = openslide.OpenSlide(os.path.join(TRAIN_MASK_DIR, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))

print_mask_details(mask, center='radboud')

mask.close()
mask = openslide.OpenSlide(os.path.join(TRAIN_MASK_DIR, '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))

print_mask_details(mask, center='karolinska')

mask.close()
mask = openslide.OpenSlide(os.path.join(TRAIN_MASK_DIR , '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))

mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])



plt.figure()

plt.title("Mask with default cmap")

plt.imshow(np.asarray(mask_data)[:,:,0], interpolation='nearest')

plt.show()



plt.figure()

plt.title("Mask with custom cmap")

# Optional: create a custom color map

cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

plt.imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)

plt.show()



mask.close()
!pip install efficientnet_pytorch

from efficientnet_pytorch import EfficientNet
TRAIN_IMG_DIR = "../input/panda2/train_images/"
##%%writefile dataset.py



import pandas as pd

import albumentations

import joblib

import numpy as np

import torch



from PIL import Image



class PANDADatasetTrain:

    def __init__(self, df, img_height, img_width, mean, std, train=True):

        

        self.image_ids = df.index.values

        self.isup_grade = df.isup_grade.values

        self.img_height = img_height

        self.img_width = img_width



        if train:

            self.aug = albumentations.Compose([

                albumentations.Resize(img_height, img_width, always_apply=True),

                albumentations.ShiftScaleRotate(shift_limit=0.0625,

                                                scale_limit=0.1, 

                                                rotate_limit=5,

                                                p=0.9),

                albumentations.Normalize(mean, std, always_apply=True)

            ])

        else:

            self.aug = albumentations.Compose([

                albumentations.Resize(img_height, img_width, always_apply=True),

                albumentations.Normalize(mean, std, always_apply=True)

            ])

            





    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, item):

        image = Image.open(f"{TRAIN_IMG_DIR}{self.image_ids[item]}.png")

        #image = image.get_thumbnail(size=(600, 400))

        #print(image.size)

        image = self.aug(image=np.array(image))["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)



        return {

            "image": torch.tensor(image, dtype=torch.float),

            "target": torch.tensor(self.isup_grade[item], dtype=torch.float)

        }
d = PANDADatasetTrain(train, 600, 400, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

out = d.__getitem__(1)

out["image"].shape, out["target"]
##%%writefile efficientnet_model.py



from efficientnet_pytorch import EfficientNet

import torch.nn as nn

from torch.nn import functional as F



class EfficientNetB1(nn.Module):

    def __init__(self, pretrained):

        super(EfficientNetB1, self).__init__()



        if pretrained is True:

            self.model = EfficientNet.from_pretrained("efficientnet-b1")

        

        self.l0 = nn.Linear(1280, 1)



    def forward(self, x):

        bs, _, _, _ = x.shape

        x = self.model.extract_features(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        out = self.l0(x)



        return out
def criterion(target, output):

    mse_loss = nn.MSELoss()(target, output)

    return torch.sqrt(mse_loss)



def train_model(model, train_loader, epoch, optimizer, scheduler, DEVICE):

    model.train()



    total_loss = 0



    t = tqdm(train_loader)

    for i, d in enumerate(t):



        image = d["image"].float().to(DEVICE)

        target = d["target"].float().to(DEVICE)



        optimizer.zero_grad()



        output = model(image)

        

        #print(output.shape, target.shape)



        loss = criterion(target, output)



        total_loss += loss



        loss.backward()

        optimizer.step()



        #print(total_loss/i+1)

        

        t.set_description(f'Epoch {epoch+1} : Loss: %.4f'%(total_loss/(i+1)))



        #if i % int(t/10) == 0:

        #    print(f'Epoch {epoch+1|i} : Loss: %.4f'%(total_loss/(i+1)))





def valid_model(model, valid_loader, epoch, scheduler, DEVICE):

    model.eval()



    total_loss = 0

    

    output_list = []

    target_list = []



    #t = tqdm(valid_loader)

    with torch.no_grad():

        for i, d in enumerate(valid_loader):



            image = d["image"].float().to(DEVICE)

            target = d["target"].float().to(DEVICE)

            

            output = model(image)



            loss = criterion(target, output)



            total_loss += loss

            

            output = output.squeeze(1)

            output = output.cpu().numpy().tolist()

            target = target.cpu().numpy().tolist()

            

            output_list.extend(output)

            target_list.extend(target)

            



            #if i == 1:

            #    break

        #print(total_loss/i+1)



    RMSE = sklearn.metrics.mean_squared_error(target_list, output_list)

    print(f" Valid RMSE : %.4f"%(RMSE))



    return RMSE
TRAIN_BATCH_SIZE = 16

VALID_BATCH_SIZE = 16

IMG_HEIGHT = 244

IMG_WIDHT = 244

MODEL_MEAN = (0.485, 0.456, 0.406)

MODEL_STD = (0.229, 0.224, 0.225)



train_df = train.iloc[:-500]

valid_df = train.iloc[-500:]







train_dataset = PANDADatasetTrain(df=train_df, 

                                  img_height=IMG_HEIGHT,

                                  img_width=IMG_WIDHT,

                                  mean=MODEL_MEAN,

                                  std=MODEL_STD, 

                                  train=True )

    

train_loader = torch.utils.data.DataLoader(

        dataset=train_dataset,

        batch_size= TRAIN_BATCH_SIZE,

        shuffle=False,

        num_workers=4,

        drop_last=False

    )



valid_dataset = PANDADatasetTrain(df=valid_df, 

                                  img_height=IMG_HEIGHT,

                                  img_width=IMG_WIDHT,

                                  mean=MODEL_MEAN,

                                  std=MODEL_STD, 

                                  train=False)



valid_loader = torch.utils.data.DataLoader(

        dataset=valid_dataset,

        batch_size= VALID_BATCH_SIZE,

        shuffle=False,

        num_workers=4,

        drop_last=False

    )
DEVICE = "cuda"

EPOCHS = 2

start_e = 0



model = EfficientNetB1(pretrained=True)

model.to(DEVICE)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



for epoch in range(start_e, EPOCHS):

    

    train_model(model, train_loader, epoch, optimizer, scheduler=None, DEVICE=DEVICE)

    rmse = valid_model(model, valid_loader, epoch, scheduler=None, DEVICE=DEVICE)

    torch.save(model.state_dict(), f"model_{epoch}_rmse_{rmse}.pth")
def qwk(a1, a2):

    """

    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168



    :param a1:

    :param a2:

    :param max_rat:

    :return:

    """

    max_rat = 3

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e



def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

        

    conf_mat = confusion_matrix1(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return (1.0 - numerator / denominator)
# qwk optimize coefficients



class OptimizedRounder(object):

    def __init__(self, init_coef):

        self.init_coef_ = init_coef

        self.coef_ = 0

    

    def _kappa_loss(self, coef, X, y):

        X_p = X.copy()

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            elif pred >= coef[3] and pred < coef[4]:

                X_p[i] = 4

            else:

                X_p[i] = 5

        

        ll = quadratic_weighted_kappa(y, X_p)

        

        return -ll

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        

        initial_coef = self.init_coef_

        

        self.coef_ = spoptimize.minimize(loss_partial, initial_coef, method="nelder-mead")

        

    def predict(self, X, coef):

        X_p = X.copy()

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            elif pred >= coef[3] and pred < coef[4]:

                X_p[i] = 4

            else:

                X_p[i] = 5

                

        return X_p

    

    def coefficients(self):

        return self.coef_['x']