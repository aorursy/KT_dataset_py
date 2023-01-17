### Directory dependencies

import os

from glob import glob



### Preprocessing dependencies: Level 1

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



### Plotting dependencies

import matplotlib as mpl

import matplotlib.pyplot as plt

import plotly.graph_objs as go # Plotly for the interactive viewer (see last section)

import seaborn as sns

from IPython.display import Image, display, HTML



### Preprocessing dependencies: Level 2

import openslide as OS

import PIL

from PIL import Image, ImageOps

import imageio

import cv2

import skimage.io as IO



### Figure Configuration

sns.set_style("whitegrid")

plt.rc("figure", titlesize=20)   # fontsize of the figure title

plt.rc("axes", titlesize=17)     # fontsize of the axes title

plt.rc("axes", labelsize=15)     # fontsize of the x and y labels

plt.rc("xtick", labelsize=12)    # fontsize of the tick labels

plt.rc("ytick", labelsize=12)    # fontsize of the tick labels

plt.rc("legend", fontsize=13)    # legend fontsize
pwd
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



print("This is an overview of the different directories and their contents\n")

count = 1

for dirname, _, filenames in os.walk("/kaggle/"):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        count += 1

        

        if count%10 == 0:

            break
### Base directory of the data

data_dir = "../input/prostate-cancer-grade-assessment"



### Directories of the training images and label masks

train_dir = os.path.sep.join([data_dir, "train_images"])

train_mask_dir = os.path.sep.join([data_dir, "train_label_masks"])
### Load the data

train_ids = pd.read_csv(f"{data_dir}/train.csv")

test_ids = pd.read_csv(f"{data_dir}/test.csv")

submission = pd.read_csv(f"{data_dir}/sample_submission.csv")



display(train_ids.tail(3).style.background_gradient(cmap="Blues"))
train_ids.set_index("image_id", inplace=True)

display(train_ids.tail(10).style.background_gradient(cmap="Blues"))
###

files = glob(f"{train_dir}/*")

print(f"Size of the samples in the train_images directory: {len(files)}")

print(f"Size of training samples whose IDs are in train.csv: {train_ids.shape[0]}")

files = glob(f"{train_mask_dir}/*")

print(f"Size of the samples masks in the train_label_masks directory: {len(files)}")

###

print(f"\nData providers IDs: {train_ids.data_provider.unique()}, Size: {len(train_ids.data_provider.unique())}")

print(f"ISUP Grades (target) used: {train_ids.isup_grade.unique()}, Size: {len(train_ids.isup_grade.unique())}")

print(f"Gleason Score used: {train_ids.gleason_score.unique()}, Size: {len(train_ids.gleason_score.unique())}")

###

#display(train_ids.tail(10).style.background_gradient(cmap="Blues"))
train_ids.isnull().sum()
display(test_ids.head())

###

print(f"Size of test samples whose IDs are in test.csv: {test_ids.shape[0]}")
df = pd.DataFrame({"data_type": ["train_images", "train_label_masks"],

                   "count": [len(glob(f"{train_dir}/*")), len(glob(f"{train_mask_dir}/*"))]}).set_index("data_type")



display(df.style.background_gradient(cmap="Blues"))

####

fig, ax = plt.subplots(figsize=[6,8])



val_ax = sns.barplot(x=df.index, y=df["count"], palette="deep", ax=ax)

for i, v in enumerate(df.values):

    ax.text(val_ax.get_xticks()[i], v, str(int(v)),

            ha="center", fontsize=13)

ax.set_ylabel("count")

ax.set_title("Training sample Size\n")

ax.set_xlabel("\ndata type")

fig.tight_layout()
def distribution_plot(df, feature, ax, title=""):

    total = float(len(df))

    sns.countplot(df[feature],

                  order=df[feature].value_counts().index,

                  ax=ax)

    

    if feature=="gleason_score":

        ax.set_xticklabels(df[feature].tolist(), rotation=45) 

    ax.set_title(title)

    if feature!="data_provider":

        ax.set_ylabel("")

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height+3,

                "{:1.1f}%".format(100*height/total),

                ha="center",

                fontsize=12) 
df = train_ids.reset_index()

temp1 = df.groupby("data_provider").count()["image_id"].reset_index().sort_values(by="image_id",ascending=False).rename(columns={"image_id": "count"})

temp2 = df.groupby("isup_grade").count()["image_id"].reset_index().sort_values(by="image_id", ascending=False).rename(columns={"image_id": "count"})

temp3 = df.groupby("gleason_score").count()["image_id"].reset_index().sort_values(by="image_id",ascending=False).rename(columns={"image_id": "count"})

display(temp1.style.background_gradient(cmap="Blues"))

display(temp2.style.background_gradient(cmap="Blues"))

display(temp3.style.background_gradient(cmap="Blues"))
df = train_ids.reset_index()

fig = plt.figure(figsize=(22, 8))

ax = [fig.add_subplot(1, 3, 1),

      fig.add_subplot(1, 3, 2),

      fig.add_subplot(1, 3, 3)]

###

title0 = "Data provider"

title1 = "ISUP grade"

title2 = "Gleason score"

###

distribution_plot(df=df, feature="data_provider", ax=ax[0], title=title0)

distribution_plot(df=df, feature="isup_grade", ax=ax[1], title=title1)

distribution_plot(df=df, feature="gleason_score", ax=ax[2], title=title2)

###

fig.suptitle("Distribution plot (count and %)", y=1.1)

fig.tight_layout()

plt.show()
df = train_ids.reset_index()

fig, ax = plt.subplots(1, 2, figsize=[20,8], sharey=True)



val1 = df.groupby(["isup_grade", "data_provider"]).count()["image_id"].unstack().plot(kind="bar", ax=ax[0])

val2 = df.groupby(["gleason_score", "data_provider"]).count()["image_id"].unstack().plot(kind="bar", ax=ax[1])



total = df.shape[0]

for k, p in enumerate(ax[0].patches):

    height = p.get_height()

    ax[0].text(p.get_x()+p.get_width()/2.,

            height + 3,

            "{:1.1f}%".format(100*height/total),

            ha="center",

            fontsize=12, rotation=0)

        

for k, p in enumerate(ax[1].patches):

    height = p.get_height()

    ax[1].text(p.get_x()+p.get_width()/2.,

               height + 3,

               "{:1.1f}%".format(100*height/total),

               ha="center",

               fontsize=12, rotation=70)

    

for label in ax[0].get_xticklabels():

    label.set_rotation(0)

for label in ax[1].get_xticklabels():

    label.set_rotation(0)





ax[0].set_ylabel("count")

ax[0].set_title("ISUP Grade/Data Provider\n")

ax[1].set_title("Gleason Score/Data Provider\n")

fig.suptitle("Relative Distribution plot (count and %)", y=1.1)



fig.tight_layout();
### Randomly sample 9 training samples from our dataframe train_ids

np.random.seed(13)

WSI9 = train_ids.sample(n=9)

display(WSI9.style.background_gradient(cmap="Blues"))
###  Let's open two files from different data provider, from the 7 selected images above

print(f"Data provider: {WSI9.data_provider.tolist()[0]}\n")

file_path1 = os.path.sep.join([train_dir, WSI9.index[0]+".tiff"]) # Full file directory

example_slide1 = OS.OpenSlide(file_path1) # Openining without reading the image into memory

print(f"Level dimensions: {example_slide1.level_dimensions}\n")



for prop in example_slide1.properties.keys():

    print("{}: -> {}".format(prop, example_slide1.properties[prop]))
print(f"Data provider: {WSI9.data_provider.tolist()[1]}\n")

file_path2 = os.path.sep.join([train_dir, WSI9.index[1]+".tiff"]) # Full file directory

example_slide2 = OS.OpenSlide(file_path2) # Openining without reading the image into memory

print(f"Level dimensions: {example_slide2.level_dimensions}\n")



for prop in example_slide2.properties.keys():

    print("{}: -> {}".format(prop, example_slide2.properties[prop]))

### Visualization based on OpenSlide and Matplolib packages

def plot_WSIs(slides): 

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18,18))

    for i, slide in enumerate(slides):

        file_path = os.path.sep.join([train_dir, slide+".tiff"]) # Full file directory

        image = OS.OpenSlide(file_path) # Openining without reading the image into memory

        

        # Creation of the patch to plot

        patch = image.read_region(location=(0,0),

                                  level=image.level_count-1,  # Get the last level/slide

                                  size=image.level_dimensions[-1]) # Get the dimension corresponding of the last level

        

        # Plot the patch

        ax[i//3, i%3].imshow(patch)

        image.close()

        ax[i//3, i%3].axis("on")

        

        image_id = slide

        data_provider = train_ids.loc[slide, "data_provider"]

        isup_grade = train_ids.loc[slide, 'isup_grade']

        gleason_score = train_ids.loc[slide, 'gleason_score']

        ax[i//3, i%3].set_title(f"\nID: ~{image_id[:7]}, Source: {data_provider}\nISUP: {isup_grade}, Gleason: {gleason_score}")



    fig.tight_layout()

    fig.suptitle("")

    plt.show()

plot_WSIs(WSI9.index)
### Visualization based on Skimage and Matplotlib packages

def plot_resized_biopsy(slides): 

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18,18))

    for i, slide in enumerate(slides):

        file_path = os.path.sep.join([train_dir, slide+".tiff"]) # Full file directory

        image = OS.OpenSlide(file_path) # Openining without reading the image into memory

        

        # Creation of the patch to plot

        patch = image.read_region(location=(0,0),

                                  level=image.level_count-1,  # Get the last level/slide

                                  size=image.level_dimensions[-1]) # Get the dimension corresponding of the last level

        

        # Resize the image patch

        image = cv2.resize(np.asarray(patch), (512, 512))

        

        # Plot the resized image patch

        ax[i//3, i%3].imshow(image)  

        ax[i//3, i%3].axis("on")

        

        image_id = slide

        data_provider = train_ids.loc[slide, "data_provider"]

        isup_grade = train_ids.loc[slide, 'isup_grade']

        gleason_score = train_ids.loc[slide, 'gleason_score']

        ax[i//3, i%3].set_title(f"\nID: ~{image_id[:7]}, Source: {data_provider}\nISUP: {isup_grade}, Gleason: {gleason_score}")



    fig.tight_layout()

    plt.show()
plot_resized_biopsy(WSI9.index)
def plot_biopsy_masks(slides): 

    fig, ax = plt.subplots(3,3, figsize=(18,18))

    for i, slide in enumerate(slides):

        

        file_path = os.path.sep.join([train_mask_dir, slide+"_mask.tiff"]) # Full file directory

        biopsy_mask = OS.OpenSlide(file_path) # Openining without reading the image into memory

        

        # Creation of the patch to plot

        mask_data = biopsy_mask.read_region(location=(0,0),

                                            level=biopsy_mask.level_count - 1,  # Get the last level/slide

                                            size=biopsy_mask.level_dimensions[-1]) # Get the dimension corresponding of the last level

    

        # Plot

        cmap = mpl.colors.ListedColormap(["black", "gray", "green", "yellow", "orange", "red"])

        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation="nearest", vmin=0, vmax=5) 

        biopsy_mask.close()       

        ax[i//3, i%3].axis("on")

        

        image_id = slide

        data_provider = train_ids.loc[slide, "data_provider"]

        isup_grade = train_ids.loc[slide, "isup_grade"]

        gleason_score = train_ids.loc[slide, "gleason_score"]

        ax[i//3, i%3].set_title(f"\nID: {image_id[:7]} Source: {data_provider}\nISUP: {isup_grade} Gleason: {gleason_score}")

    

    fig.tight_layout()    

    plt.show()
plot_biopsy_masks(WSI9.index)
train_df = train_ids.reset_index() # In train_ids, I set index to "image_id"



masks = os.listdir(train_mask_dir)

masks_df = pd.DataFrame(data={"mask_id": masks})

masks_df["image_id"] = masks_df.mask_id.apply(lambda x: x.split("_")[0]) # Recall mask_id=image_id+"_mask.tiff"



train_df = pd.merge(train_df, masks_df, on="image_id", how="outer")

print(f"We have {train_df.shape[0]} training sample images and {masks_df.shape[0]} masks. So, there will be exactly {len(train_df[~train_df.mask_id.isna()])} images in the final training samples.")

display(train_df.head(10).style.background_gradient(cmap="Blues"))
def load_and_resize_biopsy(img_id):

    

    file_path = os.path.sep.join([train_dir, img_id+".tiff"]) # Full file directory

    biopsy_img = OS.OpenSlide(file_path) # Openining without reading the image into memory



    # Creation of the patch to plot

    patch = biopsy_img.read_region(location=(0,0),

                                   level=biopsy_img.level_count-1,  # Get the last level/slide

                                   size=biopsy_img.level_dimensions[-1]) # Get the dimension corresponding of the last level



    # Resize the image patch

    image = cv2.resize(np.asarray(patch), (512, 512))

    

    return image



def load_and_resize_biopsy_mask(img_id):

    

    file_path = os.path.sep.join([train_mask_dir, img_id+"_mask.tiff"]) # Full file directory

    biopsy_mask = OS.OpenSlide(file_path) # Openining without reading the image into memory



    # Creation of the patch to plot

    patch = biopsy_mask.read_region(location=(0,0),

                                    level=biopsy_mask.level_count-1,  # Get the last level/slide

                                    size=biopsy_mask.level_dimensions[-1]) # Get the dimension corresponding of the last level



    # Resize the mask patch

    mask = cv2.resize(np.asarray(patch), (512, 512))[:,:,0]

    

    return mask
### Visualization

data_providers = train_df.data_provider.unique().tolist()

# cmap = mpl.colors.ListedColormap(["black", "gray", "green", "yellow", "orange", "red"])

cmap_rad = mpl.colors.ListedColormap(["white", "lightgrey", "green", "orange", "red", "darkred"])

cmap_kar = mpl.colors.ListedColormap(["white", "green", "red"])

labels = []

for grade in range(train_ids.isup_grade.nunique()):

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(22, 22))



    for i, row in enumerate(ax):

        idx = i//2

        temp_idx = (train_df.isup_grade == grade) & (train_df.data_provider == data_providers[idx])

        temp = train_df[temp_idx].image_id.tail(4).reset_index(drop=True)

        if i%2 < 1:

            labels.append(f"{data_providers[idx]}\n(image)")

            for j, col in enumerate(row):

                col.imshow(load_and_resize_biopsy(temp[j]))

                col.set_title(f"\nID: {temp[j][:13]} $\cdots$")

                

        else:

            labels.append(f"{data_providers[idx]}\n(mask)")

            for j, col in enumerate(row):

                if data_providers[idx] == "radboud":

                    col.imshow(load_and_resize_biopsy_mask(temp[j]), 

                               cmap = cmap_rad, 

                               norm = mpl.colors.Normalize(vmin=0, vmax=5, clip=True))

                else:

                    col.imshow(load_and_resize_biopsy_mask(temp[j]),

                               cmap = cmap_kar,

                               norm = mpl.colors.Normalize(vmin=0, vmax=2, clip=True))

                    

                gleason_score = train_ids.loc[temp[j], "gleason_score"]

                col.set_title(f"\nID: {temp[j][:13]} $\cdots$")

        

    for row, r in zip(ax[:,0], labels):

        row.set_ylabel(r, rotation=0, size="large", labelpad=30, fontsize=20)

    

    fig.tight_layout()

    fig.suptitle(f"ISUP Grade {grade}", y=1.01, fontsize=23)

    plt.show()
def mask_on_slide_overlayer(images, center="radboud", fig_title="", alpha=0.8, max_size=(800, 800)):

    """Show a mask overlayed on a slide."""

    fig, ax = plt.subplots(nrows=3,ncols=3, figsize=(18,18))

    

    

    for i, image_id in enumerate(images):

        # Open a slide/wsi

        wsi_file_path = os.path.sep.join([train_dir, image_id+".tiff"]) # Full file directory

        biopsy_img = OS.OpenSlide(wsi_file_path) # Openining without reading the image into memory

        # Open the corresponding mask

        mask_file_path = os.path.sep.join([train_mask_dir, image_id+"_mask.tiff"])

        biopsy_mask = OS.OpenSlide(mask_file_path)



        # Creation of the patch to visualize

        patch_img = biopsy_img.read_region(location=(0,0),

                                           level=biopsy_img.level_count-1,  # Get the last level/slide

                                           size=biopsy_img.level_dimensions[-1]) # Get the dimension corresponding of the last level

        

        patch_mask = biopsy_mask.read_region(location=(0,0),

                                             level=biopsy_mask.level_count-1,

                                             size=biopsy_mask.level_dimensions[-1])





        # Split the patch mask into channels

        patch_mask = patch_mask.split()[0]

        

        # Create alpha mask

        alpha_int = int(round(255*alpha))

        if center == "radboud":

            alpha_content = np.less(patch_mask.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)

        elif center == "karolinska":

            alpha_content = np.less(patch_mask.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)



        alpha_content = Image.fromarray(alpha_content)

        preview_palette = np.zeros(shape=768, dtype=int)



        if center == "radboud":

            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

        elif center == "karolinska":

            # Mapping: {0: background, 1: benign, 2: cancer}

            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)



        patch_mask.putpalette(data=preview_palette.tolist())

        mask_rgb = patch_mask.convert(mode="RGB")

        

        # Overlay the mask on its corresponding slide

        overlayed_image = Image.composite(image1=patch_img, image2=mask_rgb, mask=alpha_content)

        overlayed_image.thumbnail(size=max_size, resample=0)



        # Plot the overlayed image

        ax[i//3, i%3].imshow(overlayed_image) 

        biopsy_img.close()

        biopsy_mask.close()       

        ax[i//3, i%3].axis("on")

        

        data_provider = train_ids.loc[image_id, "data_provider"]

        isup_grade = train_ids.loc[image_id, "isup_grade"]

        gleason_score = train_ids.loc[image_id, "gleason_score"]

        ax[i//3, i%3].set_title(f"\nID: {image_id[:7]} $\cdots$, Source: {data_provider}\nISUP: {isup_grade} Gleason: {gleason_score}")

    

    fig.suptitle(fig_title, y=1.01, fontsize=23)

    fig.tight_layout()

    plt.show()
fig_title = "Some Examples of Overlayed Images"

mask_on_slide_overlayer(images=WSI9.index, fig_title=fig_title)
WSI9.index[0]
pen_marked_images = ["ca0798453868081bc8aeeabb01847d4e",

                     "ff10f937c3d52eff6ad4dd733f2bc3ac",

                     "e9a4f528b33479412ee019e155e1a197",

                     "fd6fe1a3985b17d067f2cb4d5bc1e6e1",

                     "f39bf22d9a2f313425ee201932bac91a",

                     "fb01a0a69517bb47d7f4699b6217f69d",

                     "ebb6a080d72e09f6481721ef9f88c472",

                     "feee2e895355a921f2b75b54debad328",

                     "ebb6d5ca45942536f78beb451ee43cc4"]





fig_title = "Some Examples of Pen Marked Images"

mask_on_slide_overlayer(images=pen_marked_images, fig_title=fig_title)
trn_df = train_df.copy()

dims, spacings = [], []



for img_id in trn_df.reset_index().image_id:

    # Open a slide/wsi

    wsi_file_path = os.path.sep.join([train_dir, img_id+".tiff"]) # Full file directory

    biopsy_img = OS.OpenSlide(wsi_file_path) # Openining without reading the image into memory

    

    spacing = 1 / (float(biopsy_img.properties["tiff.XResolution"]) / 10000)

    dims.append(biopsy_img.dimensions)

    spacings.append(spacing)

    biopsy_img.close()
trn_df["spacing"] = spacings

trn_df["width"]  = [i[0] for i in dims]

trn_df["height"] = [i[1] for i in dims]



display(trn_df.head(10).style.background_gradient(cmap="Blues"))
def plot_distribution_grouped(feature, feature_group, ax):

    for feat in trn_df[feature_group].unique():

        df = trn_df.loc[trn_df[feature_group] == feat]

        sns.kdeplot(df[feature], label=feat, ax=ax, shade=True)

    ax.set_title(f"Images {feature}\ngrouped by {feature_group}\n")

    ax.legend()
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,6), dpi=200, sharey=True)

###

sns.kdeplot(trn_df["width"], ax=ax[0], shade=True, label="width")

sns.kdeplot(trn_df["height"], ax=ax[0], shade=True, label="height")

ax[0].set_xlabel("dimension")

ax[0].set_title("Images Width and Height\n")

ax[0].legend()

###

plot_distribution_grouped(feature="width", feature_group="data_provider", ax=ax[1])

plot_distribution_grouped(feature="height", feature_group="data_provider", ax=ax[2])



fig.suptitle("Distribution Plots", y=1.1)

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6), dpi=200, sharey=True)

###

plot_distribution_grouped(feature="width", feature_group="isup_grade", ax=ax[0])

plot_distribution_grouped(feature="height", feature_group="isup_grade", ax=ax[1])



fig.suptitle("Distribution by ISUP Grade", y=1.1)

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6), dpi=200, sharey=True)

###

plot_distribution_grouped(feature="width", feature_group="gleason_score", ax=ax[0])

plot_distribution_grouped(feature="height", feature_group="gleason_score", ax=ax[1])



fig.suptitle("Distribution by Gleason Score", y=1.1)

fig.tight_layout()

plt.show()