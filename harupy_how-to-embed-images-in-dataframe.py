import os

from functools import reduce



import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import skimage.io
!ls -lh ../input/prostate-cancer-grade-assessment
INPUT_DIR = "../input/prostate-cancer-grade-assessment"

TRAIN_DIR = "train_images"

MASK_DIR = "train_label_masks"
train = pd.read_csv(f"{INPUT_DIR}/train.csv")

test = pd.read_csv(f"{INPUT_DIR}/test.csv")

sbm = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
# See: https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Subclassing



from IPython.display import HTML

from pandas.io.formats.style import Styler





TEMPLATE = """

{% extends "html.tpl" %}

{% block table %}

{{ super() }}

<span>Shape: {{ shape|default("") }}</span>

{% endblock table %}

"""



with open("with_shape.tpl", "w") as f:

    f.write(TEMPLATE)



WithShape = Styler.from_custom_template(".", "with_shape.tpl")



def profile(df):

    return HTML(WithShape(df.head()).render(shape=str(df.shape)))

profile(train)
profile(test)
profile(sbm)
train["gleason_score"].value_counts().sort_index().to_frame()
import base64

from io import BytesIO





def encode_image(fig):

    """

    See: https://stackoverflow.com/questions/48717794/matplotlib-embed-figures-in-auto-generated-html

    """

    b = BytesIO()

    fig.savefig(b, format="png", bbox_inches="tight", pad_inches=0.05)

    return base64.b64encode(b.getvalue()).decode("utf-8")



    

def draw_image(img_path):

    img = skimage.io.MultiImage(img_path)[-1]



    fig, ax = plt.subplots()

    ax.imshow(img)

    ax.grid(False)

    ax.set_xticks([])

    ax.set_yticks([])

    

    for pos in ["top","bottom", "left", "right"]:

        ax.spines[pos].set_linewidth(0.5)

    

    return fig





def draw_mask(mask_path):

    mask = skimage.io.MultiImage(mask_path)[-1]



    colors = ["black", "gray", "green", "yellow", "orange", "red"]

    cmap = matplotlib.colors.ListedColormap(colors)



    fig, ax = plt.subplots()

    ax.imshow(mask[:, :, 0], cmap=cmap, interpolation="nearest", vmin=0, vmax=5)

    ax.set_xticks([])

    ax.set_yticks([])

    

    for pos in ["top","bottom", "left", "right"]:

        ax.spines[pos].set_linewidth(0.5)

    

    return fig





def merge_cols(df, cols):

    return reduce(

        lambda x, y: x + "<br>" + y,

        [f"{col}: " + df[col].astype(str) for col in cols]

    )





def show_images(df):

    img_tag_tpl = '<img src="data:image/png;base64,{}" style="dispaly: block; margin: 0 auto">'



    imgs = []

    masks = []

    for idx, row in df.iloc[:9].reset_index(drop=False).iterrows():

        # Process image

        img_path = os.path.join(INPUT_DIR, TRAIN_DIR, row["image_id"] + ".tiff")

        img_fig = draw_image(img_path)

        imgs.append(img_tag_tpl.format(encode_image(img_fig)))

        plt.close(img_fig)



        # Process mask

        mask_path = os.path.join(INPUT_DIR, MASK_DIR, row["image_id"] + "_mask.tiff")

        

        if os.path.exists(mask_path):

            mask_fig = draw_mask(mask_path)

            masks.append(img_tag_tpl.format(encode_image(mask_fig)))

            plt.close(mask_fig)

        else:

            masks.append("")



    cols = ["image_id", "data_provider", "isup_grade", "gleason_score"]



    return HTML(

        df

        .assign(

            info=merge_cols(df, cols),

            image=imgs,

            mask=masks,

        )

        .drop(cols, axis=1)

        .style

        .set_properties(**{

            "background-color": "white",

            "border": "1px solid black",

            "text-align": "center",

        })

        .hide_index()

        .render()

    )
show_images(train[train["gleason_score"] == "5+5"].sample(9, random_state=42))
show_images(train[train["gleason_score"] == "5+4"].sample(9, random_state=42))
show_images(train[train["gleason_score"] == "5+3"].sample(9, random_state=42))