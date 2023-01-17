import pandas as pd

from IPython.display import Image

import json

import os

from tqdm import tqdm
d = pd.read_csv('../input/global-wheat-detection/train.csv')

d['image_id'].unique()[0]

for image_id in tqdm(list(d['image_id'].unique())):

    shapes = []

    for _, row in d[d['image_id']==image_id].iterrows():

        x, y, w, h = json.loads(row['bbox'])

        shapes.append({

          "label": "wheat",

          "points": [[x, y], [x+w, y+h]],

          "group_id": None,

          "shape_type": "rectangle",

          "flags": {}

        })

    img_filename = "{}.jpg".format(image_id)

    labelme = json.dumps({

      "version": "4.0.0",

      "flags": {},

      "shapes": shapes,

      "imagePath": img_filename,

      "imageData": None,

      "imageHeight": 1024,

      "imageWidth": 1024

    }, indent=2)

    annot_filename = "labelme/{}.json".format(image_id)

    os.makedirs(os.path.dirname(annot_filename), exist_ok=True)

    with open(annot_filename, "w") as f:

        f.write(labelme)