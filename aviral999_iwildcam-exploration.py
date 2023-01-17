# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the read-only "../input/" directory

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os as os

import json as json

import pandas as pd

import matplotlib.image as mpimg

import matplotlib.patches as patches

import matplotlib.pyplot as plt
with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_data:

    train_annotations = json.load(json_data)

    print("Train annotations: ", train_annotations.keys())



with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json') as json_data:

    megadetector_results = json.load(json_data)

    print("Megadetector results: ", megadetector_results.keys())
df_train_images = pd.DataFrame(train_annotations["images"])

display(df_train_images.sample(5))



df_train_annotations = pd.DataFrame(train_annotations["annotations"])

display(df_train_annotations.sample(5))



df_detections = pd.DataFrame(megadetector_results["images"])

display(df_detections.sample(5))
# Generate ids randomly from df_train_images

# Obtain detections of those ids from the df_detections dataframe

# Draw them and display 



sample_ids = set()



for i in range(50):

    chosen = False

    while not chosen:

        potential = df_train_images.id.sample(1).values[0]

        detections = df_detections[df_detections["id"] == potential]["detections"].values[0]

        confs = [detection["conf"] for detection in detections if detection["conf"] > 0.9]

        if len(confs) == 0 and potential not in sample_ids:

            continue

        chosen = True

        sample_ids.add(potential)

    



for sample_id in sample_ids:

    detections = df_detections[df_detections["id"] == sample_id]["detections"].values[0]

    

    img = mpimg.imread("/kaggle/input/iwildcam-2020-fgvc7/train/" + sample_id + ".jpg")

    

    _ = plt.figure(figsize = (15,20))

    _ = plt.axis('off')

    ax = plt.gca()

    # ax.text(10,100, f'{cat} {count}', fontsize=20, color='fuchsia')



    for detection in detections:

        # ref - https://github.com/microsoft/CameraTraps/blob/e530afd2e139580b096b5d63f0d7ab9c91cbc7a4/visualization/visualization_utils.py#L392

        if detection["conf"] < 0.9:

            continue

        x_rel, y_rel, w_rel, h_rel = detection['bbox']    

        img_height, img_width, _ = img.shape

        x = x_rel * img_width

        y = y_rel * img_height

        w = w_rel * img_width

        h = h_rel * img_height

        

        cat = 'animal' if detection['category'] == "1" else 'human'

        bbox = patches.FancyBboxPatch((x,y), w, h, alpha=0.8, linewidth=6, capstyle='projecting', edgecolor='fuchsia', facecolor="none")

        

        ax.text(x+1.5, y-8, f'{cat} {detection["conf"]}', fontsize=10, bbox=dict(facecolor='fuchsia', alpha=0.8, edgecolor="none"))

        ax.add_patch(bbox)



    _ = plt.imshow(img)

    