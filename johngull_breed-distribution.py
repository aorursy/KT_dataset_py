import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image
labels = pd.read_csv("../input/labels.csv")

labels.head()
unique_breeds = labels.breed.unique()

len(unique_breeds)
gr_labels = labels.groupby("breed").count()

gr_labels = gr_labels.rename(columns = {"id" : "count"})

gr_labels = gr_labels.sort_values("count", ascending=False)

gr_labels.head()
scottish_deerhound_id = labels.loc[labels.breed == "scottish_deerhound"].iloc[0, 0]

Image.open("../input/train/"+scottish_deerhound_id+".jpg")
gr_labels.tail()
eskimo_dog_id = labels.loc[labels.breed == "eskimo_dog"].iloc[1, 0] #0 row is too agressive, so i decided to take 1st :)

Image.open("../input/train/"+eskimo_dog_id+".jpg")