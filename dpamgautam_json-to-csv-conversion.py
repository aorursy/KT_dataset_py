import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import json



file = pd.read_json("../input/grocery items bounding boxes.json", lines=True)
file.to_csv("output.csv")