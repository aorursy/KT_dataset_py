%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
pd.read_csv("../input/app_labels.csv").head(10)
pd.read_csv("../input/events.csv").head(10)
pd.read_csv("../input/gender_age_test.csv").head(10)
pd.read_csv("../input/gender_age_train.csv").head(10)
pd.read_csv("../input/label_categories.csv").head(10)
pd.read_csv("../input/phone_brand_device_model.csv").head(10)
pd.read_csv("../input/sample_submission.csv").head(10)
