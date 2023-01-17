import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

EventData=pd.read_csv("../input/events.csv")
appLabelData=pd.read_csv("../input/app_labels.csv")
genderTrain=pd.read_csv("../input/gender_age_train.csv")
labelData=pd.read_csv("../input/label_categories.csv")
modelData=pd.read_csv("../input/phone_brand_device_model.csv")
appEventData=pd.read_csv("../input/app_events.csv")
