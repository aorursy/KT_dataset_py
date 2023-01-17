import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
df = pd.read_csv('../input/Dataset_spine.csv',usecols=['Col1','Col2','Col3','Col4','Col5','Col6','Col7','Col8','Col9','Col10','Col11','Col12','Class_att'])
df.columns = ['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle',

              'sacral_slope','pelvic_radius','degree_spondylolisthesis',

              'pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt',

              'sacrum_angle','scoliosis_slope','Class_att']
df.head()
features = df[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle',

              'sacral_slope','pelvic_radius','degree_spondylolisthesis',

              'pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt',

              'sacrum_angle','scoliosis_slope']]
targetVars = df.Class_att
feature_train,feature_test,target_train,target_test = train_test_split(features, targetVars, test_size=0.3)
model = LogisticRegression()
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)
accuracy_score(target_test,predictions)