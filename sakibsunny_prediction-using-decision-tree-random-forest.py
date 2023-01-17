#import modules

import pandas as pd

from sklearn import tree

%matplotlib inline
#import dataset

data = pd.read_csv("../input/Dataset_spine.csv",header=0)

data.head()
#mapping the data

d = {'Abnormal' : 1, 'Normal' : 0}

data['Class_att'] = data['Class_att'].map(d)
data = data.rename(columns={'Col1' : 'pelvic_incidence','Col2' : 'pelvic_tilt','Col3' : 'lumbar_lordosis_angle',

                           'Col4' : 'sacral_slope','Col5': 'pelvic_radius','Col6' : 'degree_spondylolisthesis',

                           'Col7' : 'pelvic_slope','Col8' : 'Direct_tilt','Col9': 'thoracic_slope',

                           'Col10' : 'cervical_tilt','Col11' : 'sacrum_angle','Col12' : 'scoliosis_slope'})
del data['Unnamed: 13']

data.head()
features = list(data.columns[:12])

features
y = data["Class_att"]

x = data[features]

Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(x,y)

output = Tree.predict([43.1,10,28.317406,40.060784,104.168725,4.91,0.843360,40.4940,15.9546,8.87237,-11.378376,20.9171])

print (output)
from sklearn.ensemble import RandomForestClassifier

Forest = RandomForestClassifier(n_estimators = 100)

Forest = Forest.fit(x,y)

output = Forest.predict([50.832021,22.218482,30.092194,26.613539,105.985135,-1.530317,0.974889,16.8343,21.4861,15.65897,-9.031888,29.2221])

print (output)
features = pd.DataFrame()

features['feature'] = data.columns[:12]

features['importance'] = Forest.feature_importances_

features.sort(['importance'],ascending=True)