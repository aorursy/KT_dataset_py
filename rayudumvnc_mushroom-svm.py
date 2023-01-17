# Loading Data and importing required Libraries 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")
   
# Importing Libraries
import numpy as np
np.random.seed(1)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

import sklearn.ensemble
import lime
import lime.lime_tabular
print(5)

# Dataset Loading

data = np.genfromtxt('/kaggle/input/mushroom-dataset/mushroom.data', delimiter=',', dtype='<U20')

labels = data[:,0]

# Categories name
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,1:]
pd.DataFrame(data)
categorical_features = range(22)

feature_names = 'cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat'.split(',')

categorical_names = '''bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
fibrous=f,grooves=g,scaly=y,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises=t,no=f
almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
attached=a,descending=d,free=f,notched=n
close=c,crowded=w,distant=d
broad=b,narrow=n
black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
enlarging=e,tapering=t
bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
fibrous=f,scaly=y,silky=k,smooth=s
fibrous=f,scaly=y,silky=k,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
partial=p,universal=u
brown=n,orange=o,white=w,yellow=y
none=n,one=o,two=t
cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'''.split('\n')

for j, names in enumerate(categorical_names):
    values = names.split(',')
    values = dict([(x.split('=')[1], x.split('=')[0]) for x in values])
    data[:,j] = np.array(list(map(lambda x: values[x], data[:,j])))
    
pd.DataFrame(data,columns=feature_names)
categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_


# Data Spliting
data = data.astype(float)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)
# Hot encoding for better results
encoder = sklearn.preprocessing.OneHotEncoder()
encoder.fit(train)
encoded_train = encoder.transform(train)
#SVM classifier Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
from sklearn import svm
clf=svm.SVC(probability=True)
clf.fit(encoded_train,labels_train)
y_pred = clf.predict(encoder.transform(test))
print(accuracy_score(labels_test,y_pred))






predict_fn = lambda x: clf.predict_proba(encoder.transform(x))

# Lime Explainer

np.random.seed(1)
explainer = lime.lime_tabular.LimeTabularExplainer(train ,class_names=['edible', 'poisonous'], feature_names = feature_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3, verbose=False)

# Example
i = 800
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook()