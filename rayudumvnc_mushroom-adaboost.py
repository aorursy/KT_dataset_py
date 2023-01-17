# Loadind data and importing required libraries 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")
   
# Importing Libraries
import numpy as np # linear algebra
np.random.seed(1)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split 
import sklearn.ensemble
import lime
import lime.lime_tabular
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


data = np.genfromtxt('/kaggle/input/mushroom-dataset/mushroom.data', delimiter=',', dtype='<U20')
labels = data[:,0]

# Categories name
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,1:]
pd.DataFrame(data)
# 
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
data2=pd.DataFrame(data,columns=feature_names)

# Data Spliting
data = data.astype(float)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.90)
#Hot Encoding for  better results
encoder = sklearn.preprocessing.OneHotEncoder()
encoder.fit(train)
encoded_train = encoder.transform(train)
#AdaBoost Model
model = DecisionTreeClassifier(max_depth=2)
AdaBoost = AdaBoostClassifier(base_estimator=model, n_estimators=400, learning_rate=1)

boostmodel = AdaBoost.fit(encoded_train, labels_train)
# Calculating Accuracy
y_pred = boostmodel.predict(encoder.transform(test))
predictions = metrics.accuracy_score(labels_test, y_pred)
#Calculating the accuracy in percentage
print('The accuracy is: ', predictions * 100, '%')
#Lime Explainer
np.random.seed(1)
explainer = lime.lime_tabular.LimeTabularExplainer(train ,class_names=['edible', 'poisonous'], feature_names = feature_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3, verbose=False)
predict_fn = lambda x: AdaBoost.predict_proba(encoder.transform(x))

# Example
i = 650
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook()