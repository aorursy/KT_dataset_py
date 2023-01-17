# Any results you write to the current directory are saved as output.

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

# Loading toy dataset

iris_data = pd.read_csv("../input/iris/Iris.csv")



# Overview of dataset using pandas-profiling.

profile = ProfileReport(iris_data.drop(["Id"],axis = 1), title='Pandas Profiling Report',minimal=False, html={'style':{'full_width':True}})

profile.to_widgets()
# Spiliting

train, test, labels_train, labels_test = train_test_split(iris_data.drop(['Id','Species'],axis=1).values, iris_data.Species, train_size=0.75)



# Modeling usimg Random forest model

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500,random_state=1234)

rf.fit(train, labels_train)



# Accuracy

sklearn.metrics.accuracy_score(labels_test, rf.predict(test))
# Explainer initialise

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris_data.drop(['Id','Species'],axis=1).columns, class_names= iris_data.Species.unique(), discretize_continuous=True)



# Explaing random instance using LIME explainer 

i = np.random.randint(0, test.shape[0])

exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)

exp.show_in_notebook(show_table=True, show_all=False)
# Loading toy dataset

data = pd.read_csv("../input/default-of-credit-card-clients/default of credit card clients.csv")



print(data.shape)

data.head()
# Spiliting

train, test, labels_train, labels_test = train_test_split(data.drop(['ID','dpnm'],axis=1).values, data.dpnm, train_size=0.75)



# Modeling usimg Random forest model

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500,random_state=1234)

rf.fit(train, labels_train)

predict_test = rf.predict(test)



# Accuracy

sklearn.metrics.accuracy_score(labels_test, predict_test)
# Explainer initialise

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=data.drop(['ID','dpnm'],axis=1).columns, class_names= ['0','1'], discretize_continuous=True)

data.dpnm.unique()
# Finding index of misclassified datapoints

np.where(labels_test!=predict_test)[0]


# Explaing random instance using LIME explainer 

idx = 4

exp = explainer.explain_instance(test[idx], rf.predict_proba, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print(f"Instance Datapoint\n{'-'*20}\n{test[idx]}\n")

print('Probability(0) =',rf.predict_proba([test[idx]])[0,0])

print('Probability(1) =',rf.predict_proba([test[idx]])[0,1])

print('True class: %s' % labels_test.iloc[idx])



exp.show_in_notebook(show_table=True, show_all=False)
# Explaing random instance using LIME explainer 

idx = 18

exp = explainer.explain_instance(test[idx], rf.predict_proba, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print(f"Instance Datapoint\n{'-'*20}\n{test[idx]}\n")

print('Probability(0) =',rf.predict_proba([test[idx]])[0,0])

print('Probability(1) =',rf.predict_proba([test[idx]])[0,1])

print('True class: %s' % labels_test.iloc[idx])



exp.show_in_notebook(show_table=True, show_all=False)
# Explaing random instance using LIME explainer 

idx = 20

exp = explainer.explain_instance(test[idx], rf.predict_proba, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print(f"Instance Datapoint\n{'-'*20}\n{test[idx]}\n")

print('Probability(0) =',rf.predict_proba([test[idx]])[0,0])

print('Probability(1) =',rf.predict_proba([test[idx]])[0,1])

print('True class: %s' % labels_test.iloc[idx])



exp.show_in_notebook(show_table=True, show_all=False)
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
# train test split

data = data.astype(float)

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)



# one hotEncoding

encoder = sklearn.preprocessing.OneHotEncoder()

encoder.fit(train)

encoded_train = encoder.transform(train)



# Modeling

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500,random_state=1234)

rf.fit(encoded_train, labels_train)

predict_fn = lambda x: rf.predict_proba(encoder.transform(x))



# Accuracy

presict_test = rf.predict(encoder.transform(test))

sklearn.metrics.accuracy_score(labels_test, rf.predict(encoder.transform(test)))

np.random.seed(1)

explainer = lime.lime_tabular.LimeTabularExplainer(train ,class_names=['edible', 'poisonous'], feature_names = feature_names,

                                                   categorical_features=categorical_features, 

                                                   categorical_names=categorical_names, kernel_width=3, verbose=False)
# finding index of misclassified points.

np.where(labels_test!=presict_test)[0]
i = 22

exp = explainer.explain_instance(test[i], predict_fn, num_features=5)

exp.show_in_notebook()
# Dataset Loading

data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head()
# Data overview

print ("Shape     : " ,data.shape)

print ("\nFeatures : \n" ,data.columns.tolist())

print ("\nMissing values :  ", data.isnull().sum().values.sum())

print ("\nUnique values :  \n",data.nunique())
# Preprocessing

totalcharges_float = []

null_idx = []

for idx,val in enumerate(data['TotalCharges'].values):

    try:

        totalcharges_float.append(np.float(val))

    

    except:

        null_idx.append(idx)

        

data = data.drop(labels=data.iloc[null_idx].index, axis=0)

data['TotalCharges'] = data['TotalCharges'].values.astype(float)

data.drop('customerID',axis=1,inplace=True)
labels = data['Churn']



le= sklearn.preprocessing.LabelEncoder()

le.fit(labels)

labels = le.transform(labels)

class_names = le.classes_

data = data.drop('Churn',axis=1)

pd.DataFrame(data)
categorical_features = range(data.shape[1]-2)



feature_names = data.columns.values



categorical_names = {}

for feature in categorical_features:

    le = sklearn.preprocessing.LabelEncoder()

    le.fit(data.iloc[:, feature])

    data.iloc[:, feature] = le.transform(data.iloc[:, feature])

    categorical_names[feature] = le.classes_

    

# data after label encoding

print(feature_names)

(categorical_names)
# train test split

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data , labels, train_size=0.80)



# one hotEncoding

encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')

encoder.fit(train.iloc[:,categorical_features])

encoded_train = encoder.transform(train.iloc[:,categorical_features])

encoded_test= encoder.transform(test.iloc[:,categorical_features])



train_concat = np.hstack((encoded_train.toarray(),train.iloc[:,-2:]))



rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500,random_state=1234)

rf.fit(train_concat, labels_train)

# predict function

predict_fn = lambda x: rf.predict(np.hstack((encoder.transform(x[:,categorical_features]).toarray(), x[:,-2:])))



# predict probability function

predict_proba_fn = lambda x: rf.predict_proba(np.hstack((encoder.transform(x[:,categorical_features]).toarray(), x[:,-2:])))

                                  

                              

# Acuuracy

print("Accuracy :")

sklearn.metrics.accuracy_score(labels_test, predict_fn(test.values))

# Explainer initialise

class_name = ['no','yes']

explainer = lime.lime_tabular.LimeTabularExplainer(train.values, class_names= class_names, feature_names= feature_names,

                                                   categorical_features= categorical_features, 

                                                   categorical_names= categorical_names, kernel_width=3, verbose=False)
# misclassified points

np.where(labels_test!=predict_fn(test.values))[0]
# Explaing random instance using LIME explainer 



idx = 19

exp = explainer.explain_instance(test.values[idx],  predict_proba_fn, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print('Probability(0) =',predict_proba_fn(test.values[idx].reshape(1,-1))[0,0])

print('Probability(1) =',predict_proba_fn(test.values[idx].reshape(1,-1))[0,1])

print('True class: %s' % labels_test[idx])



exp.show_in_notebook(show_table=True, show_all=False)
# Explaing random instance using LIME explainer 



idx = 40

exp = explainer.explain_instance(test.values[idx],  predict_proba_fn, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print('Probability(0) =',predict_proba_fn(test.values[idx].reshape(1,-1))[0,0])

print('Probability(1) =',predict_proba_fn(test.values[idx].reshape(1,-1))[0,1])

print('True class: %s' % labels_test[idx])



exp.show_in_notebook(show_table=True, show_all=False)
# Explaing random instance using LIME explainer 



idx = 51

exp = explainer.explain_instance(test.values[idx],  predict_proba_fn, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print('Probability(0) =',predict_proba_fn(test.values[idx].reshape(1,-1))[0,0])

print('Probability(1) =',predict_proba_fn(test.values[idx].reshape(1,-1))[0,1])

print('True class: %s' % labels_test[idx])



exp.show_in_notebook(show_table=True, show_all=False)
from xgboost import XGBClassifier



# Modeling

classifier = XGBClassifier()

classifier.fit(train_concat, labels_train)

# predict function

predict_fn = lambda x: classifier.predict(np.hstack((encoder.transform(x[:,categorical_features]).toarray(), x[:,-2:])))



# predict probability function

predict_proba_fn = lambda x: classifier.predict_proba(np.hstack((encoder.transform(x[:,categorical_features]).toarray(), x[:,-2:])))

                                                            

# Acuuracy

print("Accuracy :")

sklearn.metrics.accuracy_score(labels_test, predict_fn(test.values))

# Explainer initialise

class_name = ['no','yes']

explainer = lime.lime_tabular.LimeTabularExplainer(train.values, class_names= class_names, feature_names= feature_names,

                                                   categorical_features= categorical_features, 

                                                   categorical_names= categorical_names, kernel_width=3, verbose=False)
# mis classified points

np.where(labels_test!=predict_fn(test.values))[0]
idx = 3

exp = explainer.explain_instance(test.values[i],predict_proba_fn, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print('Probability(0) =',predict_proba_fn(test.values[idx].reshape(1,-1))[0,0])

print('Probability(1) =',predict_proba_fn(test.values[idx].reshape(1,-1))[0,1])

print('True class: %s' % labels_test[idx])



exp.show_in_notebook(show_table=True, show_all=False)
idx = 1404

exp = explainer.explain_instance(test.values[i],predict_proba_fn, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print('Probability(0) =',predict_proba_fn(test.values[i].reshape(1,-1))[0,0])

print('Probability(1) =',predict_proba_fn(test.values[i].reshape(1,-1))[0,1])

print('True class: %s' % labels_test[idx])



exp.show_in_notebook(show_table=True, show_all=False)
idx = 1406

exp = explainer.explain_instance(test.values[i],predict_proba_fn, num_features=5, top_labels=1)



print('\nDocument id: %d' % idx)

print('Probability(0) =',predict_proba_fn(test.values[i].reshape(1,-1))[0,0])

print('Probability(1) =',predict_proba_fn(test.values[i].reshape(1,-1))[0,1])

print('True class: %s' % labels_test[idx])



exp.show_in_notebook(show_table=True, show_all=False)