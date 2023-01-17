# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.tree import DecisionTreeClassifier

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

import shap

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/disease-prediction-using-machine-learning/Training.csv")
data=data.drop(columns=['Unnamed: 133'])
y=data['prognosis']
X=data.drop(columns=['prognosis'])
data.columns
X.columns[X.isna().any()].tolist()

X.head(3)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=5,test_size=0.25)
len(train_X),len(X)
random_f = RandomForestClassifier(n_estimators=50,max_depth=5,random_state=0).fit(train_X, train_y)
y_pred=random_f.predict(val_X)
accuracy_score(val_y, y_pred)

perm = PermutationImportance(random_f, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

feature_names = [i for i in X.columns if X[i].dtype in [np.int64]]
len(feature_names)
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=random_f, dataset=val_X, model_features=feature_names, feature='weight_loss')

# plot it
pdp.pdp_plot(pdp_goals, 'weight_loss')
plt.show()
data.prognosis.unique()[0],data.prognosis.unique()[24],data.prognosis.unique()[28],data.prognosis.unique()[36]
row_to_show = 8
nonzero=0
newi=0
for i in range(1230):
    data_for_prediction = val_X.iloc[i]  # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    x=random_f.predict_proba(data_for_prediction_array)
    if np.count_nonzero(x)>nonzero:
        nonzero=np.count_nonzero(x)
        newi=i
        print(nonzero)
        print(i)
data_for_prediction = val_X.iloc[0]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
random_f.predict_proba(data_for_prediction_array)

# Create object that can calculate shap values
explainer = shap.TreeExplainer(random_f)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
explainer = shap.TreeExplainer(random_f)
shap_values = explainer.shap_values(val_X,check_additivity=False)
shap.summary_plot(shap_values[1], val_X)

tree_model = DecisionTreeClassifier(random_state=0, max_depth=10).fit(train_X, train_y)
y_pred=tree_model.predict(val_X)
accuracy_score(val_y, y_pred)



values=np.unique(train_y)
values[:5]
Y_train=np.zeros(shape=(len(train_y),len(values)))
k=0
for x in train_y:
     for i in range(len(values)):
            if x==values[i]:
                tmp=list(np.zeros(41))
                tmp[i]=1
                Y_train[k]=tmp
                k+=1

Y_train[0]
model = Sequential()
model.add(Dense(16, input_dim=132))
model.add(Activation('tanh'))
model.add(Dense(41))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=sgd)

model.fit(train_X, Y_train, batch_size=150, epochs=30,validation_split = 0.2)
test_values=np.unique(val_y)
test_values

Y_test=np.zeros(shape=(len(val_y),len(test_values)))
k=0
for x in val_y:
     for i in range(len(test_values)):
            if x==test_values[i]:
                tmp=list(np.zeros(41))
                tmp[i]=1
                Y_test[k]=tmp
                k+=1
pre=model.predict_proba(val_X)
acc=0;
for i in range(len(pre)):
    if pre[i].argmax() == Y_test[i].argmax() :
        if pre[i].max() >= 0.6:
            acc+=1

acc=acc/len(pre)
acc

