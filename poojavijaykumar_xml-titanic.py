#####################################################################################################
## This notebook contains 5 popular techniques for XML using the Titanic dataset from kaggle

## Make sure to install the following before running the notebook:
### pip install LIME
### pip install shap
### pip install pdpbox
### pip install deeplift

#####################################################################################################
## Pooja Vijaykumar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier

import lime
import lime.lime_tabular

import shap
from xgboost import XGBClassifier

from pdpbox import pdp

from keras.models import Sequential
from keras.layers import Dense, Dropout 
from keras.models import model_from_json

import deeplift
from deeplift.layers import NonlinearMxtsMode
import deeplift.conversion.kerasapi_conversion as kc
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
train.isnull().sum()
med1 = train.Age.median()
med2 = test.Age.median()
train['Age'] = train['Age'].map(lambda x: med1 if np.isnan(x) else x)
test['Age'] = test['Age'].map(lambda x: med2 if np.isnan(x) else x)
mode = train.Embarked.value_counts()[0]
train['Embarked'] = train['Embarked'].map(lambda x: mode if pd.isnull(x) else x)
train = train.dropna()
test = test.dropna()
train.head()
train = train.join(pd.get_dummies(train['Sex'], prefix='Sex'))
train = train.join(pd.get_dummies(train['Pclass'], prefix='Pclass'))
train = train.join(pd.get_dummies(train['Embarked'], prefix='Embarked'))
train = train.join(pd.get_dummies(train['SibSp'], prefix='SibSp'))
train = train.join(pd.get_dummies(train['Parch'], prefix='Parch'))
rem = ['PassengerId','Name','Ticket','Cabin','Pclass','Sex','Embarked','SibSp','Parch']

for i in train.columns:
    if i in rem:
        train = train.drop(i, axis=1)
train.head()
test.head()
test = test.join(pd.get_dummies(test['Sex'], prefix='Sex'))
test = test.join(pd.get_dummies(test['Pclass'], prefix='Pclass'))
test = test.join(pd.get_dummies(test['Embarked'], prefix='Embarked'))
test = test.join(pd.get_dummies(test['SibSp'], prefix='SibSp'))
test = test.join(pd.get_dummies(test['Parch'], prefix='Parch'))
rem = ['PassengerId','Name','Ticket','Cabin','Pclass','Sex','Embarked','SibSp','Parch']

for i in test.columns:
    if i in rem:
        test = test.drop(i, axis=1)
xtrain = train.drop('Survived',axis=1)
ytrain = train['Survived']
xtest = test
######################################################
rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1).fit(xtrain, ytrain)
## Group all model features and obtain their importance
## In this case, the Gini importance is used
## Gini importance = total decrease in node impurity averaged over all the trees in the ensemble 

feats = {}

for x,y in zip(xtrain.columns, rfc.feature_importances_):
    feats[x] = y
## sort features based on Gini importance in descending order

imp = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0:'Gini-importance'})
imp_sort = imp.sort_values(by = 'Gini-importance', ascending=False)
imp_sort
## plot features and their Gini importance

x = imp_sort.index
y = []
for i in range(0,len(imp_sort)):
    y.append(imp_sort['Gini-importance'][i])

plt.figure(figsize=(15,7))
bar_plot = plt.bar(x,y)
plt.xticks(rotation=90)
plt.title('Feature Importance using Gini Index')
plt.xlabel('Features')
plt.ylabel('Gini-importance')
plt.show()
#########################################################################
## Extract all the features from your data and place them in different lists based on their type 
## (float in one list, int64 in another)

fl = []
inf = []

for i in xtrain.columns:
    if xtrain[i].dtype=='float64':
        fl.append(i)
    elif xtrain[i].dtype=='uint8':
        inf.append(i)
        
train_fl = xtrain[fl]
train_int = xtrain[inf]
## concatenate the lists
feats = list(train_fl) + list(train_int)
predict_fn_rf = lambda x: rfc.predict_proba(x).astype(float)
explainer = lime.lime_tabular.LimeTabularExplainer(xtrain[feats].astype(int).values, 
                                                   mode='classification',
                                                   training_labels=train['Survived'],
                                                   feature_names=feats)
## explanation for index 0 whose output is class 1
exp = explainer.explain_instance(xtest.iloc[0], predict_fn_rf, num_features=19)
exp.show_in_notebook(show_all=False)
## explanation for index 5 whose output is class 0
exp = explainer.explain_instance(xtest.iloc[5], predict_fn_rf, num_features=19)
exp.show_in_notebook(show_all=False)
###############################################################################
xtrain.columns
## CAUTION:
## rename columns with special chars ('<,>') because XGB doesn't take special chars in feature names
## do this to avoid basic syntax errors
xgb = XGBClassifier().fit(xtrain, ytrain)
## load JS visalization for SHAP plots
shap.initjs()
## create SHAP Explainer
## The TreeExplainer from the SHAP library is optimized to trace through the XGBoost tree to 
## find the Shapley value estimates of the features

explainer = shap.TreeExplainer(xgb)
## obtain the Shapley values
shap_values = explainer.shap_values(xtest)
## SHAP plot for xtest[0] where output is '1'
shap.force_plot(explainer.expected_value, 
                shap_values[0], 
                features=xtest.iloc[[0]], 
                feature_names=xtest.columns, 
                link='logit')
## SHAP plot for xtest[5] where output is '0'
shap.force_plot(explainer.expected_value, 
                shap_values[5], 
                features=xtest.iloc[[5]], 
                feature_names=xtest.columns, 
                link='logit')
# global view
shap.summary_plot(shap_values, feature_names=xtest.columns, features=xtest)
#######################################################################
## PDP for single feature - enter desired feature
## pdp_plot plots the feature impact

ff = 'Age'
pdp_feat = pdp.pdp_isolate(model=rfc, dataset=test, model_features=xtest.columns,feature=ff)
fig, axes = pdp.pdp_plot(pdp_feat,ff)
## Interaction between two features
## PDP for feature combination -- enter two features
## pdp_interact_plot plots the feature interaction 
## changing x_quantile and plot_pdp between True and False gives you two types of plots but with the same depiction

f1 = 'Age'
f2 = 'Sex_female'

inter = pdp.pdp_interact(model = rfc, dataset=test, model_features=xtest.columns, features=[f1,f2])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter,feature_names=[f1,f2] ,plot_type='contour',
                                 x_quantile=False, plot_pdp=False)
## give last two (x_quantile and plot_pdp) as True for different type of contour plot
f1 = 'Age'
f2 = 'Sex_female'
inter1 = pdp.pdp_interact(model=rfc, dataset=test, model_features=xtest.columns, features=[f1, f2])
fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=inter1, feature_names=[f1, f2], plot_type='contour', x_quantile=True, plot_pdp=True)
#########################################################################
model = Sequential()

model.add(Dense(200, activation='relu', input_dim=19))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))

model.add(Dense(1,activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=20)
## save model to json file
model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
## save model weights to HDF5
model.save_weights('model.h5')
keras_json = 'model.json'
keras_weights = 'model.h5'
km = model_from_json(open(keras_json).read())
km.load_weights(keras_weights)
## convert Keras Sequential model to DeepLIFT model
deepmodel = kc.convert_model_from_saved_files(
                    json_file = keras_json, 
                    h5_file = keras_weights, 
                    nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.GuidedBackprop)
deepmodel.get_layers()
## Specify the index of the layer to compute the importance scores of.
## In the example below, we find scores for the input layer, which is idx 0 in deeplift_model.get_layers()
find_scores_layer_idx = 0
## Compile the function that computes the contribution scores
## target_layer_idx= -1 (regression), -2 (sigmoid/softmax outputs)
deeplift_contribs_func = deepmodel.get_target_contribs_func(
                            find_scores_layer_idx=find_scores_layer_idx,
                            target_layer_idx=-2)
## choose the input row whose Contribution Score is to be observed
idx = 0
## obtain the scores
scores = np.array(deeplift_contribs_func(task_idx=0,
                                         input_data_list=[xtest.iloc[[0]]],
                                         batch_size=10,
                                         progress_update=1000))
scores
## flatten the list
flat_scores = [item for sublist in scores for item in sublist]
## Group all model features with their importance
feats = {}

for x,y in zip(xtest.columns, flat_scores):
    feats[x] = y
imp = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0:'Contribution Score'})
imp

## can sort it as well:
#imp_sort = imp.sort_values(by = 'Contribution Score', ascending=False)
#imp_sort