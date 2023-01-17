import pandas as pd

import numpy as np 

import seaborn as sb

from IPython.display import display

import matplotlib.pyplot as plt

data = pd.read_csv("../input/Churn_Modelling.csv")

overview = data.head(10)

target = data['Exited']

display(overview)
X = data.iloc[:,3:13]

target = np.array(target)

display(X.head())
target_0 = data[data['Exited'] == 0]['Exited'].count()

target_1 = data[data['Exited']== 1]['Exited'].count()

print(target_0,target_1)
%matplotlib inline

labels = [0,1]

plt.bar(labels[0],target_0, width=0.1,color = 'red',edgecolor='yellow')

plt.bar(labels[1],target_1,width=0.1,color = 'black',edgecolor='yellow')

plt.legend()
data.info()
data.describe()
fig,axis = plt.subplots(figsize=(8,6))

axis = sb.heatmap(data=data.corr(method='pearson',min_periods=1),annot=True,cmap="YlGnBu")
from itertools import chain

countmale = data[data['Gender']=='Male']['Gender'].count()

countfemale=data[data['Gender']=='Female']['Gender'].count()    

fig,aix = plt.subplots(figsize=(8,6))

#print(countmale)

#print(countfemale)

aix = sb.countplot(hue='Exited',y='Geography',data=data)
cal= data[data['IsActiveMember']==1].count()

cal2 = data[data['Exited']==1].count()

ave = (cal2/(cal+cal2))*100

va= '%.1f '  % ave[1]

print(va+'%')
age = np.array(data['Age'])

fig,axis = plt.subplots(figsize=(8,6))

axis = sb.distplot(age,kde=False,bins=200)
axis = sb.jointplot(x='Age',y='Exited',data = data)
g = sb.FacetGrid(data,hue = 'Exited')

(g.map(plt.hist,'Age',edgecolor="w").add_legend())
array1 = np.array(data['IsActiveMember'])

array2 = np.array(data['Exited'])

index = len(array1)

count = 0

for i in range(index):

    if(array1[i]==1 and array2[i]==1):

        count +=1

print(count)
France = float(data[data['Geography']=='France']['Geography'].count())

Spain = float(data[data['Geography']=='Spain']['Geography'].count())

Germany = float(data[data['Geography']=='Germany']['Geography'].count())

print(France+Spain+Germany)
import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)





data = dict(type='choropleth',

           locations=['ESP','FRA','DEU'],

           colorscale='YlGnBu',

           text = ['Spain','France','Germany'],

           z=[France,Spain,Germany],

           colorbar={'title':'number in each geography'})

layout = dict(title='Counting the numbers of each nationality',

              geo=dict(showframe=False,projection={'type':'natural earth'}))

choromap = go.Figure(data=[data],layout=layout)

iplot(choromap)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label = LabelEncoder()

X['Geography'] = label.fit_transform(X['Geography'])

X['Gender'] = label.fit_transform(X['Gender'])

print(X['Gender'].head(7))
onehotencoding = OneHotEncoder(categorical_features = [1])

X = onehotencoding.fit_transform(X).toarray()

print(X)
from sklearn.model_selection import train_test_split



train_x,test_x,train_y,test_y = train_test_split(X,target,test_size=0.25,random_state=42)
from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import ClusterCentroids

from imblearn.combine import SMOTEENN

from collections import Counter

go = RandomOverSampler(random_state=42)

train_x_resample,train_y_resample = go.fit_resample(train_x,train_y)

# before resampling the number of each catogorical

print(Counter(train_y).items())

# After resampling the number of each catogorical

print(Counter(train_y_resample).items())

# now let use under sampling 

go1 = ClusterCentroids(random_state=0)

train_x_resample1,train_y_resample1 = go1.fit_resample(train_x,train_y)

# before resampling the number of each catogorical

print(Counter(train_y).items())

# After resampling the number of each catogorical

print(Counter(train_y_resample1).items())

# now let combine two over and under resample

go2 = SMOTEENN(random_state=0) 

train_x_resample2,train_y_resample2 = go2.fit_resample(train_x,train_y)

# before resampling the number of each catogorical

print(Counter(train_y).items())

# After resampling the number of each catogorical

print(Counter(train_y_resample2).items())
from sklearn.metrics import accuracy_score,recall_score,f1_score,cohen_kappa_score,precision_score

from time import *

def choose_best(model, train_x , train_y , test_x , test_y):

    result = {}

    

    #for calculate time of fitting data

    start = time()

    model.fit(train_x,train_y)

    end = time()

    result['train_time'] = end-start

    

    #for prediction

    

    start = time()

    test_y_new = model.predict(test_x)

    train_y_new = model.predict(train_x)

    end = time()

    

    result["prediction_time"] = end - start

    

    result['acc_prediction_train'] = accuracy_score(train_y,train_y_new)

    result['recall_prediction_train'] = recall_score(train_y,train_y_new)

    result['f1_score_test'] = f1_score(test_y,test_y_new)

    result['recall_prediction_test'] = recall_score(test_y,test_y_new)

    result['cohen_kappa_score'] = cohen_kappa_score(test_y,test_y_new)

    result['precision_score'] = precision_score(test_y,test_y_new)

    print('name of model {}'.format(model))

    

    return result

    
from sklearn.linear_model import LogisticRegression



classifier_1 = LogisticRegression(random_state = 42,solver='lbfgs')

values1 = choose_best(classifier_1,train_x_resample,train_y_resample,test_x,test_y)

values1n = choose_best(classifier_1,train_x_resample1,train_y_resample1,test_x,test_y)

values1nn = choose_best(classifier_1,train_x_resample2,train_y_resample2,test_x,test_y)
from sklearn.ensemble import AdaBoostClassifier



classifier_2 = AdaBoostClassifier(random_state=42)

values2 = choose_best(classifier_2,train_x_resample,train_y_resample,test_x,test_y)

values2n = choose_best(classifier_2,train_x_resample1,train_y_resample1,test_x,test_y)

values2nn = choose_best(classifier_2,train_x_resample2,train_y_resample2,test_x,test_y)
from sklearn.ensemble import GradientBoostingClassifier



classifier_3 = GradientBoostingClassifier()

values3 = choose_best(classifier_3,train_x_resample,train_y_resample,test_x,test_y)

values3n = choose_best(classifier_3,train_x_resample1,train_y_resample1,test_x,test_y)

values3nn = choose_best(classifier_3,train_x_resample2,train_y_resample2,test_x,test_y)
from sklearn.ensemble import RandomForestClassifier

classifier_4 = RandomForestClassifier(n_estimators=100) #warning 10 to 100

values4 = choose_best(classifier_4,train_x_resample,train_y_resample,test_x,test_y)

values4n = choose_best(classifier_4,train_x_resample1,train_y_resample1,test_x,test_y)

values4nn = choose_best(classifier_4,train_x_resample2,train_y_resample2,test_x,test_y)
moduels = pd.DataFrame({'name_model':["logistic regression","adaboost","gradient boost","random forest"],\

                       'accuracy_training':[values1["acc_prediction_train"],values2['acc_prediction_train'],values3['acc_prediction_train'],values4['acc_prediction_train']],\

                       "recall_testing":[values1["recall_prediction_test"],values2["recall_prediction_test"],values3["recall_prediction_test"],values4["recall_prediction_test"]],\

                        "f1_score":[values1["f1_score_test"],values2["f1_score_test"],values3["f1_score_test"],values4["f1_score_test"]],\

                        "precision_test":[values1["precision_score"],values2["precision_score"],values3["precision_score"],values4["precision_score"]],\

                        "kappa_score":[values1["cohen_kappa_score"],values2["cohen_kappa_score"],values3["cohen_kappa_score"],values4["cohen_kappa_score"]],\

                        "timing_train":[values1["train_time"],values2["train_time"],values3["train_time"],values4["train_time"]],\

                       "timing_test":[values1["prediction_time"],values2["prediction_time"],values3["prediction_time"],values4["prediction_time"]]})

moduels.sort_values(by =["f1_score"], ascending = False)
moduels = pd.DataFrame({'name_model':["logistic regression","adaboost","gradient boost","random forest"],\

                       'accuracy_training':[values1n["acc_prediction_train"],values2n['acc_prediction_train'],values3n['acc_prediction_train'],values4n['acc_prediction_train']],\

                       "recall_testing":[values1n["recall_prediction_test"],values2n["recall_prediction_test"],values3n["recall_prediction_test"],values4n["recall_prediction_test"]],\

                        "f1_score":[values1n["f1_score_test"],values2n["f1_score_test"],values3n["f1_score_test"],values4n["f1_score_test"]],\

                        "kappa_score":[values1n["cohen_kappa_score"],values2n["cohen_kappa_score"],values3n["cohen_kappa_score"],values4n["cohen_kappa_score"]],\

                        "precision_test":[values1n["precision_score"],values2n["precision_score"],values3n["precision_score"],values4n["precision_score"]],\

                        "timing_train":[values1n["train_time"],values2n["train_time"],values3n["train_time"],values4n["train_time"]],\

                       "timing_test":[values1n["prediction_time"],values2n["prediction_time"],values3n["prediction_time"],values4n["prediction_time"]]})

moduels.sort_values(by =["f1_score"], ascending = False)
moduels = pd.DataFrame({'name_model':["logistic regression","adaboost","gradient boost","random forest"],\

                       'accuracy_training':[values1nn["acc_prediction_train"],values2nn['acc_prediction_train'],values3nn['acc_prediction_train'],values4nn['acc_prediction_train']],\

                       "recall_testing":[values1nn["recall_prediction_test"],values2nn["recall_prediction_test"],values3nn["recall_prediction_test"],values4nn["recall_prediction_test"]],\

                        "f1_score":[values1nn["f1_score_test"],values2nn["f1_score_test"],values3nn["f1_score_test"],values4nn["f1_score_test"]],\

                        "kappa_score":[values1nn["cohen_kappa_score"],values2nn["cohen_kappa_score"],values3nn["cohen_kappa_score"],values4nn["cohen_kappa_score"]],\

                        "precision_test":[values1nn["precision_score"],values2nn["precision_score"],values3nn["precision_score"],values4nn["precision_score"]],\

                        "timing_train":[values1nn["train_time"],values2nn["train_time"],values3nn["train_time"],values4nn["train_time"]],\

                       "timing_test":[values1nn["prediction_time"],values2nn["prediction_time"],values3nn["prediction_time"],values4nn["prediction_time"]]})

moduels.sort_values(by =["f1_score"], ascending = False)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import fbeta_score,make_scorer,classification_report,confusion_matrix,roc_auc_score

parameters = [{'loss':['deviance'],'learning_rate':[0.1,0.2,0.3,0.4],'n_estimators':[50,100],

              'max_depth':[3,6,10,15]}]

scorer = make_scorer(fbeta_score,beta=0.5)

grid_search =  GridSearchCV(estimator = classifier_3, param_grid  = parameters, scoring = scorer ,cv = 5)

grid_fit = grid_search.fit(train_x_resample,train_y_resample)

best_accuracy = grid_fit.best_score_

best_para = grid_fit.best_params_

best_clas = grid_fit.best_estimator_

prdict_y  = best_clas.predict(test_x)

score = fbeta_score(test_y,prdict_y,beta=0.5)

print(best_accuracy,best_para,score)
confusionMatrix = confusion_matrix(test_y,prdict_y)

sb.heatmap(confusionMatrix,annot=True,fmt='d')
print(classification_report(test_y,prdict_y))
roc = roc_auc_score(test_y,prdict_y)

print(roc)