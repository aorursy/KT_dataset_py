import sys

!{sys.executable} -m pip install sklearn
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import svm
train_data_origin = pd.read_csv("../input/data-mining-assignment-2/train.csv")

test_data_origin = pd.read_csv("../input/data-mining-assignment-2/test.csv")



train_data = train_data_origin.iloc[:,:-1]

train_class = train_data_origin.iloc[:,-1]

test_data = test_data_origin
train_data
for col in train_data:

    if train_data[col].dtypes==object:

        train_data.drop({col},axis=1,inplace=True)

        test_data.drop({col},axis=1,inplace=True)
train_data
test_data
train_data.drop('ID',1,inplace=True)

test_data.drop('ID',1,inplace=True)
#Standard scaler on seperate



from sklearn import preprocessing

#Performing Standard Normalization

standard_scaler = preprocessing.StandardScaler()

np_scaled = standard_scaler.fit_transform(train_data)

train_data_N = pd.DataFrame(np_scaled)

train_data_N



from sklearn import preprocessing

#Performing Standard Scalar Normalization

standard_scaler = preprocessing.StandardScaler()

np_scaled = standard_scaler.fit_transform(test_data)

test_data_N = pd.DataFrame(np_scaled)

test_data_N
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data_N, train_class, test_size=0.20, random_state=42)
from sklearn.utils.class_weight import compute_class_weight

classwt=compute_class_weight("balanced",[0,1,2,3],train_class)

print(classwt)
wtd={0:0.77777778, 1:3.80434783, 2:0.79545455, 3:0.83732057}
from sklearn.ensemble import RandomForestClassifier



score_train_RF = []

score_test_RF = []



for i in range(5,50,2):

    rf = RandomForestClassifier(n_estimators = 100, max_depth=i, bootstrap = False, class_weight= wtd, criterion='entropy')

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_test,y_test)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))



test_score,=plt.plot(range(5,50,2),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [test_score],["Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score





rf_temp = RandomForestClassifier(n_estimators = 100, class_weight=wtd, bootstrap = False,criterion = 'entropy')        #Initialize the classifier object

parameters = {'max_depth':[6, 9],'min_samples_split':[5,6,7,8,9,10]}    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf = RandomForestClassifier(n_estimators = 1000, max_depth=9, min_samples_split=6, bootstrap = False, class_weight= wtd, criterion='entropy')

rf.fit(X_train, y_train)

rf.score(X_test, y_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred = rf.predict(X_test)

confusion_matrix(y_test, y_pred)
rf = RandomForestClassifier(n_estimators=1000, max_depth =9 , min_samples_split=6, class_weight=wtd,bootstrap=False,criterion ='entropy')

rf.fit(train_data_N, train_class)

test_class = rf.predict(test_data_N)
arr_final = []

for idx,pred in enumerate(test_class):

    arr_final.append((idx+700,pred))

arr_final
df = pd.DataFrame(arr_final, columns =['ID', 'Class'])

df
#export_csv = df.to_csv(r'C:/Users/SV/Desktop/data mining/file10.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(df)