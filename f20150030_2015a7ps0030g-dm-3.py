import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig_0 = pd.read_csv("../input/opcode_frequency_benign.csv", sep=',')

data_0 = data_orig_0



data_orig_1 = pd.read_csv("../input/opcode_frequency_malware.csv", sep=',')

data_1 = data_orig_1



data_test_orig = pd.read_csv("../input/Test_data.csv",sep=',')

X_test = data_test_orig

X_test.info()
data_0['Class']=0
data_1['Class']=1
data_1.head()
df = data_0.append(data_1)
df.head()
df.tail()
df = df.drop(['FileName'], axis = 1)
y=df['Class']

X=df.drop(['Class'],axis=1)

X_test = X_test.drop(['FileName'], axis = 1)

X.head()

X_test.info()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
# X_test = X_test.sort_index(axis='columns')

X_test = X_test.drop(['Unnamed: 1809'], axis = 1)

X_test.head()

X_test.info()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

X_train.head()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_test)

X_test = pd.DataFrame(np_scaled)

#np_scaled_val = min_max_scaler.transform(X_val)

#X_val = pd.DataFrame(np_scaled_val)

X_test.head()

X_test.info()
from sklearn.ensemble import RandomForestClassifier
score_train_RF = []

score_test_RF = []



for i in range(1,18,1):

    rf = RandomForestClassifier(n_estimators=i, random_state = 42)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=9, random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score



rf_temp = RandomForestClassifier(n_estimators = 9)        #Initialize the classifier object



parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators = 9, max_depth = 10, min_samples_split = 2)

rf_best.fit(X_train, y_train)

rf_best.score(X_val,y_val)
y_pred_RF_best = rf_best.predict(X_val)

confusion_matrix(y_val, y_pred_RF_best)
print(classification_report(y_val, y_pred_RF_best))
y_pred_test_RF = rf_best.predict(X_test)

y_pred_test_RF

z_RF = y_pred_test_RF.tolist()

len(z_RF)

# X_test.info()
res1 = pd.DataFrame(z_RF)

final_RF = pd.concat([data_test_orig['FileName'], res1], axis=1).reindex()

final_RF = final_RF.rename(columns={0: 'Class'})

# final_RF['Class'] = final_RF.Class.astype(int)

final_RF.head(100)

res1.head(100)
final_RF.to_csv('submission_RF.csv', index = False,  float_format='%.f')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html='<a download="{filename}"href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final_RF)