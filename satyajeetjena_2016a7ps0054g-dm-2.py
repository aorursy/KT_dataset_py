# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_orig = pd.read_csv("../input/train.csv", sep=',')

data1 = data_orig

data2 = pd.read_csv("../input/test.csv", sep=',')

data1 = data1.drop(["Class"], 1)

data = pd.concat([data1, data2])

# data.info()

data.head()
for col in data.columns:

    print (col)

    string = data.dtypes[col]

    if string == "object":

        print((data[col].value_counts()/data[col].count())*100)
data = data.drop(['Worker Class', 'Enrolled', 'MIC', 'MOC', 'Hispanic', 'ID'],1)

data = data.drop(['MLU', 'Reason', 'Area', 'State', 'MSA', 'COB FATHER', 'Fill'],1)

data = data.drop(['REG', 'MOVE', 'Live', 'PREV', 'Teen', 'COB MOTHER', 'COB SELF'],1)

data = data.drop(['Timely Income', 'Married_Life', 'Weight', 'Schooling'],1)



data = data.drop(["Full/Part", "Citizen", "Detailed"],1)

data = pd.get_dummies(data, columns=["Own/Self", "Vet_Benefits", 'Cast', "Sex", "Tax Status", "Summary"])
# d1 = data.loc[data['Class'] == 1]

# d2 = data.loc[data['Class'] == 0]

# d1.head(20)

# d2.head()

data.head()
data.duplicated().sum()
data['Sex_F'].unique()
import matplotlib.pyplot as plt

import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
y=data_orig['Class']

X = data.head(100000)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

X_train.head()
from sklearn.ensemble import RandomForestClassifier
score_train_RF = []

score_test_RF = []



for i in range(1,20,1):

    rf = RandomForestClassifier(n_estimators=i, random_state = 42, class_weight="balanced")

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,20,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,20,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=19, random_state = 42, class_weight="balanced")

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 19, class_weight = "balanced")        #Initialize the classifier object



parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators = 19, max_depth = 10, min_samples_split = 5, class_weight = "balanced")

rf_best.fit(X_train, y_train)

rf_best.score(X_val,y_val)
y_pred_RF_best = rf_best.predict(X_val)

confusion_matrix(y_val, y_pred_RF_best)
print(classification_report(y_val, y_pred_RF_best))
testset = data[100000:]

abcd = rf_best.predict(testset)

abcd
final = pd.DataFrame({"ID":data2["ID"],"Class":abcd})

final.to_csv("finalsub.csv", index=False)

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(final)