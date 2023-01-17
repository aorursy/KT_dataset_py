import numpy as np

import pandas as pd

import matplotlib.pyplot as pl

%matplotlib inline
#train_data_original = pd.read_csv("/kaggle/input/data-mining-assignment-2/train.csv")

#test_data_original = pd.read_csv("/kaggle/input/data-mining-assignment-2/test.csv")

train_data_original = pd.read_csv("../input/data-mining-assignment-2/train.csv")

test_data_original = pd.read_csv("../input/data-mining-assignment-2/test.csv")

train_data = train_data_original

test_data = test_data_original
cat_columns = list(train_data.select_dtypes(include=['object']).columns)

train_data['Class'].value_counts()
#One-hot encoding data and shifting 'Class' column to the end

train_data = pd.get_dummies(train_data, columns=cat_columns)

test_data = pd.get_dummies(test_data, columns=cat_columns)

class_col = train_data['Class']

train_data.drop(labels=['Class','ID'],axis=1,inplace=True)



train_data.insert(72,'Class',class_col)
train_data.corr()['Class'].abs().sort_values(ascending=False)[:20]
train_data_X = train_data.iloc[:,:-1]

train_data_Y = train_data.iloc[:,-1]

test_data_X = test_data.iloc[:,1:]
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [5, 8, 10, 12, 14, 16, 20],

    'n_estimators': [100, 200, 300, 1000],

    'class_weight': ['balanced']#,

#    'min_samples_split': [2, 3, 4]

}



rfc = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search.fit(train_data_X,train_data_Y)

best_grid = grid_search.best_estimator_

#rfc = rfc.fit(train_data_X,train_data_Y)

#APPLY SAME TRANSFORMATIONS ON TEST DATA FIRST?

rfc_pred = best_grid.predict(test_data_X) #Not including ID column from test data
from sklearn import model_selection

from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=4, random_state=42)

rfc_kfold = model_selection.cross_val_score(best_grid, train_data_X, train_data_Y, cv=skfold)

print("Accuracy with random forest: %.2f%%" % (rfc_kfold.mean()*100.0))
final_pred = pd.DataFrame(rfc_pred,index=test_data_original.loc[:,'ID'], columns=['Class'])

final_pred.to_csv("submission.csv")
from IPython.display import HTML 

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html) 

create_download_link(final_pred)