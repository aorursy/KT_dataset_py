# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV File", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    

    html = html.format(payload=payload, title=title, filename=filename)

    return HTML(html)
train_data = pd.read_csv("/kaggle/input/data-mining-assignment-2/train.csv")
test_data = pd.read_csv("/kaggle/input/data-mining-assignment-2/test.csv")
from sklearn.preprocessing import StandardScaler
test_data_p = test_data.drop(["ID"], axis=1)

train_data_p = train_data.drop(["ID","Class"], axis=1)

for col in test_data_p.columns:

    if(test_data_p[col].dtype != object):

        scaler = StandardScaler()

        npscaled = scaler.fit_transform(train_data_p[col].values.reshape(-1,1))

        

        train_data_p[col] = npscaled

        

        npscaled = scaler.transform(test_data_p[col].values.reshape(-1,1))

        

        test_data_p[col] = npscaled
test_data_p = pd.get_dummies(test_data_p)

train_data_p = pd.get_dummies(train_data_p)
train_y = train_data["Class"]
train_data_shift = train_data_p.drop(["col56_Medium"],axis=1)

train_data_shift["col56_Medium"] = train_data_p["col56_Medium"]

train_data_shift["origin"] = 0

test_data_shift = test_data_p.drop(["col56_Medium"],axis=1)

test_data_shift["col56_Medium"] = test_data_p["col56_Medium"]

test_data_shift["origin"] = 1
training = train_data_shift.sample(450, random_state=12)

testing = test_data_shift.sample(250, random_state=11)
combine = training.append(testing)

y = combine["origin"]

combine.drop(["origin"], axis=1, inplace=True)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)

drop_list = []

for i in combine.columns:

    score = cross_val_score(model, pd.DataFrame(combine[i]),y,cv=2, scoring="roc_auc")

    if(np.mean(score) > 0.8):

        drop_list.append(i)

        print(i, np.mean(score))
drop_list.remove("col5")

drop_list.remove("col49")
dropped_train_data = train_data_p.drop(drop_list, axis=1)

dropped_test_data = test_data_p.drop(drop_list, axis=1)
#from imblearn.combine import SMOTETomek

#from imblearn.over_sampling import SMOTE



#smt = SMOTETomek()

#X_smt, y_smt = smt.fit_sample(dropped_train_data, train_y)
model = RandomForestClassifier(n_estimators=100, random_state=0)

model.fit(dropped_train_data, train_y)
predicted_y = model.predict(dropped_test_data)
finalDf = pd.DataFrame()

finalDf["ID"] = test_data["ID"]

finalDf["Class"] = predicted_y
from collections import Counter

Counter(predicted_y)
create_download_link(finalDf)
finalDf.head()
#from sklearn.model_selection import GridSearchCV

#from sklearn.metrics import make_scorer

#from sklearn.metrics import f1_score

#scorer_f1 = make_scorer(f1_score, average="micro")

#rf_temp = RandomForestClassifier(random_state=0)



#params = { "n_estimators":[10, 20, 30, 40, 50, 60, 70, 80, 100, 120] }



#grid_obj = GridSearchCV(rf_temp, params, scoring=scorer_f1)



#grid_fit = grid_obj.fit(dropped_train_data, train_y)
#grid_fit.best_params_