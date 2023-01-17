# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', 150)

pd.set_option('display.max_rows', 150)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_filepath = '../input/covid19-global-forecasting-week-2/train.csv'

test_filepath = '../input/covid19-global-forecasting-week-2/test.csv'
train_df = pd.read_csv(train_filepath, index_col="Id")

test_df = pd.read_csv(test_filepath, index_col="ForecastId")
train_df.shape
test_df.shape
train_df.info()
y_train_cc = np.array(train_df['ConfirmedCases'].astype(int))

y_train_ft = np.array(train_df['Fatalities'].astype(int))

cols = ['ConfirmedCases', 'Fatalities']



full_df = pd.concat([train_df.drop(cols, axis=1), test_df])

index_split = train_df.shape[0]

full_df = pd.get_dummies(full_df, columns=full_df.columns)



x_train = full_df[:index_split]

x_test= full_df[index_split:]

#x_train.shape, x_test.shape, y_train_cc.shape, y_train_ft.shape
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=1)
dt.fit(x_train,y_train_cc)
y_pred_cc = dt.predict(x_test)

y_pred_cc = y_pred_cc.astype(int)

y_pred_cc[y_pred_cc <0]=0
dt_f = DecisionTreeRegressor()



dt_f.fit(x_train,y_train_ft)
y_pred_ft = dt_f.predict(x_test)

y_pred_ft = y_pred_ft.astype(int)

y_pred_ft[y_pred_ft <0]=0

predicted_df_dt = pd.DataFrame([y_pred_cc, y_pred_ft], index = ['ConfirmedCases','Fatalities'], columns= np.arange(1, y_pred_cc.shape[0] + 1)).T

predicted_df_dt.to_csv('submission_rf.csv', index_label = "ForecastId")
from sklearn.tree import DecisionTreeClassifier



dtcla = DecisionTreeClassifier()

# We train model

dtcla.fit(x_train, y_train_cc)
predictions = dtcla.predict(x_test)
dtcla.fit(x_train,y_train_ft)
predictions1 = dtcla.predict(x_test)
submission = pd.DataFrame({'ForecastId': test_df.index,'ConfirmedCases':predictions,'Fatalities':predictions1})

filename = 'submission.csv'



submission.to_csv(filename,index=False)
submission