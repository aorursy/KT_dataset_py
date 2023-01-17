# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
samp = pd.read_csv('../input/sample_submission.csv')
train.head()
train.isnull().values.any()
cover_type_grouped = train.groupby('Cover_Type').mean()
cover_type_grouped
train.groupby('Cover_Type').mean()['Elevation'].plot.bar()
train.Elevation.plot.hist()
train.Horizontal_Distance_To_Hydrology.plot.hist()
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
ntrain = train.shape[0]
ntest = test.shape[0]
y = train.Cover_Type.values

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Cover_Type'], axis=1, inplace=True)
####################### all_data #############################################
all_data['HF1'] = all_data['Horizontal_Distance_To_Hydrology']+all_data['Horizontal_Distance_To_Fire_Points']
all_data['HF2'] = abs(all_data['Horizontal_Distance_To_Hydrology']-all_data['Horizontal_Distance_To_Fire_Points'])
all_data['HR1'] = abs(all_data['Horizontal_Distance_To_Hydrology']+all_data['Horizontal_Distance_To_Roadways'])
all_data['HR2'] = abs(all_data['Horizontal_Distance_To_Hydrology']-all_data['Horizontal_Distance_To_Roadways'])
all_data['FR1'] = abs(all_data['Horizontal_Distance_To_Fire_Points']+all_data['Horizontal_Distance_To_Roadways'])
all_data['FR2'] = abs(all_data['Horizontal_Distance_To_Fire_Points']-all_data['Horizontal_Distance_To_Roadways'])
all_data['ele_vert'] = all_data.Elevation-all_data.Vertical_Distance_To_Hydrology

all_data['slope_hyd'] = (all_data['Horizontal_Distance_To_Hydrology']**2+all_data['Vertical_Distance_To_Hydrology']**2)**0.5
all_data.slope_hyd=all_data.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
all_data['Mean_Amenities']=(all_data.Horizontal_Distance_To_Fire_Points + all_data.Horizontal_Distance_To_Hydrology + all_data.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
all_data['Mean_Fire_Hyd']=(all_data.Horizontal_Distance_To_Fire_Points + all_data.Horizontal_Distance_To_Hydrology) / 2 
all_data.shape
all_data.head()
col_list = list(all_data.columns)
cols_to_be_normalized = [x for x in col_list if "Soil" not in x]
cols_to_be_normalized = [x for x in cols_to_be_normalized if "Wilderness" not in x]
cols_to_be_normalized
for the_col in cols_to_be_normalized:
    all_data.loc[:, the_col]  = all_data.loc[:, the_col]/all_data.loc[:, the_col].max()
all_data.head()
x = all_data[:ntrain]
x_test = all_data[ntrain:]
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
clf1 = SVC()
clf2 = KNeighborsClassifier()
clf3 = GradientBoostingClassifier()
clf4 = XGBClassifier()
clf5 = RandomForestClassifier()
eclf1 = VotingClassifier(estimators=[('svc', clf1), ('knn', clf2), ('gbc', clf3), ('xgbc', clf4), ('rf', clf5)], voting='hard')
eclf1.fit(x, y)
y_pred = eclf1.predict(x)
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
predictions_eclf = eclf1.predict(x_test)
sub_eclf = pd.DataFrame()
sub_eclf['Id'] = test_ID
sub_eclf['Cover_Type'] = predictions_eclf
sub_eclf.to_csv('submission_eclf.csv', index=False)