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
train = pd.read_csv("/kaggle/input/melinia-2020/train.csv")
test = pd.read_csv("/kaggle/input/melinia-2020/test.csv")
submit = pd.read_csv("/kaggle/input/melinia-2020/sample_submission.csv")
#ID: A unique id for each review
#Date: The date at which the review was done in a particular assessment
#License_No: De-identified license number for a particular assessment
#Assessment_ID: De-identified unique assessment id for an assessment
#Assessment_Name: The encoded name of an assessment
#Type: The type of assessment being reviewed
#Street: The encoded street where the assessment field is located
#City: The encoded city where the assessment field is located
#State: The encoded state where the assessment field is located
#Location_ID: An encoded location feature.
#Reason: The primary reason for the review
#Section_Violations: Laws violated by the parties.
#Risk_level: The level of risk the office possesses to the consumers.
#Geo_Loc: De-identified geolocation of the assessment field
#Assessment_Results: The result of the review
import h2o
h2o.init()
train1 = h2o.H2OFrame(train)
test1 = h2o.H2OFrame(test)
train1.columns
y = 'Assessment_Results'
x = train1.col_names
x.remove(y)
train1['Assessment_Results'] = train1['Assessment_Results'].asfactor()
train1['Assessment_Results'].levels()
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models = 30, max_runtime_secs=1000, seed = 1)
aml.train(x = x, y = y, training_frame = train1)
preds = aml.predict(test1)
print(submit.columns)
ans=h2o.as_list(preds) 
submit['Assessment_Results'] = ans['predict']
submit.to_csv('Solution.csv',index=False)
train1 = h2o.H2OFrame(train)
test1 = h2o.H2OFrame(test)
train1.columns
y = 'Assessment_Results'
x = train1.col_names
x.remove(y)
train1['Assessment_Results'] = train1['Assessment_Results'].asfactor()
train1['Assessment_Results'].levels()
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models = 30, max_runtime_secs=1000, seed = 1)
aml.train(x = x, y = y, training_frame = train1)
preds = aml.predict(test1)
preds
ans=h2o.as_list(preds) 
print(submit.columns)
submit['Assessment_Results'] = ans['predict']
submit.to_csv('Solution.csv',index=False)