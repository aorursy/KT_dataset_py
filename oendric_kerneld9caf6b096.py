# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/train.csv"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()

# Any results you write to the current directory are saved as output.
digit_recog_dummies=pd.get_dummies(train)
digit_recog_dummies.head()
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train, validate = train_test_split(digit_recog_dummies,
                               test_size=0.3,
                               random_state=100)
train_y = train['label']
validate_y = validate['label']
train_x = train.drop('label', axis=1)
validate_x = validate.drop('label', axis=1)

model_rf = RandomForestClassifier(random_state=100)
model_rf.fit(train_x, train_y)


validate_pred = model_rf.predict(validate_x)

from sklearn.metrics import accuracy_score , classification_report
from sklearn.metrics import confusion_matrix

accuracy_score(validate_y,validate_pred)
confusion_matrix(validate_y,validate_pred,labels=[1,2])

df_pred_rf = pd.DataFrame({'actual': validate_y,
                        'predicted': validate_pred})
df_pred_rf['pred_status'] = df_pred_rf['actual'] == df_pred_rf['predicted']
df_pred_rf['pred_status'].sum() / df_pred_rf.shape[0] * 100

test_pred = model_rf.predict(test)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId'] = test.index + 1
df_test_pred[['ImageId','Label']].to_csv("Submission.csv",index=False)
