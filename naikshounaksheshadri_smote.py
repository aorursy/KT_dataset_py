# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix,roc_auc_score



from xgboost import XGBClassifier







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
X=df.iloc[:,1:-1]

Y=df.iloc[:,-1]



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=27)

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_val = scalar.fit_transform(X_val)
sm = SMOTE(random_state=27, sampling_strategy=0.75)

scaled_X_train,Y_train = sm.fit_sample(scaled_X_train,Y_train)
xgb = XGBClassifier(tree_method='gpu_hist',gpu_id=0,scale_pos_weight=1.33,objective ='binary:logistic',eval_metric='auc',

                    alpha = 19, n_estimators = 10,max_depth=6,max_delta_step=50,subsample=0.5,gamma=1)

xgb.fit(scaled_X_train,Y_train)
pred_val=xgb.predict(scaled_X_val)
from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(Y_val, pred_val)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)
import matplotlib.pyplot as plt

%matplotlib inline



plot_confusion_matrix(xgb, scaled_X_val, Y_val, cmap = plt.cm.Blues)
df_test=pd.read_csv("/kaggle/input/minor-project-2020/test.csv")

id=df_test.iloc[:,0]

X_test=df_test.iloc[:,1:89]

scaled_X_test = scalar.fit_transform(X_test)
pred_test=xgb.predict(scaled_X_test)
my_submission = pd.DataFrame({'Id': id, 'target': pred_test})

my_submission.to_csv('smot.csv', index=False)