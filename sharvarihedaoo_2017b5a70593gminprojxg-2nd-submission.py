# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Generating dataframe

column_names = ['id','col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12','col_13','col_14','col_15','col_16','col_17','col_18','col_19','col_20','col_21','col_22','col_23','col_24','col_25','col_26','col_27','col_28','col_29','col_30','col_31','col_32','col_33','col_34','col_35','col_36','col_37','col_38','col_39','col_40','col_41','col_42','col_43','col_44','col_45','col_46','col_47','col_48','col_49','col_50','col_51','col_52','col_53','col_54','col_55','col_56','col_57','col_58','col_59','col_60','col_61','col_62','col_63','col_64','col_65','col_66','col_67','col_68','col_69','col_70','col_71','col_72','col_73','col_74','col_75','col_76','col_77','col_78','col_79','col_80','col_81', 'col_82','col_83','col_84','col_85','col_86','col_87','target']

df = pd.read_csv("../input/minor-project-2020/train.csv",header=None, sep=',', delimiter=None, names=column_names)
df
#dropping 0th row 

df=df.drop(0)

df
#Data preprocessing

df = df.astype(np.float64)

df = df.astype(np.int64)

df.info()
y = df['target']

X = df.drop(["target"], axis=1)
#Splitting into test and train sets



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)



print(len(X_train), len(X_test))
#Feature Scaling



from sklearn.preprocessing import StandardScaler



scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_test = scalar.transform(X_test)



scaled_X_train
scaled_X_test
#Using xg_boost XGBClassifier



from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(scaled_X_train, y_train)
#Checking performance of model

#accuracy

xgb.score(scaled_X_test,  y_test)
y_pred=xgb.predict(scaled_X_test)
y_pred
#roc auc curve

from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC for skewed dataset', fontsize= 18)

plt.show()
#Without splitting into test and train
column_names1 = ['id','col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12','col_13','col_14','col_15','col_16','col_17','col_18','col_19','col_20','col_21','col_22','col_23','col_24','col_25','col_26','col_27','col_28','col_29','col_30','col_31','col_32','col_33','col_34','col_35','col_36','col_37','col_38','col_39','col_40','col_41','col_42','col_43','col_44','col_45','col_46','col_47','col_48','col_49','col_50','col_51','col_52','col_53','col_54','col_55','col_56','col_57','col_58','col_59','col_60','col_61','col_62','col_63','col_64','col_65','col_66','col_67','col_68','col_69','col_70','col_71','col_72','col_73','col_74','col_75','col_76','col_77','col_78','col_79','col_80','col_81', 'col_82','col_83','col_84','col_85','col_86','col_87']



df1 = pd.read_csv("../input/minor-project-2020/test.csv",header=None, sep=',', delimiter=None, names=column_names1)
df1
df1=df1.drop(0)

df1
y_train1 = df['target']

X_train1 = df.drop(["target"], axis=1)
X_test1 = df1

X_test1
from sklearn.preprocessing import StandardScaler



scalar = StandardScaler()

scaled_X_train1 = scalar.fit_transform(X_train1)



scaled_X_train1
from sklearn.preprocessing import StandardScaler



scalar = StandardScaler()

scaled_X_test1 = scalar.fit_transform(X_test1)



scaled_X_test1
xgb.fit(scaled_X_train1, y_train1)
xgb.score(scaled_X_test,  y_test)
y_pred1 = xgb.predict(scaled_X_test1)
y_pred1
predictions = pd.DataFrame(data=y_pred1, columns=["target"])

predictions
predictions.index = np.arange(1, len(predictions)+1)

predictions
test_id = df1['id']

predictions.insert(0, 'id',test_id , True) 

predictions
predictionXG = predictions.to_csv('predictionXG.csv',index=False)