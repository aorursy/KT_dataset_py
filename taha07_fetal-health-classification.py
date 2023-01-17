# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/fetal-health-classification/fetal_health.csv")

df.head()
df.describe()
df.info()
import missingno as msno

n = msno.bar(df,color="yellow")
plt.style.use("fivethirtyeight")

plt.figure(figsize=(9,7))

sns.countplot(x="fetal_health",data = df)

plt.show()
plt.figure(figsize=(20,10))

sns.boxplot(data = df,palette = "Set3")

plt.xticks(rotation=90)

plt.show()
lowerlimit = df.histogram_variance.mean() - 3*df.histogram_variance.std()

upperlimit = df.histogram_variance.mean() + 3*df.histogram_variance.std()

print(lowerlimit,upperlimit)

df1 = df[(df.histogram_variance > lowerlimit) & (df.histogram_variance < upperlimit)]

df.shape[0],df1.shape[0]
lowerlimit = df1.histogram_median.mean() - 3*df1.histogram_median.std()

upperlimit = df1.histogram_median.mean() + 3*df1.histogram_median.std()

print(lowerlimit,upperlimit)

df2 = df1[(df1.histogram_median > lowerlimit) & (df1.histogram_median < upperlimit)]

df1.shape[0],df2.shape[0]
lowerlimit = df2.histogram_mode.mean() - 3*df2.histogram_mode.std()

upperlimit = df2.histogram_mode.mean() + 3*df2.histogram_mode.std()

print(lowerlimit,upperlimit)

df3 = df2[(df2.histogram_mode > lowerlimit) & (df2.histogram_mode < upperlimit)]

df2.shape[0],df3.shape[0]
lowerlimit = df3.histogram_max.mean() - 3*df3.histogram_max.std()

upperlimit = df3.histogram_max.mean() + 3*df3.histogram_max.std()

print(lowerlimit,upperlimit)

df4 = df3[(df3.histogram_max > lowerlimit) & (df3.histogram_max < upperlimit)]

df3.shape[0],df4.shape[0]
lowerlimit = df4.percentage_of_time_with_abnormal_long_term_variability.mean() - 3*df4.percentage_of_time_with_abnormal_long_term_variability.std()

upperlimit = df4.percentage_of_time_with_abnormal_long_term_variability.mean() + 3*df4.percentage_of_time_with_abnormal_long_term_variability.std()

print(lowerlimit,upperlimit)

df5 = df4[(df4.percentage_of_time_with_abnormal_long_term_variability > lowerlimit) & (df4.percentage_of_time_with_abnormal_long_term_variability < upperlimit)]

df4.shape[0],df5.shape[0]
df = df5.copy()

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,10))

g = sns.heatmap(df[top_corr_features].corr(),annot = True,cmap = "RdYlGn")
x = df.drop("fetal_health",axis=1)

y =  df["fetal_health"]
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
xgb_clf = XGBClassifier()

xgb_clf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score

pred_xgb = xgb_clf.predict(x_test)

accuracy_score(y_test,pred_xgb)
from sklearn.preprocessing import LabelBinarizer

def calculate_roc_auc_score(y_test,y_pred,average="macro"):

    lb = LabelBinarizer()

    y_test1 = lb.fit_transform(y_test)

    y_pred1 =lb.transform(y_pred)

    return roc_auc_score(y_test1,y_pred1,average=average)
calculate_roc_auc_score(y_test,pred_xgb)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred_xgb))
lgb_clf = LGBMClassifier()

lgb_clf.fit(x_train,y_train)

pred_lgb = lgb_clf.predict(x_test)

accuracy_score(y_test,pred_lgb)
print(classification_report(y_test,pred_lgb))
#roc auc score

calculate_roc_auc_score(y_test,pred_lgb)
rf_clf = RandomForestClassifier(random_state=0)

rf_clf.fit(x_train,y_train)



pred_rf = rf_clf.predict(x_test)

accuracy_score(y_test,pred_rf)
print(classification_report(y_test,pred_rf))
#roc auc score

calculate_roc_auc_score(y_test,pred_rf)
from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators = [("xgb_clf",xgb_clf),("lgb_clf",lgb_clf),("rf_clf",rf_clf)],voting='soft',

                      weights =[8,7,5])

vc.fit(x_train,y_train)

pred_vc = vc.predict(x_test)
accuracy_score(y_test,pred_vc)
print(classification_report(y_test,pred_vc))
plt.style.use("ggplot")

from mlxtend.plotting import plot_confusion_matrix

cm = confusion_matrix(y_test,pred_vc)

plot_confusion_matrix(conf_mat = cm,figsize=(8,6),show_normed=True,

                      class_names =["Normal","Suspect","Pathological"])