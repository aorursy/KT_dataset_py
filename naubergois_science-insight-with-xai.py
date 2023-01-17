# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

df=pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df = df.loc[:,~df.columns.duplicated()]

df_class_0=df[df['SARS-Cov-2 exam result']=='negative']

df_class_1=df[df['SARS-Cov-2 exam result']!='negative']

df_class_0=df_class_0[:len(df_class_1)]

df_all=pd.concat([df_class_0,df_class_1],axis=0)
import re





columns=df_all.columns

new_columns=[re.sub(r"[^a-zA-Z0-9]+", ' ', x) for x in columns]

new_columns=[x.replace('_','') for x in new_columns]

new_columns=[x.replace('+','') for x in new_columns]

new_columns=[x.replace('-','') for x in new_columns]

#new_columns=[str(x, 'utf-8') for x in new_columns]

df_all.columns=new_columns

y=df_all['SARS Cov 2 exam result']

X=df_all.drop(['SARS Cov 2 exam result'],axis=1)



from sklearn.preprocessing import LabelEncoder

X_one_hot=pd.get_dummies(X)

le=LabelEncoder()

y_l=le.fit_transform(y)

X_one_hot = X_one_hot.loc[:,~X_one_hot.columns.duplicated()]

X_one_hot=X_one_hot.iloc[:,:100]

for column in X_one_hot.columns:

    if '<' in column or  '+' in column:

        X_one_hot=X_one_hot.drop([column],axis=1)
for column in X_one_hot.columns:

    

    if '<' in column:

        print(column)
X_one_hot=X_one_hot.fillna(0)
columns=X_one_hot.columns

new_columns=[re.sub(r"[^a-zA-Z0-9]+", ' ', x) for x in columns]

X_one_hot.columns=columns



for column in X_one_hot.columns:

    X_one_hot[column]=pd.to_numeric(X_one_hot[column])
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier



X_train,X_test,y_train,y_test=train_test_split(X_one_hot,y_l,random_state=10)



model=RandomForestClassifier()

model.fit(X_train,y_train)



y_pred=model.predict(X_test)





print(classification_report(y_test,y_pred))
def xai_plot_values(model,X_):

    shap.initjs()

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_)

    return explainer,shap_values



def xai_plot_values_kernel(model,X_):

    shap.initjs()

    explainer = shap.KernelExplainer(model,data=X_)

    shap_values = explainer.shap_values(X_)

    return explainer,shap_values

    
import shap 

explainer,shap_values=xai_plot_values(model,X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar")
from sklearn import svm

svm = svm.SVC(gamma='scale',decision_function_shape='ovo')

svm.fit(X_train,y_train)

# The SHAP values

svm_explainer = shap.KernelExplainer(svm.predict,X_test.iloc[:30,:])

svm_shap_values = svm_explainer.shap_values(X_test.iloc[:30,:])


shap.summary_plot(svm_shap_values, X_test.iloc[:30,:], plot_type="bar")
shap.summary_plot(svm_shap_values, X_test.iloc[:30,:])
shap.force_plot(svm_explainer.expected_value, svm_shap_values, X_train)