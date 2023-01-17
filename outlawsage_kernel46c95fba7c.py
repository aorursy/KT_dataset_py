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
import pandas as pd

import numpy as np

job_df=pd.read_csv("../input/job-classification-dataset/jobclassinfo2.csv")

job_df.info()
job_df=job_df.drop(["ID","JobFamilyDescription","JobClass","JobClassDescription","PG"],axis=1)

#all the useless variables has been droped

job_df[0:8]
import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline



influential_features=['JobFamily',

 'EducationLevel',

 'Experience',

 'OrgImpact',

 'ProblemSolving',

 'Supervision',

 'ContactLevel',

 'FinancialBudget','PayGrade']



sn.pairplot(job_df[influential_features], height=2)
job_df[influential_features].corr()
plt.figure(figsize=(10,7))

sn.heatmap(job_df[influential_features].corr(), annot=True)
x_features=list(job_df.columns)

x_features.remove("PayGrade")

x_features.remove("JobFamily")

#removing unnccesary variables after analysing correlation

x_features
categorical_vars=x_features
job_df_dum=pd.get_dummies(job_df[x_features],columns=categorical_vars,drop_first=True)

x_features=job_df_dum.columns

x_features
import statsmodels.api as sm

from sklearn.model_selection import train_test_split



y=job_df.PayGrade

x=sm.add_constant(job_df_dum)

train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.7,random_state=42)
from sklearn import linear_model

from sklearn import metrics



model_1_job=linear_model.LogisticRegression()

model_1=model_1_job.fit(train_x,train_y)
y_pred = model_1.predict(test_x)
print("Accuracy of Logistic Regression model is:",

metrics.accuracy_score(test_y, y_pred))
y_pred_df=pd.DataFrame({"actual":test_y,"predictedfrom_model":y_pred})

y_pred_df.sample(15,random_state=42)
print(metrics.classification_report( y_pred_df.actual, y_pred_df.predictedfrom_model ))