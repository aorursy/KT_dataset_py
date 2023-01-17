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


import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
stats.chisquares = lambda chisq , df : stats.chi2.sf(chisq, df)

sns.set()
#load the data
raw_data = pd.read_csv('/kaggle/input/admittancerecord/2.01. Admittance.csv')
raw_data
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes' : 1, 'No' : 0})
data
y= data['Admitted']
x1 = data['SAT']

x = sm.add_constant(x1)

reg_log= sm.Logit(y,x)
results_log = reg_log.fit()


def f(x, b0,b1):
    return np.array(np.exp(b0 + x * b1) / (1 + np.exp(b0 + x * b1)))

f_scored = np.sort(f(x1, results_log.params[0], results_log.params[1]))
x_score = np.sort(np.array(x1))
# create a scatter plot to view the data

plt.scatter(x1,y, color = 'C0')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
plt.plot(x_score, f_scored, color = 'C8')
plt.show()
results_log.summary()
raw_data = pd.read_csv('/kaggle/input/studentrecordwithsatgenderadmitted/2.02. Binary predictors.csv')
raw_data
data= raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No' : 0})
data['Gender']  = data['Gender'].map({'Female' : 1, 'Male' : 0})
data
y = data['Admitted']
x1 = data[['Gender', 'SAT']]
x = sm.add_constant(x1)

reg_log= sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary() # accessing the accuracy indicator is 'Pseudo R-squ.' from the summary below
results_log.predict()
np.array(data['Admitted'])
results_log.pred_table()
#confusion matrix

cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Prdicted 1']
cm_df = cm_df.rename(index = {0 : 'Actual 0' , 1 : 'Actual 1' })
cm_df
#calculate accuracy

cm = np.array(cm_df)
accuracy_train = (cm[0,0] + cm[1,1]/cm.sum())
accuracy_train
test =  pd.read_csv('/kaggle/input/testdataforchoosingstudent/2.03. Test dataset.csv')
test

test['Admitted'] = test['Admitted'].map({'Yes': 1, 'No' : 0})
test['Gender']  = test['Gender'].map({'Female' : 1, 'Male' : 0})
test
test_actual  = test['Admitted']
test_data = test.drop(['Admitted'], axis=1)
test_data = sm.add_constant(test_data)
test_data
# confusion matrix

def confusion_matrix(data, actual_values, model):
    pred_values = model.predict(data)
    bins= np.array([0,0.5,1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0,0] + cm[1,1]/cm.sum())
    return cm, accuracy
cm = confusion_matrix(test_data, test_actual, results_log)
cm
cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0', 'Prdicted 1']
cm_df = cm_df.rename(index = {0 : 'Actual 0' , 1 : 'Actual 1' })
cm_df