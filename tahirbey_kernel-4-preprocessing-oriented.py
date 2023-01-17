import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
data = pd.read_csv("../input/heart.csv")
data.head(20)
data.describe().T
data.columns

#We had better change column names with a more apprehensible names.
data.columns = ['age', 'sex', 'chest_pain', 'resting_bloodpress', 'cholesterol', 'fasting_bloodsugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'major_vessel_number', 'thalassemia', 'target']
data.dtypes
data['sex'] = data['sex'].astype('object')
data['chest_pain'] = data['chest_pain'].astype('object')
data['fasting_bloodsugar'] = data['fasting_bloodsugar'].astype('object')
data['rest_ecg'] = data['rest_ecg'].astype('object')
data['exercise_angina'] = data['exercise_angina'].astype('object')
data['st_slope'] = data['st_slope'].astype('object')
data['thalassemia'] = data['thalassemia'].astype('object')
data.dtypes
def bar_plot(variable):
    
    var = data[variable]
    varValue = var.value_counts()
    
    plt.figure()
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)
    plt.show()
    
categoricals = ['sex', 'chest_pain', 'fasting_bloodsugar', 'rest_ecg', 'exercise_angina', 'st_slope', 'thalassemia', 'target']

for i in categoricals:
    bar_plot(i)
def plot_hist(variable):
    plt.plot()
    plt.hist(data[variable],bins = 50)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title('{} distribution' .format(variable))
    plt.show()
    
numerical = ['age','resting_bloodpress','cholesterol' ,'max_heart_rate','st_depression','major_vessel_number']               
for i in numerical:
    plot_hist(i)
#'sex'vs 'target'

data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

data[['sex','target']].groupby(['sex']).mean()
#'chest_pain'vs 'target'

data[['chest_pain','target']].groupby(['chest_pain']).mean()
#'chest_pain'vs 'target'

data[['cholesterol','target']].groupby(['target']).mean()
#'chest_pain'vs 'target'

data[['max_heart_rate','target']].groupby(['target']).mean()

def detect_outliers(data,features):
    outlier_indices = []
    
    for c in features:
        
        Q1 = np.percentile(data[c],25)
        Q3 = np.percentile(data[c],75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = data[(data[c] < Q1 - outlier_step)|(data[c] < Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
data.loc[detect_outliers(data,['age','resting_bloodpress', 'cholesterol','fasting_bloodsugar', 'rest_ecg', 'max_heart_rate'])]
#newData = data.drop(detect_outliers(data,['age','resting_bloodpress', 'cholesterol','fasting_bloodsugar', 'rest_ecg', 'max_heart_rate']))
data.isnull().sum()
#No missing Value