import numpy as np

import pandas as pd

from fastai.tabular import *
input_path = '/kaggle/input/pima-indians-diabetes-database/'

file = 'diabetes.csv'



df = pd.read_csv(input_path + file)

df.head(5)
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)



df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 102.5

df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 169.5



df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 107

df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 140



df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27

df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 32



df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70

df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 74.5



df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.1

df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 34.3
dep_var = 'Outcome'

procs = [FillMissing, Normalize]
data = TabularDataBunch.from_df('', df, valid_idx=range(len(df)-153, len(df)), dep_var=dep_var, procs=procs)
data.show_batch(rows=10)
np.random.seed(56)

learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8, 1e-1)
learn.save('step-1')
learn.load('step-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5,1e-3)
learn.save('step-2')
learn.load('step-2')