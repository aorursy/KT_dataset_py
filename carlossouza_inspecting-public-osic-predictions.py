import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/dummyosicsubmission/submission (18).csv')
df['Patient_Week'] = df['Patient_Week'].str.strip()

df['Patient'] = df['Patient_Week'].apply(lambda x: x.split('_')[0])

df['Week'] = df['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

df = df.drop(columns='Patient_Week')

df = df.sort_values(['Patient', 'Week'])

df = df.rename(columns={'Week': 'Weeks', 'FVC': 'FVC_pred'})

df['FVC_sup'] = df['FVC_pred'] + df['Confidence']

df['FVC_inf'] = df['FVC_pred'] - df['Confidence']



truth = pd.read_csv('../input/dummyosicsubmission/train.csv')

df = pd.merge(df, truth[['Patient', 'Weeks', 'FVC']], how='left', 

              on=['Patient', 'Weeks'])

df = df.rename(columns={'FVC': 'FVC_true'})
def chart(patient_id, ax):

    data = df[df['Patient'] == patient_id]

    x = data['Weeks']

    ax.set_title(patient_id)

    ax.plot(x, data['FVC_true'], 'o')

    ax.plot(x, data['FVC_pred'])

    ax = sns.regplot(x, data['FVC_true'], ax=ax, ci=None, 

                     line_kws={'color':'red'})

    ax.fill_between(x, data["FVC_inf"], data["FVC_sup"],

                    alpha=0.5, color='#ffcd3c')

    ax.set_ylabel('FVC')



f, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, pid in enumerate(df['Patient'].unique()):

    x, y = divmod(i, 3)

    chart(pid, axes[x, y])

axes[1, 2].axis('off');