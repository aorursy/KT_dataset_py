import pandas as pd

from sklearn import metrics
pred = pd.read_csv("../input/bert-baseline/predictions.csv")

df = pd.read_csv("../input/bert-baseline/test.csv")

df['prediction'] = pred[' Toxic']

df['target'] = df['target'] >= 0.5

df['bool_pred'] = df['prediction'] >= 0.5
def auc(df):

    y = df['target']

    pred = df['prediction']

    fpr, tpr, thresholds = metrics.roc_curve(y, pred)

    return metrics.auc(fpr, tpr)



overall = auc(df)

overall
groups = ['black', 'white', 'male', 'female',

          'christian', 'jewish', 'muslim',

          'psychiatric_or_mental_illness',

          'homosexual_gay_or_lesbian']



categories = pd.DataFrame(columns = ['SUB', 'BPSN', 'BNSP'], index = groups)
import numpy as np

def Mp(data, p=-5.0):

    return np.average(data ** p) ** (1/p)



for group in groups:

    df[group] = df[group] >= 0.5

    categories.loc[group,'SUB'] = auc(df[df[group]])

    bpsn = ((~df[group] & df['target'])    #background positive

            | (df[group] & ~df['target'])) #subgroup negative

    categories.loc[group,'BPSN'] = auc(df[bpsn])

    bnsp = ((~df[group] & ~df['target'])   #background negative

            | (df[group] & df['target']))  #subgrooup positive

    categories.loc[group,'BNSP'] = auc(df[bnsp])



categories.loc['Mp',:] = categories.apply(Mp, axis= 0)

categories
leaderboard = (np.sum(categories.loc['Mp',:]) + overall) / 4

leaderboard