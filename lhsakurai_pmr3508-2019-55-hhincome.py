import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np
train_data = "../input/costa-rican-household-poverty-prediction/train.csv"

test_data = "../input/costa-rican-household-poverty-prediction/test.csv"
train_df = pd.read_csv(train_data)
train_df.shape
train_df = train_df.groupby('idhogar').first()
train_df.shape
train_df.head()
attributes = list(train_df)

attributes
ntrain_df = train_df.dropna()
ntrain_df.shape
na_per_column = train_df.isna().sum()

na_per_column
nan_indexes = na_per_column.nonzero()

nan_indexes
na_per_column[nan_indexes[0]]
train_df[['v2a1','v18q1','rez_esc','meaneduc','SQBmeaned']].describe()
train_df['v18q1'] = train_df['v18q1'].fillna(0)

train_df['meaneduc'] = train_df['meaneduc'].fillna(4)

train_df['SQBmeaned'] = train_df['SQBmeaned'].fillna(97)
train_df[['v18q1','meaneduc','SQBmeaned']].describe()
na_per_column = train_df.isna().sum()

nan_indexes = na_per_column.nonzero()

na_per_column[nan_indexes[0]]
wall_col = ['paredblolad','paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc','paredfibras', 'paredother']
def join_binary_columns(df, cols, new_column_label):

    df[new_column_label] = df[cols[0]]

    for i in range(1,len(cols)):

        df[new_column_label] = df[new_column_label] + df[cols[i]]*(2**i)

    return df.drop(columns = cols)
train_df = join_binary_columns(train_df, wall_col, 'wall')
train_df['wall'].describe()
train_df['wall'].value_counts().plot(kind="bar")
floor_col = ['pisonotiene', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisomadera']

roof_col = ['techozinc', 'techoentrepiso', 'techocane', 'techootro']

water_col = ['abastaguadentro', 'abastaguafuera', 'abastaguano']

electricity_col = ['public', 'planpri', 'noelec', 'coopele']

toilet_col = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']

energy_col = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']

rubbish_col = ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']

qua_wall_col = ['epared1', 'epared2', 'epared3']

qua_roof_col = ['etecho1', 'etecho2', 'etecho3']

qua_floor_col = ['eviv1', 'eviv2', 'eviv3']

house_state_col = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']

location_col = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']

area_col = ['area1', 'area2']

# some social variables aggregation that maybe useful to refine training (not used in this notebook)

# males_col = ['r4h1', 'r4h2', 'r4h3']

# females_col = ['r4m1', 'r4m2', 'r4m3']

# person_col = ['r4t1', 'r4t2', 'r4t3']

# civil_state_col = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']

# family_col = ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
train_df = join_binary_columns(train_df, floor_col, 'floor')

train_df = join_binary_columns(train_df, roof_col, 'roof')

train_df = join_binary_columns(train_df, water_col, 'water')

train_df = join_binary_columns(train_df, electricity_col, 'electricity')

train_df = join_binary_columns(train_df, toilet_col, 'toilet')

train_df = join_binary_columns(train_df, energy_col, 'energy')

train_df = join_binary_columns(train_df, rubbish_col, 'rubbish')

train_df = join_binary_columns(train_df, qua_wall_col, 'qua_wall')

train_df = join_binary_columns(train_df, qua_roof_col, 'qua_roof')

train_df = join_binary_columns(train_df, qua_floor_col, 'qua_roof')

train_df = join_binary_columns(train_df, house_state_col, 'house_state')

train_df = join_binary_columns(train_df, location_col, 'location')

train_df = join_binary_columns(train_df, area_col, 'area')
train_df["Target"].value_counts().plot(kind="bar")
attributes = list(train_df)

attributes
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
drop_columns = ['v2a1', 'rez_esc']

train_df = train_df.drop(columns = ['v2a1', 'rez_esc'])

attributes.remove('Target')

attributes.remove('Id')

for e in drop_columns:

    attributes.remove(e)
corr = train_df[attributes].apply(preprocessing.LabelEncoder().fit_transform).apply(lambda x: x.corr(train_df.Target))

corr
corr_thresh = 0.2

corr.where(abs(corr) > corr_thresh).dropna()
train_attr = corr.where(abs(corr) > corr_thresh).dropna().index.values
train_attr
train_df_x = train_df[train_attr].apply(preprocessing.LabelEncoder().fit_transform)

train_df_y = train_df.Target
test_df = pd.read_csv(test_data)

test_df['v18q1'] = test_df['v18q1'].fillna(0)

test_df['meaneduc'] = test_df['meaneduc'].fillna(4)

test_df['SQBmeaned'] = test_df['SQBmeaned'].fillna(97)

test_df = join_binary_columns(test_df, wall_col, 'wall')

test_df = join_binary_columns(test_df, floor_col, 'floor')

test_df = join_binary_columns(test_df, roof_col, 'roof')

test_df = join_binary_columns(test_df, water_col, 'water')

test_df = join_binary_columns(test_df, electricity_col, 'electricity')

test_df = join_binary_columns(test_df, toilet_col, 'toilet')

test_df = join_binary_columns(test_df, energy_col, 'energy')

test_df = join_binary_columns(test_df, rubbish_col, 'rubbish')

test_df = join_binary_columns(test_df, qua_wall_col, 'qua_wall')

test_df = join_binary_columns(test_df, qua_roof_col, 'qua_roof')

test_df = join_binary_columns(test_df, qua_floor_col, 'qua_roof')

test_df = join_binary_columns(test_df, house_state_col, 'house_state')

test_df = join_binary_columns(test_df, location_col, 'location')

test_df = join_binary_columns(test_df, area_col, 'area')

test_df_x = test_df[train_attr].apply(preprocessing.LabelEncoder().fit_transform)
columns = ['neighbors', 'scores']

results = [columns]

for n in range (5, 40):

    neighbors = n

    cross = 5

    knn = KNeighborsClassifier(n_neighbors = neighbors)

    scores = cross_val_score(knn, train_df_x, train_df_y, cv = cross)

    results.append([neighbors, scores])
import statistics as st



analysis = [['neighbors', 'mean', 'max', 'min']]

for i in range(1, len(results)):

    analysis.append([results[i][0], st.mean(results[i][1]), max(results[i][1]), min(results[i][1])])
analysis
neighbors = 32

cross = 5

KNeighborsClassifier(n_neighbors = neighbors)

cross_val_score(knn, train_df_x, train_df_y, cv = cross)

knn.fit(train_df_x, train_df_y)

test_pred_y = knn.predict(test_df_x)

test_pred_y
result = pd.DataFrame({'Id':test_df.Id.values, 'Target':test_pred_y})

result
result.to_csv("submission.csv", index = False)