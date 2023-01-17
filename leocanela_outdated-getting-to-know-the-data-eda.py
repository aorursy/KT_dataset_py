import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split





plt.style.use('seaborn-colorblind')



raw_data = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
raw_data.head(5)
df_ICU_list = raw_data.groupby("PATIENT_VISIT_IDENTIFIER").agg({"ICU":(list)})

raw_data['ICU_list'] = raw_data.apply(lambda row: df_ICU_list.loc[row['PATIENT_VISIT_IDENTIFIER']]['ICU'], axis=1)
raw_data['VALID_WINDOW'] = raw_data.apply(lambda row: row['ICU_list'].index(1) if 1 in row['ICU_list'] else 5, axis=1)
labs_and_vitals = raw_data.iloc[:, 13:-4].columns

before_transform = raw_data[labs_and_vitals].isna().sum()

for index, row in raw_data.iterrows():

    if index%5 > row['VALID_WINDOW']:

        raw_data.loc[index, labs_and_vitals] = np.nan
raw_data[labs_and_vitals].isna().sum() - before_transform
data = raw_data.drop(['ICU_list','VALID_WINDOW'], axis=1)
len(data)/5
temporary_grouped_data = data.groupby("PATIENT_VISIT_IDENTIFIER").agg(list)

temporary_grouped_data
## Separating features columns like the starter notebook

comorb_lst = [i for i in data.columns if "DISEASE" in i]

comorb_lst.extend(["HTN", "IMMUNOCOMPROMISED", "OTHER"])



demo_lst = [i for i in data.columns if "AGE_" in i]

demo_lst.append("GENDER")



vitalSigns_lst = data.iloc[:,193:-2].columns.tolist()



lab_lst = data.iloc[:,13:193].columns.tolist()
set(temporary_grouped_data[demo_lst].astype('str').values.ravel().tolist())
set(temporary_grouped_data[vitalSigns_lst+lab_lst].astype('str').values.ravel().tolist()[:50])
set(temporary_grouped_data['ICU'].astype('str').values.ravel().tolist())
set(temporary_grouped_data[comorb_lst].astype('str').values.ravel().tolist())
nan = np.nan

def agg_function(column):

    removed_nan = list(filter(lambda v: v==v, column))

    if str(list(column)) == '[nan, nan, nan, nan, nan]': return np.nan

    elif column.name == 'ICU': return list(column)

    elif column.name in comorb_lst: return max(column)

    elif column.name in demo_lst: return max(column)

    elif column.name in vitalSigns_lst+lab_lst: return (sum(removed_nan) / len(removed_nan) )

    else: return column

        

flattened_data = data.groupby("PATIENT_VISIT_IDENTIFIER").agg(agg_function)

flattened_data.head()
flattened_data['ICU_int'] = flattened_data.apply(lambda row: 0 if 1 not in list(row['ICU']) else 1, axis=1)
flattened_data.reset_index(inplace=True)
def plot_demo(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):

        values = flattened_data[column].value_counts().sort_values(ascending=False)

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        sns.countplot(y=column, data=dataset, order=flattened_data[column].value_counts().index)

        for i, v in enumerate(values):

            ax.text(v - (v/2), i, v, {'backgroundcolor': 'white', 'fontsize': 14})

    

plot_demo(flattened_data[demo_lst], cols=3, width=20, height=5, hspace=0.45, wspace=0.4)
def comorbs_plot(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):

        values = flattened_data[column].value_counts().sort_values(ascending=False)

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        sns.countplot(y=column, data=dataset)

        for i, v in enumerate(values):

            ax.text(v - (v/2), i, v, {'backgroundcolor': 'white'}) if len(values) > 1  else ax.text(v - (v/2),0 , v, {'backgroundcolor': 'white'})

    

comorbs_plot(flattened_data[comorb_lst], cols=3, width=20, height=20, hspace=0.4, wspace=0.4)
def vitalsandlabs_plot(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):

        values = flattened_data[column].value_counts().sort_values(ascending=False)

        ax = fig.add_subplot(rows, cols, i + 1)

        try:

            sns.distplot(dataset[column])

        except RuntimeError as re:

            if str(re).startswith("Selected KDE bandwidth is 0"):

                sns.distplot(dataset[column], kde_kws={'bw': 0.1})

            else:

                raise re

        plt.xticks(rotation=0)

vitalsandlabs_plot(flattened_data.loc[:,[i for i in flattened_data.columns if 'MEAN' in i]], cols=4, width=20, height=20, hspace=0.55, wspace=0.25)
vlabs = flattened_data.loc[:,[i for i in flattened_data.columns if 'MEAN' in i]]

vlabs['ICU'] = flattened_data['ICU_int']
fig = plt.figure(figsize=(10,7))

sns.heatmap(vlabs.corr(), cmap="RdBu_r")
def comorbs_plot(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):

        values = flattened_data[column].value_counts().sort_values(ascending=True)

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        sns.countplot(y=column, data=dataset)

        for i, v in enumerate(values):

            ax.text(v - (v/2), i, v, {'backgroundcolor': 'white'}) if len(values) > 1  else ax.text(v - (v/2),0 , v, {'backgroundcolor': 'white'})

    

comorbs_plot(flattened_data.iloc[:,-1:], cols=1, width=10, height=5, hspace=0.45, wspace=0.5)
flattened_data['AGE_PERCENTIL'] = flattened_data['AGE_PERCENTIL'].apply(lambda row: int(row[:1]) if row[:1] != 'A' else 9)
model_data = flattened_data.dropna()

X = model_data.drop(['ICU', 'ICU_int'], axis=1)

y = model_data['ICU_int']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(max_depth=4, max_leaf_nodes=5, random_state=42)

clf.fit(X_train, y_train)

importance = clf.feature_importances_

importance = pd.DataFrame(importance, index=X.columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2));
clf.score(X_test, y_test)