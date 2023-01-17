import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split as train_valid_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report



import json



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
gameplay = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")

gameplay_test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")

gameplay.info()
plt.figure(figsize=(20,8))

sns.boxplot(x='event_code',y='event_count',data=gameplay)

plt.plot();
g = gameplay.game_time.replace(0,np.nan).dropna().value_counts()

plt.hist(g.values,bins=100)

plt.xlabel('GameTime')

plt.ylabel('Count')

plt.show()
g = gameplay.world.value_counts()

plt.bar(g.index,g.values)

plt.xlabel('Worlds')

plt.ylabel('Count')

plt.show()
g = gameplay.type.value_counts()

plt.bar(g.index,g.values)

plt.xlabel('Media Type')

plt.ylabel('Count')

plt.show()
g = gameplay.title.value_counts()

plt.figure(figsize=(5,8))

plt.barh(g.index,g.values)

plt.xlabel('Count')

plt.ylabel('Media Title')

plt.show()
labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

labels.info()
labels['accuracy_group'].value_counts().plot(kind='bar');
labels['num_correct'].value_counts().plot(kind='bar');
labels['num_incorrect'].value_counts().plot(kind='bar');
labels['accuracy'].value_counts().plot(kind='hist',bins=100);
gameplay = pd.merge(gameplay,labels[['game_session','accuracy_group']],on=['game_session'],how='left')

gameplay.head()
def dt_parts(df,dt_col):

    if(df[dt_col].dtype=='O'):

        df[dt_col] = pd.to_datetime(df[dt_col])

    df['year'] = df[dt_col].dt.year.astype(np.int16)

    df['month'] = df[dt_col].dt.month.astype(np.int8)

    df['day'] = df[dt_col].dt.day.astype(np.int8)

    df['hour'] = df[dt_col].dt.hour.astype(np.int8)

    df['minute'] = df[dt_col].dt.minute.astype(np.int8)

    df['second'] = df[dt_col].dt.second.astype(np.int8)

    df.drop(dt_col,axis=1,inplace=True)

    return df

    

def category_mapping(df,map_dict):

    for col in map_dict.keys():

        df[col] = df[col].map(map_dict[col])

        df[col] = df[col].astype(np.int16)

    return df
drop_cols = ['game_session','event_data','installation_id']

gameplay.drop(drop_cols,axis=1,inplace=True)

gameplay = dt_parts(gameplay,'timestamp')



gameplay_mapping = {}

cat_cols = gameplay.select_dtypes('object').columns

for col in cat_cols:

    values = list(gameplay[col].unique())+list(gameplay_test[col].unique())

    LE = LabelEncoder().fit(values)

    gameplay_mapping[col] = dict(zip(LE.classes_, LE.transform(LE.classes_)))

    

gameplay = category_mapping(gameplay,gameplay_mapping)

gameplay['accuracy_group'] = gameplay['accuracy_group'].fillna(method='bfill')

gameplay['accuracy_group'] = gameplay['accuracy_group'].fillna(method='ffill')

gameplay.head()
target_col = 'accuracy_group'

y = gameplay[target_col]

Xs = gameplay.drop(target_col,axis=1)



X_train, X_valid, y_train, y_valid = train_valid_split(Xs, y, test_size=0.2, random_state=0)

X_train.shape,X_valid.shape
%%time

model = RandomForestClassifier(n_estimators=15,

                              random_state=0,n_jobs=-1)

model.fit(X_train,y_train)
def get_evaluations(model):

    preds = model.predict(X_train)

    plt.hist(preds)

    plt.title('training predictions')

    plt.show();

    print('train_report',classification_report(y_train,preds))

    preds = model.predict(X_valid)

    plt.hist(preds)

    plt.title('validation predictions')

    plt.show();

    print('valid_report',classification_report(y_valid,preds))



get_evaluations(model)
# gameplay_test was already loaded at the start of this notebook

gameplay_test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")

installation_id = gameplay_test['installation_id']

gameplay_test.head()
drop_cols = ['game_session','event_data','installation_id']

gameplay_test.drop(drop_cols,axis=1,inplace=True)

gameplay_test = dt_parts(gameplay_test,'timestamp')

gameplay_test = category_mapping(gameplay_test,gameplay_mapping)

gameplay_test.shape
preds_df = pd.DataFrame()

preds_df['installation_id'] = installation_id

preds_df['accuracy_group'] = model.predict(gameplay_test)



#this will be used to find which is the majority classif

preds_df['counter'] = 1

print(preds_df.shape)
preds_df = preds_df.groupby(['installation_id','accuracy_group'],as_index=False).sum()

preds_df['agg'] = preds_df.groupby(['installation_id'],as_index=False)['counter'].transform(np.mean)

preds_df = preds_df.sort_values('agg').drop_duplicates('installation_id')

preds_df = preds_df.sort_values('installation_id')

print(preds_df.shape)

preds_df.head()
sub_df = preds_df[['installation_id','accuracy_group']]

sub_df['accuracy_group'] = sub_df['accuracy_group'].astype(int)

sub_df.head()
sub_df.to_csv('submission.csv',index=False)

sub_df['accuracy_group'].hist()

plt.show()