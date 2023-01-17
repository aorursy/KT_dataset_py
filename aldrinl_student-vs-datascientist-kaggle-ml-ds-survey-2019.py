import numpy as np 

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')





import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split as train_valid_split

from sklearn.metrics import classification_report



import eli5



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
questions_only = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

with pd.option_context('display.max_colwidth', 10000):

    display(questions_only.T)
multiple_choice = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')[1:]

multiple_choice.profile_report(title='Multiple Choice Responses',style={'full_width':True})
multiple_choice['Q5'].value_counts().sort_values(ascending=True).plot(kind='barh')
condition = (multiple_choice['Q5']=='Data Scientist') | (multiple_choice['Q5']=='Student')

multiple_choice[condition].profile_report(title='Multiple Choice Responses',style={'full_width':True})
condition = (multiple_choice['Q5']=='Data Scientist') | (multiple_choice['Q5']=='Student')

df = multiple_choice[condition].reset_index(drop=True)

other_text_cols = [col for col in df.columns if 'OTHER_TEXT' in col]

df = df.drop(other_text_cols,axis=1)

df = df.rename(columns={'Time from Start to Finish (seconds)':'Duration'})

df.head()
def cat_encoding(df,map_dict):

    for col in map_dict.keys():

        df[col] = df[col].map(map_dict[col])

    return df



df['Duration'] =  df['Duration'].astype(float)

cat_cols = df.select_dtypes('object').columns



cat_mapping = {}

for col in cat_cols:

    values = list(df[col].unique())

    LE = LabelEncoder().fit(values)

    cat_mapping[col] = dict(zip(LE.classes_, LE.transform(LE.classes_)))

    

df = cat_encoding(df,cat_mapping)
y_col = 'Q5'

y = df[y_col]

Xs = df.drop(y_col,axis=1).fillna(-999)



X_train,X_valid,y_train,y_valid = train_valid_split(Xs, y, test_size = .2,

                                                    random_state=0)

X_train.shape,X_valid.shape
%%time

model = RandomForestClassifier(n_estimators=100,

                               random_state=0,n_jobs=-1)

model.fit(X_train,y_train)
preds = model.predict_proba(X_train)[:,1]

plt.hist(preds,bins=100)

plt.show();

print('train_report',classification_report(y_train,np.round(preds)))



preds = model.predict_proba(X_valid)[:,1]

plt.hist(preds,bins=100)

plt.show();

print('valid_report',classification_report(y_valid,np.round(preds)))
eli5.show_weights(model,feature_names=list(X_train.columns))
X_valid.loc[2569,:]
eli5.show_prediction(model,X_valid.loc[2569,:],feature_names=list(X_train.columns), top=20)
X_valid.loc[5683,:]
eli5.show_prediction(model,X_valid.loc[5683,:],feature_names=list(X_train.columns), top=20)