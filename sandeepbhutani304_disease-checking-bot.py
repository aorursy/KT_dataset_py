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
df = pd.read_csv('/kaggle/input/symptom-checker/Training.csv')

df_test = pd.read_csv('/kaggle/input/symptom-checker/Testing.csv')

df
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(pd.concat([df['prognosis'], df_test['prognosis']]))

le.fit_transform(df['prognosis'])

# le.classes_

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(df[df.columns.difference(['prognosis'])], le.fit_transform(df['prognosis']))
y_pred=model.predict(df_test[df_test.columns.difference(['prognosis'])])
from sklearn import metrics

from sklearn.metrics import classification_report

y_true = le.fit_transform(df_test['prognosis'])

print("Accuracy:",metrics.accuracy_score(y_true, y_pred))

print(classification_report(y_true, y_pred, target_names=df_test['prognosis']))
le.inverse_transform(model.classes_)
def get_disease(model, df, le):

    y_true = model.predict_proba(df)

#     print(y_true)

    i = np.argmax(y_true)

#     print(i)

#     print(model.score())

#     print(y_true)

#     print(model.classes_)

#     print(model.classes_[i])

    disease = le.inverse_transform([model.classes_[i]])

    return disease[0] , y_true[0][i]



def filter_cols(model, subsymptoms_df, le):

    y_true = model.predict_proba(subsymptoms_df)

    #find possible diseases

    dis = []

    for i in range(len(y_true[0])):

        if y_true[0][i] > 0:

            dis.append(i)

    disname = le.inverse_transform(dis)

#     print(dis)

    print("Possible Diseases: >>> ", disname)

    #for possible diseases find remaining symptoms

    df_fil = df[df['prognosis'].isin(disname)]

    df_fil = df_fil.loc[:, (df_fil != 0).any(axis=0)]

    df_fil = df_fil[df_fil.columns.difference(['prognosis'])]

    print("Remaining symptoms: ", df_fil.columns.to_list())

    return df_fil.columns

    

    

df_copy = df.iloc[0:1,:].copy() #pd.DataFrame().reindex_like(df)

df_copy.append(pd.Series(), ignore_index=True)

for c in df_copy.columns:

    df_copy[c] = 0

df_copy = df_copy[df_copy.columns.difference(['prognosis'])]

symptoms = df.columns

    



symptoms_covered=[]

i = 0

print("Start with any symptom out of : ", symptoms.to_list())

s = input("Mention the first symptom: ")

df_copy[s]=1

symptoms_covered.append(s)

symptoms = filter_cols(model, df_copy, le)

print("Now please type 0 or 1 for possible symptoms: ")

while (len(symptoms) > 0):

#     print("symptoms_covered", symptoms_covered)

# for s in symptoms:

    for s in symptoms.to_list():

#         print("---", s, symptoms.to_list())

        if s not in symptoms_covered:

#             print(s, "not in ", symptoms_covered)

            break

#         else:

#             print(s, "is in ", symptoms_covered)

            

    print(s + " ? (1 or 0 please)")

    a = input(s + " ? (1 or 0 please)")

    i += 1

    df_copy[s] = a

    ret = get_disease(model, df_copy, le)

    symptoms = filter_cols(model, df_copy, le)

    symptoms_covered.append(s)

    print("Suspected disease is : ", ret)

    if ret[1]>0.8:

        break