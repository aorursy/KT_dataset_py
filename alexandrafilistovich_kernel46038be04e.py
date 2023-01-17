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
df = pd.read_csv('/kaggle/input/financialinclusionafrica/Train_v2.csv', 

                 sep=',')

df.head()
df_1 = pd.get_dummies(df, columns=['year', 'country', 'bank_account', 'location_type', 'cellphone_access','gender_of_respondent',

                                  'relationship_with_head', 'marital_status', 'education_level', 'job_type'])





df_1=df_1.drop(['bank_account_No', 'location_type_Rural', 'cellphone_access_No','gender_of_respondent_Female', 'uniqueid'], axis=1)

df_1.head()
df_2 = df_1

df_2.head()
from sklearn.model_selection import train_test_split

X = df_2.drop('bank_account_Yes', axis=1)

y = df_2['bank_account_Yes']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                     test_size=0.3, 

                                                     random_state=17)

X_train_all = df_2.drop('bank_account_Yes', axis=1)

y_train_all = df_2['bank_account_Yes']
X_train.head()
from sklearn.tree import DecisionTreeClassifier # импорт класса



tree = DecisionTreeClassifier(max_depth=6, random_state=17) # создали экземпляр класса

tree.fit(X_train, y_train) # обучили модель
# картинка

from sklearn.tree import export_graphviz

tree_dot = export_graphviz(tree)

print(tree_dot)
# Прогноз

y_pred = tree.predict(X_valid) # предсказания



# Точность прогнозов

from sklearn.metrics import accuracy_score

acc1 = accuracy_score(y_pred, y_valid)

print(acc1)
df_t = pd.read_csv('/kaggle/input/financialinclusionafrica/Test_v2.csv', 

                 sep=',')

df_t.head()
df_t = pd.get_dummies(df_t, columns=['year', 'country','location_type', 'cellphone_access','gender_of_respondent',

                                  'relationship_with_head', 'marital_status', 'education_level', 'job_type'])





df_t=df_t.drop(['location_type_Rural', 'cellphone_access_No','gender_of_respondent_Female', 'uniqueid'], axis=1)

df_t.head()
tree_all = DecisionTreeClassifier(max_depth=6, random_state=17)

tree_all.fit(X_train_all, y_train_all)
X_TESTING = df_t

X_TESTING.head()


y_TESTING = tree_all.predict(X_TESTING)

print(y_TESTING)

df_sub = pd.read_csv('/kaggle/input/financialinclusionafrica/SubmissionFile.csv', 

                 sep=',')

df_sub.head()
df_t1 = pd.read_csv('/kaggle/input/financialinclusionafrica/Test_v2.csv', 

                 sep=',')

y_ans_id = df_t1['uniqueid']

y_ans_id.head()
y_ans_cou = df_t1['country']

y_ans_cou.head()
y_ans = y_ans_id + ' x ' + y_ans_cou 

y_mer =y_ans_id + ' x ' + y_ans_cou 

y_ans.head()
for i in range(len(y_TESTING)):

    y_ans[i] = y_ans[i] + ',' + str(y_TESTING[i])

y_ans.head(20)
y_ans_f = y_ans

y_ans_f.head(20)
#y_ans_f.loc[-1] = 'uniqueid,bank_account'  # adding a row

#y_ans_f.index = y_ans_f.index + 1  # shifting index

#y_ans_f = y_ans_f.sort_index()

#y_ans_f.head(20)
y_ans_f.head(20)

type(y_ans_f)
index_sub = df_sub['uniqueid']

#index.loc[-1] = 'uniqueid'  # adding a row

#index.index = index.index + 1  # shifting index

#index = index.sort_index()

index_sub.head(20)
#y_ans_fin = y_ans_f

#y_ans_fin.reindex(index = index)



#y_ans_fin = pd.DataFrame(y_ans_f['bank_account'], index = index)



#y_ans_fin['uniqueid'].reindex(index)

#df1_mer = pd.DataFrame(data=y_ans_f.index, columns=['uniqueid'])

#df2_mer = pd.DataFrame(data=y_ans_f.values, columns=['bank_account'])

#df_mer_y = pd.merge(df1_mer, df2_mer, left_index=True, right_index=True)



#df_mer_y.head()



'''y_ans   ---   "indexes of test"

y_TESTING   ---   "answer for test"

index_sub   ---   "index of submission"'''



df2_mer = pd.DataFrame({'bank_account': y_TESTING})



df1_mer = pd.DataFrame(data=y_mer.values, columns=['bank_account'])

#df2_mer = pd.DataFrame(data=y_TESTING.values, columns=['bank_account'])

df_mer_y = pd.merge(df1_mer, df2_mer, left_index=True, right_index=True)

df_mer_y.columns = ['uniqueid', 'bank_account']

df_mer_y.head()
df_mer_sub = pd.DataFrame({'uniqueid': index_sub})

df_mer_sub.head()
df_mer_ans = df_mer_sub

df_mer_ans = df_mer_ans.merge(df_mer_y, on='uniqueid', how='left')

df_mer_ans.head()
ANSWER = df_mer_ans['uniqueid']

ANSWER.head()
for i in range(len(ANSWER)):

    if (df_mer_ans['bank_account'][i] == 1):

        ANSWER[i] = ANSWER[i] + ",1"

    else:

        ANSWER[i] = ANSWER[i] + ",0"

ANSWER.head()
Answer_fin = ANSWER

Answer_fin.loc[-1] = "uniqueid,bank_account"  # adding a row

Answer_fin.index = Answer_fin.index + 1  # shifting index

Answer_fin = Answer_fin.sort_index()

Answer_fin.head()
y_ans.to_csv('y_ans.csv',index=False)

Answer_fin.to_csv('my_ans_final.csv',index=False)