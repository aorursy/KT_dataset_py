import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))
# Sample Submission

df_sam = pd.read_csv("../input/sample_submission.csv")

print(df_sam.shape)

df_sam.head()
df_X = pd.read_csv("../input/X_train.csv")

print(df_X.shape)

df_X.head()
df_y = pd.read_csv("../input/y_train.csv")

print(df_y.shape)

df_y.head()
df_test = pd.read_csv("../input/X_test.csv")

print(df_test.shape)

df_test.head()
# No null values

_ = sns.heatmap(df_X.isnull(),yticklabels=False,cbar=False,cmap='viridis')
'''

0 - orientation_X

1 - orientation_Y

2 - orientation_Z

3 - orientation_W

4 - angular_velocity_X

5 - angular_velocity_Y

6 - angular_velocity_Z

7 - linear_acceleration_X

8 - linear_acceleration_Y

9 - linear_acceleration_Z

'''



plt.figure(figsize=(15,10))

sns.set(font_scale=1.5)

sns.heatmap(df_X.corr(),annot=True);
df_X = df_X.iloc[:,3:]

mapper = {'series_id': [],

         'surface':[]}

categories = dict()

rows = df_y.iterrows()

x = 0

for each in rows:

    serid = each[0]

    surface = each[1]['surface']

    if surface not in categories.keys():

        categories[surface] = x

        x = x+1

    for i in range(0,128):

        mapper['series_id'].append(serid)

        mapper['surface'].append(categories[surface])

    

df_y2 = pd.DataFrame(mapper)

df_y2 = df_y2['surface']

print(df_y2.shape)

df_y2.head()

from sklearn.preprocessing import normalize

df_X = normalize(df_X)

df_X = pd.DataFrame(df_X)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV



clf = KNeighborsClassifier()

grid_values = {'n_neighbors': [7]}

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y2, test_size=0, random_state = 69)



# default metric to optimize over grid parameters: accuracy

grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)

grid_clf_acc.fit(X_train, y_train)



print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)

print('Grid best score (accuracy): ', grid_clf_acc.best_score_)
X_t = df_test.iloc[:,3:]

X_t = pd.DataFrame(normalize(X_t))

clf = KNeighborsClassifier(n_neighbors=7)

clf.fit(X_train, y_train)

ans = clf.predict(X_t)

ans
serser = df_test['series_id']

final = {'surf':[]}

k = list(categories.keys())

v = list(categories.values())

for each in ans:

    final['surf'].append(k[each])

final_ans = pd.DataFrame(final)

temp = final_ans['surf']

final_ans['series_id'] = serser

final_ans['surface'] = temp

final_ans = final_ans.drop(['surf'], axis=1)

print(final_ans.shape)

final_ans.head()
df_g = final_ans.groupby(by='series_id')

prediction_dict = dict()

count_dict = dict()

for ser_id, df in df_g:

    prediction_dict[ser_id] = ''

    for each in df['surface']:

        if each not in count_dict.keys():

            count_dict[each] = 1

        else:

            count_dict[each] += 1

    count_items = list(count_dict.items())

    count_items.sort(key=lambda x: x[1], reverse=True)

    prediction_dict[ser_id] = count_items[0][0]

    count_dict = dict()

print(prediction_dict)
# Convert dictionary to dataframe

modified_final_ans = pd.DataFrame(list(prediction_dict.items()), columns=['series_id', 'surface'])

print(modified_final_ans.shape)

modified_final_ans.head()
modified_final_ans.to_csv('submission5.csv')
print(os.listdir("../working"))
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





submission5 = pd.read_csv('../working/submission5.csv')

# create a link to download the dataframe

create_download_link(submission5)