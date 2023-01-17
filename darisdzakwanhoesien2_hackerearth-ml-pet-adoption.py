# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test = pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/test.csv")

train = pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/train.csv")
test
train
test
train.drop(['breed_category','pet_category'],axis=1)
dataset = pd.concat([train.drop(['breed_category','pet_category'],axis=1),test])

dataset['issue_date'] = pd.to_datetime(dataset['issue_date'])

dataset['listing_date'] = pd.to_datetime(dataset['listing_date'])

dataset['duration'] = (dataset['listing_date'] - dataset['issue_date']).dt.days  

dataset = dataset.drop(['pet_id','issue_date','listing_date'],axis=1)

dataset 
info = pd.DataFrame()

info['length(m)'] = [np.percentile(dataset['length(m)'],25*i) for i in range(1,4)]

info['height(cm)'] = [np.percentile(dataset['height(cm)'],25*i) for i in range(1,4)]

info['duration'] = [np.percentile(dataset['duration'],25*i) for i in range(1,4)]

info
info.loc[3] = [2.5*info.loc[0,column] - 1.5*info.loc[2,column] for column in info.columns]

info.loc[4] = [2.5*info.loc[2,column] - 1.5*info.loc[0,column] for column in info.columns]

info
def range_part(column,value):

    if value > info.loc[4,column]:

        return 5

    elif value > info.loc[2,column]:

        return 4

    elif value > info.loc[1,column]:

        return 3

    elif value > info.loc[0,column]:

        return 2

    elif value > info.loc[3,column]:

        return 1

    else:

        return 0
dataset.dtypes
df = dataset['color_type'].value_counts().plot(kind='barh')

df.plot(figsize=(10,10));
numerical = dataset.dtypes[dataset.dtypes != object].index
dataset['condition'].value_counts().plot(kind='barh')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,3,figsize=(10,10))



for i in range(3):

    dataset.boxplot(column=list(numerical)[i], ax=ax[0,i])

    ax[0,i].title.set_text(numerical[i])



for i in range(3):

    dataset.boxplot(column=list(numerical)[i+3], ax=ax[1,i])

    ax[1,i].title.set_text(numerical[3+i])
from tqdm import tqdm

tqdm.pandas()

dataset['length_range'] = dataset['length(m)'].progress_apply(lambda x:range_part('length(m)',x))

dataset['height_range'] = dataset['height(cm)'].progress_apply(lambda x:range_part('height(cm)',x))

dataset['duration_range'] = dataset['duration'].progress_apply(lambda x:range_part('duration',x))

dataset
from sklearn.preprocessing import LabelEncoder

dataset['color_number'] = LabelEncoder().fit_transform(dataset['color_type'])

dataset = dataset[['condition','length_range','height_range','duration_range','color_number','X1','X2']].fillna(-1)

dataset
dataset['X2'].value_counts().plot(kind='barh')
dataset['X1'].value_counts().plot(kind='barh')
from sklearn.linear_model import LogisticRegression

X_tr, y_tr = dataset.iloc[:len(train)], train['breed_category']

X_test = dataset.iloc[len(train):]

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_tr, y_tr)

y_LR = LR.predict(X_test)

pd.DataFrame(y_LR, columns=['Value'])['Value'].value_counts().plot(kind='barh')
from sklearn.svm import SVC

SVM = SVC(decision_function_shape="ovo").fit(X_tr, y_tr)

y_SVC = SVM.predict(X_test)

pd.DataFrame(y_SVC, columns=['Value'])['Value'].value_counts().plot(kind='barh')
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_tr, y_tr)

y_RF = RF.predict(X_test)

pd.DataFrame(y_RF, columns=['Value'])['Value'].value_counts().plot(kind='barh')
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_tr, y_tr)

y_NN = NN.predict(X_test)

pd.DataFrame(y_NN, columns=['Value'])['Value'].value_counts().plot(kind='barh')
z_tr = train['pet_category']

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_tr, z_tr)

z_LR = LR.predict(X_test)

SVM = SVC(decision_function_shape="ovo").fit(X_tr, z_tr)

z_SVC = SVM.predict(X_test)

RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_tr, z_tr)

z_RF = RF.predict(X_test)

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_tr, z_tr)

z_NN = NN.predict(X_test)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,2,figsize=(10,10))



title = ['LR', 'SVC', 'RF','NN']

y_value = [y_LR, y_SVC, y_RF,y_NN]

z_value = [z_LR, z_SVC, z_RF,z_NN]



for i in range(2):

    pd.DataFrame(y_value[i], columns=['Value'])['Value'].value_counts().plot(kind='barh', ax=ax[0,i])

    ax[0,i].title.set_text(title[i])



for i in range(2):

    pd.DataFrame(y_value[2+i], columns=['Value'])['Value'].value_counts().plot(kind='barh', ax=ax[1,i])

    ax[1,i].title.set_text(title[2+i])
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,2,figsize=(10,10))



title = ['LR', 'SVC', 'RF','NN']

y_value = [y_LR, y_SVC, y_RF,y_NN]

z_value = [z_LR, z_SVC, z_RF,z_NN]



for i in range(2):

    pd.DataFrame(z_value[i], columns=['Value'])['Value'].value_counts().plot(kind='barh', ax=ax[0,i])

    ax[0,i].title.set_text(title[i])



for i in range(2):

    pd.DataFrame(z_value[2+i], columns=['Value'])['Value'].value_counts().plot(kind='barh', ax=ax[1,i])

    ax[1,i].title.set_text(title[2+i])
train.boxplot(column='X1')
train['breed_category'].value_counts().plot(kind='barh')
train['pet_category'].value_counts().plot(kind='barh')
[y_LR, y_SVC, y_RF,y_NN]
test
submission = pd.DataFrame()

submission['pet_id'] = test['pet_id']

submission['breed_category'] = y_LR

submission['pet_category'] = z_LR

submission.to_csv('submission_LogisticRegression.csv',index=False)
submission = pd.DataFrame()

submission['pet_id'] = test['pet_id']

submission['breed_category'] = y_SVC

submission['pet_category'] = z_SVC

submission.to_csv('submission_SupportVectorMachine.csv',index=False)
submission = pd.DataFrame()

submission['pet_id'] = test['pet_id']

submission['breed_category'] = y_RF

submission['pet_category'] = z_RF

submission.to_csv('submission_RandomForest.csv',index=False)
submission = pd.DataFrame()

submission['pet_id'] = test['pet_id']

submission['breed_category'] = y_NN

submission['pet_category'] = z_NN

submission.to_csv('submission_NeuralNetwork.csv',index=False)