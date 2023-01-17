import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import warnings 

from sklearn.utils import shuffle

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

os.listdir('../input/testtraindata/')
base_train_dir = '../input/testtraindata/Train_Set/'

base_test_dir = '../input/testtraindata/Test_Set/'
test_data = pd.DataFrame(columns = ['activity','ax','ay','az','gx','gy','gz'])

files = os.listdir(base_test_dir)

for f in files:

    df = pd.read_csv(base_test_dir+f)

    df['activity'] = (f.split('.')[0].split('_')[-1])+' '

    test_data = pd.concat([test_data,df],axis = 0)

test_data.activity.unique()
print(test_data.info())

print('Shape: ',test_data.shape)
train_data = pd.DataFrame(columns = ['activity','ax','ay','az','gx','gy','gz'])

train_folders = os.listdir(base_train_dir)



for tf in train_folders:

    files = os.listdir(base_train_dir+tf)

    for f in files:

        df = pd.read_csv(base_train_dir+tf+'/'+f)

        train_data = pd.concat([train_data,df],axis = 0)

train_data.activity.unique()
print(train_data.info())

print('Shape: ',train_data.shape)
dataset = pd.concat([train_data, test_data], axis = 0, ignore_index=True)

print(dataset.activity.unique())
dataset = shuffle(dataset)

dataset.reset_index(drop = True,inplace = True)

dataset.head()
df_dummies = pd.get_dummies(dataset['activity'])

df_dummies.head()
final_dataset = pd.concat([df_dummies, dataset], axis = 1)

final_dataset.drop(['activity','walking '],axis = 1,  inplace = True)

final_dataset.head()
import matplotlib.pyplot as plt

import seaborn as sns

f,ax = plt.subplots(figsize = (8,10))

sns.set(style= 'whitegrid')

plt.subplot(2,1,1)

dfa = dataset.groupby('activity', as_index=False)['ax','ay','az'].mean()

dfa = dfa.melt('activity')

sns.barplot(dfa.variable,dfa.value, hue = dfa.activity)

plt.subplot(2,1,2)

dfg = dataset.groupby('activity', as_index=False)['gx','gy','gz'].mean()

dfg = dfg.melt('activity')

sns.barplot(dfg.variable,dfg.value, hue = dfg.activity)
X = np.array(final_dataset.iloc[:,3:])

y = np.array(final_dataset.iloc[:,:3])



print('X: ',X.shape)

print('y: ', y.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

#from sklearn.model_selection import train_test_split

#X_train, X_val, y_train, y_val  = train_test_split(X,y, test_size = 0.2)

#model.fit(X_train,y_train)

#model.score(X_test,y_test)

from sklearn.model_selection import cross_val_score

print('Cross Val Accuracy: {:0.2f}'.format(cross_val_score(model,X,y, cv = 5).mean()*100) + '%')
final_dataset = pd.concat([df_dummies, dataset], axis = 1)

final_dataset.drop(['activity'],axis = 1,  inplace = True)

X = np.array(final_dataset.iloc[:,4:])

y = np.array(final_dataset.iloc[:,:4])



print('X: ',X.shape)

print('y: ', y.shape)


from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier



#model_lr = LogisticRegression()

model_nb = GaussianNB()

model_dt = DecisionTreeClassifier()



#print('Cross Val Accuracy (LR): {:0.2f}'.format(cross_val_score(model_lr,X,y, cv = 5).mean()*100) + '%')

#print('Cross Val Accuracy (NB): {:0.2f}'.format(cross_val_score(model_nb,X,y, cv = 5).mean()*100) + '%')

print('Cross Val Accuracy (DT): {:0.2f}'.format(cross_val_score(model_dt,X,y, cv = 5).mean()*100) + '%')