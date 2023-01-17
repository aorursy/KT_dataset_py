# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

df_test = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df.shape
df.dtypes
for col in df.columns :

    print('Column Name : ' +str(col))

    print(df[col].value_counts())

    print('***************')

    
for col in df.columns :

    index = np.array(df[col].value_counts().index)

    index = index.astype('str')

    if(len(np.where(index == '?')[0])>0):

        print('Missing Values in : '+col)

    
df['class'] = df['class'].replace({'p':1,'e':0})
df['stalk-root'] = df['stalk-root'].replace({'?':np.NAN})
df.dropna(inplace = True)
def OneHotEncodeing (columns) :

 try :

    global df

    df_tempp = pd.DataFrame()

    df_temp = pd.DataFrame()

    flag = False

    for col in columns :

      if(not(flag)):

        df_tempp = pd.get_dummies(df[str(col)], prefix= str(col))

        flag = True

      else :

        df_temp = pd.get_dummies(df[str(col)], prefix= str(col))

        df_tempp = pd.concat([df_tempp,df_temp],axis = 1 )

      df.drop(columns= str(col),axis = 'columns',inplace = True)

    df = pd.concat([df_tempp,df],axis = 1 )

 except :

   print('Error at : ',col)
OneHotEncodeing (['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',

       'ring-type', 'spore-print-color', 'population', 'habitat'])
df_corr_mat = df.corr()['class']
feature_lst = []

corrleation_val = 0.3

for index in df_corr_mat.index :

  if((df_corr_mat[index]>=corrleation_val) or (df_corr_mat[index] <= -corrleation_val)):

    if(index != 'class') :

      feature_lst.append(index)
figure = plt.figure(figsize= (10,8))

ax = figure.add_subplot(111)

df_corr_mat[feature_lst].plot(kind = 'bar',ax=ax)
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
X_train = train[feature_lst]

X_val = validate[feature_lst]

X_test = test[feature_lst]

Y_train = train['class']

Y_val = validate['class']

Y_test = test['class']
c = 100

list = []

while c >=0.0001:

    list.append(c/2)

    c = c-c/2

    

score_list_val = []

score_list_train = []

C_list = []

for param in list :

    clf = LinearSVC(loss ="hinge", C=param,max_iter=10000)

    clf.fit(X_train,Y_train)

    Y_predict_val = clf.predict(X_val)

    score_list_val.append(f1_score(Y_val,Y_predict_val))

    

    Y_predict_train = clf.predict(X_train)

    score_list_train.append(f1_score(Y_train,Y_predict_train))

    C_list.append(str(param))

    

    

df_train_score =  pd.DataFrame(data = score_list_train,index = C_list,columns = ['Train_F1_Score'])

df_val_score =  pd.DataFrame(data = score_list_val,index = C_list,columns = ['Validation_F1_Score'])
df_train_score.reset_index(inplace = True)

df_val_score.reset_index(inplace = True)
figure = plt.figure(figsize= (18,5))

ax = figure.add_subplot(111)

df_val_score.plot(ax= ax,x='index',y='Validation_F1_Score',marker = 'o')

df_train_score.plot(ax= ax,x='index',y='Train_F1_Score',marker = 'o')

ax.set_xlabel("C parameter Values")

ax.set_title("F1_score VS parameter C")
df_val_score
C = 0.09765625



clf = LinearSVC(loss ="hinge", C=C,max_iter=10000)

clf.fit(X_train,Y_train)

Y_predict = clf.predict(X_test)
print('recall_score is : ' +str(recall_score(Y_test,Y_predict)))

print('precision_score is : ' +str(precision_score(Y_test,Y_predict)))

print('f1_score is : ' +str(f1_score(Y_test,Y_predict)))

cnf_matrix_val = confusion_matrix(Y_test,Y_predict)

sns.heatmap(pd.DataFrame(cnf_matrix_val), annot=True, cmap="YlGnBu" ,fmt='g')