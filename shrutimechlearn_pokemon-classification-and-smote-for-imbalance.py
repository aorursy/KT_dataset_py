import warnings

warnings.filterwarnings('ignore')

import pandas_profiling as pd_prof

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()



import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

poke_data = pd.read_csv('../input/Pokemon.csv')
poke_data.info()
poke_data.head()
poke_data = pd.read_csv('../input/Pokemon.csv',index_col='#')
poke_data.head()
poke_data.describe()
#pd_prof.ProfileReport(poke_data)
poke_data.Legendary.replace({True:1,False:0},inplace=True)
poke_data.describe()
p = poke_data.hist(figsize = (20,20))
poke_data['Legendary'].value_counts()
poke_data['Type 1'].value_counts()
poke_data['Type 2'].value_counts()
type_1_list = list(poke_data['Type 1'].value_counts().index)

type_2_list = list(poke_data['Type 2'].value_counts().index)
type_1_list.sort()==type_2_list.sort()
dummy_type_1 = pd.get_dummies(poke_data['Type 1'])

dummy_type_2 = pd.get_dummies(poke_data['Type 2'])
dummy_final = pd.DataFrame(index=poke_data.index)

for column_name in type_2_list:

    dummy_final[column_name] = dummy_type_1[column_name] + dummy_type_2[column_name]
dummy_final.head()
dummy_final.describe()
dummy_final.info()
# I have a dataframe 'df' like this 



# Id    v1    v2

# 0     A     0.23

# 1     B     0.65

# 2     NaN   0.87



# If I use this function



# df1 = get_dummies(df)

# df1



# Id    v1_A    v1_B    v2

# 0     1       0       0.23

# 1     0       1       0.65

# 2     0       0       0.87 .
poke_data_new = pd.concat([poke_data,dummy_final],sort=False,axis=1)
poke_data_new.head()
poke_data_new.drop(['Type 1','Type 2'],axis=1,inplace=True)

poke_data_new.info()
poke_data_new['Generation'].value_counts()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(poke_data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
poke_data_new.columns
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

numerical =  pd.DataFrame(sc_X.fit_transform(poke_data_new[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def',

       'Speed']]),columns=['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def',

       'Speed'],index= poke_data_new.index

        )
#numerical

poke_clean_standard = poke_data_new.copy(deep=True)

poke_clean_standard[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def',

       'Speed']] = numerical[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def',

       'Speed']]
poke_clean_standard.head()
poke_clean_standard.describe()
x = poke_clean_standard.drop(["Legendary","Name"],axis=1)

y = poke_clean_standard.Legendary
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y,random_state = 2,test_size=0.4,stratify=y)
from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')

#Setup a knn classifier with k neighbors

#Setup a knn classifier with k neighbors

knn = KNeighborsClassifier(7)



knn.fit(X_train,y_train)

knn.score(X_test,y_test)
y_pred = knn.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import f1_score

f1_score(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import matthews_corrcoef

print(matthews_corrcoef(y_test,y_pred))
from sklearn.metrics import roc_curve

y_pred_proba = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=7) ROC curve')

plt.show()
from imblearn.over_sampling import SMOTE
poke_clean_standard.Legendary.value_counts()
sm = SMOTE(random_state=2, ratio = 'minority')

x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(x_train_res,y_train_res)

    

    train_scores.append(knn.score(x_train_res,y_train_res))

    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
knn = KNeighborsClassifier(2)



knn.fit(x_train_res,y_train_res)

knn.score(X_test,y_test)
y_pred = knn.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import f1_score

f1_score(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import matthews_corrcoef

print(matthews_corrcoef(y_test,y_pred))
from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

x_train_res, y_train_res = smt.fit_sample(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(x_train_res,y_train_res)

    

    train_scores.append(knn.score(x_train_res,y_train_res))

    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
knn = KNeighborsClassifier(2)



knn.fit(x_train_res,y_train_res)

knn.score(X_test,y_test)
y_pred = knn.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import f1_score

f1_score(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import matthews_corrcoef

print(matthews_corrcoef(y_test,y_pred))