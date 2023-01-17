import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

data = pd.read_csv('../input/voicegender/voice.csv')
data.head()
data.shape
data.describe([.25,.50,.75,.80,.90])
data.info()
data.isna().sum() #no missing data
#univariate

def dist_male(x):

    if x == 'label':

        pass

    else:

        data[x][data['label']=='male'].plot.kde()

        plt.xlabel(x)

        plt.show()
def dist_female(y):

    if y == 'label':

        pass

    else:

        data[y][data['label']=='female'].plot.kde(color='maroon')

        plt.xlabel(y)

        plt.show()
cols = data.columns.drop('label')





for j, i in enumerate(cols):

#     print(j)

    dist_male(i)

    dist_female(i)
data_male = data[data['label']=='male'].drop('label', axis=1)

data_female = data[data['label']=='female'].drop('label', axis=1)
def box_plt_m(x,i):

    sns.boxplot(x=x, data=data_male)

    plt.show()

    



def box_plt_f(x,i):

    sns.boxplot(x=x, data=data_female)

    plt.show()
for j,i in enumerate(cols):

    plt.figure(figsize=(20,50))

    plt.subplot(21,2,j+1)

    box_plt_m(i,j)

    plt.figure(figsize=(20,50))

    plt.subplot(21,2,j+2)

    box_plt_f(i,j)

    print(j)
#check the correlation between features

#bivariate

data.corr()
plt.figure(figsize=(20,15))

sns.heatmap(data.corr(), annot=True, fmt='.2g')
# checking for more than .50 and -.50 correlation
plt.figure(figsize=(20,15))

sns.heatmap(data.corr(), annot=True, fmt='.2g', mask=~(((data.corr()) <=-.50) | ((data.corr())>=.50)))
print('The number of male in our output is: ',data[data['label']=='male'].shape[0])

print('The number of female in our output is: ',data[data['label']=='female'].shape[0])
y = data.iloc[:, -1]



from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

y
from sklearn.preprocessing import StandardScaler



X = data.iloc[:,:-2]

std_scaler = StandardScaler()

std_scaler.fit(X)



X = std_scaler.transform(X)
#splitting into train test



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=1)
from sklearn.svm import SVC

from sklearn import metrics



svc = SVC() #default parameters

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



print(f'The score for this model {svc.__class__.__name__} is {metrics.accuracy_score(y_test, y_pred)}')
svc = SVC(kernel='linear') #default parameters

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



print(f'The score for this model {svc.__class__.__name__} is {metrics.accuracy_score(y_test, y_pred)}')
svc = SVC(kernel='poly') #default parameters

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



print(f'The score for this model {svc.__class__.__name__} is {metrics.accuracy_score(y_test, y_pred)}')
from sklearn.model_selection import cross_val_score



svc = SVC()

score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(score)

print('The mean accuracy for the model on 10 K fold cross validation is: {%.3f}'%score.mean())
svc = SVC(kernel='linear')

score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(score)

print('The mean accuracy for the model on 10 K fold cross validation is: {%.3f}'%score.mean())
svc = SVC(kernel='poly')

score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(score)

print('The mean accuracy for the model on 10 K fold cross validation is: {%.2f}'%score.mean())
C_range = list(range(1,26))





acc_score = []



for i in C_range:

    svc = SVC(kernel='linear', C=i)

    score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(score.mean())

print('The best mean accuracy for the model on 10 K fold cross validation with a range of C value (0-25) is: {} and index {}'.format(max(acc_score), acc_score.index(max(acc_score))))

    
#plotting a graph



plt.plot(C_range, acc_score)

plt.xticks(np.arange(0,27,2))

plt.xlabel('C values')

plt.ylabel('Cross-Validated Accuracy')
#fine tuning to see which c is the best
C_range = list(np.arange(7,13,.1))



acc_score = []



for i in C_range:

    svc = SVC(kernel='linear', C=i)

    score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(score.mean())

print('The best mean accuracy for the model on 10 K fold cross validation with a range of C value (0-25) is: {} and index {}'.format(max(acc_score), acc_score.index(max(acc_score))))



plt.plot(C_range, acc_score)

plt.xticks(np.arange(7,14,1))

plt.xlabel('C values')

plt.ylabel('Cross-Validated Accuracy')
#checking Gamma for kernel=linear
gamma_range = [.00001,.0001,.001,.01,.1,1,10,100]





acc_score = []



for i in gamma_range:

    svc = SVC(kernel='linear', gamma=i)

    score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(score.mean())

# print('The best mean accuracy for the model on 10 K fold cross validation with a range of C value (0-25) is: {} and index {}'.format(max(acc_score), acc_score.index(max(acc_score))))

acc_score  
#checking c and Gamma for kernel=rbf
C_range = list(range(1,25))





acc_score = []



for i in tqdm(C_range):

    svc = SVC(kernel='rbf', C=i)

    score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(score.mean())

# print('The best mean accuracy for the model on 10 K fold cross validation with a range of C value (0-25) is: {} and index {}'.format(max(acc_score), acc_score.index(max(acc_score))))



acc_score
plt.plot(C_range, acc_score)

plt.xticks(np.arange(1,26,1))

plt.xlabel('C values')

plt.ylabel('Cross-Validated Accuracy')
gamma_range = [.00001,.0001,.001,.01,.1,1,10,100]





acc_score = []



for i in gamma_range:

    svc = SVC(kernel='rbf', gamma=i)

    score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(score.mean())

print('The best mean accuracy for the model on 10 K fold cross validation with a range of gamma is: {} and index {}'.format(max(acc_score), acc_score.index(max(acc_score))))

acc_score  
plt.plot(gamma_range, acc_score)

# plt.xticks(np.arange(0,9))

plt.xlabel('C values')

plt.ylabel('Cross-Validated Accuracy')
gamma_range = [.00001,.0001,.001,.01,.1,1,10,100]

C_range = list(range(1,25))





acc_score = []





for j in tqdm(gamma_range):

    for i in C_range:

        svc = SVC(kernel='rbf', C=i, gamma=j)

        score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

        acc_score.append(score.mean())

# print('The best mean accuracy for the model on 10 K fold cross validation with a range of C value (0-25) is: {} and index {}'.format(max(acc_score), acc_score.index(max(acc_score))))



temp = pd.DataFrame(acc_score)
temp['gamma_C'] = [(x,y) for x in gamma_range for y in C_range]

plt.plot(temp[0])

# plt.xticks(np.arange(0,27,2))

plt.xlabel('C values')

plt.ylabel('Cross-Validated Accuracy')
temp.sort_values(by=0,ascending=False)



temp.iloc[75,:]
degrees = [2,3,4,5,6]



acc_score = []



for i in degrees:

    svc = SVC(kernel='poly', degree=i)

    score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(score.mean())



print('The mean accuracy for the model on 10 K fold cross validation is: {}'.format(acc_score))
plt.plot( degrees, acc_score)

plt.xlabel('Power')

plt.ylabel('Cross-Validated Accuracy')
svc = SVC(kernel='rbf', C=4, gamma=.01)

score = cross_val_score(svc, X, y, cv=10, scoring='f1')

score.mean()
svc = SVC(kernel='rbf', C=4, gamma=.01)

score = cross_val_score(svc, X, y, cv=10, scoring='roc_auc')

score.mean()