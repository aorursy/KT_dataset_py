import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))

pd.read_csv('../input/train.csv').head()
df = pd.read_csv('../input/train.csv')
df = df.drop('PassengerId', axis=1)
#df = df.dropna(subset=['Embarked'])

plt.figure(figsize=(15,5))
plt.title('Null Values')
ax = sns.heatmap(df.isnull().astype('int'), cmap="inferno")
plt.show()
#df[(df.Name.str.contains('Mr.')) | (df.Name.str.contains('Mrs.')) | (df.Name.str.contains('Miss.'))].Age.plot()

#title_key = {'Mr':1, 'Mrs':2, 'Miss':3, 'None':4}
titles = []
for x in df.Name:
    if 'mrs' in x.lower():
        titles.append('mrs')
    elif 'mr' in x.lower():
        titles.append('mr')
    elif 'miss' in x.lower():
        titles.append('miss')
    else:
        titles.append('none')
df['Title'] = titles
df.head()
plt.figure(figsize=(10,5))
ax = sns.boxplot(x='Title', y='Age', data=df)
plt.show()
df = df.drop(['Age','Name','Cabin','Ticket'], axis=1)
df = df.dropna(subset=['Embarked'])
df.head()
from sklearn.preprocessing import LabelEncoder
sex_lab = LabelEncoder()
#ticket_lab = LabelEncoder()
fare_lab = LabelEncoder()
embarked_lab = LabelEncoder()
title_lab = LabelEncoder()

df.Sex = sex_lab.fit_transform(df.Sex)
#df.Ticket = ticket_lab.fit_transform(df.Ticket)
df.Embarked = embarked_lab.fit_transform(df.Embarked)
df.Title = title_lab.fit_transform(df.Title)

df.head()
import os
import itertools
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def Multi_KNN_Class(X, y, test_s=0.1, neighbors=[1,2,3], p_val=[1,2], leaf=[30], iterations=20, fig_s=(15,9), path=os.getcwd(), plot=False, verbose=True):
    """test out all combinations of hyperparameters to find the best model configuration. Returns statistics for mean and standard
          deviation of accuracy over the amount of iterations for each hyperparameter settings."""
    mu_sigma_list = []
    for s in list(set(itertools.product(neighbors, p_val, leaf))):
        i, j, k = s
        acc_list = []
        knn = KNeighborsClassifier(n_neighbors=i, p=j, leaf_size=k)
        for r in range(iterations):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_s)
            knn.fit(X_train, y_train)
            acc = knn.score(X_val, y_val)
            acc_list.append(acc)
            
        mu = np.mean(acc_list)
        sigma = np.std(acc_list)
        mu_sigma_list.append(('{}_NN__p_{}__leafsize_{}'.format(i,j,k), mu, sigma))
        if verbose: print('{}_NN__p_{}__leafsize_{}'.format(i,j,k), mu, sigma)
        
        if plot:
            x_axis = list(range(len(acc_list)))
            plt.figure(figsize = fig_s)
            text = 'Accuracy: {}_NN__p_{}__leafsize_{}'.format(i,j,k)
            plt.title(text)
            plt.plot(x_axis, acc_list)
            plt.save_fig(path + '/{}.png'.format(text))
    return mu_sigma_list

X = np.array(df.drop('Survived', axis=1))
y = np.array(df.Survived)

Multi_KNN_Class(X,y, test_s=0.1)

y = np.array(df.Survived)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
knn = KNeighborsClassifier(n_neighbors=3, p=1)
knn.fit(X_train, y_train)
acc = knn.score(X_val, y_val)
print(acc)
df_t = pd.read_csv('../input/test.csv')
titles = []
for x in df_t.Name:
    if 'mrs' in x.lower():
        titles.append('mrs')
    elif 'mr' in x.lower():
        titles.append('mr')
    elif 'miss' in x.lower():
        titles.append('miss')
    else:
        titles.append('none')
df_t['Title'] = titles

# plt.figure(figsize=(15,5))
# plt.title('Null Values')
# ax = sns.heatmap(df_t.isnull().astype('int'), cmap="inferno")
# plt.show()
df_t = df_t.drop(['Age','Name','Cabin','Ticket'], axis=1)
df_t.Fare = df_t.Fare.fillna(np.mean(df_t.Fare))

df_t.Sex = sex_lab.transform(df_t.Sex)
#df.Ticket = ticket_lab.fit_transform(df.Ticket)
df_t.Embarked = embarked_lab.transform(df_t.Embarked)
df_t.Title = title_lab.transform(df_t.Title)
df_t.head()
X_t = np.array(df_t.drop('PassengerId', axis=1))
y_t = knn.predict(X_t)
print(y_t)
print(len(y_t),'rows')
submit = df_t[['PassengerId']].copy()
submit['Survived'] = y_t
submit.head()
submit.to_csv('results_nn.csv',index=False)
print('wrote to csv')