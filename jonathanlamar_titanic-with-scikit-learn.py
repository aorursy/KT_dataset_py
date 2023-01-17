import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn import linear_model, tree, svm, naive_bayes, neighbors, ensemble

from keras.models import Sequential
from keras.layers import Dense
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
master_df = pd.concat([train_df.drop('Survived', axis=1), test_df], axis=0) # All feature engineering happens here
master_df.shape
master_df.sample(5)
def safe_isnan(x):
    # isnan, but can be applied to any type
    try:
        return np.isnan(x)
    except TypeError:
        return False # always a nonempty string in this case
def get_title(name_string):
    titles = {
        'Mrs' : 'Mrs',
        'Mr' : 'Mr',
        'Master' : 'Master',
        'Miss' : 'Miss',
        'Major' : 'Fancy',
        'Rev' : 'Fancy',
        'Dr' : 'Fancy',
        'Ms' : 'Miss',
        'Mlle' : 'Miss',
        'Col' : 'Fancy',
        'Capt' : 'Fancy',
        'Mme' : 'Mrs',
        'Countess' : 'Fancy',
        'Don' : 'Fancy',
        'Jonkheer' : 'Fancy'
    }
    for title in titles.keys():
        if name_string.find(title) != -1:
            return titles[title]
    return np.nan
master_df['title'] = master_df['Name'].apply(get_title)
master_df['fam_size'] = master_df['Parch'] + master_df['SibSp'] + 1
master_df['fare_per_person'] = master_df['Fare'] / master_df['fam_size']
def get_deck(cabin_string):
    decks = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
    if safe_isnan(cabin_string):
        return np.nan
    for deck in decks.keys():
        if cabin_string.find(deck) != -1:
            return decks[deck]
    return np.nan
master_df['deck'] = master_df['Cabin'].apply(get_deck)
D = {'C':0, 'S':1, 'Q':2, np.nan:np.nan}
master_df['Embarked'] = master_df['Embarked'].apply(lambda x: D[x])
drop_cols = ['PassengerId', 'Name', 'Cabin', 'Ticket']
master_df.drop(drop_cols, axis=1, inplace=True)
master_df.sample(5)
master_corr = master_df.corr()
master_corr
cols = list(master_corr.columns)
plt.figure()
plt.imshow(np.matrix(master_corr) - np.eye(len(cols)), cmap='seismic', norm=Normalize(vmin=-1, vmax=1))
plt.title('Training set correlation matrix')
plt.xticks(range(9),cols, rotation=90)
plt.yticks(range(9),cols)
_ = plt.colorbar()
master_df.apply(lambda x: sum(x.apply(lambda y: safe_isnan(y))), axis=0)
master_df['Embarked'].fillna(master_df['Embarked'].mode()[0], inplace=True) # Mode because categorical.
med = master_df['Fare'].median()
master_df['Fare'].fillna(med, inplace=True) # Median because numerical.
master_df.loc[master_df['fare_per_person'].isna(), 'fare_per_person'] = \
    med / master_df.loc[master_df['fare_per_person'].isna(),'fam_size']
master_corr['Age']
X = list(master_df['fam_size'].unique())
X.sort()
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.scatter(master_df['fam_size'], master_df['Age'], alpha=0.5)
meds = [master_df[master_df['fam_size']==i]['Age'].median() for i in X]
plt.plot(X, meds, c='orange', label='Median')
plt.title('Age and Family Size')
plt.ylabel('Age')
plt.xlabel('Family Size')
plt.xticks(X)
plt.legend()
plt.grid()

X = list(master_df['SibSp'].unique())
X.sort()
plt.subplot(2,2,2)
plt.scatter(master_df['SibSp'], master_df['Age'], alpha=0.5)
meds = [master_df[master_df['SibSp']==i]['Age'].median() for i in X]
plt.plot(X, meds, c='orange', label='Median')
plt.title('Age and SibSp')
plt.ylabel('Age')
plt.xlabel('SibSp')
plt.xticks(X)
plt.legend()
plt.grid()

plt.subplot(2,2,3)
plt.scatter(master_df['Pclass'], master_df['Age'], alpha=0.5)
meds = [master_df[master_df['Pclass']==i]['Age'].median() for i in range(1,4)]
plt.plot(range(1,4), meds, c='orange', label='Median')
plt.title('Age and Pclass')
plt.ylabel('Age')
plt.xlabel('Pclass')
plt.xticks(range(1,4))
plt.legend()
plt.grid()

X = list(master_df['deck'].dropna().unique())
X.sort()
plt.subplot(2,2,4)
plt.scatter(master_df['deck'], master_df['Age'], alpha=0.5)
meds = [master_df[master_df['deck']==i]['Age'].median() for i in X]
plt.plot(X, meds, c='orange', label='Median')
plt.title('Age and deck')
plt.ylabel('Age')
plt.xlabel('deck')
plt.xticks(X)
plt.legend()
_ = plt.grid()
title_dict = {'Mrs':0, 'Mr':1, 'Master':2, 'Miss':3, 'Fancy':4}
plt.figure()
plt.scatter(master_df['title'].apply(lambda x: title_dict[x]), master_df['Age'], alpha=0.5)
plt.title('Title and age')
plt.xlabel('Title')
plt.ylabel('Age')
plt.grid()
_ = plt.xticks(range(len(title_dict)), title_dict.keys())
masterdf_age = master_df[master_df['Age'].isna() == False]
masterdf_noage = master_df[master_df['Age'].isna() == True]
plt.figure()
plt.title('Most missing ages were 3rd class passengers')
plt.hist(
    [np.array(masterdf_age['Pclass']), np.array(masterdf_noage['Pclass'])],
    align='mid',
    bins=[1,2,3,4],
    edgecolor='black',
    label=['age known','age unknown']
)
plt.grid(axis='y')
plt.ylabel('count')
plt.xlabel('Class')
plt.xticks([1.5,2.5,3.5],['1st','2nd','3rd'])
_ = plt.legend()
sex_dict = {'female':0,'male':1}
plt.figure()
plt.title('More missing ages were men, but doesn\'t look significant')
plt.hist(
    [
        np.array(masterdf_age['Sex'].apply(lambda x:sex_dict[x])),
        np.array(masterdf_noage['Sex'].apply(lambda x:sex_dict[x]))
    ],
    label=['age known', 'age unknown'],
    align='mid',
    bins=[0,1,2],
    edgecolor='black'
)
plt.grid(axis='y')
plt.ylabel('count')
plt.xlabel('sex')
plt.xticks([0.5,1.5],['women','men'])
_ = plt.legend()
agedf_titles = np.array(
    masterdf_age['title'].apply(lambda x: title_dict[x]) # make titles integers in range(5)
).reshape(-1,1)
enc = OneHotEncoder()
enc.fit(agedf_titles)
agedf_titles = enc.transform(agedf_titles).toarray() # encode titles as 5D one-hot vectors
noagedf_titles = np.array(
    masterdf_noage['title'].apply(lambda x: title_dict[x]) # make titles integers in range(5)
).reshape(-1,1)
enc = OneHotEncoder()
enc.fit(noagedf_titles)
noagedf_titles = enc.transform(noagedf_titles).toarray() # encode titles as 5D one-hot vectors
X = np.concatenate((np.array(masterdf_age[['fam_size','Pclass']]), agedf_titles), axis=1)
y = np.array(masterdf_age['Age']).reshape(-1,1)
X_test = np.concatenate((np.array(masterdf_noage[['fam_size','Pclass']]), noagedf_titles), axis=1)

age_model = linear_model.LinearRegression()
age_model.fit(X, y)
master_df.loc[master_df['Age'].isna(),'Age'] = age_model.predict(X_test).reshape(-1,1)
def change_deck_entries(x):
    D = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}
    try:
        return D[x]
    except KeyError:
        return 'Unknown'
master_df['deck'] = master_df['deck'].apply(change_deck_entries)
master_df['deck'] = master_df['deck'].apply(change_deck_entries)
deck_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'Unknown':7}
train_cats = np.array(master_df.iloc[:891][['Sex','title','deck']].apply({
    'Sex' : lambda x: sex_dict[x],
    'title' : lambda x: title_dict[x],
    'deck' : lambda x: deck_dict[x]
}))
test_cats = np.array(master_df.iloc[891:][['Sex','title','deck']].apply({
    'Sex' : lambda x: sex_dict[x],
    'title' : lambda x: title_dict[x],
    'deck' : lambda x: deck_dict[x]
}))

enc = OneHotEncoder(n_values=[2,5,8])
enc.fit(np.concatenate((train_cats, test_cats)))
enc_train_cats = enc.transform(train_cats).toarray() # encoded as one-hot.  saving the unencoded versions for kNN
enc_test_cats = enc.transform(test_cats).toarray()

train_nums = np.array(master_df.iloc[:891][[
    'Pclass','Age','SibSp','Parch','Fare','Embarked','fam_size','fare_per_person'
]])
test_nums = np.array(master_df.iloc[891:][[
    'Pclass','Age','SibSp','Parch','Fare','Embarked','fam_size','fare_per_person'
]])

y = np.array(train_df['Survived']).reshape(-1,1)
X = np.concatenate((enc_train_cats, train_nums), axis=1) # This is for training the models
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
val_acc = []
for i in range(1,15):
    vali = 0
    for j in range(50):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf.fit(X_train, y_train)
        yval_pred = clf.predict(X_val).reshape(-1,1)
        vali += clf.score(X_val, y_val)
    val_acc.append(vali / 50)

plt.figure()
plt.plot(range(1,15),val_acc)
plt.title('Accuracy on validation set')
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
_ = plt.xticks(range(1,15))
acc = 0
for i in range(100): # get average accuracy on a range of splits
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X_train, y_train)
    acc += clf.score(X_val, y_val)
print('Decision tree average validation accuracy: {0:2.2f}%.'.format(acc))
acc = 0
for i in range(100):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train.flatten())
    acc += clf.score(X_val, y_val)
print('Logistic regression average validation accuracy: {0:2.2f}%.'.format(acc))
enc = OneHotEncoder(n_values=2)
enc.fit(y)
y_enc = enc.transform(y).toarray()
# normalizing the features first because neural nets are sensitive to that
X_train, X_val, y_train, y_val = train_test_split(scale(X), y_enc, test_size=0.2)
from keras.regularizers import l2
from keras.layers import Dropout
model = Sequential([
    Dense(16, activation='relu', input_dim=23, kernel_regularizer=l2(0.1)),
    Dense(8, activation='relu', kernel_regularizer=l2(0.05)),
    Dense(4, activation='relu', kernel_regularizer=l2(0.02)),
    Dense(2, activation='sigmoid')
])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
loss_val = []
acc_val = []
loss_train = []
acc_train = []
for i in range(200): # This will be slow...
    model.fit(X_train, y_train, batch_size=10, epochs=1, verbose=False)
    ev_val = model.evaluate(X_val, y_val, verbose=False)
    loss_val.append(ev_val[0])
    acc_val.append(ev_val[1])
    ev_train = model.evaluate(X_train, y_train, verbose=False)
    loss_train.append(ev_train[0])
    acc_train.append(ev_train[1])
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Loss')
plt.ylabel('Bin. Crossentropy')
plt.xlabel('Epochs')
plt.plot(loss_val, label='val')
plt.plot(loss_train, label='train')
plt.legend()

plt.subplot(1,2,2)
plt.title('Accuracy')
plt.ylabel('Proportion correct')
plt.xlabel('Epochs')
plt.plot(acc_val, label='val')
plt.plot(acc_train, label='train')
_ = plt.legend()
y_pred = model.predict(X_val)
y_pred = np.floor(y_pred / y_pred.max(axis=1).reshape(-1,1))
acc = accuracy_score(y_pred, y_val)
print('Neural network average validation accuracy: {0:2.2f}%.'.format(100*acc))
val_acc = []
for i in range(1,15):
    vali = 0
    for j in range(50):
        X_train, X_val, y_train, y_val = train_test_split(X, y.flatten(), test_size=0.25)
        clf = ensemble.RandomForestClassifier(n_estimators=i, max_depth=11)
        clf.fit(X_train, y_train)
        vali += clf.score(X_val, y_val)
    val_acc.append(vali / 50)

plt.figure()
plt.plot(range(1,15),val_acc)
plt.title('Accuracy on validation set')
plt.ylabel('Average Validation Accuracy')
plt.xlabel('Number of Trees')
_ = plt.xticks(range(1,15))
acc = 0
for i in range(100): # get average accuracy on a range of splits
    X_train, X_val, y_train, y_val = train_test_split(X, y.flatten(), test_size=0.25)
    clf = ensemble.RandomForestClassifier(n_estimators=50, max_depth=11)
    clf.fit(X_train, y_train)
    acc += clf.score(X_val, y_val)
print('Decision tree average validation accuracy: {0:2.2f}%.'.format(acc))
X_dec = np.concatenate([train_cats, scale(train_nums)], axis=1)
X_test_dec = np.concatenate([test_cats, scale(test_nums)], axis=1)
def dist(x1, x2):
    d = int(x1[0] != x2[0]) # sex
    d += int(x1[1] != x2[1]) # title
    d += int(x1[2] != x2[2]) # deck
    d += np.sqrt(sum( (x1 - x2)**2 )) # euclidean on rest
    return d
X_train, X_val, y_train, y_val = train_test_split(X_dec, y.flatten(), test_size=0.2)
valAcc = []
for n in range(5, 16):
    knn = neighbors.KNeighborsClassifier(n_neighbors=n, weights='distance', metric=dist, n_jobs=-1)
    knn.fit(X_train, y_train)
    valAcc.append(knn.score(X_val, y_val))

plt.figure(figsize=(10,4))
plt.plot(range(5, 16), valAcc, label='val. acc.')
plt.title('Accuracy vs. n_neighbors')
plt.ylabel('Average Validation Accuracy')
_ = plt.xlabel('n_neighbors')


