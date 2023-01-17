import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

import tensorflow.keras as tfk
%matplotlib inline



# == set seaborn as default == #

sns.set()



# == show all rows == #

pd.set_option('display.max_rows', None)
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test['Survived'] = np.nan



# == combine train and test data == #

dataset = pd.concat([train,test], axis=0)
train.shape
test.shape
dataset.shape
dataset.head()
dataset.tail()
dataset.describe()
dataset.corr()
dataset.isna().sum()
dataset.shape
# == define bar chart == #

def bar_chart(feature, label, stacked):

    indices = [dataset[dataset[label]==x][feature].dropna().value_counts() for x in dataset[label].dropna().unique()]

    df = pd.DataFrame(indices)

    df.index = [x for x in dataset[label].dropna().unique()]

    df.plot(kind='bar', stacked=stacked, figsize=(10,5))
# == define kde plot == #

def kde_plot(feature, label, l=None, r=None):

    if l is None: l = dataset[feature].min()

    if r is None: r = dataset[feature].max()

    facet = sns.FacetGrid(dataset, hue=label,aspect=4)

    facet.map(sns.kdeplot, feature, shade=True)

    facet.set(xlim=(0,dataset[feature].max()))

    facet.add_legend()

    plt.xlim(l,r)
bar_chart("Pclass", "Survived", True)

kde_plot('Pclass', 'Survived')
bar_chart('Sex','Survived', True)
kde_plot('Age', 'Survived')
bar_chart('SibSp', 'Survived', True)

kde_plot('SibSp', 'Survived')
bar_chart('Parch', 'Survived', True)
kde_plot('Fare', 'Survived')
bar_chart('Embarked', 'Survived', True)
from IPython.display import Image

Image(url='http://visualoop.com/media/2015/03/Titanic.jpg')
dataset.head(10)
dataset.tail(10)
dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
dataset['Title'].value_counts()
dataset.isna().sum()
dataset.head()
bar_chart('Title','Survived', True)
titles = ['Mr', 'Miss', 'Mrs', 'Master']

def title_mapping(s):

    if s in titles:

        return s

    return 'Others'



dataset['Title']=dataset['Title'].transform(title_mapping)
# == sanity check == #

dataset.head()
dataset[dataset['Title']=='Others']['Name']
bar_chart('Title','Survived', True)
dataset['Surname'] = dataset['Name'].str.extract('([A-Za-z]+)\,', expand=False)
freq_mapping = {name: value for name,value in dataset['Surname'].value_counts().items()}
dataset.head()
def surname_mapping(name):

    if freq_mapping[name] == 1:

        return 'Alone'

    else:

        return name



dataset['Surname'] = dataset['Surname'].transform(surname_mapping)
dataset['Surname'].value_counts()
plt.scatter(dataset['Age'], dataset['Title'], alpha=0.3)
dataset.isna().sum()
dataset['Age'].fillna(dataset.groupby('Title')['Age'].transform('median'),inplace=True)
dataset.isna().sum()
kde_plot('Age', 'Survived')

kde_plot('Age', 'Survived', 0, 20)

kde_plot('Age', 'Survived', 20, 40)

kde_plot('Age', 'Survived', 40, 60)
dataset['BinnedAge'] = pd.cut(dataset['Age'], bins=[0,10,17,24,34,62,dataset['Age'].max()], 

                              labels=['child','teen','young adult','adult', 'middle aged', 'elderly'])
dataset.head()
dataset.tail()
dataset['BinnedAge'].value_counts()
bar_chart('BinnedAge', 'Survived', True)
dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+1
dataset.head()
dataset.tail()
dataset.isna().sum()
kde_plot('FamilySize', 'Survived')
dataset['BinnedFamily'] = pd.cut(dataset['FamilySize'], bins=[0,2,4,dataset['FamilySize'].max()], 

                              labels=['small', 'medium', 'large'])
dataset.head(10)
dataset.isna().sum()
dataset[pd.isnull(dataset['Fare'])]
# == graph the fare against other features == #

rows = 3

cols = 1

fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

compare_to = ['Pclass', 'Age', 'Title']

for r in range(rows):

    dataset_ax = axs[r]

    dataset_ax.scatter(dataset['Fare'], dataset[compare_to[r]], alpha=0.4)

    dataset_ax.set_title(compare_to[r])



plt.tight_layout()
plt.scatter(dataset[dataset['Pclass']==3]['Fare'], dataset[dataset['Pclass']==3]['Pclass'], alpha=0.2)
dataset[dataset['Pclass']==3]['Fare'].max()
dataset[(dataset['Pclass']==3) & (dataset['Fare']==69.55)][['Name', 'FamilySize']]
dataset['FarePerPerson'] = dataset['Fare']/dataset['FamilySize']
dataset.isna().sum()
plt.scatter(dataset['FarePerPerson'], dataset['Pclass'], alpha=0.2)
dataset['FarePerPerson'].fillna(dataset.groupby('Pclass')['FarePerPerson'].transform('median'), inplace=True)
dataset.isna().sum()
kde_plot('FarePerPerson', 'Survived')

kde_plot('FarePerPerson', 'Survived', 0, 50)

kde_plot('FarePerPerson', 'Survived', 50, 100)
dataset['FarePerPerson'].describe()
dataset['BinnedFare'] = pd.cut(dataset['FarePerPerson'], bins=[-1,3,12,26,40,100,dataset['FarePerPerson'].max()],

                              labels=['giveaway','cheap','discount','normal','expensive','luxury'])
dataset['BinnedFare'].unique()
dataset.head()
dataset.tail()
bar_chart('BinnedFare', 'Survived', True)
dataset.isna().sum()
# == graph the fare against other features == #

bar_chart('BinnedAge','Embarked', True)

bar_chart('BinnedFare', 'Embarked', True)
dataset['Embarked'].fillna('S', inplace=True)
dataset.isna().sum()
dataset['Cabin'].unique()
dataset['CabinLetter'] = dataset['Cabin'].str[0]
dataset['CabinLetter'].unique()
bar_chart('CabinLetter', 'Pclass', True)
def fillna_cabin(dataset):

    cabin = dataset['CabinLetter']

    pclass = dataset['Pclass']

    if pd.isnull(cabin):

        if pclass == 1: return 'C'

        elif pclass == 2: return 'F'

        else: return 'G'

    else:

        return cabin

    

dataset['CabinLetter'] = dataset[['CabinLetter', 'Pclass']].apply(fillna_cabin, axis=1)
dataset.head()
dataset.tail()
dataset['CabinLetter'].unique()
dataset.isna().sum()
dataset[dataset['CabinLetter']=='T']
bar_chart('CabinLetter', 'Survived', True)
to_remove = ['Name','Age','SibSp','Parch','Ticket','Fare', 'Cabin', 'FamilySize', 'FarePerPerson']

dataset.drop(to_remove, axis=1, inplace=True)
dataset.head()
dataset.head(10)
to_dummy = ['Sex', 'Embarked', 'Title', 'Surname', 'BinnedAge', 'BinnedFamily', 'BinnedFare', 'CabinLetter']

dummies = [pd.get_dummies(dataset[x], drop_first=True) for x in to_dummy]

dummy_dataset = pd.concat([dataset]+dummies, axis=1)

dummy_dataset.head()
dummy_dataset.drop(to_dummy, axis=1, inplace=True)

dummy_dataset.head(10)
dummy_dataset.tail(10)
# == set the passenger id as index == #

dummy_dataset.set_index('PassengerId', inplace=True)

dummy_dataset.head()
# == separating the test and train data == #

train_data = dummy_dataset[~pd.isnull(dummy_dataset['Survived'])]

test_data = dummy_dataset[pd.isnull(dummy_dataset['Survived'])]
train_data.shape
test_data.shape
train_data.head()
test_data.head()
X_data = train_data.drop('Survived', axis=1).values

Y_data = train_data['Survived'].values

X_predict = test_data.drop('Survived', axis=1).values
#from sklearn.model_selection import train_test_split

#X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0)



X_train = X_data

Y_train = Y_data
X_train.shape
X_predict.shape
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
def models(X_train, Y_train):

    scoring = 'accuracy'

    

    # == Logistic Regression == #

    from sklearn.linear_model import LogisticRegression

    log = LogisticRegression(random_state=0)

    log_score = cross_val_score(log, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)

    log.fit(X_train, Y_train)

    

    # == KNeighbors == #

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=13)

    knn_score = cross_val_score(knn, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)

    knn.fit(X_train, Y_train)

    

    # == Support Vector Classifiers (linear) == #

    from sklearn.svm import SVC

    svc_lin = SVC(kernel='linear', random_state = 0)

    svc_lin_score = cross_val_score(svc_lin, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)

    svc_lin.fit(X_train, Y_train)

    

    # == Support Vector Classifier (RBF) == #

    svc_rbf = SVC(kernel='rbf', random_state = 0)

    svc_rbf_score = cross_val_score(svc_rbf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)

    svc_rbf.fit(X_train, Y_train)

    

    # == GaussianNB == #

    from sklearn.naive_bayes import GaussianNB

    gauss = GaussianNB()

    gauss_score = cross_val_score(gauss, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)

    gauss.fit(X_train, Y_train)

    

    # == Decision Tree == #

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

    tree_score = cross_val_score(tree, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)

    tree.fit(X_train, Y_train)

    

    # == Random Forest Classifier == #

    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

    forest_score = cross_val_score(forest, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)

    forest.fit(X_train, Y_train)

    

    # == Print accuracy for each model == #

    print(f'[0]Logistic Regression Training Accuracy: {round(np.mean(log_score)*100, 2)}, {round(np.std(log_score)*100, 2)}')

    print(f'[1]KNeighbors Training Accuracy: {round(np.mean(knn_score)*100, 2)}, {round(np.std(knn_score)*100, 2)}')

    print(f'[2]SVC (linear) Training Accuracy: {round(np.mean(svc_lin_score)*100, 2)}, {round(np.std(svc_lin_score)*100, 2)}')

    print(f'[3]SVC (RBF) Training Accuracy: {round(np.mean(svc_rbf_score)*100, 2)}, {round(np.std(svc_rbf_score)*100, 2)}')

    print(f'[4]GaussianNB Training Accuracy: {round(np.mean(gauss_score)*100, 2)}, {round(np.std(gauss_score)*100, 2)}')

    print(f'[5]Decision Tree Training Accuracy: {round(np.mean(tree_score)*100, 2)}, {round(np.std(tree_score)*100, 2)}')

    print(f'[6]Random Forest Training Accuracy: {round(np.mean(forest_score)*100, 2)}, {round(np.std(forest_score)*100, 2)}')

    



    return log, knn, svc_lin, svc_rbf, gauss, tree, forest
log, knn, svc_lin, svc_rbf, gauss, tree, forest = models(X_train, Y_train)
# == left overs from when I used train_test_split for cross validation == #

#print('[0]Logistic Regression Testing Accuracy: ', log.score(X_test, Y_test))

#print('[1]KNeighbors Testing Accuracy: ', knn.score(X_test, Y_test))

#print('[2]SVC (linear) Testing Accuracy: ', svc_lin.score(X_test, Y_test))

#print('[3]SVC (RBF) Testing Accuracy: ', svc_rbf.score(X_test, Y_test))

#print('[4]GaussianNB Testing Accuracy: ',gauss.score(X_test, Y_test))

#print('[5]Decision Tree Testing Accuracy: ', tree.score(X_test, Y_test))

#print('[6]Random Forest Testing Accuracy: ', forest.score(X_test, Y_test))
from sklearn.model_selection import StratifiedKFold

k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

cvloss = []

cvacc = []

cvprec = []

cvrecall = []
def create_model(lr, cth):

    nn = tfk.models.Sequential()

    

    nn.add(tfk.layers.Dense(units=64, activation='relu'))

    nn.add(tfk.layers.Dense(units=16, activation='relu'))

    nn.add(tfk.layers.Dense(units=1, activation='sigmoid'))

    

    METRICS = [tfk.metrics.BinaryAccuracy(name='accuracy', threshold=cth),

              tfk.metrics.Precision(name='precision'),

              tfk.metrics.Recall(name='recall')]

    nn.compile(optimizer=tfk.optimizers.Adam(lr=lr), loss=tfk.losses.BinaryCrossentropy(), metrics=METRICS)

    

    return nn
cth = 0.33

lr = 0.0002

nn = create_model(lr, cth)

early_stopping = tfk.callbacks.EarlyStopping(monitor='val_loss', patience=5)



# == cross validation == #



for train, test in k_fold.split(X_train, Y_train):

    history = nn.fit(X_train[train], Y_train[train],

                        epochs=400,

                        validation_split=0.2,

                        callbacks=[early_stopping],

                        batch_size=10

                    )

    loss, acc, prec, recall = nn.evaluate(X_train[test], Y_train[test], verbose=0)

    cvloss.append(loss)

    cvacc.append(acc)

    cvprec.append(prec)

    cvrecall.append(recall)
print(f'Loss: {np.round(np.mean(cvloss)*100,2)}, {np.round(np.std(cvloss)*100,2)}')

print(f'Accuracy: {np.round(np.mean(cvacc)*100,2)}, {np.round(np.std(cvacc)*100,2)}')

print(f'Precision: {np.round(np.mean(cvprec)*100,2)}, {np.round(np.std(cvprec)*100,2)}')

print(f'Recall: {np.round(np.mean(cvrecall)*100,2)}, {np.round(np.std(cvrecall)*100,2)}')
pred = log.predict(X_predict)

pred=pred.astype(int)

pred
submission = pd.DataFrame({"PassengerId":test_data.index, "Survived":pred})

submission.to_csv(f'/kaggle/working/log.csv', index=False)
submission = pd.read_csv('/kaggle/working/log.csv')

submission.head()
pred = svc_lin.predict(X_predict)

pred=pred.astype(int)

pred
submission = pd.DataFrame({"PassengerId":test_data.index, "Survived":pred})

submission.to_csv(f'/kaggle/working/svc_lin.csv', index=False)
submission = pd.read_csv('/kaggle/working/svc_lin.csv')

submission.head()
pred = nn.predict_classes(X_predict)

pred = pred.reshape(-1)

pred
submission = pd.DataFrame({"PassengerId":test_data.index, "Survived":pred})

submission.to_csv(f'/kaggle/working/keras.csv', index=False)
submission = pd.read_csv('/kaggle/working/keras.csv')

submission.head()