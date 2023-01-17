# Importing useful libararies



import warnings

warnings.filterwarnings("ignore")



import numpy as np  

import pandas as pd    

import seaborn as sns  

import matplotlib.pyplot as plt  

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from time import time
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")     #Loading train data

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")      #Loading test data

df_sample_sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")  #Loading sample submission data



df_all = df_train.append(df_test, ignore_index=True)     # Merging df_train and df_test dataframes



print("Titanic Dataset Summary:")

display(df_all.head())



print("Stats of some features:")

display(df_all.describe())



print(df_all.info())



plt.figure(figsize=(8,5))

sns.heatmap(df_all.isnull())                     # Heatmap of missing values

plt.show()



m = df_all.shape[0]

print('Number of missing values in',m, 'examples')

display(df_all.isnull().sum())           # Number of missing values

# creating a new feature (Title) by extracting title from the Name feature



df_all['Title'] = df_all.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

display(df_all.Title)

new_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}



df_all.Title = df_all.Title.map(new_titles)

display(df_all.Title.value_counts())

df_all.Cabin = df_all.Cabin.fillna(0)

for i in range(len(df_all.Cabin)): 

  if df_all.Cabin[i] != 0:

    df_all.Cabin[i] = 1

fig, axes = plt.subplots(1,2,figsize=(10,6))

sns.countplot(df_all.Cabin, hue=df_all.Pclass, ax= axes[0])

sns.countplot(df_all.Cabin, hue=df_all.Survived, ax= axes[1])

plt.show()
most_embarked = df_all.Embarked.value_counts().index[0]

df_all.Embarked = df_all.Embarked.fillna(most_embarked)

df_all.Fare = df_all.Fare.fillna(df_all.Fare.median())

plt.figure(figsize=(8,5))

sns.distplot(df_all["Age"] ,color='black')

plt.show()

print(df_all.Age.describe())

Age_cat = pd.cut(df_all.Age,bins=[0,10,20,30,40,50,60,70,80],labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80'])



fig, axes = plt.subplots(1,2, figsize = (10,6))

sns.countplot(Age_cat[(df_all.Sex=='female')],color= 'black',ax =axes[0])

sns.countplot(Age_cat[(df_all.Survived ==1) & (df_all.Sex=='female')],color='pink',ax =axes[0]).set_title('Female')

sns.countplot(Age_cat[(df_all.Sex=='male')],color= 'black',ax =axes[1])

sns.countplot(Age_cat[(df_all.Survived ==1) & (df_all.Sex=='male')],color='blue', ax =axes[1]).set_title('Male')

plt.show()
grouped = df_all.groupby(['Sex','Pclass', 'Title'])  

display(grouped.Age.median())

plt.figure(figsize= (10,6))

ax = grouped.Age.median().plot(kind='bar',color= 'black' )

ax.set(ylabel = 'Median Age')

plt.show()

df_all.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
print('Number of missing values in',m, 'examples')

display(df_all.isnull().sum())
plt.figure(figsize=(8,5))

sns.countplot(df_all.Sex, hue=df_all.Survived)

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(df_all.Pclass, hue=df_all.Survived)

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(df_all.SibSp, hue=df_all.Survived)

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(df_all.Parch, hue=df_all.Survived)

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(df_all.Title, hue=df_all.Survived)

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(df_all.Embarked, hue=df_all.Survived)

plt.show()
Fare_cat = pd.cut(df_all.Fare,bins=[0,10,50,100,200,550],labels=['0-10','10-50','50-100','100-200','200-550'])



plt.figure(figsize=(8,5))

sns.countplot(Fare_cat, hue=df_all.Survived)



plt.show()


df_all.drop('Name', axis =1, inplace=True)

df_all.drop('Ticket', axis =1, inplace=True)

df_all.drop('PassengerId', axis=1, inplace = True)

display(df_all)
display(df_all.head())
Sex = {"male": 0, "female":1}

df_all["Sex"] = df_all.Sex.map(Sex)

df_all['Partner'] = df_all['SibSp'] + df_all['Parch'] # 

df_all.drop(['SibSp', 'Parch'], axis=1, inplace=True)

df_all = pd.get_dummies(df_all, columns = ['Title','Embarked'])

display(df_all.head())
# Logistic Regression from scratch



def logistic_regression(X, y, alpha=1e-3, num_iter=30,random_state=42):

    

    np.random.seed(random_state) # Random_state

    d, m = X.shape 

    K = np.max(y) + 1

    w = np.random.randn(d, K)

    

    def softmax(x):

        s = np.exp(x) / np.sum(np.exp(x))

        return s

    

    def one_hot(y, k):  

        y_one_hot = np.eye(k)[y]

        return y_one_hot

    

    def h(x, w):

        p = softmax(w.T @ x)     #Using softmax for multiclass classification

        return p

    

    def cost(pred, y):

        c = np.sum(- one_hot(y, K).T * np.log(pred))

        return c



    def grad(w, x, y):

        Y = one_hot(y, K).T

        b = h(x, w) - Y

        b = np.reshape(b, (-1, 1))

        x = x.reshape(-1, 1)

        g = x @ np.reshape(b, (-1, 1)).T

        return g



    for i in range(num_iter):

        id = np.random.permutation(m)

        for j in id:

            gradient = grad(w, X[:, j], y[j])

            w -= alpha * gradient

    return w

# Least square ridge classifier

def ridge_classifier(X, y, lambd=1e-4):

    d, m = X.shape

    k = np.max(y) + 1

    w = np.linalg.inv(X @ X.T + lambd * np.eye(d)) @ X @ np.eye(k)[y]

    return w
def error(X, y, w):

    m = np.shape(y)

    y_pred = w.T @ X

    y_pred = np.argmax(y_pred, axis=0)

    err = np.sum(y_pred == y) / m

    return err
mms = MinMaxScaler()



X = df_all.drop('Survived', axis=1).iloc[:891].values

y = (df_all["Survived"].iloc[:891].values).astype(int)

X = mms.fit_transform(X)

X_test = df_all.drop('Survived', axis=1).iloc[891:].values

X_test = mms.fit_transform (X_test)


scores_lr = []

scores_ls = []

fold =1



for tr, val in KFold(n_splits=5, random_state=42).split(X,y):

    X_train = X[tr]

    X_val = X[val]

    y_train = y[tr]

    y_val = y[val]

    best_W_LR = logistic_regression(X_train.T, y_train, alpha=1e-3, num_iter=300,random_state=42)

    val_acc_LR = error(X_val.T, y_val, best_W_LR)

    scores_lr.append(val_acc_LR)

    print(f'Validation acc LR: Fold {fold}:', val_acc_LR)

    W_LS = ridge_classifier(X_train.T, y_train, lambd=1e-4)

    val_acc_LS = error(X_val.T, y_val, W_LS)

    scores_ls.append(val_acc_LS)

    print(f'Validation acc LS: Fold {fold}:', val_acc_LS)

    fold +=1 



print('-------------------------------')

print("Accuracy Logistic Regression: %0.2f (+/- %0.2f)" % (np.mean(scores_lr), np.std(scores_lr) * 2))

print("Accuracy Least Squares Ridge: %0.2f (+/- %0.2f)" % (np.mean(scores_ls), np.std(scores_ls) * 2))



# Ridge Classifier prediction on test set



y_preds_LS =  (np.argmax(W_LS.T @ X_test.T, axis=0)).astype(int)

df_sample_sub.loc[:, 'Survived'] = y_preds_LS

df_sample_sub.to_csv('submission0.csv', index=False)

display(df_sample_sub.head())

# LB score: 0.77033
def test_clfs(clfs):

    for clf in clfs:

        print('------------------------------------------')

        start = time()

        clf = clf(random_state=42)

        scores = cross_val_score(clf, X, y, cv=5)

        print(str(clf), 'results:')

        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        end = time()

        print('Processing time', end-start,'s')

    



models = [RandomForestClassifier, LogisticRegression,XGBClassifier]

test_clfs(models)

from sklearn.model_selection import GridSearchCV



# C = np.arange(1,100,1)

# fit_intercept = [True, False]

# penalty = ['l1', 'l2', 'elasticnet','none']

# class_weight = ['None', 'balanced']

# solver = ['newton-cg','lbfgs','liblinear','sag', 'saga']

# params = dict(C=C, fit_intercept=fit_intercept, penalty=penalty,

#               class_weight=class_weight, solver=solver)



# clf = GridSearchCV(estimator=LogisticRegression(random_state=42, n_jobs=-1) 

#                    , param_grid=params, cv=5, n_jobs=-1, verbose=2)



# clf.fit(X, y)

# print('best params', clf.best_params_)

# print('best score', clf.best_score_)

# best params {'C': 5, 'class_weight': 'None', 'fit_intercept': False, 'penalty': 'l2', 'solver': 'newton-cg'}

# best score 0.8349946644906158







# n_estimators =[4,5,6,7,8,9,10,11,12,13,14,15]

# min_samples_split =[2, 3, 4, 5]

# min_samples_leaf =[1, 2, 3, 4, 5]

# max_depth =[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# params = dict(n_estimators=n_estimators, min_samples_split=min_samples_split,

#                min_samples_leaf=min_samples_leaf, max_depth=max_depth)



# clf = GridSearchCV(estimator=RandomForestClassifier(random_state=42, n_jobs=-1), param_grid=params, cv=5, n_jobs=-1, verbose=0)

# clf.fit(X, y)

# print('best params', clf.best_params_)

# print('best score', clf.best_score_)



#best params {'max_depth': 9, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 9}

#best score 0.8372544096415794





# n_estimators = [10, 50,100,150]

# max_depth = [3, 5, 7, 9, 11]

# booster= ['gbtree','gblinear']

# min_child_weight = [1, 5, 10]

# gamma= [0.5, 1, 2, 5]

# subsample= [0.6, 0.8, 1.0]

# colsample_bytree = [0.6, 0.8, 1.0]

# params = dict(n_estimators=n_estimators,

#                min_child_weight=min_child_weight, max_depth=max_depth,booster=booster,gamma=gamma,

#            subsample=subsample, colsample_bytree=colsample_bytree )



# clf = GridSearchCV(estimator=XGBClassifier(random_state=42, n_jobs=-1), param_grid=params, cv=5, n_jobs=-1, verbose=2)

# clf.fit(X, y)

# print('best params', clf.best_params_)

# print('best score', clf.best_score_)

#best params {'booster': 'gbtree', 'colsample_bytree': 0.6, 'gamma': 1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.8}

#best score 0.8451321323206328
clf1 = RandomForestClassifier(max_depth=9, min_samples_leaf=4, min_samples_split=2,

                             n_estimators=9, random_state=42, n_jobs=-1) # The parameters came from the former cell results.

clf1.fit(X, y)





y_preds_RF = clf1.predict(X_test).astype(int)



df_sample_sub.loc[:, 'Survived'] = y_preds_RF

df_sample_sub.to_csv('submission1.csv', index=False)

display(df_sample_sub.head())



clf2 = LogisticRegression(C=48, class_weight='None', fit_intercept= False, penalty='l2', solver='lbfgs')

clf2.fit(X, y)



y_preds_LR = clf2.predict(X_test).astype(int)



df_sample_sub.loc[:, 'Survived'] = y_preds_LR

df_sample_sub.to_csv('submission2.csv', index=False)

display(df_sample_sub.head())

clf3 = XGBClassifier(booster='gbtree', colsample_bytree= 0.6,

                    gamma=1, max_depth=5, min_child_weight=1, n_estimators=100, subsample=0.8)



clf3.fit(X, y)

y_preds_xgb = clf3.predict(X_test).astype(int)



df_sample_sub.loc[:, 'Survived'] = y_preds_xgb

df_sample_sub.to_csv('submission3.csv', index=False)

display(df_sample_sub.head())

import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.keras.backend as K

def create_model(hid_layers ,dropout_rate, lr):

    

    inp1 = tf.keras.layers.Input(shape = (X.shape[1], ))

    x1 = tf.keras.layers.BatchNormalization()(inp1)

    

    for i, units in enumerate(hid_layers):

        x1 = tf.keras.layers.Dense(units, activation='relu')(x1)

        x1 = tf.keras.layers.Dropout(dropout_rate)(x1)    

    x1 = tf.keras.layers.Dense(1, activation='sigmoid')(x1)

    

    model = tf.keras.models.Model(inputs= inp1, outputs= x1)

    

    model.compile(optimizer ="adam", loss='binary_crossentropy', metrics='accuracy')

    

    return model 

    

    

lr=1e-5

hid_layers = [256, 256]

dr = 0.5

model = create_model(hid_layers, dr, lr)



tf.keras.utils.plot_model(model, show_shapes = True, show_layer_names= False,

                          rankdir = 'TB', expand_nested = True)



from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

df_sub_copy = df_sample_sub.copy()

df_sub_copy.loc[:, 'Survived'] = 0.0





scores=[]

fold = 0

for tr, val in KFold(n_splits=5, random_state=42).split(X,y):

    X_train = X[tr]

    X_val = X[val]

    y_train = y[tr]

    y_val = y[val]

    rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 0, 

                                min_delta = 1e-4, min_lr = 1e-6, mode = 'min')

        

    ckp = ModelCheckpoint(f'bests_weights.hdf5', monitor = 'val_loss', verbose = 0, 

                              save_best_only = True, save_weights_only = True, mode = 'min')

        

    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 15, mode = 'min', 

                           baseline = None, restore_best_weights = True, verbose = 0)

    

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[rlr,ckp,es],

                        epochs=300, verbose=0)

    scores.append(np.max(history.history['val_accuracy']))

        

    model.load_weights(f'bests_weights.hdf5')

    y_preds_nn = model.predict(X_test)

    df_sub_copy.loc[:, 'Survived'] += y_preds_nn.reshape(-1)

    print(f'fold',str(fold)+':',scores[fold])

    K.clear_session()

    fold+=1

print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2))
# NN submission

df_sample_sub.loc[:, 'Survived'] = (np.round(df_sub_copy.loc[:,'Survived']/ 5)).astype(int)

display(df_sample_sub.head())

df_sample_sub.to_csv('submission4.csv', index=False)

sub0= pd.read_csv('submission0.csv')

sub1 = pd.read_csv('submission1.csv')

sub2 = pd.read_csv('submission2.csv')

sub3 = pd.read_csv('submission3.csv')

sub4 = pd.read_csv('submission4.csv')



sub_vot = np.round((sub0['Survived']+sub1['Survived']+sub2['Survived']+sub3['Survived']+sub4['Survived'])/5).astype(int)

df_sample_sub.loc[:, 'Survived'] = sub_vot

df_sample_sub.to_csv('submission5.csv', index=False)

display(df_sample_sub.head())