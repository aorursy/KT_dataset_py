import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import resample

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RidgeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

plt.style.use('ggplot')

# %notebook inline
# Function which read csv raw data into pandas dataframe.

def process_raw_data(row_data_dir="../input/", hf=False):

    if hf:

        hf_data = pd.read_csv(row_data_dir+"HF.csv", header=None).T

        hf_ts = pd.read_csv(row_data_dir+"TimeTicksHF.csv", dtype='float64')

        hf_ts['datetime'] = pd.to_datetime(hf_ts.iloc[:, 0], unit='s')

        

        # adding the timestamps to the data.

        hf_data['datetime'] = hf_ts['datetime']

        # drop nans values.

        hf_data = hf_data.dropna()

        # set datetime as index.

        hf_data = hf_data.set_index('datetime')

        # round datetime index to seconds.

        hf_data.index = hf_data.index.floor('s')

        # drop duplicated index.

        hf_data = hf_data[~hf_data.index.duplicated(keep='first')]

#         hf_data = hf_data.drop(columns = ['ts'])

        



    else:

        lf1i_data = pd.read_csv(row_data_dir+"LF1I.csv", header=None )

        lf1v_data = pd.read_csv(row_data_dir+"LF1V.csv", header=None)

        lf2i_data = pd.read_csv(row_data_dir+"LF2I.csv", header=None)

        lf2v_data = pd.read_csv(row_data_dir+"LF2V.csv", header=None)



        lf1_ts = pd.read_csv(row_data_dir+"TimeTicks1.csv", dtype='float64')

        lf1_ts['datetime'] = pd.to_datetime(lf1_ts.iloc[:, 0], unit='s')

        lf2_ts = pd.read_csv(row_data_dir+"TimeTicks2.csv", dtype='float64')

        lf2_ts['datetime'] = pd.to_datetime(lf2_ts.iloc[:, 0], unit='s')

        



        # list of dataframes.

        data_lst = [lf1i_data, lf1v_data, lf2i_data, lf2v_data]

        # converting from str to complex.

        for data in data_lst:

            for i in range(data.shape[1]):

                data.iloc[:,i] = data.iloc[:,i].str.replace('i', 'j').apply(complex)



                

         # adding the timestamps to the data.

        lf1i_data['datetime'] = lf1_ts['datetime']

        lf1v_data['datetime'] = lf1_ts['datetime']

        lf2i_data['datetime'] = lf2_ts['datetime']

        lf2v_data['datetime'] = lf2_ts['datetime']

        

        # drop nans values.

        lf1i_data = lf1i_data.dropna()

        lf1v_data = lf1v_data.dropna() 

        lf2i_data = lf2i_data.dropna()

        lf2v_data = lf2v_data.dropna()

        

        

        # set datetime as index and round the index to seconds.

        lf1i_data.index = lf1i_data.set_index('datetime').index.floor('s')

        lf1v_data.index = lf1v_data.set_index('datetime').index.floor('s')

        lf2i_data.index = lf2i_data.set_index('datetime').index.floor('s')

        lf2v_data.index = lf2v_data.set_index('datetime').index.floor('s')     

        

        

        # remove duplicated index.

        lf1i_data = lf1i_data[~lf1i_data.index.duplicated(keep='first')]

        lf1i_data.index = lf1i_data.set_index('datetime').index.floor('s')

        

        lf1v_data = lf1v_data[~lf1v_data.index.duplicated(keep='first')]

        lf1v_data.index = lf1v_data.set_index('datetime').index.floor('s')

        

        lf2i_data.index = lf2i_data.set_index('datetime').index.floor('s')

        lf2i_data = lf2i_data[~lf2i_data.index.duplicated(keep='first')]

        

        lf2v_data.index = lf2v_data.set_index('datetime').index.floor('s')

        lf2v_data = lf2v_data[~lf2v_data.index.duplicated(keep='first')]

    

    ### tagging_data ###

    tagging_data = pd.read_csv(row_data_dir+"TaggingInfo.csv", header=None, dtype={'1':str})



    # convertion from unix to datetime.        

    tagging_data['dt_on'] = pd.to_datetime(tagging_data.iloc[:,2], unit='s')

    tagging_data['dt_off'] = pd.to_datetime(tagging_data.iloc[:,3], unit='s')

    

    if hf:

        print('hf_data shape: {0}'.format(hf_data.shape))



        return hf_data, tagging_data

        

    else:  

        print('lf1i_data shape: {0}'.format(lf1i_data.shape))

        print('lf1v_data shape: {0}'.format(lf1v_data.shape))

        print('lf2i_data shape: {0}'.format(lf2i_data.shape))

        print('lf2v_data shape: {0}'.format(lf2v_data.shape))

        

                

        return lf1i_data, lf1v_data, lf2i_data, lf2v_data, tagging_data
def assign_labels(tagging_data, row):

    for i in range(tagging_data.shape[0]):

        if row['datetime'] in pd.Interval(tagging_data.iloc[i,-2], tagging_data.iloc[i ,-1], closed='both'):

            return tagging_data.iloc[i, 1]

    return 'None'
def elec_features(lf1i_data, lf1v_data, lf2i_data, lf2v_data, idx):

    # Compute net complex power 

    # S=Sum_over(In*Vn*cos(PHIn))=Sum(Vn*complex_conjugate(In))=P+jQ

    l1_p = np.multiply(lf1v_data.iloc[:, :6], np.conj(lf1i_data.iloc[:, :6])).loc[idx]

    l2_p = np.multiply(lf2v_data.iloc[:, :6], np.conj(lf2i_data.iloc[:, :6])).loc[idx]

        

    l1_complex_power = np.sum(l1_p, axis=1).loc[idx]

    l2_complex_power = np.sum(l2_p, axis=1).loc[idx]

    

    # Real, Reactive, Apparent powers: P=Real(S), Q=Imag(S), S=Amplitude(S)=P^2+Q^2

    # l1 - stands for phase 1 S - Vector Aparent Power

    # Phase-1

    l1_real = l1_complex_power.apply(np.real).loc[idx]

    l1_imag = l1_complex_power.apply(np.imag).loc[idx]

    l1_app  = l1_complex_power.apply(np.absolute).loc[idx]



    # Real, Reactive, Apparent power currents

    # l2 - stands for phase 2 S - Vector Aparent Power

    # Phase-2

    l2_real = l2_complex_power.apply(np.real).loc[idx]

    l2_imag = l2_complex_power.apply(np.imag).loc[idx]

    l2_app  = l2_complex_power.apply(np.absolute).loc[idx]

    

    # Compute Power Factor, we only consider the first 60Hz component

    # PF=cosine(angle(S))

    l1_pf = l1_p.iloc[:,0].apply(np.angle).apply(np.cos).loc[idx]

    l2_pf = l2_p.iloc[:,0].apply(np.angle).apply(np.cos).loc[idx]

    y = lf2i_data['label'].astype(str).loc[idx] 

    

    

    return l1_real, l1_imag, l1_app, l2_real, l2_imag, l2_app, l1_pf, l2_pf, y 
def proper_index(tagging_data, hf_data, lf1i_data):

    

    print('orginal hf_data  shape: {}'.format(hf_data.shape))

    print('orginal lf_data  shape: {}'.format(lf1i_data.shape))

        

    # slice idx to be in the range of first tagging device on datetime idx to last device off datetime idx.

    idx = lf1i_data.loc[tagging_data.dt_on.iloc[0]:tagging_data.dt_off.iloc[-1],:].index

    

    print('idx len (first transformation): {}'.format(len(idx)))

    

    # idx which are common to hf_data and lf1_data/lf2_data.

    idx = hf_data.index.intersection(idx)

    

    print('idx shape (second transformation): {}'.format(len(idx)))

    

    return idx, hf_data.loc[idx]
hf_data, tagging_data = process_raw_data(hf=True)

hf_data.head()
lf1i_data, lf1v_data, lf2i_data, lf2v_data, tagging_data = process_raw_data() 

lf1i_data.head()
lf1v_data.head()
lf2i_data.head()
lf2v_data.head()
tagging_data.head()
tagging_data.shape
lf2i_data['label'] = lf2i_data.apply(lambda row: assign_labels(tagging_data, row), axis=1)
lf2i_data.shape
lf1i_data.shape
idx, hf_data = proper_index(tagging_data, hf_data, lf1i_data)
l1_real, l1_imag, l1_app, l2_real, l2_imag, l2_app, l1_pf, l2_pf, y = elec_features(lf1i_data, lf1v_data, lf2i_data, lf2v_data, idx)
y.shape
l1_real.shape
lf2i_data.shape
lf2i_data.label.unique()
y.unique().tolist()
y.value_counts() 
y.value_counts().plot(kind='bar')

plt.show()
X = pd.concat([l1_real, l1_imag, l1_app, l1_pf, l2_real, l2_imag, l2_app, l2_pf, hf_data], axis=1).values
X.shape
y.shape
X_PCA = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X_PCA)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])
pca.explained_variance_ratio_
y_PCA = y.reset_index(drop=True)

y_PCA.head()
finalDf = pd.concat([principalDf, y_PCA ], axis = 1)
finalDf.head()
# %matplotlib notebook

%matplotlib inline

fig = plt.figure(figsize = (10,7))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = y.unique().tolist()

targets = list(filter(lambda x: x!= 'None', targets))

colors = ['r', 'g', 'b', 'y', 'm', 'k', 'w', 'c', 'darkred', 'lime', 'dodgerblue', 'magenta', 'yellow' ]

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['label'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

plt.show()
clf = KNeighborsClassifier() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

kf = KFold(n_splits=5, shuffle=True, random_state=25)

scores = cross_val_score(clf, X, y, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))
clf.fit(X_train, y_train)

print(classification_report(y,clf.predict(X)))
cm = pd.DataFrame(confusion_matrix(y, clf.predict(X)), index=clf.classes_, columns=clf.classes_)

cm
plt.figure(figsize=(20,12))

sns.heatmap(cm, annot=True)

plt.show()
clf = RidgeClassifier( )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

kf = KFold(n_splits=5, shuffle=True, random_state=25)

scores = cross_val_score(clf, X, y, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))
clf.fit(X_train, y_train)

print(classification_report(y,clf.predict(X)))
cm = pd.DataFrame(confusion_matrix(y, clf.predict(X)), index=clf.classes_, columns=clf.classes_)

cm
plt.figure(figsize=(20,12))

sns.heatmap(cm, annot=True)

plt.show()
clf = RandomForestClassifier(n_estimators=100, n_jobs=3, class_weight='balanced')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

kf = KFold(n_splits=5, shuffle=True, random_state=25)

scores = cross_val_score(clf, X, y, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))
clf.fit(X_train, y_train)

print(classification_report(y,clf.predict(X)))
cm = pd.DataFrame(confusion_matrix(y, clf.predict(X)), index=clf.classes_, columns=clf.classes_)

cm
plt.figure(figsize=(20,12))

sns.heatmap(cm, annot=True)

plt.show()
clf = LogisticRegression(solver='lbfgs', n_jobs=-1, multi_class = 'auto', class_weight='balanced')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

kf = KFold(n_splits=5, shuffle=True, random_state=25)

scores = cross_val_score(clf, X, y, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))
clf.fit(X_train, y_train)

print(classification_report(y,clf.predict(X)))
cm = pd.DataFrame(confusion_matrix(y, clf.predict(X)), index=clf.classes_, columns=clf.classes_)

cm
plt.figure(figsize=(20,12))

sns.heatmap(cm, annot=True)

plt.show()
clf = DecisionTreeClassifier(class_weight='balanced')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

kf = KFold(n_splits=5, shuffle=True, random_state=25)

scores = cross_val_score(clf, X, y, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))
clf.fit(X_train, y_train)

print(classification_report(y,clf.predict(X)))
cm = pd.DataFrame(confusion_matrix(y, clf.predict(X)), index=clf.classes_, columns=clf.classes_)

cm
plt.figure(figsize=(20,12))

sns.heatmap(cm, annot=True)

plt.show()