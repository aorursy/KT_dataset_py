#Author: Aparna Vadakedathu

#email:aparnavt@yahoo.in

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("dark")

plt.rcParams['figure.figsize'] = 16, 12

import pandas as pd

pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_colwidth', 300)

pd.options.display.float_format = '{:,.6f}'.format



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from functools import reduce



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn import preprocessing

from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.preprocessing import MultiLabelBinarizer



from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import pickle

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
df_target_train = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_target_train.csv')

print('df_target_train:', df_target_train.shape)



df_sample_submit = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_sample_submit.csv')

print('df_sample_submit:', df_sample_submit.shape)



df_tracks = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_tracks.csv')

print('df_tracks:', df_tracks.shape)



df_genres = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_genres.csv')

print('df_genres:', df_genres.shape)



df_features = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_features.csv')

print('df_features:', df_features.shape)
df_target_train['track_id'].nunique()
df_tracks_numerical_features = list(df_tracks.select_dtypes([np.number]).columns)

#df_tracks_numerical_features = df_tracks_numerical_features.remove('track_id')

df_tracks_categorical_features = list(set(df_tracks.columns).difference(set(df_tracks_numerical_features)))

df_tracks_numerical_features.remove('track_id')

#df_tracks_numerical_features
#Step did not help score hence avoiding

df_tracks[df_tracks_numerical_features] = StandardScaler().fit_transform(df_tracks[df_tracks_numerical_features] )
df_features_numerical_features = list(df_features.select_dtypes([np.number]).columns)

#df_tracks_numerical_features = df_tracks_numerical_features.remove('track_id')

df_features_categorical_features = list(set(df_features.columns).difference(set(df_features_numerical_features)))

df_features_numerical_features.remove('track_id')

#df_tracks_numerical_features


df_features[df_features_numerical_features] = StandardScaler().fit_transform(df_features[df_features_numerical_features] )
pca = PCA(n_components=0.9, random_state=16).fit(df_features[df_features_numerical_features])

X_pca= pca.transform(df_features[df_features_numerical_features])

df_features_PCA= pd.DataFrame(data=X_pca, columns= ["component" + str(i) for i in range(0,X_pca.shape[1])])
df_features.drop(df_features_numerical_features, axis=1, inplace= True)


df_features_reduced = pd.merge(df_features, df_features_PCA,left_index=True, right_index = True) 
df_features_reduced.shape
df_tracks[df_tracks['album:type']==""] = np.NaN

df_tracks['album:type'].fillna("Other", inplace = True)

#df_tracks['album:type'].value_counts()

dummy_variable_1 = pd.get_dummies(df_tracks['album:type'])

dummy_variable_1.head()

df_tracks = pd.concat([df_tracks, dummy_variable_1], axis=1)

df_tracks.drop( ['album:type'], axis=1, inplace= True)

df_tracks[df_tracks['artist:location']==""] = np.NaN

df_tracks['artist:location'].fillna("Other", inplace = True)

location_list =list(set( df_tracks['artist:location']))

len(location_list)

le = preprocessing.LabelEncoder()

le.fit(location_list)

df_tracks['artist:location'] = le.transform(df_tracks['artist:location'])
df_tracks[['track:date_created']] = df_tracks[['track:date_created']].apply(pd.to_datetime) 



df_tracks['track:date_created-year'] = df_tracks['track:date_created'].dt.year

df_tracks['track:date_created-month'] = df_tracks['track:date_created'].dt.month

df_tracks.drop([ 'album:date_created', 'album:date_released',

       'artist:active_year_begin', 'artist:active_year_end', 'track:date_recorded' , 'track:date_created', 'track:language_code',], axis=1, inplace = True)
#df_features_merged = pd.merge(df_features, df_tracks, on=['track_id'])



df_features_merged = pd.merge(df_features_reduced, df_tracks, on=['track_id'])
Features = df_features_merged.columns
df_fulldataset  = pd.merge(df_features_merged ,df_target_train,  on=['track_id'],how='left')
df_test = df_fulldataset.loc[df_fulldataset['track:genres'].isna()][Features]
df_train = pd.merge(df_features_merged, df_target_train, on=['track_id'])
df_train.head()
df_train.shape
df_test.head()
df_test.shape
df_train['target:multilabel'] = np.NaN





from tqdm import tqdm_notebook

from collections import defaultdict



from tqdm import tqdm



genre = []

genre_list = []

# extract tracks for each genre

for _, row in tqdm_notebook(df_train.iterrows(), total=df_train.shape[0]):

    for g_id in row['track:genres'].split(' '):

        genre.append(g_id)

        #print(genre)

    #print(row)

    #row['track:genres']= genre

    genre_list.append(genre)

    #print(row['track:genreslist'])

    genre=[]



    
df_train['target:multilabel'] = genre_list

df_train.drop(['track:genres'], axis = 1, inplace = True)
df_train.head()
df_train.shape
mlb = MultiLabelBinarizer()

df_train = df_train.join(pd.DataFrame(mlb.fit_transform(df_train.pop('target:multilabel')),

                          columns=mlb.classes_,

                          index=df_train.index))
Target = np.setdiff1d(df_train.columns,Features) 

#df_train.columns
our_x_train = df_train[Features]

our_y_train = df_train[Target]

our_x_test = df_test[Features]

our_y_train.drop(['38','15'], axis = 1, inplace = True)
for label in our_y_train.columns:

    if our_y_train[label].sum() < 800:

        #print(label,our_y_train[label].sum() )

        our_y_train.drop([label], axis = 1, inplace = True)

        

our_y_train = our_y_train[our_y_train.sum().sort_values(ascending = True).index]
X_train, X_test, y_train, y_test = train_test_split(our_x_train,

                                                    our_y_train,

                                                    test_size=0.33,

                                                    random_state=17)
our_y_train.shape
from sklearn.metrics import hamming_loss

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from skmultilearn.problem_transform import ClassifierChain

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB

#from skml.ensemble import EnsembleClassifierChain

model_DT_simple = DecisionTreeClassifier(

                    criterion='gini',

                    splitter='best',

                    max_depth=8,

                    random_state = 0

                    )
classifier = ClassifierChain(classifier = model_DT_simple, require_dense = [True, True]) #0.218, 0.213 with PCA

#classifier = ClassifierChain(classifier = AdaBoostClassifier(n_estimators=10,learning_rate=1), require_dense = [True, True])



#require_dense = [False, True]

# train

classifier.fit(X_train, y_train)






y_pred = classifier.predict(X_test)







print("hamming loss: ")

print(hamming_loss(y_test, y_pred))



print("accuracy:")

print(accuracy_score(y_test, y_pred))



print("f1 score:")

print("micro")

print(f1_score(y_test, y_pred, average='micro'))

print("macro")

print(f1_score(y_test, y_pred, average='macro'))





print("precision:")

print("micro")

print(precision_score(y_test, y_pred, average='micro'))

print("macro")

print(precision_score(y_test, y_pred, average='macro'))



print("recall:")

print("micro")

print(recall_score(y_test, y_pred, average='micro'))

print("macro")

print(recall_score(y_test, y_pred, average='macro'))





our_y_test = classifier.predict_proba(our_x_test)
result = pd.DataFrame(columns= our_y_train.columns, data= our_y_test.toarray())
result.tail()
result['track:genres'] = result.apply(lambda x: ' '.join(x.index[x >0.1]), axis=1) 


our_x_test = our_x_test.reset_index()
final = pd.merge(our_x_test, result,  left_index=True, right_index = True)
final.tail()


df_sample_submit.drop(['track:genres'], axis=1, inplace = True)

#df_fulldataset  = pd.merge(df_features_merged ,df_target_train,  on=['track_id'],how='left')

df_submit = pd.merge(df_sample_submit, final[['track_id','track:genres']], on=['track_id'], how = 'left')
df_submit['track:genres'] = df_submit['track:genres'].apply(lambda r: r + ' 15 38' if len(r) > 0 else '15 38')
df_submit['track:genres'].apply(lambda s: len([int(x) for x in s.split(' ')])).median()
df_submit
df_submit.to_csv('./submit_12Jun_classifierchain_submission8.csv', index=False)