# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math
train_data = pd.read_csv("../input/X_train.csv")
print(train_data.shape)

train_data.head()
train_data['series_id'].nunique()
train_labels = pd.read_csv("../input/y_train.csv")
print(train_labels.shape)

print(train_labels['surface'].nunique())

train_labels.head()
train_data.info()
def get_range(data_list):

    

    return max(data_list)-min(data_list)
plt.figure(figsize=(10,4))

plt.subplot(221)

plt.hist(train_data.groupby('series_id')['orientation_X'].apply(get_range))

plt.xlabel('orientation_X_range')

plt.subplot(222)

plt.hist(train_data.groupby('series_id')['orientation_Y'].apply(get_range))

plt.xlabel('orientation_Y_range')

plt.subplot(223)

plt.hist(train_data.groupby('series_id')['orientation_Z'].apply(get_range))

plt.xlabel('orientation_Z_range')

plt.subplot(224)

plt.hist(train_data.groupby('series_id')['orientation_W'].apply(get_range))

plt.xlabel('orientation_W_range')

plt.tight_layout()
def plot_feature_variations(series_n_data, series_number, surface_type):

    

    plt.figure(figsize=(15,4))



    plt.subplot(231)

    plt.plot(series_n_data['measurement_number'],series_n_data['angular_velocity_X'])

    plt.xlabel('measurement_number')

    plt.ylabel('angular_velocity_X')



    plt.subplot(232)

    plt.plot(series_n_data['measurement_number'],series_n_data['angular_velocity_Y'])

    plt.xlabel('measurement_number')

    plt.ylabel('angular_velocity_Y')



    plt.subplot(233)

    plt.plot(series_n_data['measurement_number'],series_n_data['angular_velocity_Z'])

    plt.xlabel('measurement_number')

    plt.ylabel('angular_velocity_Z')



    plt.subplot(234)

    plt.plot(series_n_data['measurement_number'],series_n_data['linear_acceleration_X'])

    plt.xlabel('measurement_number')

    plt.ylabel('linear_acceleration_X')



    plt.subplot(235)

    plt.plot(series_n_data['measurement_number'],series_n_data['linear_acceleration_Y'])

    plt.xlabel('measurement_number')

    plt.ylabel('linear_acceleration_Y')



    plt.subplot(236)

    plt.plot(series_n_data['measurement_number'],series_n_data['linear_acceleration_Z'])

    plt.xlabel('measurement_number')

    plt.ylabel('linear_acceleration_Z')



    plt.tight_layout()
series_0_data=train_data[train_data['series_id']==0]

surface_type=train_labels['surface'][0]

print("Feature Variations for Surface Type {}".format(surface_type))

plot_feature_variations(series_0_data, 0 , surface_type)





series_1_data=train_data[train_data['series_id']==1]

surface_type=train_labels['surface'][1]

print("Feature Variations for Surface Type {}".format(surface_type))

plot_feature_variations(series_1_data, 1 , surface_type)



series_4_data=train_data[train_data['series_id']==4]

surface_type=train_labels['surface'][4]

print("Feature Variations for Surface Type {}".format(surface_type))

plot_feature_variations(series_4_data, 4 , surface_type)
x = np.arange(9)

counts = train_labels['surface'].value_counts()

 

plt.figure(figsize=(15,4))

plt.bar(x, counts, align='center', alpha=0.5)

plt.xticks(x, train_labels['surface'].value_counts().index.tolist())

plt.ylabel('Counts in Training Data')

plt.title('Surface Data Occurences')



print(train_labels['surface'].value_counts())



y=train_labels['surface'].values
group_ids=train_labels['group_id']

print(group_ids.shape)

print(group_ids.nunique())



group_ids=np.array(group_ids)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
train_features=train_data.drop(['row_id','measurement_number'],axis=1)
train_features.columns
sc= MinMaxScaler()



def feature_transform(features_data):

    all_features=pd.DataFrame()

    

    features_data['orientation']=np.sqrt(features_data['orientation_X']**2+features_data['orientation_Y']**2+

                                         features_data['orientation_Z']**2+features_data['orientation_W']**2)

    

    features_data['ang_vel_mag']=np.sqrt(features_data['angular_velocity_X']**2 + 

                                features_data['angular_velocity_Y']**2 + features_data['angular_velocity_Z']**2)

    

    features_data['lin_acc_mag']=np.sqrt(features_data['linear_acceleration_X']**2 + 

                                features_data['linear_acceleration_Y']**2 + features_data['linear_acceleration_Z']**2)

    

    

    for col in features_data.columns:

        if col=='series_id':

            continue

        all_features[col+'_mean']=features_data.groupby('series_id')[col].mean()

        all_features[col+'_median']=features_data.groupby('series_id')[col].median()

        all_features[col+'_min']=features_data.groupby('series_id')[col].min()

        all_features[col+'_max']=features_data.groupby('series_id')[col].max()

        all_features[col+'_std']=features_data.groupby('series_id')[col].std()

        #all_features[col+'_q25']=features_data.groupby('series_id')[col].quantile(0.25)

        #all_features[col+'_q50']=features_data.groupby('series_id')[col].quantile(0.5)

        #all_features[col+'_q75']=features_data.groupby('series_id')[col].quantile(0.75)

        all_features[col+'_maxByMin']=all_features[col+'_max']/all_features[col+'_min']

        all_features[col+'_range']=all_features[col+'_max']-all_features[col+'_min']

       

        

    all_features=all_features.reset_index()

    all_features=all_features.drop(['series_id'],axis=1)

    all_features=sc.fit_transform(all_features)

    

    return all_features
all_train_features=feature_transform(train_features)
enc = LabelEncoder()

y_transformed=enc.fit_transform(np.reshape(y,(-1,1)))
y_transformed[:25]
X=np.array(all_train_features)

y=y_transformed
test_data= pd.read_csv("../input/X_test.csv")
test_data.shape
test_features=test_data.drop(['row_id','measurement_number'],axis=1)
all_test_features=feature_transform(test_features)
all_test_features=np.array(all_test_features)



print(len(all_test_features))

print(len(all_test_features[0]))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)

predicted = np.zeros((len(all_test_features),9))

measured= np.zeros(len(X))

score = 0



model = RandomForestClassifier(n_estimators=500, random_state=123, max_depth=15, min_samples_split=5)



for t, (trn_idx, val_idx) in enumerate(folds.split(X,y)):    

    model.fit(X[trn_idx],y[trn_idx])

    measured[val_idx] = model.predict(X[val_idx])

    predicted += model.predict_proba(all_test_features)/folds.n_splits

    score += model.score(X[val_idx],y[val_idx])

    print("Fold: {} score: {}".format(t,model.score(X[val_idx],y[val_idx])))
print(confusion_matrix(measured,y))
print('Average Accuracy is ',score/folds.n_splits)
submission_file=pd.read_csv("../input/sample_submission.csv")
results=pd.DataFrame(enc.inverse_transform(predicted.argmax(axis=1)))
results.head()
final_submission=submission_file.drop(['surface'],axis=1)
final_submission=pd.concat([final_submission,results],axis=1,ignore_index=True)
final_submission.to_csv("submission_final.csv",header=['series_id','surface'],index=False)