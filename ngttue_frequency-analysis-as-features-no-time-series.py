# Importing the libraries

import numpy as np  

import matplotlib.pyplot as plt  

import pandas as pd  

from sklearn.preprocessing import LabelEncoder, StandardScaler
# Importing the dataset

X = pd.read_csv('../input/X_train.csv')

X_test=pd.read_csv('../input/X_test.csv')

target=pd.read_csv('../input/y_train.csv')

y=pd.read_csv('../input/sample_submission.csv')

target['surface'].head()
# Get a list of unique surface types

surfaces_list=target['surface'].unique()
seriesID_list=X['series_id'].unique()
# Define a list of column names to store fft data

fft_names_list=['offt_X','offt_Y','offt_Z','offt_W','afft_X','afft_Y','afft_Z','lfft_X','lfft_Y','lfft_Z']

# Prefix offt, afft, and lfft are assgined as the names for fft data of orientation, angular_velocity, and linear_acceleration, respectively
# Scaling train and test sets

pre_scaler = StandardScaler()

X_prescaled     = pd.DataFrame(pre_scaler.fit_transform(X.loc[:,X.columns[3:]]),columns=X.columns[3:])

X_test_prescaled= pd.DataFrame(pre_scaler.transform(X_test.loc[:,X_test.columns[3:]]),columns=X_test.columns[3:])
X_prescaled     =X[X.columns[0:3]].merge(X_prescaled,left_index=True,right_index=True)

X_test_prescaled=X_test[X_test.columns[0:3]].merge(X_test_prescaled,left_index=True,right_index=True)

X_test_prescaled.head()
def fft_calculate(X):

# Calculate FFT data for each series (Each series consists of 128 units of time)

# Each series has 10 parameters (orientation_X,...,angular_velocity_X,....,linear_acceleration_X,...)

# Each of the parameters will be fourier transformed using numpy fft function

    fft_data={}

    fft_names_list=['offt_X','offt_Y','offt_Z','offt_W','afft_X','afft_Y','afft_Z','lfft_X','lfft_Y','lfft_Z']

    for seriesID in range(round(len(X)/128)):

        fft_data[seriesID]={}

        i=-1

        for col in X.columns[3:]:   #stepping through each parameter columns

            c=np.fft.rfft(X[X['series_id']==seriesID][col])  # Calculate real fft

            x=np.real(np.abs(c))  # Calculate the amplitude of fft

            i+=1

            fft_name=fft_names_list[i]  # Assign names for fft data (orientation_X-->offt_X, etc.)

            fft_data[seriesID][fft_name]=x

    return fft_data
# Calculate FFT data for train and test sets



fft_data=fft_calculate(X_prescaled)

fft_test_data=fft_calculate(X_test_prescaled)
# Group series_id's into their respective surface types

seriesID_group={}

for floor_type in surfaces_list:

    seriesID_group[floor_type]=target[target['surface']==floor_type].series_id
#Calculate fft average for each parameter_X,Y,Z for each surface.  

#These average values are for viewing only, not used for training.

fft_average={}

for floor_type in surfaces_list:

    count=len(seriesID_group[floor_type])

    fft_average[floor_type]={}

    cumsum={}

    for fft in fft_names_list:

        cumsum[fft]=np.zeros(65)

        for seriesID in seriesID_group[floor_type]:

            cumsum[fft]+=fft_data[seriesID][fft]

        fft_average[floor_type][fft]=np.zeros(65)

        fft_average[floor_type][fft]=cumsum[fft]/count                
# Preview of fft spectra

plt.figure(figsize=(26, 26))

i=0

for fft in fft_names_list:

    i+=1

    if i==8:

        i+=1

    plt.subplot(3,4,i)

    if fft in ['offt_X','offt_Y','offt_Z','offt_W']:

        plt.ylim(0,0.4)

    plt.title(fft, fontsize=20)

#    plt.yscale('log')

    for floor_type in surfaces_list:

        plt.plot(fft_average[floor_type][fft][:])

    plt.legend(surfaces_list)
len(fft_data)
def fft_stats(fft_data):

# Calculate the mean, sum, and standard deviation of spectra 

    df_fft=pd.DataFrame()

    for fft in fft_names_list:

            sum_=fft+'_sum'

            mean=fft+'_mean'

            std=fft+'_std'

            for seriesID in range(len(fft_data)):

                    df_fft.loc[seriesID,sum_]=np.sum(fft_data[seriesID][fft])

                    df_fft.loc[seriesID,mean]=np.mean(fft_data[seriesID][fft])

                    df_fft.loc[seriesID,std]=np.std(fft_data[seriesID][fft])

    return df_fft
df_fft=fft_stats(fft_data)

df_fft_test=fft_stats(fft_test_data)
# Make a copy of fft dataframe

df=df_fft.copy()

df=df.reset_index(drop=True)

df_test=df_fft_test.copy()

df_test=df_test.reset_index(drop=True)
df=target.merge(df,left_index=True,right_index=True)

df.head()
le = LabelEncoder()

df['surface'] = le.fit_transform(df['surface'])
#Split train and test sets

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df.drop(['surface','group_id','series_id'],axis=1), df['surface'], test_size = 0.2, random_state = 10)
# List of features

X_train.columns
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

clf=RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_val_scaled = pd.DataFrame(scaler.transform(X_val),columns=X_val.columns)

X_test_scaled = pd.DataFrame(scaler.transform(df_test),columns=df_test.columns)
clf.fit(X_train_scaled, y_train)
y_val_predict = clf.predict(X_val_scaled)

y_test_pred = clf.predict(X_test_scaled)
#Result from including full set

from sklearn.metrics import confusion_matrix, accuracy_score

print(round(accuracy_score(y_val, y_val_predict),3))
feature_importances = pd.DataFrame(clf.feature_importances_,

                                   index = X_train_scaled.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)
features = X_train_scaled.columns.values

importances = clf.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize=(15, 10))

plt.title('Feature Importances', fontsize=24)

plt.barh(range(len(indices)), importances[indices], color='r', align='center')

plt.yticks(range(len(indices)), features[indices], fontsize=18)

plt.xlabel('Importance', fontsize=18)

plt.show()
confusion_matrix(y_val,y_val_predict)
X_train=df.drop(['surface','group_id','series_id'],axis=1)

y_train=df['surface']
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
from sklearn.model_selection import cross_val_score

score=cross_val_score(clf, X_train_scaled, y_train, cv=10, scoring="accuracy")

print(np.around(score,3))

print('The average score is: ',round(score.mean(),3))
y_test_pred=le.inverse_transform(y_test_pred)

y_test_pred[0:10]
y.head()
y['surface']=y_test_pred
y.head(10)
y.to_csv('/sample_submission.csv',index=False)