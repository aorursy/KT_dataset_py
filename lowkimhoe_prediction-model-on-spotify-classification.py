#Import the neccessary library for the task

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from warnings import simplefilter # ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline
#Read the dataset from the kaggle

dataset = pd.read_csv('/kaggle/input/spotifyclassification/data.csv', index_col = 0)
dataset.head()
dataset.tail()
print('Dataset: ', dataset.shape[0], 'Rows', dataset.shape[1], 'Features')
dataset['target'].value_counts()
dataset.columns.values
SpeechinessMusic = dataset[['instrumentalness','speechiness']]

Energetic = dataset[['danceability','energy']]

MusicAttribute = dataset[['tempo','mode','key','time_signature']]

Environment = dataset[['acousticness','liveness','loudness']]



print(SpeechinessMusic.head(2))

print(Energetic.head(2))

print(MusicAttribute.head(2))

print(Environment.head(2))
dataset.describe()
dataset.describe(include = 'O')
#Check the null value for the string variable

print('song_title:' ,dataset['song_title'].isnull().sum())

print('artist:' ,dataset['artist'].isnull().sum())
#Check how many of duplicate values in song_title & artist features



def DuplicatedFunction(data,column):

    result = data[column].duplicated().sum()

    return result



print('Duplicate Values:' ,DuplicatedFunction(dataset,'song_title'))

print('Duplicate Values:' ,DuplicatedFunction(dataset,'artist'))
print(dataset[['mode','target']].groupby(['mode']).mean().sort_values(by = 'target', ascending = False))
sns.factorplot('mode','target', data = dataset)

plt.show()
dataset[['key','target']].groupby('key').mean().sort_values(by = 'target', ascending = False)
Explode = [.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1]



f, ax = plt.subplots(figsize = (7,10))

dataset[['key','target']].groupby('key').mean().plot.pie(subplots = True, explode = Explode, autopct = '%.2f%%',ax = ax)

plt.legend(loc = 'lower left') 

plt.show()
sns.factorplot('key','target', data = dataset)

plt.show()
dataset[['time_signature','target']].groupby(['time_signature']).mean().sort_values(by = 'target', ascending = False)
sns.factorplot('time_signature','target', data = dataset)

plt.show()
f, ax = plt.subplots(2,2, figsize = (10,10))

dataset[dataset['target'] == 0].instrumentalness.plot.hist(bins = 10, ax = ax[0,0])

ax[0,0].set_title('target = 0 | instrumentalness')

dataset[dataset['target'] == 1].instrumentalness.plot.hist(bins = 10, ax = ax[0,1])

ax[0,1].set_title('target = 1 | instrumentalness')

dataset[dataset['target'] == 0].speechiness.plot.hist(bins = 10, ax = ax[1,0])

ax[1,0].set_title('target = 0 | speechiness')

dataset[dataset['target'] == 1].speechiness.plot.hist(bins = 10, ax = ax[1,1])

ax[1,1].set_title('target = 1 | speechiness')

plt.show()
f, ax = plt.subplots(2,2,figsize = (10,10))

dataset[dataset['target'] == 0].danceability.plot.hist(bins = 10, ax = ax[0,0])

ax[0,0].set_title('target = 0 | danceability')

dataset[dataset['target'] == 1].danceability.plot.hist(bins = 10, ax = ax[0,1])

ax[0,1].set_title('target = 1 | danceability')

dataset[dataset['target'] == 0].energy.plot.hist(bins = 10, ax = ax[1,0])

ax[1,0].set_title('target = 0 | energy')

dataset[dataset['target'] == 1].energy.plot.hist(bins = 10, ax = ax[1,1])

ax[1,1].set_title('target = 1 | energy')



plt.show()
f,ax = plt.subplots(1,2,figsize = (10,5))

dataset[dataset['target'] == 0].tempo.plot.hist(bins = 10, ax = ax[0])

ax[0].set_title('target = 0 | tempo')

dataset[dataset['target'] == 1].tempo.plot.hist(bins = 10, ax = ax[1])

ax[1].set_title('target = 1 | tempo')



plt.show()
f,ax = plt.subplots(2,2,figsize = (10,10))

dataset[dataset['target'] == 0].acousticness.plot.hist(bins = 10, ax = ax[0,0])

ax[0,0].set_title('target = 0 | acousticness')

dataset[dataset['target'] == 1].acousticness.plot.hist(bins = 10, ax = ax[0,1])

ax[0,1].set_title('target = 1 | acousticness')

dataset[dataset['target'] == 0].liveness.plot.hist(bins = 10, ax = ax[1,0])

ax[1,0].set_title('target = 0 | liveness')

dataset[dataset['target'] == 1].liveness.plot.hist(bins = 10, ax = ax[1,1])

ax[1,1].set_title('target = 1 | liveness')



plt.show()
f,ax = plt.subplots(1,2,figsize = (10,5))

dataset[dataset['target'] == 0].loudness.plot.hist(bins = 10, ax = ax[0])

ax[0].set_title('target = 0 | loudness')

dataset[dataset['target'] == 1].loudness.plot.hist(bins = 10, ax = ax[1])

ax[1].set_title('target = 1 | loudness')



plt.show()
f,ax = plt.subplots(figsize = (10,10)) #the size of the heat map

sns.heatmap(dataset.corr(), annot = True, fmt = '.2g', cmap = 'RdYlGn', ax= ax) #annot: values, fmt: decimal points of values

sns.set(font_scale = 0.75) #the font size of the value in the heat map

plt.xlabel('Features')

plt.show()
print('The Dimension of the dataset before drop the features:', dataset.shape)

dataset = dataset.drop(['song_title','artist','duration_ms'], axis = 1)

print('The Dimension of the dataset after drop the features:', dataset.shape)
#1. instrumentalness

dataset['InstrumentalnessBand'] = pd.cut(dataset['instrumentalness'],4)

dataset[['InstrumentalnessBand','target']].groupby('InstrumentalnessBand',as_index = False).mean().sort_values(by = 'InstrumentalnessBand', ascending = True)
dataset['instrumentalness2'] = 0

dataset.loc[dataset['instrumentalness'] <= 0.244,'instrumentalness2'] = 0

dataset.loc[(dataset['instrumentalness'] > 0.244) & (dataset['instrumentalness'] <= 0.488), 'instrumentalness2'] = 1

dataset.loc[(dataset['instrumentalness'] > 0.488) & (dataset['instrumentalness'] <= 0.732), 'instrumentalness2'] = 2

dataset.loc[dataset['instrumentalness'] > 0.732, 'instrumentalness2'] = 3
#2. speechiness

dataset['SpeechinessBand'] = pd.cut(dataset['speechiness'],4)

dataset[['SpeechinessBand','target']].groupby('SpeechinessBand',as_index = False).mean().sort_values(by = 'SpeechinessBand', ascending = True)
dataset['speechiness2'] = 0

dataset.loc[dataset['speechiness'] <= 0.221, 'speechiness2'] = 0

dataset.loc[(dataset['speechiness'] > 0.221) & (dataset['speechiness'] <= 0.42), 'speechiness2'] = 1

dataset.loc[(dataset['speechiness'] > 0.42) & (dataset['speechiness'] <= 0.618), 'speechiness2'] = 2

dataset.loc[dataset['speechiness'] > 0.618, 'speechiness2'] = 3
#3. danceability

dataset['DanceabilityBand'] = pd.cut(dataset['danceability'],4)

dataset[['DanceabilityBand','target']].groupby('DanceabilityBand',as_index = False).mean().sort_values(by = 'DanceabilityBand', ascending = True)
dataset['danceability2'] = 0

dataset.loc[dataset['danceability'] <= 0.338, 'danceability2'] = 0

dataset.loc[(dataset['danceability'] > 0.338) & (dataset['danceability'] <= 0.553), 'danceability2'] = 1

dataset.loc[(dataset['danceability'] > 0.553) & (dataset['danceability'] <= 0.769), 'danceability2'] = 2

dataset.loc[dataset['danceability'] > 0.769, 'danceability2'] = 3
#4. energy

dataset['EnergyBand'] = pd.cut(dataset['energy'],4)

dataset[['EnergyBand','target']].groupby('EnergyBand',as_index = False).mean().sort_values(by = 'EnergyBand', ascending = True)
dataset['energy2'] = 0

dataset.loc[dataset['energy'] <= 0.261, 'energy2'] = 0

dataset.loc[(dataset['energy'] > 0.261) & (dataset['energy'] <= 0.506), 'energy2'] = 1

dataset.loc[(dataset['energy'] > 0.506) & (dataset['energy'] <= 0.752), 'energy2'] = 2

dataset.loc[dataset['energy'] > 0.752, 'energy2'] = 3
#5. acousticness

dataset['AcousticnessBand'] = pd.cut(dataset['acousticness'],4)

dataset[['AcousticnessBand','target']].groupby('AcousticnessBand',as_index = False).mean().sort_values(by = 'AcousticnessBand', ascending = True)
dataset['acousticness2'] = 0

dataset.loc[dataset['acousticness'] <= 0.249, 'acousticness2'] = 0

dataset.loc[(dataset['acousticness'] > 0.249) & (dataset['acousticness'] <= 0.498), 'acousticness2'] = 1

dataset.loc[(dataset['acousticness'] > 0.498) & (dataset['acousticness'] <= 0.746), 'acousticness2'] = 2

dataset.loc[dataset['acousticness'] > 0.746, 'acousticness2'] = 3
#6. liveness

dataset['LivenessBand'] = pd.cut(dataset['liveness'],4)

dataset[['LivenessBand','target']].groupby('LivenessBand', as_index = False).mean().sort_values(by = 'LivenessBand', ascending = True)
dataset['liveness2'] = 0

dataset.loc[dataset['liveness'] <= 0.256,'liveness2'] = 0

dataset.loc[(dataset['liveness'] > 0.256) & (dataset['liveness'] <= 0.494),'liveness2'] = 1

dataset.loc[(dataset['liveness'] > 0.494) & (dataset['liveness'] <= 0.731),'liveness2'] = 2

dataset.loc[dataset['liveness'] > 0.731, 'liveness2'] = 3
#7. loudness

dataset['LoudnessBand'] = pd.cut(dataset['loudness'], 4)

dataset[['LoudnessBand','target']].groupby('LoudnessBand').mean()
dataset['loudness2'] = 0

dataset.loc[dataset['loudness'] <= -24.9, 'loudness2'] = 0

dataset.loc[(dataset['loudness'] > -24.9) & (dataset['loudness'] <= -16.702), 'loudness2'] = 1

dataset.loc[(dataset['loudness'] > -16.702) & (dataset['loudness'] <= -8.504), 'loudness2'] = 2

dataset.loc[dataset['loudness'] > -8.504, 'loudness2'] = 3
#8. tempo



dataset['TempoBand'] = pd.cut(dataset['tempo'],4)

dataset[['TempoBand','target']].groupby('TempoBand',as_index = False).mean().sort_values(by = 'TempoBand', ascending = True)
dataset['tempo2'] = 0

dataset.loc[dataset['tempo'] <= 90.727, 'tempo2'] = 0

dataset.loc[(dataset['tempo'] > 90.727) & (dataset['tempo'] <= 133.595), 'tempo2'] = 1

dataset.loc[(dataset['tempo'] > 133.595) & (dataset['tempo'] <= 176.463), 'tempo2'] = 2

dataset.loc[ dataset['tempo'] > 176.463, 'tempo2'] = 3
#9. valence

dataset['valenceband'] = pd.cut(dataset['valence'], 4)

dataset[['valenceband','target']].groupby('valenceband').mean().sort_values(by = 'valenceband')
dataset['valence2'] = 0

dataset.loc[dataset['valence'] <= 0.274, 'valence2'] = 0

dataset.loc[(dataset['valence'] > 0.274) & (dataset['valence'] <= 0.513), 'valence2'] = 1

dataset.loc[(dataset['valence']> 0.513) & (dataset['valence'] <= 0.753), 'valence2'] = 2

dataset.loc[dataset['valence'] > 0.753, 'valence2'] = 3
dataset.head()
#Drop the range features

dataset = dataset.drop(['InstrumentalnessBand','SpeechinessBand','DanceabilityBand','EnergyBand',

                        'AcousticnessBand','LivenessBand','LoudnessBand','TempoBand','valenceband'], axis = 1)



dataset.columns
#Drop all the numerical features without process through binning method



dataset = dataset.drop(['acousticness','danceability','energy','instrumentalness',

                        'liveness','loudness','speechiness','tempo','valence'],axis = 1)



dataset.columns
#Rename the binning features

dataset = dataset.rename(columns = {'instrumentalness2':'instrumentalness','speechiness2': 'speechiness', 'danceability2': 'danceability',

                                   'energy2': 'energy','acousticness2':'acousticness', 'liveness2':'liveness', 'loudness2':'loudness',

                                   'tempo2': 'tempo', 'valence2': 'valence'})

dataset.columns
#Change the time_signature features from numerical type features to Int type features



dataset['time_signature'] = dataset['time_signature'].astype(int)

dataset.head()
dataset.describe()
#drop the features which doesn't have the good result in average mean

dataset = dataset.drop(['instrumentalness','speechiness','acousticness','liveness'], axis = 1)

print('The dimension of the dataset after drop the features: ', dataset.shape)
df_key = pd.get_dummies(dataset['key'])

df_time_signature = pd.get_dummies(dataset['time_signature'])

df_danceability = pd.get_dummies(dataset['danceability'])

df_energy = pd.get_dummies(dataset['energy'])

df_loudness = pd.get_dummies(dataset['loudness'])

df_tempo = pd.get_dummies(dataset['tempo'])

df_valence = pd.get_dummies(dataset['valence'])



dummy_variables = pd.concat([df_key,df_time_signature,df_danceability,df_energy,df_loudness,df_tempo,df_valence], axis = 1)

dataset = pd.concat([dataset,dummy_variables], axis = 1)

print('The dimension of the dataset after create the dummy variables: ', dataset.shape)
#Replace the numerical features by dummy variables, but the target class labels

dataset = dataset.drop(['key','time_signature','danceability','energy','loudness','tempo','valence'], axis = 1)

print('The dimension of the dataset after drop the numerical features: ', dataset.shape)
#Import the library we need to use for the following step



from sklearn.linear_model import LogisticRegression #Logistic Regression

from sklearn.naive_bayes import GaussianNB #Naive Bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.neighbors import KNeighborsClassifier #KNN



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix



#1. Create the X_train without the target class label & Y_train (target)

X_train = dataset.drop('target', axis = 1)

Y_train = dataset['target']



#2. Split the X_Train & Y_train into training set & testing set by train_test_split function

x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size = 0.2, random_state = 0)
print('the dimension of the x_train: ', x_train.shape)

print('the dimension of the x_test: ', x_test.shape)
#3. Fit the model into the training set



#i. Logistic Regression

log = LogisticRegression()

log.fit(x_train,y_train)

log_y_pred = log.predict(x_test)

log_result_train = round(log.score(x_train,y_train)*100,2)



#ii. Gaussian Naive Bayes

NB = GaussianNB()

NB.fit(x_train,y_train)

NB_y_pred = NB.predict(x_test)

NB_result_train = round(NB.score(x_train,y_train)*100,2)



#iii. Decision Tree

DT = DecisionTreeClassifier()

DT.fit(x_train,y_train)

DT_y_pred = DT.predict(x_test)

DT_result_train = round(DT.score(x_train,y_train)*100, 2)



#iv. K-Nearest Neighbors (K-NN)

KNN = KNeighborsClassifier()

KNN.fit(x_train,y_train)

KNN_y_pred = KNN.predict(x_test)

KNN_result_train = round(KNN.score(x_train,y_train)*100,2)



print('1. Logistic Regression: ', log_result_train)

print('2. Gaussian Naive Bayes: ', NB_result_train)

print('3. Decision Tree Classifier: ', DT_result_train)

print('4. K-NN: ', KNN_result_train)
#4. Fit the model into the testing dataset





#i. Logistic Regression

log_result_test = round(log.score(x_test,y_test)*100,2)



#ii. Gaussian Naive Bayes

NB_result_test = round(NB.score(x_test,y_test)*100,2)



#iii. Decision Tree

DT_result_test = round(DT.score(x_test,y_test)*100,2)



#iv. K-Nearest Neighbors

KNN_result_test = round(KNN.score(x_test,y_test)*100,2)



print('1. Logistic Regression: {}'.format(log_result_test))

print('2. Gaussian Naive Bayes: {}'.format(NB_result_test))

print('3. Decision Tree: {}'.format(DT_result_test))

print('4. K-NN: {}'.format(KNN_result_test))
#5. Apply K-fold Cross Validation method into the model (testing data)



#1. Logistic Regression

Kfold = KFold(n_splits = 10)

logregScore = cross_val_score(log,x_test,y_test, cv = Kfold)

avglogregScore = np.mean(logregScore)



#2. Gaussien Naive Bayes

NBScore = cross_val_score(NB,x_test,y_test, cv = Kfold)

avgNBScore = np.mean(NBScore)



#3. Decision Tree Classifier

DTScore = cross_val_score(DT, x_test,y_test, cv =Kfold)

avgDTScore = np.mean(DTScore)



#4. K-NN

KNNScore = cross_val_score(KNN, x_test,y_test, cv = Kfold)

avgKNNScore = np.mean(KNNScore)

#for i in range(len(logregScore)): 

  #  print(i+1, 'Logistic Regression:',logregScore[i])

    

print('1. Logistic Regression: ', round(avglogregScore*100,2))

print('2. Gaussian Naive Bayes:  ', round(avgNBScore*100,2))

print('3. Decision Tree Classifier: ', round(avgDTScore*100,2))

print('4. K-NN: ', round(avgKNNScore*100,2))

#6. Create the confusion matrix table for the performance of model



f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize = (12,3))

#1. Logistic Regression

LogregCM = confusion_matrix(y_test,log_y_pred)

sns.heatmap(LogregCM, annot = True, fmt = 'd', vmin = 0, vmax = 150,cmap = 'viridis', ax = ax1)

#Annot: the value of the heatmap

#fmt: the decimal point of value of heatmap

#vmin, vmax: the limits of the colorbar

ax1.set_title('Logistic Regression')

ax1.set_xlabel('Features')



#2. Gaussian Naive Bayes

NBCM = confusion_matrix(y_test,NB_y_pred)

sns.heatmap(NBCM, annot = True, fmt = 'd', vmin = 0, vmax = 150, cmap = 'YlGnBu', ax = ax2)

ax2.set_title('Gaussian Naive Bayes')

ax2.set_xlabel('Features')





#3. Decision Tree

DTCM = confusion_matrix(y_test, DT_y_pred)

sns.heatmap(DTCM, annot = True, fmt = 'd', vmin = 0, vmax = 150, cmap = 'viridis', ax = ax3)

ax3.set_title('Decision Tree')

ax3.set_xlabel('Features')





#4. KNN

KNNCM = confusion_matrix(y_test, KNN_y_pred)

sns.heatmap(KNNCM, annot = True, fmt = 'd', vmin = 0 , vmax = 150, cmap = 'YlGnBu', ax = ax4)

ax4.set_title('KNN')

ax4.set_xlabel('Features')





plt.show()