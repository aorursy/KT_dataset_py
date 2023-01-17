# Usual Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn



# Librosa (the mother of audio files)

import librosa

import librosa.display

import IPython.display as ipd

import warnings

warnings.filterwarnings('ignore')
import os

general_path = '../input/gtzan-dataset-music-genre-classification/Data'

print(list(os.listdir(f'{general_path}/genres_original/')))
# Importing 1 file

y, sr = librosa.load(f'{general_path}/genres_original/reggae/reggae.00036.wav')



print('y:', y, '\n')

print('y shape:', np.shape(y), '\n')

print('Sample Rate (KHz):', sr, '\n')



# Verify length of the audio

print('Check Len of Audio:', 661794/22050)
# Trim leading and trailing silence from an audio signal (silence before and after the actual audio)

audio_file, _ = librosa.effects.trim(y)



# the result is an numpy ndarray

print('Audio File:', audio_file, '\n')

print('Audio File shape:', np.shape(audio_file))
plt.figure(figsize = (16, 6))

librosa.display.waveplot(y = audio_file, sr = sr, color = "#A300F9");

plt.title("Sound Waves in Reggae 36", fontsize = 23);
# Default FFT window size

n_fft = 2048 # FFT window size

hop_length = 512 # number audio of frames between STFT columns (looks like a good default)



# Short-time Fourier transform (STFT)

D = np.abs(librosa.stft(audio_file, n_fft = n_fft, hop_length = hop_length))



print('Shape of D object:', np.shape(D))
plt.figure(figsize = (16, 6))

plt.plot(D);
# Convert an amplitude spectrogram to Decibels-scaled spectrogram.

DB = librosa.amplitude_to_db(D, ref = np.max)



# Creating the Spectogram

plt.figure(figsize = (16, 6))

librosa.display.specshow(DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log',

                        cmap = 'cool')

plt.colorbar();
y, sr = librosa.load(f'{general_path}/genres_original/metal/metal.00036.wav')

y, _ = librosa.effects.trim(y)





S = librosa.feature.melspectrogram(y, sr=sr)

S_DB = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize = (16, 6))

librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log',

                        cmap = 'cool');

plt.colorbar();

plt.title("Metal Mel Spectrogram", fontsize = 23);
y, sr = librosa.load(f'{general_path}/genres_original/classical/classical.00036.wav')

y, _ = librosa.effects.trim(y)





S = librosa.feature.melspectrogram(y, sr=sr)

S_DB = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize = (16, 6))

librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log',

                        cmap = 'cool');

plt.colorbar();

plt.title("Classical Mel Spectrogram", fontsize = 23);
# Total zero_crossings in our 1 song

zero_crossings = librosa.zero_crossings(audio_file, pad=False)

print(sum(zero_crossings))
y_harm, y_perc = librosa.effects.hpss(audio_file)



plt.figure(figsize = (16, 6))

plt.plot(y_harm, color = '#A300F9');

plt.plot(y_perc, color = '#FFB100');
tempo, _ = librosa.beat.beat_track(y, sr = sr)

tempo
# Calculate the Spectral Centroids

spectral_centroids = librosa.feature.spectral_centroid(audio_file, sr=sr)[0]



# Shape is a vector

print('Centroids:', spectral_centroids, '\n')

print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')



# Computing the time variable for visualization

frames = range(len(spectral_centroids))



# Converts frame counts to time (seconds)

t = librosa.frames_to_time(frames)



print('frames:', frames, '\n')

print('t:', t)



# Function that normalizes the Sound Data

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_file, sr=sr, alpha=0.4, color = '#A300F9');

plt.plot(t, normalize(spectral_centroids), color='#FFB100');
# Spectral RollOff Vector

spectral_rolloff = librosa.feature.spectral_rolloff(audio_file, sr=sr)[0]



# The plot

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_file, sr=sr, alpha=0.4, color = '#A300F9');

plt.plot(t, normalize(spectral_rolloff), color='#FFB100');
mfccs = librosa.feature.mfcc(audio_file, sr=sr)

print('mfccs shape:', mfccs.shape)



#Displaying  the MFCCs:

plt.figure(figsize = (16, 6))

librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
# Perform Feature Scaling

mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

print('Mean:', mfccs.mean(), '\n')

print('Var:', mfccs.var())



plt.figure(figsize = (16, 6))

librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
# Increase or decrease hop_length to change how granular you want your data to be

hop_length = 5000



# Chromogram

chromagram = librosa.feature.chroma_stft(audio_file, sr=sr, hop_length=hop_length)

print('Chromogram shape:', chromagram.shape)



plt.figure(figsize=(16, 6))

librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm');
data = pd.read_csv(f'{general_path}/features_30_sec.csv')

data.head()
# Computing the Correlation Matrix

spike_cols = [col for col in data.columns if 'mean' in col]

corr = data[spike_cols].corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(16, 11));



# Generate a custom diverging colormap

cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.title('Correlation Heatmap (for the MEAN variables)', fontsize = 25)

plt.xticks(fontsize = 10)

plt.yticks(fontsize = 10);

plt.savefig("Corr Heatmap.jpg")
x = data[["label", "tempo"]]



f, ax = plt.subplots(figsize=(16, 9));

sns.boxplot(x = "label", y = "tempo", data = x, palette = 'husl');



plt.title('BPM Boxplot for Genres', fontsize = 25)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 10);

plt.xlabel("Genre", fontsize = 15)

plt.ylabel("BPM", fontsize = 15)

plt.savefig("BPM Boxplot.jpg")
from sklearn import preprocessing



data = data.iloc[0:, 1:]

y = data['label']

X = data.loc[:, data.columns != 'label']



#### NORMALIZE X ####

cols = X.columns

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X)

X = pd.DataFrame(np_scaled, columns = cols)





#### PCA 2 COMPONENTS ####

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])



# concatenate with target label

finalDf = pd.concat([principalDf, y], axis = 1)



pca.explained_variance_ratio_



# 44.93 variance explained
plt.figure(figsize = (16, 9))

sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7,

               s = 100);



plt.title('PCA on Genres', fontsize = 25)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 10);

plt.xlabel("Principal Component 1", fontsize = 15)

plt.ylabel("Principal Component 2", fontsize = 15)

plt.savefig("PCA Scattert.jpg")
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier, XGBRFClassifier

from xgboost import plot_tree, plot_importance



from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE
data = pd.read_csv(f'{general_path}/features_3_sec.csv')

data = data.iloc[0:, 1:] 

data.head()
y = data['label'] # genre variable.

X = data.loc[:, data.columns != 'label'] #select all columns but not the labels



#### NORMALIZE X ####



# Normalize so everything is on the same scale. 



cols = X.columns

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X)



# new data frame with the new scaled data. 

X = pd.DataFrame(np_scaled, columns = cols)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
def model_assess(model, title = "Default"):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    #print(confusion_matrix(y_test, preds))

    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')
# Naive Bayes

nb = GaussianNB()

model_assess(nb, "Naive Bayes")



# Stochastic Gradient Descent

sgd = SGDClassifier(max_iter=5000, random_state=0)

model_assess(sgd, "Stochastic Gradient Descent")



# KNN

knn = KNeighborsClassifier(n_neighbors=19)

model_assess(knn, "KNN")



# Decission trees

tree = DecisionTreeClassifier()

model_assess(tree, "Decission trees")



# Random Forest

rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)

model_assess(rforest, "Random Forest")



# Support Vector Machine

svm = SVC(decision_function_shape="ovo")

model_assess(svm, "Support Vector Machine")



# Logistic Regression

lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

model_assess(lg, "Logistic Regression")



# Neural Nets

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)

model_assess(nn, "Neural Nets")



# Cross Gradient Booster

xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)

model_assess(xgb, "Cross Gradient Booster")



# Cross Gradient Booster (Random Forest)

xgbrf = XGBRFClassifier(objective= 'multi:softmax')

model_assess(xgbrf, "Cross Gradient Booster (Random Forest)")
# Final model

xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)

xgb.fit(X_train, y_train)





preds = xgb.predict(X_test)



print('Accuracy', ':', round(accuracy_score(y_test, preds), 5), '\n')



# Confusion Matrix

confusion_matr = confusion_matrix(y_test, preds) #normalize = 'true'

plt.figure(figsize = (16, 9))

sns.heatmap(confusion_matr, cmap="Blues", annot=True, 

            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],

           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);

plt.savefig("conf matrix")
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(estimator=xgb, random_state=1)

perm.fit(X_test, y_test)



eli5.show_weights(estimator=perm, feature_names = X_test.columns.tolist())
# Libraries

import IPython.display as ipd

from sklearn.metrics.pairwise import cosine_similarity

from sklearn import preprocessing



# Read data

data = pd.read_csv(f'{general_path}/features_30_sec.csv', index_col='filename')



# Extract labels

labels = data[['label']]



# Drop labels from original dataframe

data = data.drop(columns=['length','label'])

data.head()



# Scale the data

data_scaled=preprocessing.scale(data)

print('Scaled data type:', type(data_scaled))
# Cosine similarity

similarity = cosine_similarity(data_scaled)

print("Similarity shape:", similarity.shape)



# Convert into a dataframe and then set the row index and column names as labels

sim_df_labels = pd.DataFrame(similarity)

sim_df_names = sim_df_labels.set_index(labels.index)

sim_df_names.columns = labels.index



sim_df_names.head()
def find_similar_songs(name):

    # Find songs most similar to another song

    series = sim_df_names[name].sort_values(ascending = False)

    

    # Remove cosine similarity == 1 (songs will always have the best match with themselves)

    series = series.drop(name)

    

    # Display the 5 top matches 

    print("\n*******\nSimilar songs to ", name)

    print(series.head(5))
# pop.00019 - Britney Spears "Hit me baby one more time"

find_similar_songs('pop.00019.wav') 



ipd.Audio(f'{general_path}/genres_original/pop/pop.00019.wav')
ipd.Audio(f'{general_path}/genres_original/pop/pop.00023.wav')
ipd.Audio(f'{general_path}/genres_original/pop/pop.00034.wav')
ipd.Audio(f'{general_path}/genres_original/pop/pop.00078.wav')
ipd.Audio(f'{general_path}/genres_original/pop/pop.00088.wav')
ipd.Audio(f'{general_path}/genres_original/pop/pop.00091.wav')
# metal.00002 - Iron Maiden "Flight of Icarus"

find_similar_songs('metal.00002.wav') 



ipd.Audio(f'{general_path}/genres_original/metal/metal.00002.wav')
ipd.Audio(f'{general_path}/genres_original/metal/metal.00028.wav')
ipd.Audio(f'{general_path}/genres_original/metal/metal.00059.wav')
ipd.Audio(f'{general_path}/genres_original/rock/rock.00018.wav')
ipd.Audio(f'{general_path}/genres_original/rock/rock.00017.wav')
ipd.Audio(f'{general_path}/genres_original/rock/rock.00016.wav')