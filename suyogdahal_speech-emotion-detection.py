! apt-get install -y libsndfile-dev
import glob

import os



# For manipulating audio

import librosa

import librosa.display as disp

import soundfile

import numpy as np

import pandas as pd



# For machine learning models

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.utils.multiclass import unique_labels

import optuna



# For displaying

import matplotlib.pyplot as plt

import IPython.display as ipd

import seaborn as sns

import tqdm



input_folder = "/kaggle/input/"
sample_audio_path = input_folder + 'Actor_01/03-01-03-02-02-02-01.wav'



# play a sample audio

ipd.Audio(sample_audio_path)
# Load the audio file as an array without resampling

audio, sr = librosa.load(sample_audio_path, sr=None)



####### Visualizing Waveform #######

plt.figure(figsize=(14, 5))

plt.title("Waveform")

disp.waveplot(audio, sr=sr)





####### Zooming in in the Waveform #######

plt.figure(figsize=(14, 5))

plt.title("Zoomed in Waveform")

plt.plot(audio[30000:30050])





####### Visualizing Spectogram #######

X = librosa.stft(audio)

Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))

plt.title("Spectogram")

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

plt.colorbar()
# Extract features (mfcc, chroma, mel) from a sound file

def extract_feature(file_name, mfcc, chroma, mel):

    with soundfile.SoundFile(file_name) as sound_file:

        X = sound_file.read(dtype="float32")

        sample_rate = sound_file.samplerate

        if chroma:

            stft = np.abs(librosa.stft(X))

        result = np.array([])

        if mfcc:

            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

            result = np.hstack((result, mfccs))

        if chroma:

            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

            result = np.hstack((result, chroma))

        if mel:

            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

            result = np.hstack((result, mel))

            

    return result
emotions={

  '01':'neutral',

  '02':'calm',

  '03':'happy',

  '04':'sad',

  '05':'angry',

  '06':'fearful',

  '07':'disgust',

  '08':'surprised'

}
def load_data(test_size=0.2):

    x,y=[],[]

    for file in glob.glob(input_folder + "Actor_*/*.wav"):

        file_name = os.path.basename(file)

        emotion = emotions[file_name.split("-")[2]]

        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)

        x.append(feature)

        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
x_train, x_test, y_train, y_test = load_data(test_size=0.25)
print(f'Number of training data: {x_train.shape[0]}')

print(f'Number of testing data: {x_test.shape[0]}')

print(f'Number of features extracted: {x_train.shape[1]}')
# Scale data

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
mlp_model = MLPClassifier(activation='relu',

                         solver='sgd',

                         hidden_layer_sizes=100,

                         alpha=0.839903176695813,

                         batch_size=150,

                         learning_rate='adaptive',

                         max_iter=100000)
# Fit mlp model

mlp_model.fit(x_train,y_train)
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, model='clf'):

    """

    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix,

    as a seaborn heatmap. 

    """

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt="d", cmap=plt.cm.Oranges)

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

        

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    # fix for mpl bug that cuts off top/bottom of seaborn viz

    b, t = plt.ylim() 

    b += 0.5 

    t -= 0.5 

    plt.ylim(b, t) 

    plt.show()
def get_model_performance(model):

    y_pred = model.predict(x_test)

    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

    # Print the accuracy

    print("\nModel:{}    Accuracy: {:.2f}%".

          format(type(model).__name__ , accuracy*100))

    

    # Print Confusion Matrix

    print_confusion_matrix(confusion_matrix(y_test, y_pred), unique_labels(y_test, y_pred), model=model)
# Get Accuracy of individual models

get_model_performance(mlp_model)
# def objective_mlp(trial):



#     params = {

#         'activation': trial.suggest_categorical('activation', ['logistic', 'tanh', 'relu']),

#         'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),

#         'hidden_layer_sizes':trial.suggest_int('hidden_layer_sizes', 100, 300, 1500),

#         'alpha': trial.suggest_uniform('alpha', 0.001, 0.99),

#         'batch_size':trial.suggest_int('batch_size', 150, 256, 300), 

#         'learning_rate': trial.suggest_categorical('learning_rate', ['adaptive', 'constant', 'invscaling']),

#         'max_iter': 1000

#         }

  

#     model = MLPClassifier(**params, random_state = 22) 

    

#     model.set_params(**params)



#     return np.mean(cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy'))
# study = optuna.create_study(direction='maximize')

# study.optimize(objective_mlp, n_trials=10)