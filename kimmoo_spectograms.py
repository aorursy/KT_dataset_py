import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
print(os.listdir("../input"))
data_path = '../input/'
train_root = '../input/audio_train/'
test_root = '../input/audio_test/'
def to_log_S(fname, PATH):
    y, sr = librosa.load(os.path.join(PATH, fname))
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    return log_S
to_log_S('65b299e9.wav', train_root)[:10, :4]
mean = (0.485+0.456+0.406)/3
std = (0.229+0.224+0.225)/3
mean, std
def normalize(x):
    x = -x/80
    x = (x-mean)/std
def display_spectogram(log_S):
    sr = 22050
    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
display_spectogram(to_log_S('65b299e9.wav', train_root)[:10, :4])
display_spectogram(to_log_S('65b299e9.wav', train_root))