import os
import numpy as np
import IPython
import matplotlib
from matplotlib import pyplot as plt
import IPython.display as ipd  # To play sound in the notebook
import librosa
from librosa import display
import pandas as pd

%matplotlib inline
matplotlib.style.use('ggplot')
class Config(object):
    def __init__(self,
                 sampling_rate=22050, n_classes=41,
                 train_dir='../input/audio_train',
                 n_mels=64, frame_weigth=80, frame_shift=10):

        self.sampling_rate = sampling_rate
        self.n_classes = n_classes
        self.train_dir = train_dir

        self.n_fft = int(frame_weigth / 1000 * sampling_rate)
        self.n_mels = n_mels
        self.frame_weigth = frame_weigth
        self.frame_shift = frame_shift
        self.hop_length = int(frame_shift / 1000 * sampling_rate)

        
def get_subset(csv_file, n):
    """
    Train set is too big. We get a subset to display.
    """
    train = pd.read_csv(csv_file)
    train = train.sort_values(by=['label', 'fname'])

    LABELS = list(train.label.unique())

    subset = train.iloc[[0]]
    
    # choose n files from each category randomly.
    for label in LABELS:
        subset = subset.append(train[train.label == label].sample(n))

    subset = subset.iloc[1:]
    return subset


def add_subplot_axes(ax, position):
    box = ax.get_position()
    
    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]
    
    return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]], axisbg='w')


def plot_one_file(row, ax):
    fname = os.path.join(config.train_dir, row['fname'])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax_waveform = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
    ax_logmel = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])
    
    data, _ = librosa.load(fname, sr=config.sampling_rate)
    melspec = librosa.feature.melspectrogram(data, sr=config.sampling_rate,
                                             n_fft=config.n_fft, hop_length=config.hop_length,
                                             n_mels=config.n_mels)

    logmel = librosa.core.power_to_db(melspec)

    ax_waveform.plot(data, '-')
    ax_waveform.get_xaxis().set_visible(False)
    ax_waveform.get_yaxis().set_visible(False)
    ax_waveform.set_title('{0} \n {1}'.format(row['label'], row['fname']), {'fontsize': 8}, y=1.03)

    librosa.display.specshow(logmel, sr=config.sampling_rate)
    ax_logmel.get_xaxis().set_visible(False)
    ax_logmel.get_yaxis().set_visible(False)
# frame_weight: 80ms, frame_shift:10ms
config = Config(frame_weigth=80, frame_shift=10, train_dir='../input/audio_train')

files_shown = 5 
subset = get_subset('../input/train.csv', files_shown)

f, axes = plt.subplots(config.n_classes, files_shown, figsize=(files_shown * 3, config.n_classes * 3), sharex=True, sharey=True)
f.subplots_adjust(hspace = 0.35)

for i, (_, row) in enumerate(subset.iterrows()):
    plot_one_file(row, axes[i//files_shown, i%files_shown])
