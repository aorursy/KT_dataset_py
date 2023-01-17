import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import librosa

from scipy.io import wavfile

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import IPython.display as ipd



# Any results you write to the current directory are saved as output.
meta = pd.read_csv("../input/train_curated.csv",sep=",")

meta.head()
meta.shape
meta.labels.value_counts()
labels = meta.labels.str.split(",",expand=True)

labels.head()
labels_onehot = pd.get_dummies(labels.loc[:,0])
labels_onehot.shape
for i in range(1, labels.shape[1]):

    for lab in labels.loc[:,i].unique():

        if lab is not None:

            if not lab in list(labels_onehot):

                print(lab)

                labels_onehot[lab] = 0

            labels_onehot.loc[labels.loc[:,i]==lab,lab] = 1
labels_onehot[["Strum","Water_tap_and_faucet"]].sum(axis=0)
labels_onehot.sum(axis=0)
plt.figure(figsize=(15,15))

plt.matshow(labels_onehot.corr(),fignum=1)

plt.xticks(range(len(labels_onehot.columns)), labels_onehot.columns, rotation='vertical');

plt.yticks(range(len(labels_onehot.columns)), labels_onehot.columns);

plt.show()
meta.loc[labels_onehot.Electric_guitar == 1,"fname"].head()
ipd.Audio(wavfile.read("../input/train_curated/03730245.wav")[1], rate=44100)
sampling, audio = wavfile.read("../input/train_curated/03730245.wav")
plt.figure(figsize = (20,5))

plt.plot(audio)

plt.annotate("Attack",(2500,8000),(0,25000), arrowprops = {"arrowstyle":"->"})

plt.annotate("Beginning of note",(20000,-20000),(30000,-25000), arrowprops = {"arrowstyle":"->"})

plt.annotate("Muting",(90000,-5000),(90000,-25000), arrowprops = {"arrowstyle":"->"})

y, sr = librosa.load("../input/train_curated/03730245.wav")

mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

mel_spec.shape
# since specshow does not seem to work from librosa library, just copying it from : https://github.com/librosa/librosa/blob/master/librosa/display.py

from matplotlib.cm import get_cmap

from matplotlib.axes import Axes

from matplotlib.ticker import Formatter, ScalarFormatter

from matplotlib.ticker import LogLocator, FixedLocator, MaxNLocator

from matplotlib.ticker import SymmetricalLogLocator



def specshow(data, x_coords=None, y_coords=None,

             x_axis=None, y_axis=None,

             sr=22050, hop_length=512,

             fmin=None, fmax=None,

             bins_per_octave=12,

             ax=None,

            **kwargs):

    if np.issubdtype(data.dtype, np.complexfloating):

        warnings.warn('Trying to display complex-valued input. '

                      'Showing magnitude instead.')

        data = np.abs(data)

    kwargs.setdefault('cmap', cmap(data))

    kwargs.setdefault('rasterized', True)

    kwargs.setdefault('edgecolors', 'None')

    kwargs.setdefault('shading', 'flat')

    all_params = dict(kwargs=kwargs,

                      sr=sr,

                      fmin=fmin,

                      fmax=fmax,

                      bins_per_octave=bins_per_octave,

                      hop_length=hop_length)

    # Get the x and y coordinates

    y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)

    x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)

    axes = __check_axes(ax)

    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)

    __set_current_image(ax, out)

    axes.set_xlim(x_coords.min(), x_coords.max())

    axes.set_ylim(y_coords.min(), y_coords.max())

    # Set up axis scaling

    __scale_axes(axes, x_axis, 'x')

    __scale_axes(axes, y_axis, 'y')

    # Construct tickers and locators

    __decorate_axis(axes.xaxis, x_axis)

    __decorate_axis(axes.yaxis, y_axis)

    return axes



def __set_current_image(ax, img):

    plt.sci(img)



    

def __mesh_coords(ax_type, coords, n, **kwargs):

    if coords is not None:

        if len(coords) < n:

            raise ParameterError('Coordinate shape mismatch: '

                                 '{}<{}'.format(len(coords), n))

        return coords



    coord_map = {

                 'mel': __coord_mel_hz,

                 None: __coord_n}



    if ax_type not in coord_map:

        raise ParameterError('Unknown axis type: {}'.format(ax_type))

    return coord_map[ax_type](n, **kwargs)





def __check_axes(axes):

    '''Check if "axes" is an instance of an axis object. If not, use `gca`.'''

    if axes is None:

        import matplotlib.pyplot as plt

        axes = plt.gca()

    elif not isinstance(axes, Axes):

        raise ValueError("`axes` must be an instance of matplotlib.axes.Axes. "

                         "Found type(axes)={}".format(type(axes)))

    return axes





def __scale_axes(axes, ax_type, which):

    '''Set the axis scaling'''



    kwargs = dict()

    if which == 'x':

        thresh = 'linthreshx'

        base = 'basex'

        scale = 'linscalex'

        scaler = axes.set_xscale

        limit = axes.set_xlim

    else:

        thresh = 'linthreshy'

        base = 'basey'

        scale = 'linscaley'

        scaler = axes.set_yscale

        limit = axes.set_ylim



    # Map ticker scales

    if ax_type == 'mel':

        mode = 'symlog'

        kwargs[thresh] = 1000.0

        kwargs[base] = 2



    elif ax_type == 'log':

        mode = 'symlog'

        kwargs[base] = 2

        kwargs[thresh] = core.note_to_hz('C2')

        kwargs[scale] = 0.5



    elif ax_type in ['cqt', 'cqt_hz', 'cqt_note']:

        mode = 'log'

        kwargs[base] = 2



    elif ax_type == 'tempo':

        mode = 'log'

        kwargs[base] = 2

        limit(16, 480)

    else:

        return



    scaler(mode, **kwargs)





def __decorate_axis(axis, ax_type):

    '''Configure axis tickers, locators, and labels'''



    if ax_type == 'tonnetz':

        axis.set_major_formatter(TonnetzFormatter())

        axis.set_major_locator(FixedLocator(0.5 + np.arange(6)))

        axis.set_label_text('Tonnetz')



    elif ax_type == 'chroma':

        axis.set_major_formatter(ChromaFormatter())

        axis.set_major_locator(FixedLocator(0.5 +

                                            np.add.outer(12 * np.arange(10),

                                                         [0, 2, 4, 5, 7, 9, 11]).ravel()))

        axis.set_label_text('Pitch class')



    elif ax_type == 'tempo':

        axis.set_major_formatter(ScalarFormatter())

        axis.set_major_locator(LogLocator(base=2.0))

        axis.set_label_text('BPM')



    elif ax_type == 'time':

        axis.set_major_formatter(TimeFormatter(unit=None, lag=False))

        axis.set_major_locator(MaxNLocator(prune=None,

                                           steps=[1, 1.5, 5, 6, 10]))

        axis.set_label_text('Time')



    elif ax_type == 's':

        axis.set_major_formatter(TimeFormatter(unit='s', lag=False))

        axis.set_major_locator(MaxNLocator(prune=None,

                                           steps=[1, 1.5, 5, 6, 10]))

        axis.set_label_text('Time (s)')



    elif ax_type == 'ms':

        axis.set_major_formatter(TimeFormatter(unit='ms', lag=False))

        axis.set_major_locator(MaxNLocator(prune=None,

                                           steps=[1, 1.5, 5, 6, 10]))

        axis.set_label_text('Time (ms)')



    elif ax_type == 'lag':

        axis.set_major_formatter(TimeFormatter(unit=None, lag=True))

        axis.set_major_locator(MaxNLocator(prune=None,

                                           steps=[1, 1.5, 5, 6, 10]))

        axis.set_label_text('Lag')



    elif ax_type == 'lag_s':

        axis.set_major_formatter(TimeFormatter(unit='s', lag=True))

        axis.set_major_locator(MaxNLocator(prune=None,

                                           steps=[1, 1.5, 5, 6, 10]))

        axis.set_label_text('Lag (s)')



    elif ax_type == 'lag_ms':

        axis.set_major_formatter(TimeFormatter(unit='ms', lag=True))

        axis.set_major_locator(MaxNLocator(prune=None,

                                           steps=[1, 1.5, 5, 6, 10]))

        axis.set_label_text('Lag (ms)')



    elif ax_type == 'cqt_note':

        axis.set_major_formatter(NoteFormatter())

        axis.set_major_locator(LogLocator(base=2.0))

        axis.set_minor_formatter(NoteFormatter(major=False))

        axis.set_minor_locator(LogLocator(base=2.0,

                                          subs=2.0**(np.arange(1, 12)/12.0)))

        axis.set_label_text('Note')



    elif ax_type in ['cqt_hz']:

        axis.set_major_formatter(LogHzFormatter())

        axis.set_major_locator(LogLocator(base=2.0))

        axis.set_minor_formatter(LogHzFormatter(major=False))

        axis.set_minor_locator(LogLocator(base=2.0,

                                          subs=2.0**(np.arange(1, 12)/12.0)))

        axis.set_label_text('Hz')



    elif ax_type in ['mel', 'log']:

        axis.set_major_formatter(ScalarFormatter())

        axis.set_major_locator(SymmetricalLogLocator(axis.get_transform()))

        axis.set_label_text('Hz')



    elif ax_type in ['linear', 'hz']:

        axis.set_major_formatter(ScalarFormatter())

        axis.set_label_text('Hz')



    elif ax_type in ['frames']:

        axis.set_label_text('Frames')



    elif ax_type in ['off', 'none', None]:

        axis.set_label_text('')

        axis.set_ticks([])





def __coord_fft_hz(n, sr=22050, **_kwargs):

    n_fft = 2 * (n - 1)

    # The following code centers the FFT bins at their frequencies

    # and clips to the non-negative frequency range [0, nyquist]

    basis = core.fft_frequencies(sr=sr, n_fft=n_fft)

    fmax = basis[-1]

    basis -= 0.5 * (basis[1] - basis[0])

    basis = np.append(np.maximum(0, basis), [fmax])

    return basis





def __coord_mel_hz(n, fmin=0, fmax=11025.0, **_kwargs):

    '''Get the frequencies for Mel bins'''



    if fmin is None:

        fmin = 0

    if fmax is None:

        fmax = 11025.0

    basis = librosa.mel_frequencies(n, fmin=fmin, fmax=fmax)

    basis[1:] -= 0.5 * np.diff(basis)

    basis = np.append(np.maximum(0, basis), [fmax])

    return basis



def cmap(data, robust=True, cmap_seq='magma', cmap_bool='gray_r', cmap_div='coolwarm'):

    data = np.atleast_1d(data)



    if data.dtype == 'bool':

        return get_cmap(cmap_bool)



    data = data[np.isfinite(data)]



    if robust:

        min_p, max_p = 2, 98

    else:

        min_p, max_p = 0, 100



    max_val = np.percentile(data, max_p)

    min_val = np.percentile(data, min_p)



    if min_val >= 0 or max_val <= 0:

        return get_cmap(cmap_seq)



    return get_cmap(cmap_div)



def __coord_n(n, **_kwargs):

    '''Get bare positions'''

    return np.arange(n+1)


D = librosa.amplitude_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(10,10))

specshow(D, x_coords=None, y_coords=None,

             x_axis=None, y_axis="mel",

             sr=sr, hop_length=512,

             fmin=None, fmax=None,

             bins_per_octave=12,

             ax=None

            )