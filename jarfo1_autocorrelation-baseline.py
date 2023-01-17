# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from types import SimpleNamespace

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.io import wavfile

from scipy.signal import correlate



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/test"))



# Any results you write to the current directory are saved as output.
def autocorr_method(frame, sfreq, threshold=0.46, fmin=50, fmax=400):

    """Estimate pitch using autocorrelation

    """



    # Calculate autocorrelation using scipy correlate

    frame = frame.astype(np.float)

    frame -= frame.mean()

    amax = np.abs(frame).max()

    if amax > 0:

        frame /= amax

    else:

        return 0



    corr = correlate(frame, frame)

    # keep the positive part

    corr = corr[len(corr)//2:]



    # Find the first minimum

    dcorr = np.diff(corr)

    rmin = np.where(dcorr > 0)[0]

    if len(rmin) > 0:

        rmin1 = rmin[0]

    else:

        return 0



    # Find the next peak

    peak = np.argmax(corr[rmin1:]) + rmin1

    rmax = corr[peak]/corr[0]

    f0 = sfreq / peak



    if rmax > threshold and f0 >= fmin and f0 <= fmax:

        return f0

    else:

        return 0
class Counters:

    def __init__(self, gross_threshold=0.2):

        self.num_voiced = 0

        self.num_unvoiced = 0

        self.num_voiced_unvoiced = 0

        self.num_unvoiced_voiced = 0

        self.num_voiced_voiced = 0

        self.num_gross_errors = 0

        self.fine_error = 0

        self.e2 = 0

        self.gross_threshold = gross_threshold

        self.nfiles = 0



    def add(self, other):

        if other is not None:

            self.num_voiced += other.num_voiced

            self.num_unvoiced += other.num_unvoiced

            self.num_voiced_unvoiced += other.num_voiced_unvoiced

            self.num_unvoiced_voiced += other.num_unvoiced_voiced

            self.num_voiced_voiced += other.num_voiced_voiced

            self.num_gross_errors += other.num_gross_errors

            self.fine_error += other.fine_error

            self.e2 += other.e2

            self.nfiles += 1



    def __repr__(self):

        nframes = self.num_voiced + self.num_unvoiced

        if self.nfiles > 0:

            self.fine_error /= self.nfiles

        str = [

            f"Num. frames:\t{self.num_unvoiced + self.num_voiced} = {self.num_unvoiced} unvoiced + {self.num_voiced} voiced",

            f"Unvoiced frames as voiced:\t{self.num_unvoiced_voiced}/{self.num_unvoiced} ({100*self.num_unvoiced_voiced/self.num_unvoiced:.2f}%)",

            f"Voiced frames as unvoiced:\t{self.num_voiced_unvoiced}/{self.num_voiced} ({100*self.num_voiced_unvoiced/self.num_voiced:.2f}%)",

            f"Gross voiced errors (>{100*self.gross_threshold}%):\t{self.num_gross_errors}/{self.num_voiced_voiced} ({100*self.num_gross_errors/self.num_voiced_voiced:.2f}%)",

            f"MSE of fine errors:\t{100*self.fine_error:.2f}%",

            f"RMSE:\t{np.sqrt(self.e2/nframes):.2f}"

        ]

        return '\n'.join(str)
def compare(fref, pitch):

    vref = np.loadtxt(fref)

    vtest = np.array(pitch)



    diff_frames = len(vref) - len(vtest)

    if abs(diff_frames) > 5:

        print(f"Error: number of frames in ref ({len(vref)}) != number of frames in test ({len(vtest)})")

        return None

    elif diff_frames > 0:

        vref = np.resize(vref, vtest.shape)

    elif diff_frames < 0:

        vtest = np.resize(vtest, vref.shape)



    counters = Counters()

    counters.num_voiced = np.count_nonzero(vref)

    counters.num_unvoiced = len(vref) - counters.num_voiced

    counters.num_unvoiced_voiced = np.count_nonzero(np.logical_and(vref == 0, vtest != 0))

    counters.num_voiced_unvoiced = np.count_nonzero(np.logical_and(vref != 0, vtest == 0))



    voiced_voiced = np.logical_and(vref != 0, vtest != 0)

    counters.num_voiced_voiced = np.count_nonzero(voiced_voiced)



    f = np.absolute(vref[voiced_voiced] - vtest[voiced_voiced])/vref[voiced_voiced]

    gross_errors = f > counters.gross_threshold

    counters.num_gross_errors = np.count_nonzero(gross_errors)

    fine_errors = np.logical_not(gross_errors)

    counters.fine_error = np.sqrt(np.square(f[fine_errors]).mean()) if np.count_nonzero(fine_errors) > 0 else 0

    counters.e2 = np.square(vref - vtest).sum()



    return counters
def wav2f0(options, gui):

    fs = open(options.submission, 'w') if options.submission is not None else None

    totalCounters = Counters()

    with open(gui) as f:

        if fs is not None:

            print('id,frequency', file=fs)

        for line in f:

            line = line.strip()

            if len(line) == 0:

                continue

            filename = os.path.join(options.datadir, line + ".wav")

            f0ref_filename = os.path.join(options.datadir, line + ".f0ref")

            print("Processing:", filename)

            sfreq, data = wavfile.read(filename)

            nsamples = len(data)



            # From miliseconds to samples

            ns_windowlength = int(round((options.windowlength * sfreq) / 1000))

            ns_frameshift = int(round((options.frameshift * sfreq) / 1000))

            ns_left_padding = int(round((options.left_padding * sfreq) / 1000))

            ns_right_padding = int(round((options.right_padding * sfreq) / 1000))

            pitch = []

            for ini in range(-ns_left_padding, nsamples - ns_windowlength + ns_right_padding + 1, ns_frameshift):

                first_sample = max(0, ini)

                last_sample = min(nsamples, ini + ns_windowlength)

                frame = data[first_sample:last_sample]

                f0 = autocorr_method(frame, sfreq)

                pitch.append(f0)

                

            if fs is not None:

                for frame_id, f0 in enumerate(pitch):

                    print(line + '_' + str(frame_id) + ',', f0, file=fs)



            if os.path.isfile(f0ref_filename):

                counters = compare(f0ref_filename, pitch)

                totalCounters.add(counters)



    if totalCounters.num_voiced + totalCounters.num_unvoiced > 0:

        print("### Summary")

        print(totalCounters)

        print("-------------------------------\n")
fda_ue_options = SimpleNamespace(

    windowlength=32, frameshift=15, left_padding=16, right_padding=16, datadir='../input', submission=None)

wav2f0(fda_ue_options, '../input/fda_ue.gui')
ptdb_tug_options = SimpleNamespace(

    windowlength=32, frameshift=10, left_padding=0, right_padding=0, datadir='../input', submission=None)

#wav2f0(ptdb_tug_options, '../input/ptdb_tug.gui')
test_options = SimpleNamespace(

    windowlength=26.5, frameshift=10, left_padding=13.25, right_padding=7, datadir='../input/test', submission='submission.csv')

wav2f0(test_options, '../input/test.gui')