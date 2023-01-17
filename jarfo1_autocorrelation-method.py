%pylab inline
from scipy.io import wavfile

from scipy.signal import correlate

rcParams['figure.figsize'] = (16, 6)
# Read and plot a speech file from the FDA_UE database

name = "sb034.wav"

filename="../input/fda_ue/" + name

sfreq, data = wavfile.read(filename)

plot(data)

xlim(0, len(data))

title(name)
# Pick a short segment

windowlength = 32 # 32ms

ns_windowlength = int(round((windowlength * sfreq) / 1000))

pos = 31000

frame_length = ns_windowlength



frame = data[pos:pos+frame_length]



plot(frame, linewidth=1.0)

xlim(0, frame_length)

title("A frame")
frame = frame.astype(np.float)

frame -= frame.mean()

amax = np.abs(frame).max()

frame /= amax

xlim(-(frame_length-1), frame_length-1)

bcorr = correlate(frame, frame)

plot(range(-(frame_length-1), frame_length), bcorr, linewidth=1.0)

title("The complete autocorrelation")
# keep the positive indexes of the autocorrelation

corr = bcorr[len(bcorr)//2:]

xlim(0, frame_length-1)

plot(corr)

title("The autocorrelation for nonnegative indexes")
# Find the first minimum

dcorr = np.diff(corr)

xlim(0, len(dcorr))

plot(dcorr)

title("The first difference of the autocorrelation (corr[n+1] - corr[n])")
# Find the first minimum

rmin = np.where(dcorr > 0)[0]

if len(rmin) > 0:

    rmin1 = rmin[0]

plot(range(rmin1,len(corr)), corr[rmin1:])

title("Autocorrelation without the initial values up to the first minimum")
# Find the next peak

peak = np.argmax(corr[rmin1:]) + rmin1

rmax = corr[peak]/corr[0]

f0 = sfreq / peak
print("corr[peak]/corr[0] = {:.1f}".format(rmax))

print("Pitch frequency = {:.1f} Hz".format(f0))

print("Pitch period {:.1f} ms ({:d} samples)".format((1/f0)*1000, peak))