import matplotlib.patches as mpatches
from pylab import plt, np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.transform import downscale_local_mean
%matplotlib inline
t = np.linspace(0, 10, 100)
n_samples = 100
sines = []
for _idx in range(n_samples):
    period = np.random.rand() + 0.5
    shift = 3 * np.random.rand()
    w1 = np.sin(t / period - shift)
    sines.append(w1)
rectangles = []
for _idx in range(n_samples):
    period = np.random.rand() + 0.5
    shift = 3 * np.random.rand()
    w2 = np.array([2*float(np.sin(_t / period - shift) > 0) - 1 for _t in t])
    rectangles.append(w2)
data = sines + rectangles
target = [0] * n_samples + [1] * n_samples
for k in range(5):
    plt.plot(t, sines[k], color='blue')
for k in range(5):
    plt.plot(t, rectangles[k], color='orange')
data[0]
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(data, target)
period = np.random.rand() + 0.5
shift = 3 * np.random.rand()
w1 = np.sin(t / period - shift)
clf.predict(w1.reshape(1, -1))
period = np.random.rand() + 0.5
shift = 3 * np.random.rand()
w2 = np.array([2*float(np.sin(_t / period - shift) > 0) - 1 for _t in t])
clf.predict(w2.reshape(1, -1))

