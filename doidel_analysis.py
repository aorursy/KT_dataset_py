import pandas as pd
import numpy as np

base_path = '../input/sdobenchmark_full/'
train = pd.read_csv(base_path + 'training/meta_data.csv', sep=",", parse_dates=["start","end"], index_col="id")
test = pd.read_csv(base_path + 'test/meta_data.csv', sep=",", parse_dates=["start","end"], index_col="id")
print(f'We have {len(train)} training and {len(test)} test samples')
# optimal fixed point prediction, solved by optimization
# Always predict the same small flare ("B5")
predict_val = 5.29411764705883E-07

# evaluate
print('Mean absolute errors:')
print(f'train: {np.mean(np.abs(train.peak_flux-predict_val))}')
print(f'test:  {np.mean(np.abs(test.peak_flux-predict_val))}')
# statistics.py
import numpy as np
import math
from sklearn.metrics import confusion_matrix

goes_classes = ['quiet','A','B','C','M','X']

def class_to_flux(c: str):
    'Inverse of flux_to_class \
    Maps a flare class (e.g. B6, M, X9) to a GOES flux value'
    if c == 'quiet':
        return 1e-9
    decade = goes_classes.index(c[0])-9
    sub = float(c[1:]) if len(c) > 1 else 1
    return round(10 ** decade * sub, 10)

def flux_to_class(f: float, only_main=False):
    'maps the peak_flux of a flare to one of the following descriptors: \
    *quiet* = 1e-9, *B* >= 1e-7, *C* >= 1e-6, *M* >= 1e-5, and *X* >= 1e-4\
    See also: https://en.wikipedia.org/wiki/Solar_flare#Classification'
    decade = int(min(math.floor(math.log10(f)), -4))
    sub = round(10 ** -decade * f)
    if decade < -4: # avoiding class 10
        decade += sub // 10
        sub = max(sub % 10, 1)
    main_class = goes_classes[decade + 9] if decade >= -8 else 'quiet'
    sub_class = str(sub) if main_class != 'quiet' and only_main != True else ''
    return main_class + sub_class

def true_skill_statistic(y_true, y_pred, threshold='M'):
    'Calculates the True Skill Statistic (TSS) on binarized predictions\
    It is not sensitive to the balance of the samples\
    This statistic is often used in weather forecasts (including solar weather)\
    1 = perfect prediction, 0 = random prediction, -1 = worst prediction'
    separator = class_to_flux(threshold)
    y_true = [1 if yt >= separator else 0 for yt in y_true]
    y_pred = [1 if yp >= separator else 0 for yp in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) - fp / (fp + tn)


y_pred = np.repeat(predict_val, len(train.peak_flux))
print(f'Train TSS of fixed point baseline: {true_skill_statistic(train.peak_flux, y_pred)}')

y_pred = np.repeat(predict_val, len(test.peak_flux))
print(f'Test TSS of fixed point baseline: {true_skill_statistic(test.peak_flux, y_pred)}')
train["noaa_num"] = [t[0] for t in train.index.str.split('_')]
noaa_nums = np.unique(train["noaa_num"])
classes_samples = [flux_to_class(pf, only_main=True) for pf in train["peak_flux"]]
max_classes_ARs = [flux_to_class(max(train[train["noaa_num"] == noaa_num]["peak_flux"]), only_main=True) for noaa_num in noaa_nums]
classes, counts_samples = np.unique(classes_samples, return_counts=True)
classes, counts_ARs = np.unique(max_classes_ARs, return_counts=True)
pd.DataFrame(data={"samples":counts_samples, "unique ARs":counts_ARs}, index=classes)
import os
from skimage import io
import statistics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

base_path = '../input/sdobenchmark_full/training/'
train = pd.read_csv(base_path + 'meta_data.csv', sep=",", parse_dates=["start","end"], index_col="id")

# just picking the quiet regions might reduce the noise a bit
train_degrad = train.loc[train['peak_flux'] == 1e-9]
train_degrad = train_degrad.sort_values(by=['start'])
dict_degrad = {'date': [], '94': [], '131': [], '171': [], '193': [], '211': [], '304': [], '335': [], '1700': [], 'continuum': [], 'magnetogram': []}
wavelengths = list(dict_degrad.keys())
wavelengths = wavelengths[1:]
for row in train_degrad.iterrows():
    pixel_vals = {}
    ar_nr, p = row[0].split("_", 1)
    path = os.path.join(base_path, ar_nr, p)
    if os.path.isdir(path):
        for img in [name for name in os.listdir(path) if name.endswith('.jpg')]:
            img_wavelength = os.path.splitext(img)[0].split("__")[1]
            im = io.imread(os.path.join(path, img))
            mid = im.shape[0] // 2
            im = im[mid - 20:mid + 20,mid - 20:mid + 20]
            if img_wavelength not in pixel_vals:
                pixel_vals[img_wavelength] = []
            pixel_vals[img_wavelength].append(np.mean(im))
        for wl in wavelengths:
            if wl not in pixel_vals:
                if len(dict_degrad[wl]) > 0:
                    dict_degrad[wl].append(dict_degrad[wl][-1])
                else:
                    dict_degrad[wl].append(0)
            else:
                dict_degrad[wl].append(sum(pixel_vals[wl]) / float(len(pixel_vals[wl])))
        dict_degrad['date'].append(row[1]['start'])

display_dict = dict(dict_degrad)
del display_dict['date']
df_degrad = pd.DataFrame(data=display_dict, index=dict_degrad['date'])
df_degrad = df_degrad.sort_index(ascending=True)
print('This is the (very noisy) data:')
degrad_plot = df_degrad.plot()
degrad_plot.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()
print('And by averaging aggressively you can see some trends in it:')
degrad_plot = df_degrad.rolling(window=200).mean().plot()
degrad_plot.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()
