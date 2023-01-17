import numpy as np

import random
import joblib



X = joblib.load('../input/pixelartdata/dataset/pixelart-48x48.dump')['X']
X = np.moveaxis(X, -1, 1)



joblib.dump({'X': X}, 'pixelart-48x48.dump')

del X
!python ../input/ganspytorch/ct-gan/ct_gan.py --images_filename pixelart-48x48.dump --sample_interval 25 --gamma 0.00003 --lambda_1 2 --lambda_2 1 --n_epochs 5