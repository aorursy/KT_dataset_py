import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib
%matplotlib inline
matplotlib.rcParams.update({'font.size': 12})
img_list = pd.read_csv('../input/driver_imgs_list.csv')

img_list['class_type'] = img_list['classname'].str.extract('(\d)',expand=False).astype(np.float)
pyplot.figure()
img_list.hist('class_type',alpha=0.5,layout=(1,1),bins=9)
pyplot.title('class distribution')
pyplot.draw()
img_list['subject_type'] = img_list['subject'].str.extract('(\d\d\d)',expand=False).astype(np.int32)
n_unique_sub=len(img_list['subject_type'].unique())
pyplot.figure()
img_list.hist('subject_type',layout=(1,1),bins=n_unique_sub)
pyplot.title('subject_type, # unique subject {}'.format(n_unique_sub))
