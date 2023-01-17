# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load train set

trn = pd.read_csv("../input/train.csv")



# Split labels and images

trn_img = (trn.ix[:,1:].values).astype('float32')

trn_lbl = trn.ix[:,0].values.astype('int32')



# Load test set

tst_img = (pd.read_csv("../input/test.csv").values).astype('float32')



# Show a few labels to pick a subset

trn_lbl[1:100]
# Take subset and reshape

vis_img = trn_img[30006:30014,:]

vis_lbl = trn_lbl[30006:30014]

vis_img = vis_img.reshape(vis_img.shape[0], 28, 28)

vis_img.shape
# Visualize greyscale numbers

plt.subplots(1, 8, figsize=(12,2))

for i in range(0, 8):

    plt.subplot(180 + (i+1))

    plt.imshow(vis_img[i], cmap=plt.get_cmap('gray'))

    plt.title(vis_lbl[i]);

    plt.axis('off')
# Threshold will be from 64 brightness to 192 in steps of 16

plt.subplots(9, 9, figsize=(13, 14))

for k in range(0, 9):

	threshold = k * 16 + 64

	baw_img = vis_img.__ge__(threshold)

	plt.subplot(9,9,k*9+1)

	plt.axis('off')

	plt.title(threshold)

	for i in range(0, 8):

		plt.subplot(9,9,k*9+i+2)

		plt.imshow(baw_img[i], cmap=plt.get_cmap('gray'))

		plt.axis('off')

		if k == 0:

			plt.title(vis_lbl[i])
# Picking 112 as threshold

#threshold = 112

#csv_trn_lbl = trn_lbl

#csv_trn_img = trn_img.__ge__(threshold)

#csv_tst_img = tst_img.__ge__(threshold)



#np.savetxt('csv_trn_lbl.csv', csv_trn_lbl, fmt='%1d', delimiter=',')

#np.savetxt('csv_trn_img.csv', csv_trn_img, fmt='%1d', delimiter=',')

#np.savetxt('csv_tst_img.csv', csv_tst_img, fmt='%1d', delimiter=',')
# Picking 176 as threshold (skipping test set this time)

#threshold = 176

#csv_trn_lbl = trn_lbl

#csv_trn_img = trn_img.__ge__(threshold)

#csv_tst_img = tst_img.__ge__(threshold)



#np.savetxt('csv_trn_lbl_176.csv', csv_trn_lbl, fmt='%1d', delimiter=',')

#np.savetxt('csv_trn_img_176.csv', csv_trn_img, fmt='%1d', delimiter=',')

#np.savetxt('csv_tst_img_176.csv', csv_tst_img, fmt='%1d', delimiter=',')