import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split



%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

# take all the rows, take all columns starting with 1

images = train_df.iloc[:,1:]

# take all the rows, take all columns from 0 to 1, exclusive

labels = train_df.iloc[:,:1]



train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=123)
train_images.head()
train_labels.head()
images = train_images.as_matrix()

imgs = [i.reshape(28, 28) for i in images]



def show(ids):

    fig, ax = plt.subplots(1, len(ids), figsize=(10,5))

    for i in ids:

        ax[i].imshow(imgs[i], cmap='gray')

        ax[i].set_title('Label: ' + str(train_labels.iloc[i, 0]))
show([0,1,2,3,4])
import warnings

warnings.filterwarnings('ignore')



import xgboost as xgb





xg_train = xgb.DMatrix(train_images, label=train_labels.values)

xg_test = xgb.DMatrix(test_images, label=test_labels.values)



params = {'eta':0.1, 

          'nround':10, 

          'nthread':4, 

          'num_class':10, 

          'max_depth':6}



num_round = 5

watchlist = [(xg_train,'train'), (xg_test, 'test')]



bst = xgb.train(params, xg_train, num_round, watchlist)
bst.predict(xg_test).astype(int)
test_images = pd.read_csv('../input/test.csv')
result = bst.predict(xgb.DMatrix(test_images))

result.astype(int)
np.savetxt('predictions.csv', np.dstack((np.arange(1, result.size+1), result))[0], delimiter=',', header='ImageId,Label', fmt='%d', comments='')
predictions = pd.read_csv('./predictions.csv', index_col='ImageId')

predictions.head()