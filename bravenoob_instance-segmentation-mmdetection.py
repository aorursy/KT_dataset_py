import os

import sys
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

#from catboost import CatBoostClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder, StandardScaler,KBinsDiscretizer

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



# Evaluation

from sklearn.metrics import cohen_kappa_score,make_scorer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
%reload_ext autoreload

%autoreload 2

%matplotlib inline
!pip install mmcv terminal tables tabulate
!git clone https://github.com/open-mmlab/mmdetection.git

os.chdir('mmdetection')

#!git checkout v1.0rc0

!pip install -v -e .

os.chdir('..')
sys.path.append('mmdetection') # To find local version

from mmdet.apis import init_detector, inference_detector, show_result

import mmcv
!pip install wget

import wget

#url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_64x4d_fpn_2x_20181218-c22bdd70.pth'

url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'

wget.download(url, 'checkpoint_file.pth')
config_file = 'mmdetection/configs/rpn_x101_64x4d_fpn_1x.py'

config_file = 'mmdetection/configs/faster_rcnn_r50_fpn_1x.py'



checkpoint_file = 'checkpoint_file.pth'



# build the model from a config file and a checkpoint file

model = init_detector(config_file, checkpoint_file, device='cuda:0')



# test a list of images and write the results to image files

imgs = ['../input/petfinder-adoption-prediction/train_images/de993d8ad-3.jpg']

for i, result in enumerate(inference_detector(model, imgs)):

    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i), show=False)
%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img=mpimg.imread('../working/result_0.jpg')

imgplot = plt.imshow(img)

plt.show()
import shutil

shutil.rmtree('mmdetection')