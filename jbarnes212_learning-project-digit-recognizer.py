import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline

# Input data files are available in the "../input/" directory.
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,1:]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size = 0.8, random_state=0)



# Any results you write to the current directory are saved as output.
