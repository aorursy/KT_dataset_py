# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load data

train_df = pd.read_csv('../input/train.csv')
from sklearn.model_selection import train_test_split

number_samples = 5000



train_x_df = train_df.iloc[0:number_samples,1:]

train_y_df = train_df.iloc[0:number_samples,:1]



train_x_df_in_training, test_x_df_in_training, train_y_df_in_training, test_y_df_in_training = train_test_split(train_x_df, train_y_df, train_size=0.8, random_state=0)
import matplotlib.pyplot as plt, matplotlib.image as mpimg

%matplotlib inline



i=1

img=train_x_df_in_training.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_y_df_in_training.iloc[i,0])
plt.hist(train_x_df_in_training.iloc[i])
from sklearn.svm import SVC

clf = SVC(decision_function_shape='ovo')

clf.fit(train_x_df_in_training, train_y_df_in_training)

clf.score(test_x_df_in_training,test_y_df_in_training)
def binarize(image_data):

    image_data[image_data > 0] = 1

    return image_data
binarized_train_x_df_in_training = binarize(train_x_df_in_training)

binarized_test_x_df_in_training = binarize(test_x_df_in_training)





i=1

img=binarized_train_x_df_in_training.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_y_df_in_training.iloc[i,0])

plt.show()



plt.hist(binarized_train_x_df_in_training.iloc[i])

plt.show()
clf = SVC(decision_function_shape='ovo')

clf.fit(binarized_train_x_df_in_training, train_y_df_in_training)

clf.score(binarized_test_x_df_in_training,test_y_df_in_training)
test_df = pd.read_csv('../input/test.csv')
test_x_df = test_df

binarized_test_x_df = binarize(test_x_df)
test_predicted = clf.predict(test_x_df)

test_predicted
result_df = pd.DataFrame({'Label': test_predicted})

result_df.index+=1

result_df = result_df.reindex(result_df.index.rename('ImageId'))

result_df
result_df.to_csv('results.csv', header=True)