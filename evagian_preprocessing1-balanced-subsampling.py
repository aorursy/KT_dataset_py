# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def balanced_subsample(x,y,subsample_size=1.0):



    class_xs = []

    min_elems = None



    for yi in np.unique(y):

        elems = x[(y == yi)]

        class_xs.append((yi, elems))

        if min_elems == None or elems.shape[0] < min_elems:

            min_elems = elems.shape[0]



    use_elems = min_elems

    if subsample_size < 1:

        use_elems = int(min_elems*subsample_size)



    xs = []

    ys = []



    for ci,this_xs in class_xs:

        if len(this_xs) > use_elems:

            this_xs.reindex(np.random.permutation(this_xs.index))



        x_ = this_xs[:use_elems]

        y_ = np.empty(use_elems)

        y_.fill(ci)



        xs.append(x_)

        ys.append(y_)



    xs = pd.concat(xs)

    ys = pd.Series(data=np.concatenate(ys), name='sirna')



    return xs,ys
data = pd.read_csv("/kaggle/input/recursion-cellular-image-classification/train.csv")

# Preview the first 5 lines of the loaded data

print(data.head())

print(data.shape)
# Split train set

xstrain,ystrain = balanced_subsample(data.drop('sirna', axis=1),data['sirna'],subsample_size=1/3)



balanced_sample_train = pd.concat([xstrain,ystrain], axis=1)

print(balanced_sample_train.head())

print(balanced_sample_train.shape)

balanced_sample_train = balanced_sample_train.dropna()



balanced_sample_train = balanced_sample_train.astype({"plate": int, "sirna": int})



balanced_sample_train.to_csv('/kaggle/input/recursion-cellular-image-classification/input/balanced_sample_train.csv', index=False)
# Split test set

common = data.merge(balanced_sample_train,on=['id_code','sirna'])

print(common)

test_data = data[(~data.id_code.isin(common.id_code))&(~data.sirna.isin(common.sirna))]



print(test_data.head())

print(test_data.shape)



xstest,ystest = balanced_subsample(test_data.drop('sirna', axis=1),test_data['sirna'],subsample_size=1/6)



balanced_sample_test = pd.concat([xstest,ystest], axis=1)

print(balanced_sample_test.head())

print(balanced_sample_test.shape)



balanced_sample_test = balanced_sample_test.dropna()

balanced_sample_test = balanced_sample_test.astype({"plate": int, "sirna": int})



balanced_sample_test.to_csv('/kaggle/input/recursion-cellular-image-classification/input/balanced_sample_test.csv', index=False)