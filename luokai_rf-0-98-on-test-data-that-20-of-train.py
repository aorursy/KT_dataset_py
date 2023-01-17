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
#split train data & test data

train = pd.read_csv('../input/train.csv')



sort_train = train.sort_values(['species'])



train_mask = ([True]*8 + [False]*2)*len(train.species.value_counts().index)

test_mask = ([False]*8 + [True]*2)*len(train.species.value_counts().index)



train_data = sort_train[train_mask]



test_data = sort_train[test_mask]
#random forest

from sklearn import ensemble

rf = ensemble.RandomForestClassifier(n_estimators=200)

rf.fit(train_x, train_y)

rf.score(test_x, test_y)
#读取图片，随机保存500个图像边沿坐标值，返回每点坐标值到中心点的欧式距离

def get_edge(imgfile):

    img = mpimg.imread(imgfile)

    cy, cx = ndi.center_of_mass(img)

    contours = measure.find_contours(img, .8)

    contour = max(contours, key=len)

    np.random.shuffle(contour)

    con = contour[:500,:]

    dis = [np.sqrt( (i[1]-cx)**2 + (i[0]-cy)**2 ) for i in con]

    return con, dis



imglist = os.walk('images').__next__()[-1]

imgfile = ['images/' + i for i in imglist]



all_distance = [get_edge(i)[1] for i in imgfile]



distance = np.array(all_distance)



imgid = np.array([i.split('.')[0] for i in imglist])

imgid = imgid.astype('int64')



df = pd.DataFrame(distance, index = imgid)

alldata = df.sort_index()



train = pd.read_csv('train.csv')

train = train.set_index(train.id)

train = train.join(alldata, how='inner')



sort_train = train.sort_values(['species'])



train_mask = ([True]*8 + [False]*2)*len(train.species.value_counts().index)

test_mask = ([False]*8 + [True]*2)*len(train.species.value_counts().index)



train_data = sort_train[train_mask]



test_data = sort_train[test_mask]