import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np

import matplotlib.pyplot as plt, matplotlib.image as mpimg

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

from random import choice

import seaborn as sns

%matplotlib inline

sns.set(color_codes=True)
labeled_images = pd.read_csv('../input/train.csv')

working_size = 5000

images = labeled_images.iloc[:working_size,1:]

labels = labeled_images.iloc[:working_size,:1]

seed = 21

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.7, random_state=seed)
i=2

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
def mean_split(data, std_weight):

    dt = data.copy()

    midVal = dt[dt > 0].mean()

    std_width = dt[dt > 0].std() * std_weight

    lowerVal = midVal - std_width

    higherVal = midVal + std_width

    dt[(dt < lowerVal) & (dt > 0)] = 1

    dt[((dt < higherVal) & (dt >= lowerVal))] = 2

    dt[dt >= higherVal] = 3

    return dt

std_weight = 0.1

bt_tst_imgs = mean_split(test_images,std_weight)

bt_trn_imgs = mean_split(train_images,std_weight)



img=bt_trn_imgs.iloc[i].as_matrix().reshape((28,28))

plt.imshow(img, cmap='viridis')

plt.title(train_labels.iloc[i])
plt.hist(bt_trn_imgs.iloc[i])
clf_bucket = svm.SVC()

clf_bucket.fit(bt_trn_imgs, train_labels.values.ravel())

score = clf_bucket.score(bt_tst_imgs,test_labels)

round(score,3)
def grid_search(gridVals):

    best_weight = [0.0]

    best_score = [0.0]

    for n in gridVals:

        bt_tst_imgs = mean_split(test_images,n)

        bt_trn_imgs = mean_split(train_images,n)

        clf_bucket = svm.SVC(random_state=seed)

        clf_bucket.fit(bt_trn_imgs, train_labels.values.ravel())

        score = clf_bucket.score(bt_tst_imgs,test_labels)

        if(score > best_score[0]):

            best_score = [score]

            best_weight = [n]

        elif(score == best_score[0]):

            best_score.append(score)

            best_weight.append(n)

    return best_weight,best_score



grid_vals = np.arange(0.0,1.025,0.025)

bw,bs = grid_search(grid_vals)

print()

print('winning std weight: ' + str(bw))

print('got score: ' + str(bs))
best_w = []

best_s = [bs[0]]

for w in bw:

    minutiae_grid = np.arange(w - 0.024, w + 0.026,0.001)

    mbw,mbs = grid_search(minutiae_grid)

    print()

    print('winning std weight: ' + str(mbw))

    print('got score: ' + str(mbs))

    print()

    if(max(mbs) > best_s[0]):

        best_w = mbw

        best_s = mbs

    elif(max(mbs) == best_s[0]):

        best_w.extend(mbw)

        best_s.extend(mbs)



print()

print('after minutiae grid search weights, scores: ')

print(best_w)

print(best_s)



final_weight = choice(best_w)

final_grid_score = best_s[0]

print('final weight: ' + str(final_weight))

print('final score: ' + str(final_grid_score))
final_clf = svm.SVC(random_state=seed)

final_test_images = mean_split(test_images,final_weight)

final_train_images = mean_split(train_images,final_weight)

final_clf.fit(final_train_images,train_labels.values.ravel())

final_score = final_clf.score(final_test_images,test_labels)

round(final_score,3)
test_data=pd.read_csv('../input/test.csv')

test_data[test_data>0]=1

results=clf_bucket.predict(test_data[:working_size])
results
df = pd.DataFrame(results)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv', header=True)