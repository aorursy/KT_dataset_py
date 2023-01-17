# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import plotly.plotly as py

import plotly.tools as tls

import cv2

from sklearn.feature_extraction.text import CountVectorizer

from operator import itemgetter





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import os

#os.remove('2017-05-4-extd_ftrs_amazon_02.csv')
from subprocess import check_output

print(check_output(["ls"]).decode("utf8"))
df = pd.read_csv('../input/train_v2.csv')

#print(np.array(df)[:10])

print(df.head())
labels = np.array(df['tags'])



vect = CountVectorizer()

vect.fit(labels)

vect.get_feature_names()



labels_dtm = vect.transform(labels)

df_labels = pd.DataFrame(labels_dtm.toarray(), columns = vect.get_feature_names())

#print(df_labels.head())

#print(labels_dtm[:10])



#print(len(df_labels))

#df_labels['image_name'] = df['image_name']

#print(df_labels.head())

#print(df_labels.columns.values)



# create a dict to collect total values of each class of label

amazon_condition = {}

for col in df_labels.columns.values:

#    print(col)

    

    z = df_labels.groupby([col])[col].count().astype(int)

#    print(type(z))

#    print(col, [np.array(k.astype(int)) for l, k in enumerate(z)])

#    print([z])

    amazon_condition[col] = 0

    for i, j in enumerate(z):

        if i != 0:

#            print ('j =', j) 

            amazon_condition[col] += j

#print(amazon_condition)



amazon_condition_labels = [x for x in amazon_condition.keys()]

amazon_condition_values = [x for x in amazon_condition.values()]



print(amazon_condition_labels)

print(amazon_condition_values)





fig, ax = plt.subplots()



N = len(amazon_condition_labels)

ind = np.arange(N)    # the x locations for the groups

width = 0.5       # the width of the bars



p1 = ax.bar(ind, amazon_condition_values, width, color='r')

ax.set_ylabel('Count', size=20)

ax.set_xlabel('Amazon Condition', size=20)

ax.set_title('Count by label type', size = 25)

ax.set_xticks(ind + width/2.)

ax.set_xticklabels(amazon_condition_labels)



#plotly_fig = tls.mpl_to_plotly( fig )



# For Legend

#plotly_fig["layout"]["showlegend"] = True

#plotly_fig["data"][0]["name"] = "Count"

#plotly_fig["data"][1]["name"] = "Test"



for tick in ax.get_xmajorticklabels():

    tick.set_rotation(45)

    tick.set_horizontalalignment("right")

    tick.set_fontsize(15)



    

for tick in ax.get_ymajorticklabels():

#    tick.set_rotation(45)

    tick.set_horizontalalignment("right")

    tick.set_fontsize(15)

    

plt.show()
# a much better way of creating bar plot using pandas

df_labels[amazon_condition_labels].sum().sort_values().plot.bar()
counts = len(df_labels.columns)

heatmapdata = np.zeros([counts]*2)





#for i in np.array(df_labels)[:10]:

#    print(i[0])



for i in np.array(df_labels):

    for col, n in enumerate(df_labels.columns.values):

        if i[col] > 0:

            for row, m in enumerate(df_labels.columns.values):

                if i[row] != 0: 

                    heatmapdata[row, col] += 1





#for m, n in enumerate(df_labels.columns.values):

#    for i in np.array(df_labels):

#        if i[m] > 0:

#            for j, k in enumerate(df_labels.columns.values):

#                for l in np.array(df_labels):

#                    if l[m] > 0:

#                        heatmapdata[i,j] += l[j]
#print(heatmapdata.astype(int))

#print(heatmapdata.shape)

heatmapdatanew = []

for i in heatmapdata:

    heatmapdatanew.append(i/max(i).astype(float))



#print(np.array(heatmapdatanew).astype(float))    
N = len(amazon_condition_labels)

ind = np.arange(N)    

width = 0.5       



fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(np.array(heatmapdatanew), cmap='magma', interpolation='nearest')

ax.set_xticks(ind + width/2.)

ax.set_yticks(ind + width/2.)

ax.set_xticklabels(amazon_condition_labels)

ax.set_yticklabels(amazon_condition_labels)



for tick in ax.get_xmajorticklabels():

    tick.set_rotation(45)

    tick.set_horizontalalignment("right")

    tick.set_fontsize(10)



for tick in ax.get_ymajorticklabels():

    tick.set_verticalalignment("center")

    tick.set_fontsize(10)

    

plt.show()
# heatmap using seaborn!



sns.heatmap(heatmapdata)
def heatmap(labellist):

    heatmatrix = df_labels[labellist].T.dot(df_labels[labellist])

    sns.heatmap(heatmatrix)

    return heatmatrix

    

# Compute the heatmap matrix

heatmap(amazon_condition_labels)
import cv2

import matplotlib.pyplot as plt

#img = cv2.imread('../input/train-jpg/train_6.jpg')



#cv2.imshow('img', img)



fig, axes = plt.subplots(2,2, figsize = (32,32))

fig.subplots_adjust(hspace = 0.1, wspace = 0.1)



#axes[0,0].imshow(img, cmap = 'binary')

#print(axes.shape)



# Plot the impages starting from i = 1

for i, ax in enumerate(axes.flat):

    a = i

#    im = np.reshape(X_test[a], (32,32))

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(df['image_name'][a]))

#    print(df['image_name'][a])

    ax.imshow(img, cmap = 'binary')

    ax.text(0.95, 0.05, 'Labels={0}'.format(df['tags'][a]), ha='right', transform = ax.transAxes, color = 'yellow', size=16)

#    ax.set_xticks([])

#    ax.set_yticks([])
#f, ax = plt.subplots(figsize=(7, 6))

#print(X_test_pd.shape)



#plt.title('Correlation plot of a 100 columns in the MNIST dataset')

# Draw the heatmap using seaborn

#sns.heatmap(X_test_pd.ix[:,0:200].astype(float).corr(),linewidths=0, square=True, cmap="viridis", xticklabels=False, yticklabels= False, annot=True)
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import fbeta_score

from PIL import Image

from PIL import ImageStat

import glob



def extract_features(path):

    features = []

    image_stat = ImageStat.Stat(Image.open(path))

    features += image_stat.sum

    features += image_stat.mean

    features += image_stat.rms

    features += image_stat.var

    features += image_stat.stddev

    img = cv2.imread(path)

    cv2img = cv2.imread(path,0)

    features += list(cv2.calcHist([cv2img],[0],None,[256],[0,256]).flatten())

    mean, std = cv2.meanStdDev(img)

    features += list(mean)

    features += list(std)

    return features
X_train = pd.DataFrame()

input_path = '../input/'

df['path'] = df['image_name'].map(lambda x: input_path + 'train-jpg/' + x + '.jpg')



f_list = []



for i in df['path']:

    f = np.array(extract_features(i)).astype(int)

    f_list.append(f)
#print(list(f))

#z = [' '.join([str(x)]) for x in range(len(f))]

#print(z)

#X_train = pd.DataFrame.from_items(f)  
f_list_arr = np.array(f_list)

X_train = pd.DataFrame(f_list_arr)

print(type(X_train))

print(X_train.head())
print(type(X_train))

print(type(df_labels))
#X_train.to_csv('2017-05-4-extd_ftrs_amazon_02.csv', index=False)
from subprocess import check_output

print(check_output(["ls"]).decode("utf8"))
etr = ExtraTreesRegressor(n_estimators=75, max_depth=35, n_jobs=-1, random_state=1)

etr.fit(X_train, df_labels); 

print('fit process done')
print(df_labels.head())

print(X_train.head())
train_pred = etr.predict(X_train)

train_pred [train_pred >0.24] = 1

train_pred [train_pred < 1] = 0

print('step1')

print(train_pred.shape, type(train_pred))

z = np.array(df_labels)

print(z.shape, type(z))
print(z[5])

print(train_pred[5].astype(int))

q = train_pred.astype(int)

print(type(z), type(q), z.shape, q.shape)
test_images = glob.glob(input_path + 'test-jpg-v2/*')

print(test_images[:5])

a = test_images[0].split('/')[3].replace('.jpg','')

b = test_images[0].split('/')

print(a, b)



X_test = pd.DataFrame([[x.split('/')[3].replace('.jpg',''),x] for x in test_images])

print(X_test[:5])



X_test.columns = ['image_name','path']

print(X_test[:5])

print(X_test.shape, type(X_test))
ftr_list = []

for i in X_test['path']:

    ftr = np.array(extract_features(i)).astype(int)

    ftr_list.append(ftr)



ftr_list_arr = np.array(ftr_list)

test_pred = pd.DataFrame(ftr_list_arr)

print(type(test_pred))

print(test_pred.head())

    

result = etr.predict(test_pred); 

tags = []
result [result >0.24] = 1

result [result < 1] = 0

print(result[:5])
#print(type(result))

result_df = pd.DataFrame(result)

result_df.columns = df_labels.columns.values

print(result_df.head())

tags = []



for i,j in enumerate(np.array(result_df)):

    temp_tags = []

    #print(temp_tags)

    for c, col in enumerate(result_df.columns.values):

        if j[c] == 1:

            temp_tags.append(col)

    tags.append(temp_tags)



tags1 = []    



for x in tags:

    st = ''    

    for y in x:

        st += y + ' '

    tags1.append(st[:(len(st)-1)])         
print(len(tags), type(tags))

print(tags[:10])

print(len(tags1), type(tags1))

print(tags1[:10])

X_test['tags'] = tags1



X_test[:10]

print(X_test.columns.values)
X_test[['image_name','tags']].to_csv('2017-06-12-submission_amazon_02.csv', index=False)