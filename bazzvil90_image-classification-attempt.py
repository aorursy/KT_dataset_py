import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from matplotlib import pyplot as plt

import glob

import random 

import itertools #group in dict

from subprocess import check_output

from sklearn.cluster import KMeans

import math  #ceil

from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline

print(check_output(["ls", "../input/train"]).decode("utf8"))
select = 2

#random.seed(10)



files_imgs = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:select]

print("Lenght of train {}".format(len(files_imgs)))



imgs = {}

for i, f in enumerate(files_imgs) :

    imgs[f] = cv2.imread(f)

    

color = ('b','g','r')



#function to calculate image channel distribution

def show_hist(img,ax,x,y):

 for ch, col in enumerate(color):

   histo = cv2.calcHist([img],[ch],None,[256],[0,256])

   ax[x,y].plot(histo,color=col)

    

print(files_imgs)



# plotting original images with their channel distribution

fig,ax = plt.subplots(len(imgs),2, figsize=(10,6))

for i, f in enumerate(files_imgs) :

  ax[i,0].set_ylabel(f.split('/')[3])

  ax[i,0].imshow(imgs[f])

  show_hist(imgs[f],ax,i,1)
#calculate image histogram

histo_grams = {}

#

def cal_hist(f) :

    for ch, col in enumerate(color):

      img = imgs[f]

      v = cv2.calcHist([img],[ch],None,[256],[0,256])

      v = v.flatten()

      hist = v/sum(v)

      histo_grams[f] = hist

    

#

for f in files_imgs :

    cal_hist(f)
#matrix features of each image - size is images * hist 

histo_matrix = np.zeros((len(files_imgs), len(histo_grams[files_imgs[0]])))

#

for i,ifile in enumerate(files_imgs):

        histo_matrix[i,:] = histo_grams[ifile]
len(histo_grams[files_imgs[0]])

str(histo_grams)
#histo_grams['../input/train/ALB/img_07262.jpg']

#nbr_occurences = np.sum( (histo_grams['../input/train/ALB/img_07262.jpg'] > 0) * 1, axis = 0)

#nbr_occurences

#histo_grams.values()

#kmeans = KMeans(n_clusters = 8, random_state = 0).fit(histo_grams.values())



#img = imgs['../input/train/ALB/img_07262.jpg']

#f = '../input/train/ALB/img_07262.jpg'

#img = imgs[f]

#img = cv2.imread(f)

#v = cv2.calcHist([img],[0],None,[256],[0,256])

#v

#v = v.flatten()

#v

#hist = v/sum(v)

#histo_grams[f] = hist
#compute distances between histograms and store them in a dictionary

#compare methode is cv2.cv.CV_COMP_INTERSECT = 2

dist_matrix = np.zeros((len(files_imgs),len(files_imgs)))

for i , ifile in enumerate(files_imgs):

    for j, jfile in enumerate(files_imgs):

        if i <= j :

            c = cv2.compareHist(histo_grams[ifile], histo_grams[jfile],2)

            dist_matrix[i,j] = c

            dist_matrix[j,i] = c

#



#display distance histograms

plt.hist(dist_matrix.flatten(), bins=50)

plt.title('distance matrix histogram')
#clustering image with kmeans

#number of clusters

nclusters = 2



#algorithm terminaison criteria = (type, max_iter, epsilon )

criteria = (cv2.TERM_CRITERIA_EPS, 100, 0.01)



#number of times the algorithm is executed

nattemps= 10



#ret: compactness sum of squared distance from each point to their centers

#labels: label array each element is marked

#centers: centers of clusters

ret,labels,centers = cv2.kmeans(np.float32(histo_matrix), nclusters,None, criteria, nattemps, cv2.KMEANS_RANDOM_CENTERS)



#displaying labels matrix

labels
#transform n dimensional array in list

label_list = []

for i,l in enumerate(labels) :

    label_list.append(l[0])

print(label_list)
#mapping image with it s label

#map_lab = zip(label_list,files_imgs)
map_lab = [ (lab,f) for (lab, f) in (zip(label_list, files_imgs))]

map_lab
#get unique keys

#t = {}

#ukeys = [ k for k,v in map_lab if k not in t ]

#ukeys
#number of keys by label

acounter = dict()

for lab in label_list :

    acounter.setdefault(lab, 0)

    acounter[lab] += 1

    

acounter  
def get_l(lab) :

    return([v for k,v in map_lab if k == lab] )
for i in range(nclusters):

    print(i)

    l=get_l(i)

    print(l)
def show_imgs(flist) :

        if (len(flist) >= 8) :

            tmp_files = [flist[np.random.choice(len(flist))] for _ in range(8)]

            fig,ax = plt.subplots(2,4, figsize=(10,6))

            for i, f in enumerate(tmp_files) : 

                ax[i // 4, i % 4].imshow(imgs[f])

                ax[i // 4, i % 4].set_ylabel(f.split('/')[3])

        elif (len(flist) >= 4) :

            tmp_files = [flist[np.random.choice(len(flist))] for _ in range(4)]

            fig,ax = plt.subplots(1,4, figsize=(10,6))

            for i, f in enumerate(tmp_files) : 

                ax[i].imshow(imgs[f])

                ax[i].set_ylabel(f.split('/')[3])

        elif (len(flist) >= 2) :

            tmp_files = [flist[np.random.choice(len(flist))] for _ in range(2)]

            fig,ax = plt.subplots(1,2, figsize=(10,6))

            for i, f in enumerate(tmp_files) : 

                ax[i].imshow(imgs[f])

                ax[i].set_ylabel(f.split('/')[3])

        else :

            fig,ax = plt.subplots(1,1, figsize=(10,3))

            ax.imshow(imgs[flist[0]])   

            ax.set_ylabel(flist[0].split('/')[3])

            
for i in range(nclusters):

    #print(i)

    show_imgs(get_l(i))
#get short file name

def get_short(f):

    t = f.split('/')

    return(t[len(t) - 1])



def get_group(f) :

    t = f.split('/')

    return(t[len(t) - 2])



#calculate image features  

def cal_feat(f) :

    img = imgs[f]

    v = get_short(f) + ','

    for ch, col in enumerate(color):

        h = cv2.calcHist([img],[ch],None,[256], [0,256])

        m,s = cv2.meanStdDev(h)

        if (ch == (len(color) - 1) ):

            v = v + str(int(m[0][0])) + ',' + str(int(s[0][0])) + ',' + str(int(h.min())) + ',' + str(int(h.max())) + ',' + get_group(f)

        else :

            v = v + str(int(m[0][0])) + ',' + str(int(s[0][0])) + ',' + str(int(h.min())) + ',' + str(int(h.max())) + ','

    return(v.split(','))

        

#calculate features for all images to a list

w = list()

for f in files_imgs :

    w.append(cal_feat(f))

 

#create a dataframe for image features as training data

df = pd.DataFrame.from_records(w, columns=['img','mean1','stdev1','min1', 'max1','mean2','stdev2','min2', 'max2','mean3','stdev3','min3', 'max3','group'])



#add label columns as target data

df['lab'] = df['group']

df['lab'].replace(['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'],[0,1,2,3,4,5,6,7], inplace=True)

df

df['lab'].unique()
knn = KNeighborsClassifier(n_neighbors = 1)
X = df.drop(['img','lab','group'], axis=1)

#target/response vector 

Y = df['lab']

df
knn.fit(X,Y)
pred = [[3600,	4223,	6,	35211,	3600,	4410,	9,	42053,	3600,	3530,	122,	25519],

[3600,	4620,	4,	21343,	3600,	4917,	5,	26403,	3600,	4961,	31,	28796]]



pred

#knn.predict([[3600,5624,41,23558,3600,2947,6,20909,3600,3927,55,15350]])
prob = knn.predict_proba(pred)
prob

#submission

#image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT

#img_00005.jpg,0.45500264690312336,0.05293806246691371,0.03096876654314452,0.017734250926416093,0.12308099523557438,0.07914240338803599,0.046585494970884066,0.1942826892535733



#l = [[3600,5624,41,23558,3600,2947,6,20909,3600,3927,55,15350]]

#type(l)

#m = knn.predict_proba(l)

#m

prob = prob.tolist()

prob


#l = []

#l = [[0.375, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.125],[1,2,3,4,5,6,7,8]]

#v = [['img1'],['img3']]

#k = [ i+j for i,j in zip(v,l)]

#','.join(map(str,k[0]))

#','.join(map(str,k[1]))
y = []

for i, j in enumerate (df['img']):

    y.append([j])

y
w = [ i+j for i,j in zip(y, prob)]

w
','.join(map(str,w[0]))
','.join(map(str,w[1]))
len(y)
r=''

for i, j in enumerate(y):

    r = r + ','.join(map(str,w[i]))+'\n'

    

o = print(r)

o
def ral() :

    r=''

    for i, j in enumerate(y):

        r = r + ','.join(map(str,w[i])) +'\n'

    return(r)

ral()

q = ral()

print(q)