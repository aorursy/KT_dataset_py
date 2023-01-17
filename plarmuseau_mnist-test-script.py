import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))







df_train = pd.read_csv(os.path.join(dirname, 'train.csv'))

train_y = df_train['label'].reset_index()

train_x = df_train.drop(['label'], axis=1)

test_x = pd.read_csv(os.path.join(dirname, 'test.csv'))

test_x=test_x.drop(['id'],axis=1)



print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

total=train_x.append(test_x)

total=total.values.reshape(-1,28,28)

print(total.shape)
import cv2

import numpy as np

from matplotlib import pyplot as plt



# loading image

#img0 = cv2.imread('SanFrancisco.jpg',)

img0 = train_x.iloc[8].values.reshape(28,28)

img = np.array(img0 ,dtype=np.uint8)



# converting to gray scale

gray = img0

# remove noise

#img = cv2.GaussianBlur(gray,(3,3),0)



# convolute with proper kernels

laplacian = cv2.Laplacian(img,cv2.CV_64F)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y



plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')

plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')

plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')

plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')

plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])



plt.show()
from scipy import ndimage

def moments(image):

    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid

    totalImage = np.sum(image) #sum of pixels

    m0 = np.sum(c0*image)/totalImage #mu_x

    m1 = np.sum(c1*image)/totalImage #mu_y

    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)

    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)

    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)

    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively

    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix

    return mu_vector, covariance_matrix



def deskew(image):

    c,v = moments(image)

    alpha = v[0,1]/v[0,0]

    affine = np.array([[1,0],[alpha,1]])

    ocenter = np.array(image.shape)/2.0

    offset = c-np.dot(affine,ocenter)

    img=ndimage.interpolation.affine_transform(image,affine,offset=offset)

    return (img - img.min()) / (img.max() - img.min())*255



trainx=train_x.values.reshape(-1,28,28)

for xi in range(len(train_x)):

    #print(train[xi])

    trainx[xi]= deskew(trainx[xi]) 



from PIL import Image

imgP=Image.fromarray(img, mode='L')

imgP.show()

imgP.thumbnail((14,14), resample=3)

Image.frombuffer('L', (14,14), imgP.tobytes(), 'raw', 'L', 0, 1) #.thumbnail(14, resample=3)

#np.str(imgP.tobytes()).split('\\x')
from PIL import Image

from numpy import *



def pca(X):

  """  Principal Component Analysis

    input: X, matrix with training data stored as flattened arrays in rows

    return: projection matrix (with important dimensions first), variance

    and mean."""



  # get dimensions

  num_data,dim = X.shape



  # center data

  mean_X = X.mean(axis=0)

  X = X - mean_X



  if dim>num_data:

    # PCA - compact trick used

    M = dot(X,X.T) # covariance matrix

    e,EV = linalg.eigh(M) # eigenvalues and eigenvectors

    tmp = dot(X.T,EV).T # this is the compact trick

    V = tmp[::-1] # reverse since last eigenvectors are the ones we want

    S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order

    for i in range(V.shape[1]):

      V[:,i] /= S

  else:

    # PCA - SVD used

    print('svd')

    U,S,V = linalg.svd(X,full_matrices=False)

    #V = V[:num_data] # only makes sense to return the first num_data



  # return the projection matrix, the variance and the mean

  return U,V,S,mean_X



from PIL import Image

from numpy import *

from pylab import *

#import pca



im = train_x.iloc[8].values.reshape(28,28) #array(Image.open(imlist[0])) # open one image to get size

m,n = im.shape[0:2] # get the size of the images

imnbr = 10 #len(imlist) # get the number of images



# create matrix to store all flattened images

immatrix = trainx[:42000].reshape(42000,-1) #SkelTR.reshape(42000,-1) #  #array([array(Image.open(im)).flatten()

           #   for im in imlist],'f')



# perform PCA

U,V,S,immean = pca(immatrix)



# show some images (mean and 7 first modes)

figure()

gray()

subplot(5,4,1)

imshow(immean.reshape(m,n))

for i in range(19):

    subplot(5,4,i+1)

    #print(U[i].shape,V.shape,S.shape,np.dot(U[i],V.T/S).shape)

    imshow(np.dot(U[:,:100][i]*S[:100],V[:100,:]).reshape(m,n) )



show()
from sklearn.decomposition import PCA, FastICA

pca = PCA(n_components=100)

TRANS = pca.inverse_transform(pca.fit_transform(immatrix))

#ica = FastICA(random_state=42)

#TRANS = ica.fit_transform(train_x.values.reshape(-1,28*28))  # Estimate the sources



# show some images (mean and 7 first modes)

figure()

gray()

subplot(5,4,1)



for i in range(19):

    subplot(5,4,i+1)

    #print(U[i].shape,V.shape,S.shape,np.dot(U[i],V.T/S).shape)

    imshow(TRANS[i].reshape(28,28) )



show()
from sklearn.decomposition import PCA, FastICA,NMF

#nmf = NMF()

#TRANS = nmf.inverse_transform(nmf.fit_transform(immatrix[:1000]))

ica = FastICA(n_components=50,random_state=42)

TRANS = ica.inverse_transform(ica.fit_transform(train_x.values.reshape(-1,28*28)) )  # Estimate the sources



# show some images (mean and 7 first modes)

figure()

gray()

subplot(5,4,1)



for i in range(19):

    subplot(5,4,i+1)

    #print(U[i].shape,V.shape,S.shape,np.dot(U[i],V.T/S).shape)

    imshow(TRANS[i].reshape(28,28) )



show()
SVDTR=np.dot(U*S,V)# 
from skimage.morphology import skeletonize

from skimage import data

import matplotlib.pyplot as plt

from skimage.util import invert



# Invert the horse image

image = np.array(train_x.iloc[14].values.reshape(28,28)  ,dtype=np.uint8)>mean(train_x.iloc[14])



# perform skeletonization

skeleton = skeletonize(image)



# display results

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),

                         sharex=True, sharey=True)



ax = axes.ravel()



ax[0].imshow(image, cmap=plt.cm.gray)

ax[0].axis('off')

ax[0].set_title('original', fontsize=20)



ax[1].imshow(skeleton, cmap=plt.cm.gray)

ax[1].axis('off')

ax[1].set_title('skeleton', fontsize=20)



fig.tight_layout()

plt.show()
from skimage.morphology import skeletonize

from tqdm import tqdm_notebook



SkelTR=np.asarray([])

for yi in range(21):

    Lapl=np.asarray([])

    for xi in tqdm_notebook(range(2000)):

        gray = np.array(train_x.iloc[yi*2000+xi].values.reshape(28,28)  ,dtype=np.uint8)>mean(train_x.iloc[yi*2000+xi])

        # remove noise

        #img = cv2.GaussianBlur(gray,(3,3),0)

        img=gray

        # convolute with proper kernels

        skelet = skeletonize(img)

        Lapl=np.append(Lapl,skelet.reshape(-1,1))

    SkelTR=np.append(SkelTR,Lapl)

print(SkelTR.reshape(-1,28,28).shape)
from tqdm import tqdm_notebook

LapTR=np.asarray([])

for yi in range(21):

    Lapl=np.asarray([])

    for xi in tqdm_notebook(range(2000)):

        gray = np.array(train_x.iloc[yi*2000+xi].values.reshape(28,28)  ,dtype=np.uint8)

        # remove noise

        #img = cv2.GaussianBlur(gray,(3,3),0)

        img=gray

        # convolute with proper kernels

        laplacian = cv2.Laplacian(img,cv2.CV_64F)

        Lapl=np.append(Lapl,laplacian.reshape(-1,1))

    LapTR=np.append(LapTR,Lapl)

print(LapTR.reshape(-1,28,28).shape)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

def sigmoid(z):

    return 1 / (1 + np.exp(-z))



class classGS():  

    # fit should give a pandas database X and a pandas y

    # define 'veld' as the labelfield and add the name in the y data

    # define 'index' as an index field and add the name in the y data

    #defining constructor  

    def __init__(self, clf=[KNeighborsClassifier(10)],thres=0.1,probtrigger=False,ncomp=5,neighU=5,ncompU=5,midiU=0.3,veld='label',idvld='index',lentrain=30000):  

        self.clf2=clf

        self.thres=thres

        self.probtrigger=probtrigger

        self.ncomp=ncomp

        self.neighU=neighU

        self.ncompU=ncompU

        self.midiU=midiU

        self.lentrain=lentrain

        self.veld=veld

        self.idvld=idvld

        

    def get_params(self, deep=True):

        # suppose this estimator has parameters "alpha" and "recursive"

        return {"clf":self.clf2,'thres':self.thres,'probtrigger':self.probtrigger,'ncomp':self.ncomp,'neighU':self.neighU,'ncompU':self.ncompU,'midiU':self.midiU,'lentrain':self.lentrain}



    def set_params(self, **parameters):

        for parameter, value in parameters.items():

            setattr(self, parameter, value)

        return self

    

    def fit(self, e__,mtrain_):

        from umap import UMAP

        from sklearn.decomposition import PCA

        from sklearn import preprocessing

        min_max_scaler = preprocessing.MinMaxScaler() #.RobustScaler() #





        #klasseerGS(e_,mtrain,mtest,veld,idvld,thres,probtrigger,ncomp,neighU,ncompU,midiU):

        mtest=pd.DataFrame( e__[self.lentrain:].index,columns=[self.idvld] )

        mtest[self.veld]=mtrain_[self.lentrain:][self.veld]

        label = mtrain_[:self.lentrain][self.veld]

        print('1:',e__.shape,mtrain_.shape,label.shape,self.lentrain,e__[self.lentrain:].shape)        

        e__ = min_max_scaler.fit_transform(e__)

        e__ = sigmoid(e__) 

        

        e__ = PCA(n_components=self.ncomp).fit_transform(e__)

        e__ = UMAP(n_neighbors=self.neighU,n_components=self.ncompU, min_dist=self.midiU,metric='minkowski').fit_transform(e__)

        self.e__=e__

        

        pd.DataFrame(e__[:self.lentrain]).plot(x=0,y=1,c=mtrain_[:self.lentrain][self.veld]+1,kind='scatter',title='classesplot',colormap ='jet')

        pd.DataFrame(e__).plot.scatter(x=0,y=1,c=['r' for x in range(self.lentrain)] +['g' for x in range(len(e__[self.lentrain:]))])   

        print('Model with threshold',self.thres/1000,mtrain_[:self.lentrain].shape,e__.shape,self.ncomp,self.neighU,self.ncompU,self.midiU,)

    

        for clf2 in self.clf2:

            #train

            fit=clf2.fit(e__[:self.lentrain],label)

            print(fit)



            #if clf.__class__.__name__=='DecisionTreeClassifier':

                #treeprint(clf)

            pred=fit.predict(e__)

            #Model.append(self.clf2.__class__.__name__)

            #Accuracy.append(accuracy_score(mtrain[:self.lentrain][self.veld],pred))

            #predict

            print('3:',self.idvld,self.veld,len(mtest))

            self.sub = pd.DataFrame({self.idvld: mtest[self.idvld],self.veld: pred[-len(mtest):]})

            self.sub.plot(x=self.idvld,kind='kde',title=clf2.__class__.__name__ +str(( mtrain_[:self.lentrain][self.veld]==pred[:self.lentrain]).mean()) +'prcnt') 

            sub2=pd.DataFrame(pred,columns=[self.veld])



            #estimate sample if  accuracy

            if self.veld in mtest.columns:

                print( clf2.__class__.__name__ +str(round( accuracy_score(mtrain_[:self.lentrain][self.veld],pred[:self.lentrain]),2)*100 )+'prcnt accuracy versus unknown',(mtrain_[self.lentrain:][self.veld]==pred[self.lentrain:]).mean() )

                from sklearn.metrics import confusion_matrix

                print(confusion_matrix(mtrain_[self.lentrain:][self.veld],pred[self.lentrain:]))

                #write results

            if self.probtrigger:

                pred_prob=fit.predict_proba(e__[-len(mtest):])

                sub=pd.DataFrame(pred_prob)        

        #defining class methods  

            self.f1score=((mtrain_[self.veld]==pred[:len(mtrain_)]).mean())

            self.treshold_=pred



            print(self.sub.shape)

        return self

        

    def _meaning(self, _e1):

        # returns True/False according to fitted classifier

        # notice underscore on the beginning

        print('meaning')

        return( True if _e1 >= self.treshold_ else True )



    def predict(self, e__, mtrain_):

        try:

            getattr(self, "treshold_")

        except AttributeError:

            raise RuntimeError("You must train classifer before predicting data!")

        print('predict',e__.shape,mtrain_.shape)

        return([True for _e1 in e__])



    def score(self, e__, mtrain_):

        # counts number of values bigger than mean

        print('score',self.e__.shape,mtrain_.shape,self.f1score)

        return(self.f1score) 
cGS=classGS(ncomp=30,midiU=0.1,ncompU=7,neighU=5,lentrain=30000)

result=cGS.fit(pd.DataFrame(SVDTR[:40000]),train_y[:40000])
LapTR=LapTR.reshape(-1,784)

cGS=classGS(ncomp=30,midiU=0.1,ncompU=7,neighU=5,lentrain=30000)

result=cGS.fit(pd.DataFrame(train_x[:40000]),train_y[:40000])
cGS=classGS(ncomp=30,midiU=0.1,ncompU=7,neighU=5,lentrain=30000)

result=cGS.fit(pd.DataFrame(TRANS[:40000]),train_y[:40000])
trainx.shape
print(train_x.shape,LapTR.reshape(42000,-1).shape)

mtotal=np.concatenate( (SVDTR, LapTR.reshape(42000,-1),trainx[:42000].reshape(42000,28*28)), axis=1)

mtotal.shape

cGS=classGS(ncomp=30,midiU=0.1,ncompU=7,neighU=5,lentrain=30000)

result=cGS.fit(pd.DataFrame(mtotal[:40000]),train_y[:40000])
from PIL import Image

from numpy import *

from pylab import *



#trainx=pd.DataFrame(trainx.reshape(60000,-1))

subplot(1,3,1)

for ii in range(3):

    subplot(1,3,ii+1)

    print( train_x.iloc[ii].mean(),train_x.iloc[ii].median() ,train_x.iloc[ii][:3])

    marr=train_x.iloc[ii].values.reshape(28,28) #-train_x.iloc[ii].median()

    img2 = Image.fromarray(np.uint8(marr)) #-train_x.iloc[i].median())

    imgres = img2.resize((7,28), Image.ANTIALIAS)

    img3=np.array(imgres.getdata(),np.uint8).reshape(imgres.size[1], imgres.size[0])

    print(img3)

    imshow(img3)

    #img2 = img2.resize( (28, 28), Image.ANTIALIAS)

    #img2.show()



xtrain1=[]

for ii in tqdm_notebook(range(len(train_x))):

    marr=train_x.iloc[ii].values.reshape(28,28) #-train_x.iloc[ii].median()

    img2 = Image.fromarray(np.uint8(marr)) #-train_x.iloc[i].median())

    imgres = img2.resize((7,28), Image.ANTIALIAS)

    #img3=np.array(imgres.getdata(),np.uint8) #.reshape(imgres.size[1], imgres.size[0])

    xtrain1.append(np.array(imgres.getdata(),np.uint8).reshape(-1,1) )

    

xtrain1=np.reshape(xtrain1, (-1, 7*28))



xtrain2=[]

for ii in tqdm_notebook(range(len(train_x))):

    marr=train_x.iloc[ii].values.reshape(28,28) #-train_x.iloc[ii].median()

    img2 = Image.fromarray(np.uint8(marr)) #-train_x.iloc[i].median())

    imgres = img2.resize((28,7), Image.ANTIALIAS)

    #img3=np.array(imgres.getdata(),np.uint8).reshape(imgres.size[1], imgres.size[0])

    xtrain2.append(np.array(imgres.getdata(),np.uint8).reshape(-1,1) )



    

xtrain2=np.reshape(xtrain2, (-1, 7*28))



xtrain3=[]

for ii in tqdm_notebook(range(len(train_x))):

    marr=train_x.iloc[ii].values.reshape(28,28) #-train_x.iloc[ii].median()

    img2 = Image.fromarray(np.uint8(marr)) #)

    imgres = img2.resize((14,14), Image.ANTIALIAS)

    #img3=np.array(imgres.getdata(),np.uint8).reshape(imgres.size[1], imgres.size[0])

    xtrain3.append(np.array(imgres.getdata(),np.uint8).reshape(-1,1) )



    

xtrain3=np.reshape(xtrain3, (-1,14*14))
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

mtotal=np.concatenate( (train_x,xtrain1,xtrain2,xtrain3), axis=1)

mtotal.shape



if True: #for nco in range(5,10):

    cGS=classGS(clf=[KNeighborsClassifier(10)],ncomp=60,midiU=0.1,ncompU=8,neighU=8,lentrain=30000)  #OneVsRestClassifier(SVC(kernel='linear', probability=True)),

    result=cGS.fit(pd.DataFrame(mtotal[:40000]),train_y[:40000])
from scipy.sparse.linalg import svds

from scipy.sparse import coo_matrix

from scipy  import sparse



img=train_x.iloc[1].astype('float32').values.reshape(28,28) #-train_x.iloc[1].median()

img=sparse.csr_matrix(img)



print(min(img.shape))

nco = 3

u, s, v = svds(img, k=nco)

#X = np.dot(u*s,v)

print(u.shape,s.shape,v.shape)

s=s/s.max()

us=np.concatenate((u,s.reshape(1,-1)),0)

vs=np.concatenate((v,s.reshape(-1,1)),1)

print(us.shape,vs.shape)

uv=np.concatenate((us,vs.T),1)

plt.imshow(uv,cmap = 'gray')

plt.show()





trainsv=[]

for xi in tqdm_notebook(range(len(train_x)) ):

    img=train_x.iloc[xi].astype('float32').values.reshape(28,28) #-train_x.iloc[1].median()

    img=sparse.csr_matrix(img)

    u, s, v = svds(img, k=nco)    

    us=np.concatenate((u,s.reshape(1,-1)),0)

    vs=np.concatenate((v,s.reshape(-1,1)),1)



    uv=np.concatenate((us,vs.T),1)

    trainsv.append(uv)



trainsv=np.reshape(trainsv, (-1, 28*nco*2+nco*2)) 

print(trainsv.shape)
if True: #for nco in range(5,10):

    cGS=classGS(clf=[KNeighborsClassifier(10)],ncomp=60,midiU=0.1,ncompU=8,neighU=8,lentrain=30000)  #OneVsRestClassifier(SVC(kernel='linear', probability=True)),

    result=cGS.fit(pd.DataFrame(trainsv[:40000]),train_y[:40000])