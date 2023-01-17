from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Any results you write to the current directory are saved as output.
import os
sounds = os.listdir("../input/cats_dogs")
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")

from scipy.io import wavfile
import IPython.display as ipd 

#look and some data and hear a sound

w = wavfile.read('../input/cats_dogs/'+sounds[0])
plt.plot(w[1])
plt.show()
print(sounds[0])
ipd.Audio('../input/cats_dogs/'+sounds[0])

print(w)
#ipd.(w)
#seperate cats from dogs
train_test = pd.read_csv('../input/train_test_split.csv', index_col=0)

train_dog = train_test["train_dog"]
train_dog = train_dog.dropna()
train_cat = train_test["train_cat"]
train_cat = train_cat.dropna()

#print(train_cat)

dog_files =[]
cat_files =[]
for dog in train_dog:
    w = wavfile.read('../input/cats_dogs/'+str(dog))
    dog_files.append(w[1])
for cat in train_cat:
    w = wavfile.read('../input/cats_dogs/'+cat)
    cat_files.append(w[1])


#minimum length
dog_l=[]
for dog in dog_files:
    dog_l.append(len(dog))
print(sorted(dog_l))
plt.title("Soundlength of barkings")
plt.hist(dog_l)
plt.show()
#minimum length
cat_l=[]
for cat in cat_files:
    cat_l.append(len(cat))
print(sorted(cat_l))

plt.title("Soundlength of meows")

plt.hist(cat_l)
plt.show()
for dog in dog_files:
    plt.plot(dog,c="blue")
plt.title("All dog sounds overlapped")
plt.show()
for cat in cat_files:
    plt.plot(dog,c="red")
plt.title("All cat sounds overlapped")
plt.show()
cat_loudness=[]
dog_loudness=[]
x_pos = np.arange(2)
for dog in dog_files:
    dog_loudness.append(sum(abs(dog)))
for cat in cat_files:
    cat_loudness.append(sum(abs(cat)))
    
print([sum(dog_loudness),sum(cat_loudness)])
plt.bar(x_pos,[sum(dog_loudness),sum(cat_loudness)])
plt.xticks(x_pos, ["dog","cat"])
plt.show()
print(dog_loudness)
df_dog = pd.DataFrame(dog_files).fillna(int(0))
df_cat = pd.DataFrame(cat_files).fillna(int(0))[:64]
print(df_dog.head()) 
#calculating variance
dog_vars =[]
cat_vars = []
for row in df_dog.as_matrix():
    dog_vars.append(np.var(row))

plt.hist(dog_vars,bins=64)
plt.show()
for row in df_cat.as_matrix():
    cat_vars.append(np.var(row))

plt.hist(cat_vars,bins=64)
plt.show()
#calculating sum
dog_sum =[]
cat_sum = []
for row in df_dog.as_matrix():
    dog_sum.append(sum(row))

plt.hist(dog_sum,bins=64)
plt.show()
for row in df_cat.as_matrix():
    cat_sum.append(sum(row))

plt.hist(cat_sum,bins=64)
plt.show()
pca = PCA(n_components=2)
pca_2d_dog = pca.fit_transform(df_dog)
pca_df_dog = pd.DataFrame(pca_2d_dog,columns=["pc1","pc2"])

pca = PCA(n_components=2)
pca_2d_cat = pca.fit_transform(df_cat)
pca_df_cat = pd.DataFrame(pca_2d_cat,columns=["pc1","pc2"])
print("finished pca")

plt.scatter(pca_df_dog["pc1"],pca_df_dog["pc2"],c="blue")
plt.scatter(pca_df_cat["pc1"],pca_df_cat["pc2"],c="red")

pca_df_dog["variance"]=dog_vars
pca_df_cat["variance"]=cat_vars
pca_df_dog["sum"]=dog_sum
pca_df_cat["sum"]=cat_sum

pca_df_dog["loudness"]=dog_loudness
pca_df_cat["loudness"]= cat_loudness[:64]
pca_df_dog["label"]=np.zeros(len(pca_df_dog))
pca_df_cat["label"]=np.ones(len(pca_df_cat))

print(len(pca_df_dog))
print(len(pca_df_cat))
pca_df_cat=pca_df_cat
df = pd.concat([pca_df_dog,pca_df_cat])

print(df.head())
print(df.tail())

#little help function to measure accuracy
def predict_acc(X_test,y_test,clf):
    predictions=[]       
    for x in X_test.as_matrix(): 
    
        predictions.append(float(clf.predict([x])))   
    
    acc = accuracy_score(y_test, predictions)
    return [acc,predictions]
X = df[["pc1","pc2","variance","sum","loudness"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12,stratify=y)
# using Random Forest CLassification
classifier =ExtraTreesClassifier()

clf = classifier
clf = clf.fit(X_train, y_train)
acc = predict_acc(X_test,y_test,clf)[0]
print("Accuracy of Model is "+str(acc))
fi = clf.feature_importances_
plt.bar(range(len(fi)),fi)
#plt.xticks(range(len(fi)),["pc1","pc2","variance","sum","loudness"])

#bagging
accs=[]
n=100
for i in range(n):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12,stratify=y)
    # using Random Forest CLassification
    classifier =ExtraTreesClassifier()

    clf = classifier
    clf = clf.fit(X_train, y_train)
    acc = predict_acc(X_test,y_test,clf)[0]
    accs.append(acc)
print("Mean accuracy of "+str(n)+" sample splits is "+str(np.mean(accs)))
from scipy.io.wavfile import write
row = np.array(df_dog.loc[0]).astype(int)
print(row)
test =write('test.wav',rate = 10000 , data = row)
w = wavfile.read("test.wav")
plt.plot(w[1])
ipd.Audio("test.wav")
sr = 22050 # sample rate
ipd.Audio(w[1], rate=sr) 
#now the mean sound of all dogs
soundlen = len(row)
soundstack_dog = np.zeros(soundlen)
for i in range(len(df_dog)):
    soundstack_dog +=df_dog.loc[i]
soundstack_dog /=len(df_dog)

test =write('test.wav',rate = 10000 , data = soundstack_dog)
w = wavfile.read("test.wav")
plt.plot(w[1])
ipd.Audio("test.wav")
sr = 22050 # sample rate
ipd.Audio(w[1], rate=sr)
    
#now the mean sound of all cats

row = np.array(df_cat.loc[0]).astype(int)
soundlen = len(row)
soundstack_cat = np.zeros(soundlen)
for i in range(len(df_cat)):
    soundstack_cat+=df_cat.loc[i]
soundstack_cat/=len(df_cat)

test =write('test.wav',rate = 10000 , data = soundstack_cat)
w = wavfile.read("test.wav")
plt.plot(w[1])
ipd.Audio("test.wav")
sr = 22050 # sample rate
ipd.Audio(w[1], rate=sr)