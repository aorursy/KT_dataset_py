# Import necessory libraries (Feel free to ask if you don't know 'em)
%matplotlib notebook
import os, sys
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score,classification_report
import json
# Load The Word Vectors (GloVE)
# Have some patience to let them load :-P.
vectors_path = '../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt'
glove = dict()
with open(vectors_path,'r') as f:
    for l in f:
        if l.strip == '':
            continue
        l = l.strip().split()
        w,v = l[0], np.asarray([float(i) for i in l[1:]])
        glove[w] = v
Favourites = ['Men','Women','King','Queen','Uncle','Aunt','boy','girl']
# Favourites = ['Men','Women','King','Queen'] 
# Favourites = ['America','Potato','Men','Mexican','Canadian','Indian','Women','Kangaroo','Dog','Carrot','Raddish','cat','mouse','King','Queen'] # Add the words you love
Favourites = [i.lower() for i in Favourites]
Vectors = [glove[w] if w in glove else np.zeros_like(glove['a']) for w in Favourites]
for w,v in zip(Favourites[:2],Vectors[:2]):
    print(w,v)
# Visualize these vectors on your own in 2d
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(np.asarray(Vectors))

z, y = vectors_2d[:,0], vectors_2d[:,1]
labels = Favourites
fig, ax = plt.subplots()
ax.scatter(z, y)
for i, txt in enumerate(labels):
    ax.annotate(txt, (z[i], y[i]))
data_path = '../input/polarity/examples.json'
with open(data_path, 'r') as data_file:
    all_data  = json.loads(data_file.read()) 
    
# spliting the data
splitPer = 0.8
trainingData = all_data[:int(len(all_data)*splitPer)]
testData     = all_data[int(len(all_data)*splitPer):]

print(trainingData[4])
def sent2vec(sent):
    sentvec = np.zeros(glove['is'].shape)
    sentlen = len(sent.split())
    for w in sent.split():
        w = w.strip().lower()
#         print(w,)
        try:
            sentvec += glove[w]
        except KeyError as e:
            sentlen -=1
#         print()
    return sentvec/sentlen
def process_data(Data):
    X = [sent2vec(raw['text']) for raw in Data]
    y = [raw['label'] for raw in Data]
    return (X,y)
# Train an SVM
X,y = process_data(trainingData)
clf = svm.SVC(verbose=True)
clf.fit(X, y)
X,y = process_data(testData)
predicted_y = clf.predict(X)
print('Test Accuracy is :',accuracy_score(y,predicted_y))
print(classification_report(y, predicted_y))
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30, 10, 5), random_state=1)
X,y = process_data(trainingData)
clf.fit(X,y)
X,y = process_data(testData)
predicted_y = clf.predict(X)
print('Test Accuracy is :',accuracy_score(y,predicted_y))
print(classification_report(y, predicted_y))
list_of_sample = ['Fantastic movie, great action','Awesome cast and story','X-men sucks','Fuck this shit','Total time waste']
sampleX = np.asarray([sent2vec(text) for text in list_of_sample])
sampley_predicted = clf.predict(sampleX)


for text,label in zip(list_of_sample,sampley_predicted):
    print(text+":"+str(label))
