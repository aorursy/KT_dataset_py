#Libraries - many are not used in final code

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import json
import gc


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Dataset Preparation
print("Welcome to Salrite's Cookin'")
print ("Read Dataset ... ")
def read_dataset(path):
    return json.load(open(path)) 

train = read_dataset('../input/train.json') #json
test = read_dataset('../input/test.json')


# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
    text_data = [" ".join(doc['ingredients']).lower() for doc in data]
    return text_data 

train_text = generate_text(train)
test_text = generate_text(test)

target = [doc['cuisine'] for doc in train]


# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)

def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 

X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")
X = X.astype('float32')
X_test = X_test.astype('float32')


# Label Encoding - Target 
print ("Label Encoding the Target Variable for tSNE... ")
lb = LabelEncoder()  #lb.inverse_transform()
y = pd.DataFrame(np.array(target),columns=['Cuisine'])  
y['Label'] = lb.fit_transform(y['Cuisine'])
gc.collect()
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
%matplotlib inline

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 100


# Utility function to visualize the outputs of  PCA & t-SNE
def v_scatter(x, c,name,label):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(c[label]))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(15, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[c['Label'].astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.
        xtext, ytext = np.median(x[c[label] == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(c[c[label] == i][name].unique()), fontsize=10)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

        
#PCA
from sklearn.decomposition import PCA

def makePCA(X):
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(X.toarray())
    pca_df = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])

    pca_df['pca1'] = pca_result[:,0]
    pca_df['pca2'] = pca_result[:,1]
    pca_df['pca3'] = pca_result[:,2]
    pca_df['pca4'] = pca_result[:,3]
    
    return pca_df 

pca_df = makePCA(X)
print('PCA...')
#print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

top_two_comp = pca_df[['pca1','pca2']] # taking first and second principal component

v_scatter(top_two_comp.values,y,'Cuisine','Label')
gc.collect()
from sklearn.manifold import TSNE
print('tSNE...')
v_tsne = TSNE(random_state=RS).fit_transform(X.toarray() )
v_scatter(v_tsne, y,'Cuisine','Label')
gc.collect()
print ("Label Encoding the Target Variable for Training... ")
lb = LabelEncoder()
yt = lb.fit_transform(target)

#train-val split and pca for validation set
X_train, X_val, y_train, y_val = train_test_split(X, yt, test_size=0.20, random_state=40, shuffle=True)

#Model definition
etc = ExtraTreesClassifier(n_estimators=1000,warm_start =True)
etc.fit(X_train, y_train)
y_pred = etc.predict(X_train)
print("\nTraining Accuracy ",accuracy_score(y_train, y_pred))
y_pred = etc.predict(X_val)
print("\nValidation Accuracy ",accuracy_score(y_val, y_pred))
model = etc #current model to be used for prediction 
# Prediction
print ("Predicting on test data ... ")
y_test = model.predict(X_test)
y_pred = lb.inverse_transform(y_test)

# Submission
print ("Generating Submission File! ")
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('output.csv', index=False)
