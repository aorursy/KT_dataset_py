import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer 
df = pd.read_json("../input/train.json")
testset = pd.read_json("../input/test.json")
df.head()
testset.head()
df.isnull().sum()
testset.isnull().sum()
df.cuisine.unique()
df.ingredients = df.ingredients.astype('str')
testset.ingredients = testset.ingredients.astype('str')
df.ingredients[0]
testset.ingredients[0]
df.ingredients = df.ingredients.str.replace("["," ")
df.ingredients = df.ingredients.str.replace("]"," ")
df.ingredients = df.ingredients.str.replace("'"," ")
df.ingredients = df.ingredients.str.replace(","," ")
testset.ingredients = testset.ingredients.str.replace("["," ")
testset.ingredients = testset.ingredients.str.replace("]"," ")
testset.ingredients = testset.ingredients.str.replace("'"," ")
testset.ingredients = testset.ingredients.str.replace(","," ")
df.ingredients[0]
testset.ingredients[0]
df.ingredients = df.ingredients.str.lower()
testset.ingredients = testset.ingredients.str.lower()
df.ingredients = df.ingredients.apply(lambda x: word_tokenize(x))
testset.ingredients = testset.ingredients.apply(lambda x: word_tokenize(x))
lemmatizer = WordNetLemmatizer()
def lemmat(wor):
    l = []
    for i in wor:
        l.append(lemmatizer.lemmatize(i))
    return l
df.ingredients = df.ingredients.apply(lemmat)
testset.ingredients = testset.ingredients.apply(lemmat)
df.ingredients[0]
testset.ingredients[0]
type(df.ingredients[0])
df.ingredients = df.ingredients.astype('str')
df.ingredients = df.ingredients.str.replace("["," ")
df.ingredients = df.ingredients.str.replace("]"," ")
df.ingredients = df.ingredients.str.replace("'"," ")
df.ingredients = df.ingredients.str.replace(","," ")
testset.ingredients = testset.ingredients.astype('str')
testset.ingredients = testset.ingredients.str.replace("["," ")
testset.ingredients = testset.ingredients.str.replace("]"," ")
testset.ingredients = testset.ingredients.str.replace("'"," ")
testset.ingredients = testset.ingredients.str.replace(","," ")
type(df.ingredients[0])
df.ingredients[0]
#vect = HashingVectorizer ()
vect = TfidfVectorizer()
features = vect.fit_transform(df.ingredients)
features
#vect.get_feature_names()
testfeatures = vect.transform(testset.ingredients)
testfeatures
encoder = LabelEncoder()
labels = encoder.fit_transform(df.cuisine)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#logreg = LogisticRegression(C=10,solver='lbfgs', multi_class='multinomial',max_iter=400)
#logreg.fit(X_train,y_train)
#print("Logistic Regression accuracy",logreg.score(X_test, y_test))
#logreg.predict(X_test)
#from sklearn import linear_model
#sgd = linear_model.SGDClassifier()
#sgd.fit(X_train, y_train)
#print("SGD classifier accuracy",sgd.score(X_test, y_test))
#from sklearn.svm import LinearSVC
#linearsvm = LinearSVC(C=1.0,random_state=0,multi_class='crammer_singer',dual = False, max_iter = 1500)
#linearsvm.fit(X_train, y_train)
#print("Linear SVM accuracy", linearsvm.score(X_test, y_test))
#labelsNN = df.cuisine
#labelsNN = pd.get_dummies(labelsNN)
#labelsNN = labelsNN.values
#labelsNN[0]
#from scipy.sparse import csr_matrix
#sparse_dataset = csr_matrix(features)
#featuresNN = sparse_dataset.todense()
#featuresNN[0]
#X_trainNN, X_testNN, y_trainNN, y_testNN = train_test_split(featuresNN, labelsNN, test_size=0.2)
#print(X_trainNN.shape, X_testNN.shape, y_trainNN.shape, y_testNN.shape)
#numfeat = X_trainNN.shape[1]
#import keras
#from keras.layers import *
#model = keras.models.Sequential()
#model.add(Dense(300,input_dim = numfeat,activation = 'relu'))
#model.add(Dense(500,activation = 'relu'))
#model.add(Dense(400,activation = 'relu'))
#model.add(Dense(20,activation='softmax'))
#model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['categorical_accuracy'])
#model.fit(X_trainNN,y_trainNN,epochs=50,shuffle=True, verbose =2,batch_size=500)
#print("Accuracy with KERAS" ,model.evaluate(X_testNN,y_testNN)[1])
#linearsvmfinal = LinearSVC(C=1.0,random_state=0,multi_class='crammer_singer',dual = False, max_iter = 1500)
#linearsvmfinal.fit(features,labels)
import lightgbm as lgb
gbm = lgb.LGBMClassifier(objective="mutliclass",n_estimators=10000,num_leaves=512)
gbm.fit(X_train,y_train,verbose = 300)
pred = gbm.predict(testfeatures)
#pred = linearsvmfinal.predict(testfeatures)
predconv = encoder.inverse_transform(pred)
sub = pd.DataFrame({'id':testset.id,'cuisine':predconv})
output = sub[['id','cuisine']]
output.to_csv("outputfile.csv",index = False)
