import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# It's been created a new folder (train-n-test-data) with duplicates of train and test data because some minor issues with the competition folder
import sklearn 
import os
print(os.listdir("../input/train-n-test-data"))
#The train data
train_data = pd.read_csv("../input/train-n-test-data/train_data.csv")
train_data.drop('Id',axis = 'columns')
#As the data Id collumn just refers to a specific email, we need this information since the email will be classified in terms of the features and the target(ham) columns
train_data.ham
# The features to be classified 
test_features = pd.read_csv("../input/train-n-test-data/test_features.csv")
test_features_with_id = test_features
test_features = test_features.drop('Id',axis = 'columns')
test_features

train_data.shape


train_data['ham'].value_counts()
train_data
#for special characters let's create the excerpt from the dataset containing the Id, ham and char_freq
special_char = train_data[['char_freq_;','char_freq_(', 'char_freq_[','char_freq_!','char_freq_$','char_freq_#','ham','Id']]
special_char
test_special_char = test_features_with_id[['char_freq_;','char_freq_(', 'char_freq_[','char_freq_!','char_freq_$','char_freq_#','Id']]


word_freq = train_data.loc[:,"word_freq_make":"word_freq_conference"]
test_word_freq = test_features_with_id.loc[:,"word_freq_make":"word_freq_conference"]
#Adding the ham and the id collumn
word_freq.loc[:,'ham'] = train_data[['ham']]
word_freq.loc[:,'Id'] = train_data[['Id']]
test_word_freq.loc[:,'Id'] = test_features_with_id[['Id']]
word_freq.shape
test_word_freq


word_freq
capital_run = train_data[['capital_run_length_average','capital_run_length_longest','capital_run_length_total','ham','Id']]
test_capital_run_with_id = test_features_with_id[['capital_run_length_average','capital_run_length_longest','capital_run_length_total', 'Id']]
test_capital_run = test_capital_run_with_id.drop('Id', axis = 'columns')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
X_train_data= train_data
X_train_data = X_train_data.drop('ham',axis = 'columns')
X_train_data= X_train_data.drop('Id',axis = 'columns')
Y_train_data= train_data.loc[:,'ham':'ham']
#setting a binary value for ham
Y_train_data = Y_train_data.astype(int)
# to reset for boolean, just try ~ Y_train_data = Y_train_data.astype(bool)
Y_train_data.ham
X_train_data
Scores =[]
for i in range (1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, X_train_data, Y_train_data.ham, cv=10)
    Scores.append(scores.mean())
Scores


#But scores is not a pd.DataFrame, so:
Scores_df = pd.DataFrame()
Scores_df['Knn Scores']= Scores
Scores_df




Scores_df.plot()

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train_data, Y_train_data.ham)
Y_test_Pred = knn.predict(test_features)

Y_test_Pred = Y_test_Pred.astype(bool)
Predict = pd.DataFrame()
Predict['Id'] = test_features_with_id['Id']
Predict['ham'] = Y_test_Pred
Predict.set_index('Id', inplace=True)
Predict
#Predict.drop('', axis = 'columns')
Predict.to_csv("Prediction_Knn.csv")
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB

#For Gaussian Naive Bayes, the classification(clf1) is 

clf1 = GaussianNB()
clf1.fit(X_train_data, Y_train_data.ham)
GaussianNB(priors=None, var_smoothing=1e-09)
Y_Pred_G_NB = clf1.predict(test_features)
Y_Pred_G_NB = Y_Pred_G_NB.astype(bool)
Y_Pred_G_NB


Pred_G_NB = pd.DataFrame()
Pred_G_NB['Id'] = test_features_with_id['Id']
Pred_G_NB['ham'] = Y_Pred_G_NB
Pred_G_NB.set_index('Id', inplace=True)
Pred_G_NB
Pred_G_NB.to_csv("Prediction_GaussianNB.csv")
#For the Multinomial Naive Bayes Classifier (clf2)
clf2 = MultinomialNB()
clf2.fit(X_train_data, Y_train_data.ham)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
Y_Pred_M_NB = clf2.predict(test_features)
Y_Pred_M_NB = Y_Pred_M_NB.astype(bool)
Y_Pred_M_NB

Pred_M_NB = pd.DataFrame()
Pred_M_NB['Id'] = test_features_with_id['Id']
Pred_M_NB['ham'] = Y_Pred_M_NB
Pred_M_NB.set_index('Id', inplace=True)
Pred_M_NB
Pred_M_NB.to_csv("Prediction_MultinomialNB.csv")
#For the ComplementNB, the clf3 (now when can make a little more compact)
  
clf3 = ComplementNB()
clf3.fit(X_train_data, Y_train_data.ham)
ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
Y_Pred_C_NB = clf3.predict(test_features)
Y_Pred_C_NB = Y_Pred_C_NB.astype(bool)
Pred_C_NB = pd.DataFrame()
Pred_C_NB['Id'] = test_features_with_id['Id']
Pred_C_NB['ham'] = Y_Pred_C_NB
Pred_C_NB.set_index('Id', inplace=True)
Pred_C_NB.to_csv("Prediction_ComplementNB.csv")
Pred_C_NB
#And finally, the bernoulli NB classifier :

clf4 = BernoulliNB()
clf4.fit(X_train_data, Y_train_data.ham)
BernoulliNB(alpha=0.5, binarize=0.0, fit_prior=True)
Y_Pred_B_NB = clf4.predict(test_features)
Y_Pred_B_NB = Y_Pred_B_NB.astype(bool)
Pred_B_NB = pd.DataFrame()
Pred_B_NB['Id'] = test_features_with_id['Id']
Pred_B_NB['ham'] = Y_Pred_B_NB
Pred_B_NB.set_index('Id', inplace=True)
Pred_B_NB.to_csv("Prediction_BernoulliNB.csv")
Pred_B_NB
def KNN(n,train_data,x_test,doc_name):
    x_train_data = train_data.drop('ham',axis = 'columns')
    x_train_data= x_train_data.drop('Id',axis = 'columns')
    y_train_data= train_data.loc[:,'ham':'ham']
    x_test_with_id = x_test
    x_test = x_test.drop('Id',axis = 'columns')
    #setting a binary value for ham
    y_train_data = y_train_data.astype(int)
    # to reset for boolean, just try ~ Y_train_data = Y_train_data.astype(bool)
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(x_train_data, y_train_data.ham)
    y_test_Pred = knn.predict(x_test)
    y_test_Pred = y_test_Pred.astype(bool)
    Predict = pd.DataFrame()
    Predict['Id'] = x_test_with_id['Id']
    Predict['ham'] = y_test_Pred
    Predict.set_index('Id', inplace=True)
    Predict.to_csv(doc_name)

    


#now using the function ...

KNN(1,word_freq,test_word_freq,"KNN_word_freq.csv")
#for special characters...
KNN(1,special_char,test_special_char,"KNN_special_char.csv")

#Now, summarizing the four types of naive bayes used above
def Gaussian_NB(train_dt,test_feat,name,priors,smooth):
    train_dt = train_dt.drop('Id', axis = 'columns')
    test_feat = test_feat.drop('Id', axis = 'columns')
    y_train_dt = train_dt.ham
    x_train_dt = train_dt.drop('ham', axis ='columns')
    clf = GaussianNB()
    clf.fit(x_train_dt,y_train_dt)
    GaussianNB(priors=priors, var_smoothing=smooth) #None, 1e-09
    Y_Pred_GNB = clf.predict(test_feat)
    Y_Pred_GNB = Y_Pred_GNB.astype(bool)
    Y_Pred_GNB
    Pred_GNB = pd.DataFrame()
    Pred_GNB['Id'] = test_features_with_id['Id']
    Pred_GNB['ham'] = Y_Pred_GNB
    Pred_GNB.set_index('Id', inplace=True)
    print(Pred_GNB)
    Pred_GNB.to_csv(name)
def  Multinomial_NB(train_dt,test_feat,name,smooth,class_prior, fit_prior):
    train_dt = train_dt.drop('Id', axis = 'columns')
    test_feat = test_feat.drop('Id', axis = 'columns')
    y_train_dt = train_dt.ham
    x_train_dt = train_dt.drop('ham', axis ='columns')
    clf = MultinomialNB()
    clf.fit(x_train_dt,y_train_dt)
    MultinomialNB(alpha=smooth, class_prior=class_prior, fit_prior=fit_prior) # Default = (1.0,None, True)
    Y_Pred_MNB = clf.predict(test_feat)
    Y_Pred_MNB = Y_Pred_MNB.astype(bool)
    Y_Pred_MNB
    Pred_MNB = pd.DataFrame()
    Pred_MNB['Id'] = test_features_with_id['Id']
    Pred_MNB['ham'] = Y_Pred_MNB
    Pred_MNB.set_index('Id', inplace=True)
    print(Pred_MNB)
    Pred_MNB.to_csv(name)
def  Complement_NB(train_dt,test_feat,name,smooth,class_prior,fit_prior,norm):
    train_dt = train_dt.drop('Id', axis = 'columns')
    test_feat = test_feat.drop('Id', axis = 'columns')
    y_train_dt = train_dt.ham
    x_train_dt = train_dt.drop('ham', axis ='columns')
    clf = ComplementNB()
    clf.fit(x_train_dt,y_train_dt)
    ComplementNB(alpha=smooth, class_prior=class_prior, fit_prior=fit_prior, norm=norm) #Default = (1.0,None, True,False)
    Y_Pred_CNB = clf.predict(test_feat)
    Y_Pred_CNB = Y_Pred_CNB.astype(bool)
    Y_Pred_CNB
    Pred_CNB = pd.DataFrame()
    Pred_CNB['Id'] = test_features_with_id['Id']
    Pred_CNB['ham'] = Y_Pred_CNB
    Pred_CNB.set_index('Id', inplace=True)
    print(Pred_CNB)
    Pred_CNB.to_csv(name)
def  Bernoulli_NB(train_dt,test_feat,name,smooth,b,fit_prior):
    train_dt = train_dt.drop('Id', axis = 'columns')
    test_feat = test_feat.drop('Id', axis = 'columns')
    y_train_dt = train_dt.ham
    x_train_dt = train_dt.drop('ham', axis ='columns')
    clf = BernoulliNB()
    clf.fit(x_train_dt,y_train_dt)
    BernoulliNB(alpha=smooth, binarize=b, fit_prior=fit_prior) # Default (1.0,0.0,True)
    Y_Pred_BNB = clf.predict(test_feat)
    Y_Pred_BNB = Y_Pred_BNB.astype(bool)
    Y_Pred_BNB
    Pred_BNB = pd.DataFrame()
    Pred_BNB['Id'] = test_features_with_id['Id']
    Pred_BNB['ham'] = Y_Pred_BNB
    Pred_BNB.set_index('Id', inplace=True)
    print(Pred_BNB)
    Pred_BNB.to_csv(name)
    


word_freq

Gaussian_NB(word_freq,test_word_freq,"GaussianNB_word_freq.csv",None, 1e-09)
#0.74871
Multinomial_NB(word_freq,test_word_freq,"MultinomialNB_word_freq.csv",1.0,None, True)
#0.83815


Complement_NB(word_freq,test_word_freq,"ComplementNB_word_freq.csv",1.0,None, True,False)
#0.82773
Bernoulli_NB(word_freq,test_word_freq,"BernoulliNB_word_freq.csv",1.0,0.0,True)
#0.92374

Gaussian_NB(special_char, test_special_char, "GaussianNB_special_char.csv",None, 1e-09)
#0.91857
Multinomial_NB(special_char, test_special_char, "MultinomialNB_special_char.csv",1.0,None, True)
#0.93577
Complement_NB(special_char, test_special_char, "ComplementNB_special_char.csv",1.0,None, True,False)
#0.89619
Bernoulli_NB(special_char, test_special_char, "BernoulliNB_special_char.csv",1.0,0.0,True)
#0.92638
Bernoulli_NB(capital_run, test_capital_run_with_id, "BernoulliNB_capital_run.csv",1.0,0.0,True)
#I thought that changes in the smoothing parameter and the fit_prior would afect the classification but result remains the same
#Still, the code used to test is presented below 
x=0.0
y=str(0)+str(int(x*10))+ "BNB"
name = y + (".csv")
print(name)
for i in range(0,5):
    x=i/5
    y=str(0)+str(int(x*10))+ "BNB"
    name = y +(".csv")
    Bernoulli_NB(train_data, test_features_with_id, name,x,0.0,False)
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import roc_curve, auc
train_w = train_data.drop('ham', axis = 'columns')
train_w = train_data.drop('Id', axis = 'columns')
f3 = make_scorer(fbeta_score, beta=3)
train_f3 = cross_val_score(BernoulliNB(), train_w, train_data.ham, cv=10, scoring = f3 )
train_f3.mean()
capital_run_w = capital_run.drop('ham', axis = 'columns')
capital_run_w  = capital_run.drop('Id', axis = 'columns')
f3 = make_scorer(fbeta_score, beta=3)
train_f3 = cross_val_score(BernoulliNB(), capital_run_w, capital_run.ham, cv=10, scoring = f3 )
train_f3.mean()
prob = cross_val_predict(BernoulliNB(), train_w, train_data.ham, cv=10, method = 'predict_proba')
fpr, tpr ,thresholds =roc_curve(train_data.ham,prob[:,1]);
lw=2
plt.plot(fpr,tpr, color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()
word_freq
x_train_data = train_data.drop('Id', axis = 'columns')
x_train_data = x_train_data.drop('ham', axis = 'columns')
x_train_data.loc[:,'char_freq_!2'] = train_data['char_freq_!']
test_features.loc[:,'char_freq_!2'] = train_data['char_freq_!']
x_train_data.loc[:,'char_freq_!3'] = train_data['char_freq_!']
test_features.loc[:,'char_freq_!3'] = train_data['char_freq_!']
x_train_data.shape
test_features.shape
x_train_data

train_data.ham.shape
clf5 = BernoulliNB()
clf5.fit(x_train_data, train_data.ham)
BernoulliNB(alpha=0.1, binarize=0.0, fit_prior=True)
Y_Pred_B_NB = clf5.predict(test_features)
Y_Pred_B_NB = Y_Pred_B_NB.astype(bool)
Pred_B_NB = pd.DataFrame()
Pred_B_NB['Id'] = test_features_with_id['Id']
Pred_B_NB['ham'] = Y_Pred_B_NB
Pred_B_NB.set_index('Id', inplace=True)
Pred_B_NB.to_csv("Weight_BernoulliNB.csv")
Pred_B_NB 
