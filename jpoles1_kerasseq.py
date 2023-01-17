#Import general prereqs
import pandas as pd
import numpy as np
from ggplot import *
from matplotlib import pyplot as plt
%matplotlib inline
#First round of analysis
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
def plotROC(true_list, prob_list):
    #ROC Curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(true_list, prob_list)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
#Second round
from sklearn import svm
#Read in data
raw_data = pd.read_csv("../input/train.csv")
n_comp = 370;
#Separate into explanatory and dependent vars (X and y respectively)
y = raw_data["TARGET"]
X = raw_data.ix[:, raw_data.columns != "TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
target_names = ["Satisfied", "Unsatisfied"]
if(n_comp<X.shape[1]):
	pca = PCA(n_components=n_comp)
	pca_model = pca.fit(X_train)
	X_train_pca = pca_model.transform(X_train)
	X_test_pca = pca_model.transform(X_test)
else:
    X_train_pca = X_train.as_matrix()
    X_test_pca = X_test.as_matrix()

#Setup neural net
from keras.models import Model
from keras.layers import Input, Dense, Activation
inputs = Input(shape=(n_comp,))
x = Dense(600, activation='tanh')(inputs)
x = Dense(300, activation='tanh')(x)
x = Dense(150, activation='tanh')(x)
x = Dense(30, activation='tanh')(x)
cat_prob = Dense(2, activation='sigmoid', name='cat_prob')(x)
category = Dense(1, activation='softmax', name='category')(cat_prob)
#Compile models
cat_model = Model(input=inputs, output=category)
cat_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
prob_model = Model(input=inputs, output=cat_prob)
prob_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

cat_model.fit(X_train_pca, y_train.as_matrix(), nb_epoch=10, verbose=0)
pred_cat = pd.DataFrame(cat_model.predict(X_train_pca))
pred_cat[0].value_counts()
probs = pd.concat([y_train, 1-y_train], axis=1)
pred_prob = prob_model.predict(X_train_pca)
pred_prob = pd.DataFrame(pred_prob)
pred_prob.describe()
pred_prob_plotdat = pd.melt(pred_prob)
ggplot(aes(fill="variable", x="value"), data=pred_prob_plotdat)+geom_histogram()
#Test classifier on training data
true_train = (pred_cat[0] == y_train)
correct_predictions = true_train.value_counts()
print(y_train.value_counts())
print(correct_predictions)
print("Percent accurate (train): "+str(100*correct_predictions[1]/(correct_predictions[0]+correct_predictions[1]))+"%")
plotROC(true_train, pred_prob[:,1])