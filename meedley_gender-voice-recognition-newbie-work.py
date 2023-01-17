import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns
import pandas as pd
import numpy as np


vdf = read_csv('/kaggle/input/voicegender/voice.csv')
print(vdf.shape)
vdf.info()
#NO null data, all numeric except label
print(vdf['label'].unique())
vdf["label"] = vdf["label"].astype('category')
y = vdf["label"].cat.codes #save label code as y variabl

#drop label from dataframe
x = vdf.drop(['label'],axis=1)
features = x.columns.tolist() #save all the features
print(x[features].round(2).describe().transpose())
feature_mean = x.mean()
feature_std = x.std()
#center and scale the data
x = (x - feature_mean)/feature_std
print(x[features].round(2).describe().transpose())
from numpy.linalg import matrix_rank
print(matrix_rank(x))
max_corr = 0.9 #largest acceptable correlation value
corr_matrix = x.corr().abs() #get absolute values for correlation
#work with upper triangular matrix, corr_matrix is symmetric
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
sns.heatmap(upper>max_corr); #check for high collinearity
#drop columns/features
to_drop = [column for column in upper.columns if any(upper[column] > max_corr)]
x.drop(to_drop, axis=1, inplace=True)
print('Drop features: ', to_drop)
print('Rank: ', matrix_rank(x), '\nShape: ', x.shape)
#check the new correlation matrix
corr_matrix = x.corr().abs();
sns.heatmap(corr_matrix);
sns.countplot(x=y); #equal counts of male and female data
plt.xticks(np.arange(2), ('Male','Female'));
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
#split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)
from sklearn.svm import SVC
#create classifier objects.
svm = SVC(kernel='linear')
#fit the model
svm.fit(x_train,y_train)
#perform cross validation
scores = cross_val_score(svm,x,y)#get cross validation score
#do prediction
y_pred = svm.predict(x_test)
print("SVM training accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), 100*scores.std()))
print("SVM prediction accuracy: %0.2f" % accuracy_score(y_test, y_pred))
#check confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap="Greens");
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=200)
LR.fit(x_train, y_train)
scores = cross_val_score(LR,x,y)
y_pred = svm.predict(x_test)
print("LR training accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), 100*scores.std()))
print("LR prediction accuracy: %0.2f" % accuracy_score(y_test, y_pred))
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap="Greens");
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(max_depth=10)
RF.fit(x_train, y_train)
y_pred = RF.predict(x_test)
scores = cross_val_score(RF,x,y)
print("RF training accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), 100*scores.std()))
print("RF prediction accuracy: %0.2f" % accuracy_score(y_test, y_pred))
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap="Greens");
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(random_state = 100,max_iter=500)
NN.fit(x_train, y_train);
scores = cross_val_score(NN,x,y)
y_pred = NN.predict(x_test)
print("NN training accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("NN prediction accuracy: %0.2f" % accuracy_score(y_test, y_pred))
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap="Greens");