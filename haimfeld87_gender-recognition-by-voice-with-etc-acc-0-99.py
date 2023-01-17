import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



voice_df = pd.read_csv("../input/voice.csv")
voice_df.columns
voice_df['label']=voice_df['label'].map( {'female': 0, 'male': 1} ).astype(int)
plt.figure()

voice_df['label'].value_counts().plot(kind = 'bar')

plt.ylabel("Count")

plt.title('female=0, male=1')
#extra features with PCA

extra = np.vstack(voice_df[["meanfun", "meanfreq"]].values)

pca = PCA().fit(extra)

voice_df['pca0'] = pca.transform(extra)[:, 0]

voice_df['pca1'] = pca.transform(extra)[:, 1]
colormap = plt.cm.viridis

plt.figure(figsize=(15,15))

sns.heatmap(voice_df.corr(), annot=True, linecolor='white', cmap=colormap)
X=voice_df.drop(["label"],axis=1)

Y=voice_df["label"]
from sklearn.model_selection import train_test_split, learning_curve

X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1,random_state=1)
ETC = ExtraTreesClassifier()

ETC.fit(X_train,Y_train)
features_list = X_train.columns.values

feature_importance = ETC.feature_importances_

sorted_idx = np.argsort(feature_importance)



plt.figure(figsize=(5,9))

#plt.subplot(1, 2, 2)

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()

plt.show()
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
Y_pred = ETC.predict(X_test)



print("Extra Tree Classifier report \n",classification_report(Y_pred,Y_test))



print('Extra Tree Classifier accuracy: %0.3f'% accuracy_score(Y_pred,Y_test))



print("Extra Tree Classifier confusion matrix \n",confusion_matrix(Y_pred,Y_test))
parameters = {"n_estimators":list(range(10,130,10))}

clf = GridSearchCV(ETC, parameters,cv=10,scoring='accuracy')

clf.fit(X_train, Y_train)

print(clf.best_params_)
Y_pred=clf.predict(X_test)

print("Extra Tree Classifier report after GridSearchCV \n",classification_report(Y_pred,Y_test))
print("Extra Tree Classifier confusion matrix after GridSearchCV \n",confusion_matrix(Y_pred,Y_test))

cfm=confusion_matrix(Y_pred,Y_test)

sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None)

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title('Confusion matrix')
print('Extra Tree Classifier accuracy after GridSearchCV: %0.3f'% accuracy_score(Y_pred,Y_test))
#scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')

#print('Extra Tree Classifier cross_val_score: %0.3f'% scores.mean())