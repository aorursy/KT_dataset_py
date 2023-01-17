import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



X_train, y_train = train.ix[:,1:]/255.0, train.ix[:,0]

X_test = test/255.0



print ("Train set: %i" % X_train.shape[0] )

print ("Test set: %i" % X_test.shape[0])
plt.imshow(X_train.iloc[30].values.reshape((28, 28)), cmap='gray')

plt.show()
sns.countplot(train['label'])
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

#from sklearn.svm import LinearSVC 



pipe = Pipeline([

    ('pca', PCA()),

    ('clf', KNeighborsClassifier()),

])



parameters = {

    'pca__n_components': [2, 3, 4, 5, 6, 7],

    #'clf__C': [1, 10, 100],

    }



from sklearn.model_selection import GridSearchCV, StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



gs = GridSearchCV(pipe, parameters, cv=kf, n_jobs=-1, verbose=1)

gs.fit(X_train, y_train)



print("Best score: %0.3f" % gs.best_score_)

print("Best parameters set:")

best_parameters = gs.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
def results(y_pred, y_test):

    from sklearn.metrics import classification_report, confusion_matrix

    print (classification_report(y_test, y_pred))

    

    cm = confusion_matrix(y_test, y_pred)

    df_cm = pd.DataFrame(100*cm/float(cm.sum()))

    ax = sns.heatmap(df_cm.round(2), annot=True, cmap='Blues', fmt='g', linewidths=1)

    ax.set_title("Confusion Matrix - per 100 predictions")

    ax.set_xlabel('Predicted', fontsize=16)

    ax.set_ylabel('True', fontsize=16, rotation=90)

    plt.show()

    

results(gs.predict(X_train), y_train)
y_pred = gs.predict(X_test)

df = pd.DataFrame(y_pred)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv', header=True)