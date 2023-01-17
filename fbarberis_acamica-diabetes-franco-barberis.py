import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

path_dataset = '../input/acamica_diabetes.csv'

df = pd.read_csv(path_dataset)
df.shape
df.head()
df.isna().sum()
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



sns.set(style="white")
corr = df.corr()
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
(df['Outcome']==1).sum()
df2 = df.filter(['Glucose','BMI','Outcome','Pregnancies','Age','Insulin','DiabetesPedegreeFunction'], axis=1)
df2.head()
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(df)

transformer
#scaled_df = transformer.transform(df2)

scaled_df=df2
scaled_df = pd.DataFrame(scaled_df, columns=df2.columns)

scaled_df.head()
import numpy as np

np.random.seed(123)

from sklearn.model_selection import train_test_split

X = scaled_df.drop(['Outcome'], axis=1)

y = scaled_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)



print(X_train.shape[0], X_test.shape[0])
X_train.shape
y_train.head()
(y_train==1).sum()
from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC
param_grid=[

    {'C':[0.001, 0.01, 0.1, 1, 10], 'loss': ['hinge', 'squared_hinge']},

]
linear_clf=LinearSVC()
grid_search = GridSearchCV (linear_clf, param_grid, cv=5, scoring = 'f1', refit= True, return_train_score=True,n_jobs=-1)

grid_search.fit(X_train, y_train)
grid_search.scorer_
grid_search.best_params_
grid_search.cv_results_['mean_train_score']
grid_search.best_score_
best_linearSCV = grid_search.best_estimator_

best_linearSCV
y_pred_linear=best_linearSCV.predict(X_test)
import numpy as np

import itertools

import matplotlib.pylab as plt

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()



# Mostrá la matriz de confusión en esta celda

import itertools

import numpy as np

import matplotlib.pyplot as plt



from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



class_names=['0', '1']

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred_linear)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Matriz de confusión sin normalizar')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Matriz de confusión normalizada')



plt.show()
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.metrics import classification_report
roc_auc_score (y_test, y_pred_linear)
print(classification_report(y_test, y_pred_linear))
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(X_train, y_train)
y_pred_knn=neigh.predict(X_test)
neigh.score(X_test, y_test)
class_names=['0', '1']

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred_knn)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Matriz de confusión sin normalizar')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Matriz de confusión normalizada')



plt.show()
roc_auc_score (y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))
from keras.models import Sequential

from keras.layers import Dense
# create model

model = Sequential()

model.add(Dense(12, input_dim=5, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=50)


# evaluate the model

scores = model.evaluate(X_train, y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_pred_model = model.predict(X_test)
roc_auc_score (y_test, y_pred_model)
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
from keras import backend as K

K.tensorflow_backend._get_available_gpus()