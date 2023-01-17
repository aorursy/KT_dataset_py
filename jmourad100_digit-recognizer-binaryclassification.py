import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Conv2D, Input, Dense, Flatten
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

def warn(*args, **kwargs): pass ; import warnings; warnings.warn = warn
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

## Transform problem to binary classificaton [<=5 et >5]
data['label'] = data['label'].apply(lambda x: 0 if x<=5 else 1)
    
X = data.drop('label', axis = 1).values
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Every image is stored as a row in the dataframe
train_images = X_train.reshape((-1, 28, 28, 1))
test_images = X_test.reshape((-1, 28, 28, 1))

# Showing the 10th image in the training data
plt.imshow(train_images[10,:,:,0]);
def build_model(NbFilters = 16):
    inputLayer = Input((28,28,1), name="InputLayer")
    conv1 = Conv2D(filters=NbFilters, kernel_size=(3,3),
                   strides=(1,1), padding="valid", name="ConvolutionalLayer1")(inputLayer)
    conv2 = Conv2D(filters=2*NbFilters, kernel_size=(3,3),
                   strides=(1,1), name="ConvolutionalLayer2")(conv1)
    conv2 = Flatten()(conv2)
    output = Dense(1, activation='sigmoid', name="OutputLayer")(conv2)
    
    model = Model(inputLayer, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
build_model().summary()
m = build_model()
m.fit(train_images, y_train, batch_size=96, epochs=3, verbose = 1)

m.save("trainedModel.h5")
m_ = load_model("trainedModel.h5")

y_pred = m_.predict(test_images).reshape((1,-1))[0].astype(int)
print("Accuracy = {}".format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
estimator = KerasClassifier(build_fn=build_model, batch_size=64, epochs=5, verbose = 1)
Kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
results = cross_val_score(estimator, train_images, y_train, cv=Kfold, n_jobs=1)
print("Results: {} ({}%)".format(round(results.mean()*100,2), round(results.std()*100, 2)))
estimator = KerasClassifier(build_fn=build_model, verbose = 0)

tuned_parameters = { 'batch_size': [96], 'epochs': [2, 3] }
scores = ['accuracy','precision', 'recall']

for score in scores:
    print("="*39, "\n# Tuning hyper-parameters for {}:\n".format(score), "="*39)

  # clf = GridSearchCV(estimator, tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf = GridSearchCV(estimator, tuned_parameters, cv=5, scoring=score)
    clf.fit(train_images, y_train)

    print("\n# Best parameters set found on development set: ", clf.best_params_)
    print("# Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("\t\t\t\t %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("# Detailed classification report: ")
    # The model is trained on the full development set. & The scores are computed on the full evaluation set.
    y_true, y_pred = y_test, clf.predict(test_images)
    print("Accuracy = {}".format(accuracy_score(y_true, y_pred)))
    print(classification_report(y_true, y_pred), "\n")
