from keras.initializers import he_normal, normal

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler



from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



from sklearn.metrics import f1_score, precision_score, recall_score

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import RandomizedSearchCV

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('../input/spambase/realspambase.data', header=None)



# features first 57 columns

# last column is labels

X_features, y_labels = df.iloc[:, :57].values, df.iloc[:, 57].values



# Scale the features

X_features = StandardScaler().fit_transform(X_features)



x_train, x_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.1, random_state=15)



callbacks = [

    ReduceLROnPlateau(),

    EarlyStopping(patience=4),

    ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)

]
def model(activation='tanh', units_per_layer=10, n_hidden_layers=2):

    model = Sequential()

    # First hidden layer, taking 57 features as inputs

    model.add(Dense(input_dim=57,

                    units=units_per_layer,

                    kernel_initializer='uniform',

                    activation=activation))



    # Additional hidden layers

    for i in range(1, n_hidden_layers):

        model.add(Dense(units=units_per_layer,

                    kernel_initializer='uniform',

                    activation=activation))





    # binary classification layer

    model.add(Dense(units=1,

                    kernel_initializer='uniform',

                    activation='sigmoid',

                    name='output'))



    model.compile(optimizer='adam', loss='binary_crossentropy',

                  metrics=['acc'])



    #print(model.summary())



    return model
def model_eval(model, X_test, y_test):



    predicted_probs = model.predict(X_test, verbose=0)

    predicted_classes = model.predict_classes(X_test, verbose=0)



    # sklearn metrics require 1D array of actual and predicted vals

    # need to transform the data into 1D from 2D



    predicted_probs = predicted_probs[:, 0]

    predicted_classes = predicted_classes[:, 0]



    # accuracy: (tp + tn) / (p + n)

    accuracy = accuracy_score(y_test, predicted_classes)

    print('Accuracy: {}'.format(accuracy))

    # precision tp / (tp + fp)

    precision = precision_score(y_test, predicted_classes)

    print('Precision: {}'.format(precision))

    # recall: tp / (tp + fn)

    recall = recall_score(y_test, predicted_classes)

    print('Recall: {}'.format(recall))

    # f1: 2 tp / (2 tp + fp + fn)

    f1 = f1_score(y_test, predicted_classes)

    print('F1 score: {}'.format(f1))

    

    return accuracy
def cross_val(chosen_model):



    model = KerasClassifier(chosen_model, verbose=True)

    

    activation=['tanh', 'relu']

    units_per_layer=[10, 20]

    n_hidden_layers=[2, 3, 4]

    epochs=[20]



    param_grid = dict(activation=activation, units_per_layer=units_per_layer, 

                      n_hidden_layers=n_hidden_layers, epochs=epochs)



    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,n_jobs=-1,

                              cv=5, verbose=5, n_iter=12, scoring=['accuracy', 'f1', 'precision', 'recall'], refit='accuracy')

    grid_result = grid.fit(x_train, y_train)



    test_accuracy = grid.score(x_train, y_train)



    print(grid_result.best_score_)

    print(grid_result.best_params_)



    print(test_accuracy)



    return grid_result.best_params_
def plot_model(history):



    # training & validation accuracy

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    # training & validation loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
from sklearn.metrics import roc_curve
def plot_roc_curve(model, X_test, y_test):

    y_score = model.predict(X_test, verbose=0)

    fpr, tpr, _ = roc_curve(y_test, y_score)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr)

    ax.set_ylabel('True Positive Rate')

    ax.set_xlabel('False Positive Rate')

    ax.set_title('ROC Curve')    
model_original = model()

history = model_original.fit(x_train, y_train, validation_split=0.10, epochs=20)

original_testing_acc = model_eval(model_original, x_test, y_test)

plot_model(history)

plot_roc_curve(model_original, x_test, y_test)
print('Original Model:\nTraining Acc: {:0.4}\nValidation Acc: {:0.4}\nTesting Acc: {:0.4}'.format(history.history['acc'][-1], history.history['val_acc'][-1], original_testing_acc))
param_results = cross_val(model)

print(param_results)
model_optimized = model(activation=param_results['activation'], units_per_layer=param_results['units_per_layer'], n_hidden_layers=param_results['n_hidden_layers'])

model_history = model_optimized.fit(x_train, y_train, validation_split=0.10, epochs=param_results['epochs'])

optimized_acc = model_eval(model_optimized, x_test, y_test)

plot_model(model_history)

plot_roc_curve(model_optimized, x_test, y_test)
print('Optimized Model:\nTraining Acc: {:0.4}\nValidation Acc: {:0.4}\nTesting Acc: {:0.4}'.format(model_history.history['acc'][-1], model_history.history['val_acc'][-1], optimized_acc))
#Decision Tree

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=2, min_samples_leaf=7)

clf_gini.fit(x_train, y_train)

pred_gini = clf_gini.predict(x_test)

print("DECISION TREE")

print("Accuracy")

print (accuracy_score(pred_gini, y_test))

print("Report")

print(classification_report(pred_gini, y_test))
#Random Forest

print("RANDOM FOREST")

clf_random_forest = RandomForestClassifier(max_depth=2, random_state=0)

clf_random_forest.fit(x_train, y_train)

pred_random_forest = clf_random_forest.predict(x_test)

print("Accuracy")

print(accuracy_score(pred_random_forest, y_test))

print("Report")

print(classification_report(pred_random_forest, y_test))
#K-Nearest Neighbor

print("K-NEAREST NEIGHBORS")

clf_knn = KNeighborsClassifier(n_neighbors=3)

clf_knn.fit(x_train, y_train)

pred_knn = clf_knn.predict(x_test)

print("Accuracy")

print(accuracy_score(pred_knn, y_test))

print("Report")

print(classification_report(pred_knn, y_test))