import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
def my_scorer(clf, X, y_true):
    y_pred_proba = clf.predict_proba(X)
    class_labels = clf.classes_

    # Sort the classifications based on highest probability,
    # making sure to keep track of class labels properly
    p_pairs = [sorted([(c, p) for c, p in zip(class_labels, y_pred_proba[j])],
                      key=lambda x: x[1], reverse=True)
                for j in range(y_pred_proba.shape[0])]
    
    # Get the top 3 predictions by selecting the class
    # label piece of the tuple (element 0) for each sample
    y_top_3 = [[p[i][0] for i in range(3)]
               for p in p_pairs]
    
    # Calculate the MAP@3 score,
    # where the sum over P(k) is equal to
    # 1 if correct prediction is 1st
    # 1/2 if correct prediction is 2nd
    # 1/3 if correct prediction is 3rd
    MAP_score = [[(y == y_true[j])/(i+1) for i, y in enumerate(y_sample)]
                 for j, y_sample in enumerate(y_top_3)]
    MAP_score = np.sum(MAP_score, axis=1).mean()
    
    return MAP_score

gs = GridSearchCV(estimator=KNeighborsClassifier(),
                  param_grid=[{'n_neighbors': [2, 4, 6]}],
                  cv=5,
                  scoring=my_scorer)