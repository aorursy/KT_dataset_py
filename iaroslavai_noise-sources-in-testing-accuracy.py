from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm

mean_ts = []
mean_nd = []

for i in tqdm(range(100)):
    X, y = load_boston(True)
    X, X_new, y, y_new = train_test_split(X, y)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    best_score = 0
    best_model = None

    for depth in range(1, 20):
        model = DecisionTreeRegressor(max_depth=depth)
        score = np.mean(cross_val_score(model, X_train, y_train, cv=5))
        model.fit(X_train, y_train)

        if score > best_score:
            best_score = score
            best_model = model

    y_pred = best_model.predict(X_test)
    test_score = r2_score(y_test, y_pred)
            
    mean_ts.append(test_score)
    mean_nd.append(best_model.score(X_new, y_new))
    #render_tree(model)

print('Optimized:', np.mean(mean_ts))
print('New data:', np.mean(mean_nd))
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm

test_scores = []

for repetition in tqdm(range(64)):
    X, y = load_boston(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    best_score = 0
    best_model = None

    for depth in range(1, 20):
        model = DecisionTreeRegressor(max_depth=depth)
        score = np.mean(cross_val_score(model, X_train, y_train, cv=5))
        model.fit(X_train, y_train)

        if score > best_score:
            best_score = score
            best_model = model

    y_pred = best_model.predict(X_test)
    test_score = r2_score(y_test, y_pred)

    test_scores.append(test_score)

print(np.mean(test_scores))
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm

test_scores = []
cv_scores = []

for repetition in range(64):
    X, y = load_boston(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    best_score = 0
    best_model = None

    for depth in range(1, 40):
        model = DecisionTreeRegressor(max_depth=depth)
        score = np.mean(cross_val_score(model, X_train, y_train, cv=3))
        model.fit(X_train, y_train)

        if score > best_score:
            best_score = score
            best_model = model
            
    y_pred = best_model.predict(X_test)
    test_score = r2_score(y_test, y_pred)
    cv_scores.append(best_score)
    test_scores.append(test_score)

print('CV Scores:', np.mean(cv_scores))
print('Test scores:', np.mean(test_scores))