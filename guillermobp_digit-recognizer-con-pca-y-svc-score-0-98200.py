import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
from time import time
from scipy import stats

filename = '../input/train.csv'
test_filename = '../input/test.csv'

train_df = pd.read_csv(filename)
test_df = pd.read_csv(test_filename)

images = train_df.iloc[:,1:]
labels = train_df.iloc[:,:1]
images.describe()
train_df.shape
X = images.values/255.0
y = labels['label'].values
#images = train_df.iloc[0:5000,1:]
#labels = train_df.iloc[0:5000,:1]

#Partición del set de entrenamiento para realizar ajuste hiperparamétrico

#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=7)
##Tuning PCA (Valores optimos: n_components = 0.72,whiten=False)
#if __name__ == '__main__':
#    svc = SVC()
#    pca =PCA()
#    steps = [('PCA', pca),('SVC', svc)]
#    pipeline = Pipeline(steps)
#    param_grid = { "PCA__n_components" : [0.70,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78], "PCA__whiten" :[True,False]}
#    grid = GridSearchCV(pipeline, param_grid,cv=3,scoring = "accuracy",iid=False, n_jobs=-1)
#    start = time()
#    grid.fit(X_train, Y_train)
#    print("Tuned Model Parameters: {}".format(grid.best_params_))
#    print("Tuned  Accuracy: {}".format(grid.best_score_))
#    y_pred = grid.predict(X_test)
#    print("Accuracy (usando y_test): {}".format(grid.score(X_test, Y_test)))
#    print(time() - start)
#Tuning SVC (Valores optimos: C=3, gamma=0.04)
#if __name__ == '__main__':
#    svc = SVC()
#    pca =PCA(0.72,whiten=False)
#    steps = [('PCA', pca),('SVC', svc)]
#    pipeline = Pipeline(steps)
#    param_grid = { "SVC__C" : [1,2,3,4,5,6,7,8,9,10] , "SVC__gamma" : [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]}
#    grid = GridSearchCV(pipeline, param_grid,cv=3,scoring = "accuracy",iid=False, n_jobs=-1)
#    start = time()
#    grid.fit(X_train, Y_train)
#    print("Tuned Model Parameters: {}".format(grid.best_params_))
#    print("Tuned  Accuracy: {}".format(grid.best_score_))
#    y_pred = grid.predict(X_test)
#    print("Accuracy (usando y_test): {}".format(grid.score(X_test, Y_test)))
#    print(time() - start)
X_test_final = test_df.values/255.0

#Tuned Model Parameters: {'SVC__C': 3, 'SVC__gamma': 0.04} PCA(0.72,whiten=False)  Score Kaggle: 0.98200
pca =PCA(0.72,whiten=False)
svc = SVC(C=3, gamma=0.04)
steps = [('PCA', pca),('SVC', svc)]
pipeline_final = Pipeline(steps)
pipeline_final.fit(X, y)
y_pred_final = pipeline_final.predict(X_test_final)
my_solution = pd.DataFrame(y_pred_final)
my_solution.index+=1
my_solution.index.name='ImageId'
my_solution.columns=['Label']
print(my_solution.shape)
my_solution.to_csv("my_solution_SVM_PCA.csv",header=True)


print("Numero de componentes:")
print(pca.n_components_)
print("Varianza explicada:")
print(pca.explained_variance_ratio_.sum())