import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn.metrics import accuracy_score, log_loss

from time import clock_gettime_ns



np.random.seed(77)



modelTypes = ["randomForest", "logisticRegression", "SVM"]



def sample_data(size = 1000, stratified=True):

    X = np.random.multivariate_normal(mean = np.zeros(2), cov = np.diag(np.ones(2)), size = size)

    y = np.random.randint(0,2, size=size)

    if stratified:

        y = np.concatenate([np.zeros(size//2), np.ones(size - size//2)]).astype(np.int64)

    return X, y



def sample_model(modelType):

    model_dict = {"randomForest": RandomForestClassifier(n_estimators=10, max_depth=5),

                 "logisticRegression": LogisticRegression(solver="liblinear"),

                 "SVM": svm.SVC(gamma="scale", C=0.1)}

    return model_dict[modelType]

    

def run_train_validation_split(X,y, model, stratified = True, ratio_test = 0.2):

    if stratified:

        stratify = y

    else:

        stratify = None

    

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=stratify, test_size=ratio_test)

    model.fit(X_train, y_train)

    y_val_predicted = model.predict(X_val)

    val_acc = accuracy_score(y_val, y_val_predicted)

    return val_acc



def run_KFold_CV(X,y, model, num_folds = 5, stratified = True):    

    folds = KFold(n_splits=num_folds, shuffle=True)

    if stratified:

        folds = StratifiedKFold(n_splits = num_folds, shuffle=True)

        

    y_predicted = np.zeros_like(y)

    for ind_train, ind_val in folds.split(X,y):

        X_train = X[ind_train]

        y_train = y[ind_train]

        X_val = X[ind_val]

        model.fit(X_train, y_train)

        y_predicted[ind_val] = model.predict(X_val)

    val_acc = accuracy_score(y, y_predicted)

    return val_acc



def plot_data(data):

    X, y = data

    df = pd.DataFrame(data=X, columns=["x1","x2"]) 

    df["y"] = y

    sns.scatterplot(data=df, x="x1", y="x2", hue="y")
# Generating and plotting the distribution of our toy dataset.

X, y = sample_data(stratified=False, size=1000)

plt.figure(figsize=(7,7))

plot_data((X,y))

plt.show()
print("Ratio of positive samples: {}".format(y.sum() / y.shape[0]))
# Plotting the distribution of positive samples based on the dataset size.

plt.figure(figsize=(16,16))

plt.subplot(2,2,1)

for size in [200, 1000, 5000]:

    dist_positives = np.zeros(10000)

    for i in range(10000):

        _, y = sample_data(size=size, stratified=False)

        dist_positives[i] = y.sum() / size

    sns.kdeplot(dist_positives, label="size={}".format(size))

plt.legend()

plt.title("Distribution of positives based on dataset size")



# Plotting the distribution of validation accuracies for different stratified datasets and different model types.

num_datasets = 5

num_runs = 1000



datasets = [sample_data(size = 1000, stratified=True) for i in range(num_datasets)]

for i,modelType in enumerate(modelTypes):

    plt.subplot(2,2,i+2)

    val_accs_kfold = np.zeros(num_runs)

    for dataset_index in range(num_datasets):    

        X, y = datasets[dataset_index]

        for run in range(num_runs):

            model = sample_model(modelType)

            val_accs_kfold[run] = run_KFold_CV(X, y, model, num_folds=5, stratified=True)

        sns.kdeplot(val_accs_kfold, label="dataset {}".format(dataset_index))

    plt.title("{} validation accuracy distribution for each stratified dataset".format(modelType))



plt.show()
# Plotting the worst and best dataset (w.r.t validation accuracy) for each model type.

num_datasets = 100

num_runs = 10



plt.figure(figsize=(16,24))



datasets = [sample_data(size=20, stratified=True) for i in range(num_datasets)]



for i,modelType in enumerate(modelTypes):

    val_accs_kfold = np.zeros(num_datasets)

    for dataset_index in range(num_datasets):

        X, y = datasets[dataset_index]

        for run in range(num_runs):

            model = sample_model(modelType)

            val_accs_kfold[dataset_index] += run_KFold_CV(X, y, model, num_folds=5, stratified=True) / num_runs

    best_dataset_index = np.argmax(val_accs_kfold)

    plt.subplot(3,2,2*i+1)

    plot_data(datasets[best_dataset_index])

    plt.title("best dataset for {}, average validation acc: {:.2f}".format(modelType, val_accs_kfold[best_dataset_index]))

    

    worst_dataset_index = np.argmin(val_accs_kfold)

    plt.subplot(3,2,2*i+2)

    plot_data(datasets[worst_dataset_index])    

    plt.title("worst dataset for {}, average validation acc: {:.2f}".format(modelType, val_accs_kfold[worst_dataset_index]))



plt.show()
# Plotting validation accuracy distribution for K-fold CV and train/val split. 

num_runs = 1000



val_accs_train_val = np.zeros(num_runs)

val_accs_kfold = np.zeros(num_runs)



X, y = sample_data(size = 1000, stratified=True)



plt.figure(figsize=(15,10))



for modelType in modelTypes:

    for run in range(num_runs):

        model = sample_model(modelType=modelType)

        val_accs_train_val[run] = run_train_validation_split(X, y, model, ratio_test=0.2, stratified=True)

        val_accs_kfold[run] = run_KFold_CV(X, y, model, num_folds=5, stratified=True)





    sns.kdeplot(val_accs_train_val, label = "{} with train val split, 0.2 val ratio".format(modelType))

    sns.kdeplot(val_accs_kfold, label = "{} with 5-fold CV".format(modelType))

plt.title("Distribution of validation accuracy for different validation methods and model.")

plt.legend()