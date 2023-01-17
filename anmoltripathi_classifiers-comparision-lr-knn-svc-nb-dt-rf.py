# For Data Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap



# For Data Manipulation

import numpy as np 

import pandas as pd

import sklearn

from itertools import cycle





# For Data Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# For Classification Results

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import plot_roc_curve

from sklearn.preprocessing import label_binarize

from scipy import interp

from sklearn.exceptions import NotFittedError



# Dimensionality Reduction

from sklearn.decomposition import PCA



# Importing Models

from sklearn.linear_model import LogisticRegression #Logistic Regression

from sklearn.neighbors import KNeighborsClassifier as KNN #K-Nearest Neighbors

from sklearn.svm import SVC #Support Vector Classifier

from sklearn.naive_bayes import GaussianNB #Naive Bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree Classifier

from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df.head()
ax = df["quality"].value_counts().plot.bar(figsize=(7,5))

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    

print(df["quality"].value_counts(normalize=True)*100)
df.describe()
df.isnull().sum() #No missing values
df["is good"] = 0

df.loc[df["quality"]>=7,"is good"] = 1
ax = df["is good"].value_counts().plot.bar(figsize=(7,5))

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height() * 0.5), color="white")

    

print(df["is good"].value_counts(normalize=True)*100)
features = df.columns[:-2]

output = df.columns[-1]

print("Features: \n{}, \n\nLabels: \n{}".format(features.values,output))
# sns.pairplot(df[features],palette='coolwarm')

# plt.show()
for f in features:

    print('Feature:{}\n Skew = {} \n\n'.format(f,df[f].skew()))
corr = df[features].corr()

plt.figure(figsize=(16,16))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           xticklabels= features, yticklabels= features, alpha = 0.7,   cmap= 'coolwarm')

plt.show()
for f in features:

    df.boxplot(column=f, by=output)

    plt.title(f)

plt.show()
X = df[features].values

y = df[output].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)

print('Training size: {}, Testing size: {}'.format(X_train.size,X_test.size))
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
def get_probabilty_output(X_test, model_fitted, value_count=10):

    def highlight_max(data, color='yellow'):

        attr = 'background-color: {}'.format(color)

        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1

            is_max = data == data.max()

            return [attr if v else '' for v in is_max]

        else:  # from .apply(axis=None)

            is_max = data == data.max().max()

            return pd.DataFrame(np.where(is_max, attr, ''), index=data.index, columns=data.columns)

        

    y_scores = model_fitted.predict_proba(X_test)

    prob_df = pd.DataFrame(y_scores*100).head(value_count)

    styled_df = prob_df.style.background_gradient(cmap='Reds')

    styled_df = styled_df.highlight_max(axis=1, color='green')

    return styled_df
def get_classification_report(y_test,predictions,average="macro"):

    #Confusion Matrix

    cm = confusion_matrix(y_test, predictions)

    sns.heatmap(cm, annot=True)

    plt.title("Confusion Matrix")

    

    acc = accuracy_score(y_test, predictions)

    pre = precision_score(y_test, predictions, average=average)

    rec = recall_score(y_test, predictions, average=average)

    # Prediction Report

    print(classification_report(y_test, predictions, digits=3))

    print("Overall Accuracy:", acc)

    print("Overall Precision:", pre)

    print("Overall Recall:", rec)

    

    return acc,pre,rec

    
def get_classification_ROC(X,y,model,test_size,model_fitted=False,random_state=0):

    

    def check_fitted(clf): 

        return hasattr(clf, "classes_")

    

    if(len(np.unique(y)) == 2):

        #Binary Classifier

        if not check_fitted(model):

            model = model.fit(X,y)

        

        plot_roc_curve(model, X, y)

        y_score = model.predict_proba(X)[:, 1]

        fpr, tpr, threshold = roc_curve(y, y_score)

        auc = roc_auc_score(y, y_score)

        return auc

#         print("False Positive Rate: {} \nTrue Positive Rate: {} \nThreshold:{}".format(fpr,tpr,threshold))

    

    else:

        #Multiclass Classifier

        y_bin = label_binarize(y, classes=np.unique(y))

        n_classes = y_bin.shape[1]



        # shuffle and split training and test sets

        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=test_size, random_state=random_state)



        # Learn to predict each class against the other

        classifier = OneVsRestClassifier(model)

        model_fitted = classifier.fit(X_train, y_train)

        try:

            y_score = model_fitted.decision_function(X_test)

        except:

            y_score = model_fitted.predict_proba(X_test)







        # Compute ROC curve and ROC area for each class

        fpr = dict()

        tpr = dict()

        roc_auc = dict()

        for i in range(n_classes):

            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

            roc_auc[i] = auc(fpr[i], tpr[i])





        # Compute micro-average ROC curve and ROC area

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])





        plt.figure()

        lw = 2

        plt.plot(fpr[2], tpr[2], color='darkorange',

                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('Receiver operating characteristic averaged')

        plt.legend(loc="lower right")

        plt.show()







        # First aggregate all false positive rates

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



        # Then interpolate all ROC curves at this points

        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):

            mean_tpr += interp(all_fpr, fpr[i], tpr[i])



        # Finally average it and compute AUC

        mean_tpr /= n_classes



        fpr["macro"] = all_fpr

        tpr["macro"] = mean_tpr

        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



        # Plot all ROC curves

        plt.figure(figsize=(10,10))

        plt.plot(fpr["micro"], tpr["micro"],

                 label='micro-average ROC curve (area = {0:0.2f})'

                       ''.format(roc_auc["micro"]),

                 color='deeppink', linestyle=':', linewidth=4)



        plt.plot(fpr["macro"], tpr["macro"],

                 label='macro-average ROC curve (area = {0:0.2f})'

                       ''.format(roc_auc["macro"]),

                 color='navy', linestyle=':', linewidth=4)



        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue', 'purple', 'green'])

        for i, color in zip(range(n_classes), colors):

            plt.plot(fpr[i], tpr[i], color=color, lw=lw,

                     label='ROC curve of class {0} (area = {1:0.2f})'

                     ''.format(i, roc_auc[i]))



        plt.plot([0, 1], [0, 1], 'k--', lw=lw)

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('multi-class ROC (One vs All)')

        plt.legend(loc="lower right")

        plt.show()
def visualisation_through_PCA(X_PCA, y, model_PCA, model_name="Classification Model"):

    X_set, y_set = X_PCA, y

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, model_PCA.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                 alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue', 'yellow', 'purple', 'grey')))

    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):

        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                    c = ListedColormap(('red', 'green', 'blue', 'yellow', 'purple', 'grey'))(i), label = j)

    plt.title(model_name)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend()

    plt.show()
pca = PCA(n_components = 2)

X_train_PCA_2 = pca.fit_transform(X_train)

X_test_PCA_2 = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

print("Variance Explained by each of the Principal Components: {:.{prec}f}% and {:.{prec}f}%, \nTotal Variance Explained: {:.{prec}f}%".format((explained_variance*100)[0],

                                                                                                                                               (explained_variance*100)[1],

                                                                                                                                                  explained_variance.sum()*100,prec=3))
parameters_LR = {

    "solver" : ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),

    "penalty" : ('l1', 'l2', 'elasticnet', 'none'),

    "C" : [0.01, 0.1, 1, 10, 1000]

    

}



model_LR = LogisticRegression()

model_LR_with_best_params = GridSearchCV(model_LR, parameters_LR)

model_LR_with_best_params.fit(X_train,y_train)

model_LR_best_params = model_LR_with_best_params.best_params_
model_LR_best_params
predictions_LR = model_LR_with_best_params.predict(X_test)

print("Predictions:",predictions_LR[:10])

print("Actual:",y_test[:10])
get_probabilty_output(X_test=X_test, model_fitted=model_LR_with_best_params, value_count=15)
acc_LR,pre_LR,rec_LR = get_classification_report(y_test,predictions_LR)
auc_LR = get_classification_ROC(X_test,y_test,model_LR_with_best_params,test_size=0.3,random_state=0)
# model_LR_PCA = LogisticRegression(random_state = 0)

# model_LR_PCA.fit(X_train_PCA_2, y_train)

# predictions_LR_PCA = model_LR_PCA.predict(X_test_PCA_2)
# visualisation_through_PCA(X_train_PCA_2, y_train, model_LR_PCA, model_name="Logisitic Regression (Training Set)")
# visualisation_through_PCA(X_test_PCA_2, y_test, model_LR_PCA, model_name="Logisitic Regression (Test Set)")
parameters_KNN = {

    "n_neighbors" : [2,5,7,15],

    "weights" : ('uniform','distance'),

    "algorithm" : ('auto','ball_tree','kd_tree','brute'),

    'p': [1,2,5]

    

    

}



model_KNN = KNN(n_jobs=-1)

model_KNN_with_best_params = GridSearchCV(model_KNN, parameters_KNN)

model_KNN_with_best_params.fit(X_train,y_train)

model_KNN_best_params = model_KNN_with_best_params.best_params_
model_KNN_best_params
predictions_KNN = model_KNN_with_best_params.predict(X_test)

print("Predictions:",predictions_KNN[:10])

print("Actual:",y_test[:10])
get_probabilty_output(X_test=X_test, model_fitted=model_KNN_with_best_params)
acc_KNN,pre_KNN,rec_KNN = get_classification_report(y_test,predictions_KNN)
auc_KNN = get_classification_ROC(X_test,y_test,model_KNN_with_best_params,test_size=0.3,random_state=0)
# model_KNN_PCA = KNN(5)

# model_KNN_PCA.fit(X_train_PCA_2, y_train)

# predictions_KNN_PCA = model_KNN_PCA.predict(X_test_PCA_2)
# visualisation_through_PCA(X_train_PCA_2, y_train, model_KNN_PCA, model_name="k-Nearest Neighbors (Training Set)")
# visualisation_through_PCA(X_test_PCA_2, y_test, model_KNN_PCA, model_name="k-Nearest Neighbors (Test Set)")
parameters_SVC = {

    "C": [0.1, 1, 10],

    "kernel": ('linear','poly','rbf'),

    "degree": [2,4] 

    

}



model_SVC = SVC(probability=True)

model_SVC_with_best_params = GridSearchCV(model_SVC, parameters_SVC)

model_SVC_with_best_params.fit(X_train,y_train)

model_SVC_best_params = model_SVC_with_best_params.best_params_
model_SVC_best_params
predictions_SVC = model_SVC_with_best_params.predict(X_test)

print("Predictions:",predictions_SVC[:10])

print("Actual:",y_test[:10])
get_probabilty_output(X_test=X_test, model_fitted=model_SVC_with_best_params)
acc_SVC,pre_SVC,rec_SVC = get_classification_report(y_test,predictions_SVC)
auc_SVC = get_classification_ROC(X_test,y_test,model_SVC_with_best_params,test_size=0.3,random_state=0)
# model_SVC_PCA = model_SVC = SVC(kernel=kernel, random_state=random_state, probability=True)

# model_SVC_PCA.fit(X_train_PCA_2, y_train)

# predictions_SVC_PCA = model_SVC_PCA.predict(X_test_PCA_2)
# visualisation_through_PCA(X_train_PCA_2, y_train, model_SVC_PCA, model_name="Support Vector Classifier (Training Set)")
# visualisation_through_PCA(X_test_PCA_2, y_test, model_SVC_PCA, model_name="Support Vector Classifier (Test Set)")
model_NB = GaussianNB()

model_NB.fit(X_train, y_train)
predictions_NB = model_NB.predict(X_test)

print("Predictions:",predictions_NB[:10])

print("Actual:",y_test[:10])
get_probabilty_output(X_test=X_test, model_fitted=model_NB)
acc_NB,pre_NB,rec_NB = get_classification_report(y_test,predictions_NB)
auc_NB = get_classification_ROC(X_test,y_test,model_NB,test_size=0.3,random_state=0)
# model_NB_PCA = GaussianNB()

# model_NB_PCA.fit(X_train_PCA_2, y_train)

# predictions_NB_PCA = model_NB_PCA.predict(X_test_PCA_2)
# visualisation_through_PCA(X_train_PCA_2, y_train, model_NB_PCA, model_name="Naive Bayes Classifier (Training Set)")
# visualisation_through_PCA(X_test_PCA_2, y_test, model_NB_PCA, model_name="Naive Bayes Classifier (Test Set)")
parameters_DT = {

    'criterion':('gini','entropy'),

    'max_features': ('auto','sqrt','log2')

}





model_DT = DecisionTreeClassifier()

model_DT_with_best_params = GridSearchCV(model_DT, parameters_DT)

model_DT_with_best_params.fit(X_train,y_train)

model_DT_best_params = model_DT_with_best_params.best_params_

model_DT_with_best_params.fit(X_train,y_train)
model_DT_best_params
predictions_DT = model_DT_with_best_params.predict(X_test)

print("Predictions:",predictions_DT[:10])

print("Actual:",y_test[:10])
get_probabilty_output(X_test=X_test, model_fitted=model_DT_with_best_params)
acc_DT,pre_DT,rec_DT = get_classification_report(y_test,predictions_DT)
auc_DT = get_classification_ROC(X_test,y_test,model_DT_with_best_params,test_size=0.3,random_state=0)
# model_DT_PCA = DecisionTreeClassifier(criterion="entropy", random_state=0)

# model_DT_PCA.fit(X_train_PCA_2, y_train)

# predictions_DT_PCA = model_DT_PCA.predict(X_test_PCA_2)
# visualisation_through_PCA(X_train_PCA_2, y_train, model_DT_PCA, model_name="Decision Tree Classifier (Training Set)")
# visualisation_through_PCA(X_test_PCA_2, y_test, model_DT_PCA, model_name="Decision Tree Classifier (Test Set)")
parameters_RF = {

    'criterion':('gini','entropy'),

    'max_features': ('auto','sqrt','log2'),

    'n_estimators': [100,150,200,250,300]

}





model_RF = RandomForestClassifier(n_jobs=-1)

model_RF_with_best_params = GridSearchCV(model_RF, parameters_RF)

model_RF_with_best_params.fit(X_train,y_train)

model_RF_best_params = model_RF_with_best_params.best_params_

model_RF_with_best_params.fit(X_train,y_train)
model_RF_best_params
predictions_RF = model_RF_with_best_params.predict(X_test)

print("Predictions:",predictions_DT[:10])

print("Actual:",y_test[:10])
get_probabilty_output(X_test=X_test, model_fitted=model_RF_with_best_params)
acc_RF,pre_RF,rec_RF = get_classification_report(y_test,predictions_RF)
auc_RF = get_classification_ROC(X_test,y_test,model_RF_with_best_params,test_size=0.3,random_state=0)
# model_RF_PCA = RandomForestClassifier(n_estimators = 10, criterion="entropy", random_state=0)

# model_RF_PCA.fit(X_train_PCA_2, y_train)

# predictions_RF_PCA = model_RF_PCA.predict(X_test_PCA_2)
# visualisation_through_PCA(X_train_PCA_2, y_train, model_RF_PCA, model_name="Random Forest Classifier (Training Set)")
# visualisation_through_PCA(X_test_PCA_2, y_test, model_RF_PCA, model_name="Random Forest Classifier (Test Set)")
result = pd.DataFrame(

    [["LogisticRegression",auc_LR,acc_LR,pre_LR,rec_LR],

    ["kNearestNeighbor",auc_KNN,acc_KNN,pre_KNN,rec_KNN],

    ["SupportVectorClassifier",auc_SVC,acc_SVC,pre_SVC,rec_SVC],

    ["NaiveBayes",auc_NB,acc_NB,pre_NB,rec_NB],

    ["DecisionTree",auc_DT,acc_DT,pre_DT,rec_DT],

    ["RandomForest",auc_RF,acc_RF,pre_RF,rec_RF]],

    columns=["Classifier","AUC","Accuracy","Precision","Recall"]

)



result
fig = plt.figure(figsize=(10,5))

ax = fig.add_axes([0,0,1,1])

x = result.Classifier

y = result.AUC

sns.barplot(x=x, y=y)

plt.title("AUC Score Comparision")

plt.show()
fig = plt.figure(figsize=(10,5))

ax = fig.add_axes([0,0,1,1])

x = result.Classifier

y = result.Accuracy

sns.barplot(x=x, y=y)

plt.title("Accuracy Comparision")

plt.show()
fig = plt.figure(figsize=(10,5))

ax = fig.add_axes([0,0,1,1])

x = result.Classifier

y = result.Precision

sns.barplot(x=x, y=y)

plt.title("Precision Comparision")

plt.show()
fig = plt.figure(figsize=(10,5))

ax = fig.add_axes([0,0,1,1])

x = result.Classifier

y = result.Recall

sns.barplot(x=x, y=y)

plt.title("Recall Comparision")

plt.show()