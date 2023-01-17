#import cleaning and visualization modules
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import rcParams, gridspec

#import analysis modules
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import linear_model
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from xgboost import plot_importance
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

#pandas configuration
import warnings
warnings.filterwarnings('ignore')
#read into dataframe, display first 5 values
df = pd.read_csv('../input/mushrooms.csv')
df.head()
#Look into df for datatypes, if nulls exist 
null_count = 0
for val in df.isnull().sum():
    null_count += val
print('There are {} null values.\n'.format(null_count))
df.info()
def show_features(df):
    '''Takes a dataframe and outputs the columns, number of classes and category variables.'''
    col_count, col_var = [], []
    for col in df:
        col_count.append(len(df[col].unique()))
        col_var.append(df[col].unique().sum())
    df_dict = {'Count': col_count, 'Variables': col_var}
    df_table = pd.DataFrame(df_dict, index=df.columns)
    print(df_table)
    
show_features(df)
df['stalk-root'].value_counts()
df_dum = pd.get_dummies(df, drop_first=True)
df_dum.head()
plt.figure(figsize=[16,12])

plt.subplot(231)
sns.countplot(x='odor', hue='class', data=df)
plt.title('Odor')
plt.xticks(np.arange(10),('Pungent', 'Almond', 'Anise', 'None', 'Foul', 'Creosote', 'Fish', 'Spicy', 'Musty'), rotation='vertical')
plt.ylabel('Count')

plt.subplot(232)
sns.countplot(x='spore-print-color', hue='class', data=df)
plt.title('Spore Print Color')
plt.xticks(np.arange(10),('Black', 'Brown','Purple','Chocolate','White','Green','Orange','Yellow','Brown'), rotation='vertical')
plt.legend(loc='upper right')

plt.subplot(233)
sns.countplot(x='cap-color', hue='class', data=df)
plt.title('Cap Color')
plt.xticks(np.arange(11),('Brown', 'Yellow','White','Gray','Red','Pink','Buff','Purple','Cinnamon','Green'), rotation='vertical')
plt.legend(loc='upper right')

plt.subplot(234)
sns.countplot(x='bruises', hue='class', data=df)
plt.title('Bruising')
plt.xticks(np.arange(2),('Bruise', 'No Bruise'), rotation='vertical')
plt.legend(loc='upper right')

plt.subplot(235)
sns.countplot(x='habitat', hue='class', data=df)
plt.title('Habitat')
plt.xticks(np.arange(8),('Urban', 'Grasses','Meadows','Woods','Paths','Waste','Leaves'), rotation='vertical')
plt.legend(loc='upper right')

plt.subplot(236)
sns.countplot(x='population', hue='class', data=df)
plt.title('Population')
plt.xticks(np.arange(7),('Scattered', 'Numerous','Abundant','Several','Solitary','Clustered'), rotation='vertical')
plt.legend(loc='upper right')

plt.tight_layout()
sns.despine()
#set features
X = df_dum.drop('class_p', axis=1)
#set independent variable
y = df_dum['class_p']
#split the training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#print shapes of training/testing sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#visualize edible vs poison classes
pca = PCA(n_components=2)

x_pca = X.values
x_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y, s=40, edgecolor='k')
plt.title('Visualizing Edible vs. Poison Classes')
from sklearn import metrics
from sklearn.cluster import KMeans

#Specify the model and fit to training set
km = KMeans(n_clusters = 2)
km.fit(X_train)

#PCA X_test for visualization
pca_test = PCA(n_components = 2)
pca_test.fit(X_test)
X_test_pca = X_test.values
X_test_pca = pca_test.fit_transform(X_test)

#KMeans prediction
y_pred_km = km.predict(X_test)

#Plot the data
plt.figure(figsize=(8,6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_km, 
            s=40, edgecolor='k')
plt.title('KMeans: Test Data')
plt.show();
def plot_confusion_matrix(cm, classes, fontsize=15,
                          normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cm_num = cm
    cm_per = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title.replace('_',' ').title()+'\n', size=fontsize)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=fontsize)
    plt.yticks(tick_marks, classes, size=fontsize)

    fmt = '.5f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Set color parameters
        color = "white" if cm[i, j] > thresh else "black"
        alignment = "center"

        # Plot perentage
        text = format(cm_per[i, j], '.5f')
        text = text + '%'
        plt.text(j, i,
            text,
            fontsize=fontsize,
            verticalalignment='baseline',
            horizontalalignment='center',
            color=color)
        # Plot numeric
        text = format(cm_num[i, j], 'd')
        text = '\n \n' + text
        plt.text(j, i,
            text,
            fontsize=fontsize,
            verticalalignment='center',
            horizontalalignment='center',
            color=color)
        
    plt.tight_layout()
    plt.ylabel('True label'.title(), size=fontsize)
    plt.xlabel('Predicted label'.title(), size=fontsize)

    return None
cm_km = metrics.confusion_matrix(y_test, y_pred_km)
plot_confusion_matrix(cm_km, classes=['Edible','Poison'])
print(f'KMeans accuracy: {str(accuracy_score(y_test, y_pred_km)*100)[:5]}%')
#set the model and fit entire data to RFECV--train/test splits are done automatically and cross-validated.
lm = linear_model.LogisticRegression()
rfecv = RFECV(estimator=lm, step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print('Optimal number of features: %d' % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

#plot features vs. validation scores
plt.figure(figsize=(10,6))
plt.xlabel('Number of features selected')
plt.ylabel('Cross validation score')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
#set optimal features and assign new X, train/test split
opt_features = ['odor_c', 'odor_f', 'odor_l', 'odor_n', 'odor_p', 
                'gill-spacing_w', 'gill-size_n', 'stalk-surface-above-ring_k', 
                'ring-type_f', 'spore-print-color_k', 'spore-print-color_n', 
                'spore-print-color_r', 'spore-print-color_u', 'population_c']
#new dependent variables
X_opt = X[opt_features] 

#split the training and test data
Xo_train, Xo_test, yo_train, yo_test = train_test_split(X_opt, y, test_size=0.3)
#print shapes of training/testing sets
print(Xo_train.shape, Xo_test.shape, yo_train.shape, yo_test.shape)
#logistic regression
lm = linear_model.LogisticRegression()
lm.fit(Xo_train, yo_train)
log_probs = lm.predict_proba(Xo_test)
loss = log_loss(yo_test, log_probs)
print(f'Loss value: {loss}')
print(f'Training accuracy: {str(lm.score(Xo_train, yo_train)*100)[:5]}%')
print(f'Test accuracy: {str(lm.score(Xo_test, yo_test)*100)[:5]}%')
y_pred_lm = lm.predict(Xo_test)

cm_lm = metrics.confusion_matrix(yo_test, y_pred_lm)
plot_confusion_matrix(cm_lm, ['Edible','Poison'])
print(f'Logistic Regression accuracy: {str(accuracy_score(yo_test, y_pred_lm)*100)[:5]}%')
#test out different SVMs using the different kernals
kerns = ['linear', 'rbf', 'sigmoid']
for i in kerns:
    #Kernel trick
    svm_kern = SVC(kernel=f'{i}')
    svm_kern.fit(Xo_train,yo_train)
    
    #Get the score
    print(f'{i} kernal SVM score: {str(100*svm_kern.score(Xo_test,yo_test))[:6]}%')
#fit SVM model to scaled data
svm = LinearSVC()
svm.fit(Xo_train, yo_train)
print(f'Linear SVM Training accuracy is: {svm.score(Xo_train, yo_train)*100}%')
print(f'Linear SVM Test accuracy is: {svm.score(Xo_test, yo_test)*100}%')
y_pred_svm = svm.predict(Xo_test)

cm_svm = metrics.confusion_matrix(yo_test, y_pred_svm)
plot_confusion_matrix(cm_svm, ['Edible','Poison'])
print(f'SVM accuracy: {str(accuracy_score(yo_test, y_pred_svm)*100)[:5]}%')
#initialize gradientboost and xgboost
gb = GradientBoostingClassifier()
xgb = XGBClassifier()
#fit models
gb.fit(Xo_train,yo_train)
xgb.fit(Xo_train,yo_train)
#score models
print(f'Gradient Boost score: {(100 * gb.score(Xo_test,yo_test))}%')
print(f'XG Boost score: {(100 * xgb.score(Xo_test,yo_test))}%')
#plot feature importance XGBoost
plot_importance(xgb)
plt.show()
#fitting a random forest
rf = RandomForestClassifier()
rf.fit(Xo_train, yo_train)
print("Default RFR: %3.1f" % (rf.score(Xo_test, yo_test)*100))
param_grid = { 
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 10)
CV_rfc.fit(Xo_train, yo_train)
CV_rfc.best_params_
rfcv = RandomForestClassifier(criterion= 'gini',
 max_depth= 6,
 max_features= 'auto',
 n_estimators= 50)
rfcv.fit(Xo_train, yo_train)
print(f'GridSearchCV RFR: {(rfcv.score(Xo_test, yo_test)*100)}%')
feature_imp = pd.Series(rfcv.feature_importances_,index=Xo_train.columns).sort_values(ascending=False)
print(feature_imp)
#plot feature importance for RFR
plt.figure(figsize=(12,8))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title('Random Forest Feature Importance');
