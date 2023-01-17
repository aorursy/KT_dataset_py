import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For visualizing data
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style='darkgrid')
import plotly.graph_objs as go
import plotly.offline as py

# For preprocessing dataset
from sklearn.preprocessing import LabelEncoder
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# For building models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination with Cross-Validation
# To identify the best features by reducing less important features
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# For model evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve, auc
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.head()
df.describe()
df.info()
df.isnull().sum()
def drop_and_encode_features(data):
    """
    - Drop 'id' and 'Unnamed: 32' columns
    - Encode 'diagnosis' to numerical variable
    """
    data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    data.diagnosis = [1 if result=='M' else 0 for result in df.diagnosis]
    return data
def get_pie_chart(data):
    """
    Visualize "diagnosis" with pie chart
    """
    result = data['diagnosis'].value_counts()
    values = [result['M'], result['B']]
    labels = ['Malignant', 'Benign']
    trace = go.Pie(labels=labels, values=values)
    py.iplot([trace])
    
get_pie_chart(df);
drop_and_encode_features(df);
def histograms(data, features, rows, columns):
    """
    Histograms of all features
    """
    fig = plt.figure(figsize=(20,20))
    for idx, feature in enumerate(features):
        ax = plt.subplot(rows, columns, idx+1)
        plt.hist(data[feature], bins='auto')
        ax.set_title(data.columns[idx], color='Red')

    plt.tight_layout()
    plt.show()

histograms(df, df.columns, 8, 4)
def mean_dist(data):   
    """
    Mean Distribution (Malignant vs. Benign)    
    """
    feature_means = list(data.columns[1:11])
    fig = plt.figure(figsize=(20,20))
    for idx, feature_mean in enumerate(feature_means):
        plt.subplot(5, 2, idx+1)
        sns.distplot(data[data['diagnosis']==1][feature_mean], label='Malignant', color='red')
        sns.distplot(data[data['diagnosis']==0][feature_mean], label='Benign', color='green')
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
mean_dist(df);
def correlation_heatmap(data):
    plt.figure(figsize=(25,18))
    sns.heatmap(data.corr(), annot=True, lw=0, fmt='.1f', cmap='Reds', linewidth=0.5)
    plt.title('Correlation Heatmap', fontsize=40, color='Blue')
    plt.tight_layout()

correlation_heatmap(df);
def feature_scaling(data):
    """
    Split dataset and standardize the dataset
    """
    y = data['diagnosis']
    x = data.drop('diagnosis', axis = 1)
    x = (x - x.mean()) / x.std()
    return x, y
def features_vs_diagnosis(data, column1, column2):
    x, y = feature_scaling(data);
    data_std = pd.concat([y, x.iloc[:, column1 : column2]], axis=1)
    data_std = pd.melt(data_std, id_vars="diagnosis", var_name="features", value_name='value')
    
    plt.figure(figsize=(10, 7))
    sns.violinplot(x="features", y="value", hue="diagnosis", data=data_std,split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
features_vs_diagnosis(df, 0, 10);
features_vs_diagnosis(df, 10, 20);
features_vs_diagnosis(df, 20, 30);
def get_data_split(data):
    """
    Train-test split (Train : Test = 75% : 25%)
    """
    X, y = feature_scaling(data);
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print("X train: ", X_train.shape)
    print("X test: ", X_test.shape)
    print("y train: ", y_train.shape)
    print("y test: ", y_test.shape)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_data_split(df);
def get_test_score(model, X_test, y_test):
    """
    Get test accuracy score for models
    """
    model_score = model.score(X_test, y_test)
    return model_score
# Logistic Regression Model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

print('Logistic Regression Model')
print('Training Accuracy Score:', logreg_model.score(X_train, y_train))
print('Test Accuracy Score:', logreg_model.score(X_test, y_test))

y_pred_logreg = logreg_model.predict(X_test)
print(classification_report(y_test, y_pred_logreg))
plot_roc_curve(logreg_model, X_test, y_test);
def conf_matrix(y_test, y_predict):
    """
    Plot a confusion matrix of models
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d')
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.show()
    return plt
# Confusion matrix of logistic regression
conf_matrix(y_test, y_pred_logreg);
rfecv_logreg = RFECV(estimator=logreg_model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv_logreg = rfecv_logreg.fit(X_train, y_train)

print('Optimal number of features in LogisticRegression:', rfecv_logreg.n_features_)
print('Best featuures in LogisticRegression:', X_train.columns[rfecv_logreg.support_])
def rfecv_grid_scores(model):
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(model.grid_scores_) + 1), model.grid_scores_);
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross validation score')
    plt.show()
rfecv_grid_scores(rfecv_logreg);
y_rfe_logreg = rfecv_logreg.predict(X_test)

print('Logistic Regression with RFECV')
print( 'Training Accuracy Score:', rfecv_logreg.score(X_train, y_train))
print( 'Test Accuracy Score:', rfecv_logreg.score(X_test, y_test))
print(classification_report(y_test, y_rfe_logreg))
# Confusion matrix of logistic regression with RFECV
conf_matrix(y_test, y_rfe_logreg);
# Random Forest Classification Model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

print('Random Forest Model')
print('Training Accuracy Score:', rf_model.score(X_train, y_train))
print('Test Accuracy Score:', rf_model.score(X_test, y_test))

y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))
plot_roc_curve(rf_model, X_test, y_test);
# Confusion matrix of random forest
conf_matrix(y_test, y_pred_rf);
rfecv_rf = RFECV(estimator=rf_model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv_rf = rfecv_rf.fit(X_train, y_train)

print('Optimal number of features in RandomForest:', rfecv_rf.n_features_)
print('Best featuures in RandomForest:', X_train.columns[rfecv_rf.support_])
rfecv_grid_scores(rfecv_rf);
y_rfe_rf = rfecv_rf.predict(X_test)

print('Logistic Regression with RFECV')
print( 'Training Accuracy Score:', rfecv_rf.score(X_train, y_train))
print( 'Test Accuracy Score:', rfecv_rf.score(X_test, y_test))
print(classification_report(y_test, y_rfe_rf))
# Confusion matrix of random forest with RFECV
conf_matrix(y_test, y_rfe_rf);