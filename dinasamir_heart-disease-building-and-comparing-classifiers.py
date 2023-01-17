import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier 
from sklearn.metrics import accuracy_score,roc_auc_score
#read data
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
#Get number of rows and columns of data
df.shape
df.info()
#get the statistics from data
df.describe()
#Get count and percentage for independant variable (target)
display(df.target.value_counts())
df.target.value_counts()/len(df)
sns.countplot(x='target',data=df)
#Get count and percentage for variable (sex)
display(df.sex.value_counts())
df.sex.value_counts()/len(df)
sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()
def get_ferq_visulaization(x, color , xlabel ,title, figsize= (8,6)):
    pd.crosstab(x,df.target).plot(kind="bar",figsize=figsize,color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()
#Heart Disease Frequency for age
get_ferq_visulaization(df.age,color=['#5bc0de','#d9534f'],xlabel='Age',title='Heart Disease Frequency for Ages',figsize=(15,8))
#Heart Disease Frequency for sex
get_ferq_visulaization(df.sex,color=['#11A5AA','#AA1190'],xlabel='Sex',title='Heart Disease Frequency for Sex')
#Heart Disease Frequency for fbs
get_ferq_visulaization(df.fbs,color=['#FFC300','#581845'],xlabel='fbs',title='Heart Disease Frequency for FBS')
get_ferq_visulaization(df.slope,color=['#11A5AA','#AA1190'],xlabel='Slope',title='Heart Disease Frequency for Slopes')
get_ferq_visulaization(df.cp,color=['#1CA53B','#AA1111'],xlabel='cp',title='Heart Disease Frequency for cp')
sns.pairplot(df)
#take copy from data and apply on in data preprocessing
df_copy=df.copy()
df_copy.shape
#remove outliers by removing data geater than 3 std
mean = df_copy[['chol','thalach','oldpeak']].mean()
std = df_copy[['chol','thalach','oldpeak']].std()
cut_off = std * 3
lower, upper = mean - cut_off, mean + cut_off
new_df = df_copy[(df_copy[['chol','thalach','oldpeak']] < upper) & (df_copy[['chol','thalach','oldpeak']] > lower)]
new_df.shape
#get one-hot-encoding to categorical variables
a = pd.get_dummies(df_copy['cp'], prefix = "cp")
b = pd.get_dummies(df_copy['thal'], prefix = "thal")
c = pd.get_dummies(df_copy['slope'], prefix = "slope")


# concat the one-hot-encoding variables with data
data = [df_copy, a, b, c]
df_copy = pd.concat(data, axis = 1)
df_copy = df_copy.drop(columns = ['cp', 'thal', 'slope'])
df_copy.head()
y=df_copy['target']
X=df_copy.drop(['target'],axis=1)
#split the data to train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
#scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#LogisticRegression
lr=LogisticRegression(random_state = 1)
# KNN Model
knn = KNeighborsClassifier(n_neighbors = 2)  
#DecisionTree model
dt = DecisionTreeClassifier(random_state=1)
# Random Forest Classification
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers[0:4])
# Instantiate a BaggingClassifier 'bc'
bc_knn = BaggingClassifier(base_estimator=knn, n_estimators=300, n_jobs=-1)
# Instantiate a BaggingClassifier 'bc'
bc_lr = BaggingClassifier(base_estimator=lr, n_estimators=300, n_jobs=-1)
# Instantiate a classification-tree 'dt' for AdaBoost
dt_adb = DecisionTreeClassifier(max_depth=1, random_state=1)
# Instantiate an AdaBoost classifier 'adab_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt_adb, n_estimators=100)
# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingClassifier(n_estimators=300, max_depth=1, random_state=1)

# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt),
               ('Random Forest',rf),
               ('Voting Classifier',vc),
               ('Bagging Classifier for knn',bc_knn),
               ('Bagging Classifier for logistic regression',bc_lr),
               ('AdaBoost Classifier',adb_clf),
               ('GradientBoosting Classifier',gbt)]

# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    #fit clf to the training set
    clf.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))

