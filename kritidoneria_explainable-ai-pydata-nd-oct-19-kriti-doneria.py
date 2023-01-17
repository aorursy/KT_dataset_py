#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
%matplotlib inline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

#Loading dataset
wine = pd.read_csv('../input/winequality-red.csv')
#Let's check how the data is distributed
wine.head()
#Information about the data columns
wine.info()
#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
#Composition of citric acid go higher as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)
#Composition of chloride also go down as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)
#Sulphates level goes higher with the quality of wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wine)
#Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)
wine.quality.describe()
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2,6,8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()
sns.countplot(wine['quality'])
#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine.quality
#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Applying Standard scaling to get optimized result
sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
#commented on purpose to demonstrate something
type(X_train)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
#Testing time
print(classification_report(y_test,pred_logreg))
#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_logreg))
import eli5
eli5.show_weights(logreg)
#uses permutation importance to compute feature weights
feat_names=wine.columns[:-1].tolist()
feat_names
eli5.show_weights(logreg, feature_names=feat_names)
import numpy as np
i = np.random.randint(1,100)
i
X_test.iloc[i]
y_test.iloc[i]
eli5.show_prediction(logreg, 
                     X_test.iloc[i],
                     feature_names=feat_names, show_feature_values=True)
rfc=RandomForestClassifier(n_estimators=200)
rfm = rfc.fit(X_train,y_train)
pred_rfc= rfc.predict(X_test)
#Testing time
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
#looking at eli5 first
eli5.show_weights(rfm, 
                  feature_names=feat_names)
from lime.lime_tabular import LimeTabularExplainer

X_train.head()
explainer = LimeTabularExplainer(X_train.values,
                                 mode="classification",
                                 feature_names=X_train.columns.tolist(),
                                 categorical_names=None,
                                 categorical_features=None,
                                 discretize_continuous=True,
                                 random_state=42)
prob=lambda x:rfm.predict_proba(X_test[[i]]).astype(float)

X_test.iloc[i]
#prediction function
pred_fn = lambda x: rfm.predict_proba(x).astype(float)
explanation = explainer.explain_instance(X_test.iloc[i], pred_fn)
explanation.show_in_notebook(show_table=True, show_all=False,)
print(explanation.score)
!pip install https://github.com/adebayoj/fairml/archive/master.zip
# Installing another package called fairML
from fairml import audit_model
importances, _ = audit_model(rfm.predict, X_test)
print(importances)
#inbuilt methods to visualize it
total, _ = audit_model(logreg.predict, X_test)
# print feature importance
print(total)

# generate feature dependence plot
fig = plot_dependencies(
    total.median(),
    reverse_values=False,
    title="FairML feature dependence"
)
plt.savefig("fairml_ldp.eps", transparent=False, bbox_inches='tight')