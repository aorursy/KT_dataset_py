import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
df = pd.read_csv(r'../input/iris-dataset/Iris.csv')
df
df.info()
df.shape
target = df['Species']
df1 = df.copy()
df1 = df1.drop('Species' , axis = 1)
df1.shape
df.isnull().sum()
sns.pairplot(df , hue='Species')
sns.heatmap(df.corr()) 
X = df.iloc[: , [0,1,2,3]].values

le = LabelEncoder()

df['Species'] = le.fit_transform(df['Species'])

y = df['Species'].values

df.shape
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
X_train.shape
X_test.shape
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
print('Classification Report \n' , classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test , y_pred)
confusion_matrix(y_test , y_pred)
cm = confusion_matrix(y_test , y_pred)
plt.figure(figsize=(9,9))

sns.heatmap(cm , annot=True , fmt='.3f' , 
            linewidths=.5 , 
            square=True,
            cmap='Blues');

plt.xlabel('Original Label');
plt.ylabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(clf.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
from sklearn.tree import plot_tree
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 80 , 50

print(plot_tree(clf))
plt.figure(figsize = (20,20))
dec_tree = plot_tree(clf, feature_names = df1.columns, 
                     class_names = target.values, filled = True , precision = 4, rounded = True);
