import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Loading dataset
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.info()
df['quality'].unique()
df1 = df.select_dtypes([np.int, np.float])

for i, col in enumerate(df1.columns):
    plt.figure(i)
    sns.barplot(x='quality',y=col, data=df1)
# quality > 6 is good and less is bad
bins = [2, 6.5, 8] 
print(bins)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
print(df['quality'])
# converting 'bad' and 'good' to labels 0 and 1
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
df.quality = encode.fit_transform(df.quality)
df['quality'].value_counts()
y_data = df['quality']
x_data = df
x_data.drop(['residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'quality'], axis=1, inplace=True)
#x_data.drop(['quality'], axis=1, inplace=True)
x_data.head()
y_data.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
sns.heatmap(cnf_matrix, annot=True)
plt.title("Confusion Matrix")
plt.show()
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

accuracy_score(y_test, y_pred)
from sklearn.svm import SVC

clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy_score(y_test, y_pred)
from sklearn.neighbors import KNeighborsClassifier
k = 2
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
yhat = neigh.predict(x_test)

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
print("Test set Accuracy (real acc): ", metrics.accuracy_score(y_test, yhat))

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
model.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier()
train_x = x_train
train_y = y_train
test_x = x_test
test_y = y_test
# fit the model with the training data
model.fit(train_x,train_y)

# number of trees used
print('Number of Trees used : ', model.n_estimators)

# predict the target on the train dataset
predict_train = model.predict(train_x)
print('\nTarget on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(test_x)
print('\nTarget on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)