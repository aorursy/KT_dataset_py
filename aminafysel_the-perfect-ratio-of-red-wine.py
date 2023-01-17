import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/red-wine-quality/winequality-red.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df.shape
df['quality'].value_counts()
import seaborn as sns

sns.countplot(x = 'quality',data = df)
f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax,annot=True)
new_df=df.replace(0,np.NaN)
new_df.isnull().sum()
new_df["citric acid"].fillna(new_df["citric acid"].mean(), inplace = True)
new_df.describe().T
# converting the response variables(3-7) as binary response variables that is either good or bad



names = ['bad', 'good']

bins = (2, 6.5, 8)



df['quality'] = pd.cut(df['quality'], bins = bins, labels = names)



df['quality'].value_counts()
#We have now labelled the quality into good and bad,now to convert them into numerical values



from sklearn.preprocessing import LabelEncoder

label_quality=LabelEncoder()

df['quality']= label_quality.fit_transform(df['quality'])

df['quality'].value_counts()
sns.countplot(df['quality'])
#FeatureSelection

X=df.iloc[:,:11].values

y=df.iloc[:,11].values

#splitting X and y

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.20, random_state = 44 )
#Checking dimensions

print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)

print("y_test shape:", y_test.shape)
# standard scaling 

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

# Logistic Regression Algorithm

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state = 44)

logreg.fit(X_train, y_train)



# Support Vector Classifier Algorithm

from sklearn.svm import SVC

svc = SVC(kernel = 'linear', random_state = 44)

svc.fit(X_train, y_train)
# Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

# Decision tree Algorithm

from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 44)

dectree.fit(X_train, y_train)

# Random forest Algorithm

from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 44)

ranfor.fit(X_train, y_train)

# Making predictions on test dataset

Y_pred_logreg = logreg.predict(X_test)

#Y_pred_knn = knn.predict(X_test)

Y_pred_svc = svc.predict(X_test)

Y_pred_nb = nb.predict(X_test)

Y_pred_dectree = dectree.predict(X_test)

Y_pred_ranfor = ranfor.predict(X_test)
# Evaluating using accuracy_score metric

from sklearn.metrics import accuracy_score

accuracy_logreg = accuracy_score(y_test, Y_pred_logreg)

#accuracy_knn = accuracy_score(y_test, Y_pred_knn)

accuracy_svc = accuracy_score(y_test, Y_pred_svc)

accuracy_nb = accuracy_score(y_test, Y_pred_nb)

accuracy_dectree = accuracy_score(y_test, Y_pred_dectree)

accuracy_ranfor = accuracy_score(y_test, Y_pred_ranfor)
# Accuracy on test set

print("Logistic Regression: " + str(accuracy_logreg * 100))

#print("K Nearest neighbors: " + str(accuracy_knn * 100))

print("Support Vector Classifier: " + str(accuracy_svc * 100))

print("Naive Bayes: " + str(accuracy_nb * 100))

print("Decision tree: " + str(accuracy_dectree * 100))

print("Random Forest: " + str(accuracy_ranfor * 100))
# Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, Y_pred_ranfor)

cm
# Heatmap of Confusion matrix

sns.heatmap(pd.DataFrame(cm), annot=True)
# Classification report

from sklearn.metrics import classification_report

print(classification_report(y_test, Y_pred_ranfor))