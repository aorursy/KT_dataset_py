# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA,KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

import warnings
warnings.filterwarnings("ignore")
    



# Importing the dataset
data = pd.read_csv('../input/winequality-red.csv')
data.head(5)

reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews

data.head()
x = data.iloc[:, 0:-2].values
y = data.iloc[:, -1].values
sc_x=StandardScaler()
x = sc_x.fit_transform(x)

pca=PCA()
x_pca = pca.fit_transform(x)
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.2, random_state = 1)
def classifier(model):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    score=accuracy_score(y_pred,y_test)
    return score*100
classifier(KNeighborsClassifier(n_neighbors=100)),classifier(RandomForestClassifier(n_estimators=100)),classifier(LogisticRegression()),classifier(GaussianNB())

