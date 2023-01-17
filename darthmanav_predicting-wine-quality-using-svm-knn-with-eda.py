!pip install pycaret==2.0
!pip install mplcyberpunk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mplcyberpunk
from sklearn.datasets import load_wine
plt.style.use("cyberpunk")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import warnings
warnings.filterwarnings("ignore")
wine=load_wine()
df = pd.DataFrame(wine['data'],columns = wine['feature_names'])
df.head()
df = df.rename(columns={'od280/od315_of_diluted_wines': '% of diluted_wines'})
wine.keys()
y=wine['target']
y
df['label']=y
data_pycaret=df.copy()
df.head()
df.describe()
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)
plt.show()
df.isnull().sum() # checking for null values
## plotting the target values
sns.countplot(df['label'])
mplcyberpunk.add_glow_effects()
plt.show()
#Plot a boxplot to check for Outliers
#Target variable is Label. So will plot a boxplot each column against target variable
sns.boxplot('label', 'alcohol', data = df)
sns.boxplot('label', 'malic_acid', data = df)
sns.boxplot('label', 'ash', data = df)
sns.boxplot('label', 'alcalinity_of_ash', data = df)
sns.boxplot('label', 'magnesium', data = df)
sns.boxplot('label', 'color_intensity', data = df)
sns.boxplot('label', 'hue', data = df)
le = LabelEncoder()
df['label']=le.fit_transform(df['label'])
sc = StandardScaler()
x = sc.fit_transform(df.iloc[:,:-1])
X_train, X_test, y_train,y_test = train_test_split(df.iloc[:,:-1], y, test_size = 0.2,random_state=2)
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
print(classification_report(y_test,lr_predict))
print(confusion_matrix(y_test,lr_predict))
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
svc = SVC()
svc.fit(X_train, y_train)
y_pred=svc.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
mlp=MLPClassifier()
mlp.fit(X_train,y_train)
y_pred=mlp.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from pycaret.classification import *
wine= setup(data = data_pycaret, target = 'label',
            remove_outliers=True,
            session_id=1)
compare_models()