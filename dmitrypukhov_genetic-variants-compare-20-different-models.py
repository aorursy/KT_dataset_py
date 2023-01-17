import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Set random seed to make results reproducable
np.random.seed(42)
plt.style.use('seaborn')
df = pd.read_csv('../input/clinvar_conflicting.csv',dtype={0: object, 38: str, 40: object})
df.fillna(0,inplace=True)
df.head()
# Features histograms
df.drop('CLASS',axis=1).hist(figsize=(12,7))
plt.suptitle("Features histograms", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
sns.countplot(x='CLASS',data=df)
plt.title("Target label histogram")
plt.show()
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seed to make results reproducable
np.random.seed(42)

# Balance
g = df.groupby('CLASS')
df_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
# Extract smaller sample to avoid memory error later, when training starts
df_balanced = df_balanced.sample(1000)

# Illustrate balancing results on plots
f, ax = plt.subplots(1,2)
# Before balancing plot
df.CLASS.value_counts().plot(kind='bar', ax=ax[0])
ax[0].set_title("Before")
ax[0].set_xlabel("CLASS value")
ax[0].set_ylabel("Count")
# After balanced plot
df_balanced.CLASS.value_counts().plot(kind='bar',ax=ax[1])
ax[1].set_title("After")
ax[1].set_xlabel("CLASS value")
ax[1].set_ylabel("Count")

plt.suptitle("Balancing data by CLASS column value")
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()
# Features - all columns except 'CLASS'
# Target label = 'CLASS' column
X=df_balanced.drop('CLASS',axis=1)
# One hot encoding
X=pd.get_dummies(X, drop_first=True)
y=df_balanced['CLASS']
y=pd.get_dummies(y, drop_first=True)

# Train/test split
train_X, test_X, train_y, test_y = train_test_split(X, y)

# Normalize using StandardScaler
scaler=StandardScaler()
train_X=scaler.fit_transform(train_X)
test_X=scaler.transform(test_X)

# Histogram of target labels distribution
test_y.hist()
plt.title("Target feature distribution: CLASS values")
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()
# Models to try
models = [LogisticRegression(), 
          LogisticRegressionCV(), 
          PassiveAggressiveClassifier(),
          RidgeClassifier(),
          RidgeClassifierCV(),
          KNeighborsClassifier(),
          #RadiusNeighborsClassifier(),
          NearestCentroid(),
          DecisionTreeClassifier(), 
          AdaBoostClassifier(), 
          BaggingClassifier(),
          ExtraTreesClassifier(),
          GradientBoostingClassifier(),
          RandomForestClassifier(), 
          SGDClassifier(),
          GaussianNB(),
          GaussianProcessClassifier(),
          LinearDiscriminantAnalysis(),
          QuadraticDiscriminantAnalysis(),
          MLPClassifier(),
          SVC()
         ]
# Gather metrics here
accuracy_by_model={}

# Train then evaluate each model
for model in models:
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    score = accuracy_score(test_y, pred_y)
    # Fill metrics dictionary
    model_name = model.__class__.__name__
    accuracy_by_model[model_name]=score  

# Draw accuracy by model chart
acc_df = pd.DataFrame(list(accuracy_by_model.items()), columns=['Model', 'Accuracy']).sort_values('Accuracy', ascending=False).reset_index(drop=True)
acc_df.index=acc_df.index+1
sns.barplot(data=acc_df,y='Model',x='Accuracy')
plt.xlim(0,1)
plt.title('Accuracy of models with default settings')
plt.xticks(rotation=45)
plt.show()

# Print table
acc_df
best_model = acc_df[acc_df.Accuracy==acc_df.Accuracy.max()]
best_model