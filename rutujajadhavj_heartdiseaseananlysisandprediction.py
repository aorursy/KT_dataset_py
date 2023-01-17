# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
data=pd.read_csv('../input/heart-disease-uci/heart.csv')
print('Data First 5 Rows Show\n')
data.head()
data.isnull().values.any()
data.describe()
ref_col = ["age",
"sex",
"chest pain type",
"resting blood pressure",
"serum cholestoral in mg/dl",
"fasting blood sugar > 120 mg/dl",
"resting electrocardiographic results (values 0,1,2)",
"maximum heart rate achieved",
"exercise induced angina",
"oldpeak = ST depression induced by exercise relative to rest",
"the slope of the peak exercise ST segment",
"number of major vessels (0-3) colored by flourosopy",
"thal(3 = normal; 6 = fixed defect; 7 = reversable defect)"]


ref_dictionary = dict()
i = 0
for col in data.columns[:13] :
        ref_dictionary[col] = ref_col[i]
        i+=1

print(ref_dictionary)
data.info()
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()
#plot correlation between input features and target value

data.drop('target', axis=1).corrwith(data.target).plot(kind='bar', grid=True, figsize=(12, 8), 
                                                   title="Correlation with target")
sns.pairplot(data)
plt.show()

# Create another figure
plt.figure(figsize=(10, 8))

# Scatter with postivie examples
plt.scatter(data.age[data.target==1],
            data.thalach[data.target==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data.age[data.target==0],
            data.thalach[data.target==0],
            c="lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);

plt.figure(figsize=(10, 8))

# Scatter with postivie examples
plt.scatter(data.chol[data.target==1],
            data.trestbps[data.target==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data.chol[data.target==0],
            data.trestbps[data.target==0],
            c="lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Cholestrol and Resting blood pressure")
plt.xlabel("Cholestrol")
plt.ylabel("Resting blood pressure")
plt.legend(["Disease", "No Disease"]);
sns.countplot('sex', hue="sex", data=data, palette="bwr")
hist_data = [data['age']]
sns.distplot(hist_data)

sns.swarmplot(x="target", y="age", data=data, color=".25")
#outlier analysis through PCA

pca = PCA(2)  
projected = pca.fit_transform(data)

plt.scatter(projected[:, 0], projected[:, 1],c=data.target) 
# label 0 is no disease 1 is disease
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        

        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
from sklearn.model_selection import train_test_split
y = data.target
X = data.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #30-70 division
from sklearn.linear_model import LogisticRegression
results_df = []
log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_train, y_train)
print_score(log_reg, X_train, y_train, X_test, y_test, train=True)
print_score(log_reg, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, log_reg.predict(X_test)) * 100
train_score = accuracy_score(y_train, log_reg.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
from sklearn.tree import DecisionTreeClassifier


tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

print_score(tree, X_train, y_train, X_test, y_test, train=True)
print_score(tree, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, tree.predict(X_test)) * 100
train_score = accuracy_score(y_train, tree.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Decision Tree Classifier", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rand_forest = RandomForestClassifier(n_estimators=1000, random_state=42)
rand_forest.fit(X_train, y_train)

print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)
print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, rand_forest.predict(X_test)) * 100
train_score = accuracy_score(y_train, rand_forest.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Random Forest Classifier", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

print_score(xgb, X_train, y_train, X_test, y_test, train=True)
print_score(xgb, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, xgb.predict(X_test)) * 100
train_score = accuracy_score(y_train, xgb.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["XGBoost Classifier", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)

params = {"criterion":("gini", "entropy"), 
          "splitter":("best", "random"), 
          "max_depth":(list(range(1, 20))), 
          "min_samples_split":[2, 3, 4], 
          "min_samples_leaf":list(range(1, 20))
          }

tree = DecisionTreeClassifier(random_state=42)
grid_search_cv = GridSearchCV(tree, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3, iid=True)
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=3,
                              min_samples_leaf=2, 
                              min_samples_split=2, 
                              splitter='random')
tree.fit(X_train, y_train)

print_score(tree, X_train, y_train, X_test, y_test, train=True)
print_score(tree, X_train, y_train, X_test, y_test, train=False)
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster = ['gbtree', 'gblinear']
base_score = [0.25, 0.5, 0.75, 0.99]
learning_rate = [0.05, 0.1, 0.15, 0.20]
min_child_weight = [1, 2, 3, 4]

hyperparameter_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,
                       'learning_rate' : learning_rate, 'min_child_weight' : min_child_weight, 
                       'booster' : booster, 'base_score' : base_score
                      }

xgb_model = XGBClassifier()

xgb_cv = RandomizedSearchCV(estimator=xgb_model, param_distributions=hyperparameter_grid,
                               cv=5, n_iter=650, scoring = 'accuracy',n_jobs =-1, iid=True,
                               verbose=1, return_train_score = True, random_state=42)
xgb_best = XGBClassifier(base_score=0.25, 
                         booster='gbtree',
                         learning_rate=0.05, 
                         max_depth=5,
                         min_child_weight=2, 
                         n_estimators=100)
xgb_best.fit(X_train, y_train)

print_score(xgb_best, X_train, y_train, X_test, y_test, train=True)
print_score(xgb_best, X_train, y_train, X_test, y_test, train=False)

results_df
import shap
import numpy as np
shap.initjs()
np.random.seed(27)
tf.random.set_seed(27)
df_train = data.dropna()
df_target = df_train.pop('target')

X_train, X_test, y_train, y_test = train_test_split(df_train, df_target, test_size=0.2, random_state=42)

normalized_X_train=(X_train-X_train.mean())/X_train.std()
normalized_X_test=(X_test-X_train.mean())/X_train.std()
def build_model():
    inputs = keras.Input(shape=(len(X_train.keys()),))
    x = layers.Dense(64, activation='tanh')(inputs) 
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(10, activation='tanh')(x)
    x = layers.Dense(5, activation='tanh')(x)
    x = layers.Dense(3, activation='tanh')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name="death_rate_model")

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
  
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)   

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
            
        print('.', end='')
model = build_model()
model.fit(X_train, y_train, epochs=3000, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
print("")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
kernel_explainer = shap.KernelExplainer(model.predict, X_test)
shap_values = kernel_explainer.shap_values(X_test, nsamples=100, l1_reg="aic")[0]
for i in np.random.choice(range(len(X_test)),5):
    x = shap.force_plot(kernel_explainer.expected_value, shap_values[i], X_train.iloc[i])
    display(x)
shap.force_plot(kernel_explainer.expected_value, shap_values, X_test)

shap.summary_plot(shap_values, X_test)

import joblib 
filename = 'finalized_model.sav'
joblib.dump(log_reg, filename)
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)

print(result)
#sklearn pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(LogisticRegression())
pipe.fit(X_train, y_train)
joblib.dump(pipe, 'model.pkl')
pipe = joblib.load('model.pkl')

# New data to predict
pipe.predict(X_test)
#pred_cols = list(pr.columns.values)[:-1]

# apply the whole pipeline to data
#pred = pd.Series(pipe.predict(pr[pred_cols]))
#print (pred)