import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sklearn
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression,Perceptron,SGDClassifier
from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report,make_scorer,roc_curve, auc
from sklearn.model_selection import train_test_split
from category_encoders.one_hot import OneHotEncoder
import keras

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
numeric_features = ['age','trestbps','thalach','oldpeak','chol']
categorical_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
df[numeric_features].describe()
numerics = df[numeric_features]
fig, ax = plt.subplots(5,1,figsize=(22, 30))
for i, col in enumerate(numerics):
    plt.subplot(5,1,i+1)
    plt.xlabel(col, fontsize=10)
    sns.distplot(numerics[col].values)
plt.show() 
corr = numerics.corr()
sns.heatmap(corr,annot=True,fmt='.3f',linewidths=2)
plt.title('Correlation Matrix')
plt.gcf().set_size_inches(11,7)
plt.show()
categorical = df[categorical_features]
fig, axes = plt.subplots(round(len(categorical.columns) / 4), 3, figsize=(22, 10))

for i, ax in enumerate(fig.axes):
    if i < len(categorical.columns):
        sns.countplot(x=categorical.columns[i], data=categorical, ax=ax)

fig.tight_layout()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(18,5))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('N')
plt.show()

plt.subplots(figsize=(18, 5))
plt.title('Heart Disease Distribution for Ages')
sns.distplot(df[df['target'] == 1]['age'], label="Target = 1 ")
sns.distplot(df[df['target'] == 0]['age'], label="Target = 0 ")
plt.legend()
sns.set(style="ticks")
sns.pairplot(df)
plt.show()
pd.options.mode.chained_assignment = None  # default='warn'

scaler = sklearn.preprocessing.StandardScaler().fit(df[numeric_features])
df[numeric_features]= scaler.transform(df[numeric_features])

df['sex'][df['sex'] == 0] = 'female'
df['sex'][df['sex'] == 1] = 'male'

df['cp'][df['cp'] == 1] = 'typical angina'
df['cp'][df['cp'] == 2] = 'atypical angina'
df['cp'][df['cp'] == 3] = 'non-anginal pain'
df['cp'][df['cp'] == 4] = 'asymptomatic'

df['fbs'][df['fbs'] == 0] = 'lower than 120mg/ml'
df['fbs'][df['fbs'] == 1] = 'greater than 120mg/ml'

df['restecg'][df['restecg'] == 0] = 'normal'
df['restecg'][df['restecg'] == 1] = 'ST-T wave abnormality'
df['restecg'][df['restecg'] == 2] = 'left ventricular hypertrophy'

df['exang'][df['exang'] == 0] = 'no'
df['exang'][df['exang'] == 1] = 'yes'

df['slope'][df['slope'] == 1] = 'upsloping'
df['slope'][df['slope'] == 2] = 'flat'
df['slope'][df['slope'] == 3] = 'downsloping'

df['thal'][df['thal'] == 1] = 'normal'
df['thal'][df['thal'] == 2] = 'fixed defect'
df['thal'][df['thal'] == 3] = 'reversable defect'

df = pd.get_dummies(df, drop_first=True)
df_eda = df.copy()

dummies = OneHotEncoder(cols= 'ca',use_cat_names=True)
dummies.fit(df)
df = dummies.transform(df)
df.head()
#Split data in train and test
y = df[['target']]
x = df.drop(['target'],axis = 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
import os
from IPython.display import SVG
from graphviz import Source
import itertools

tree = DecisionTreeClassifier(max_depth=3)
tree.fit(x,y)
graph = Source(export_graphviz(tree
                               , feature_names=x.columns
                               , filled = True))
display(SVG(graph.pipe(format='svg')))
pd.crosstab(df['thal_fixed defect'],df.target).plot(kind="bar",figsize=(18,5))
plt.title('Thal Fixed defect x Target')
plt.xlabel('thal_fixed defect')
plt.ylabel('N')
plt.show()
pd.crosstab(df_eda['ca'],df_eda.target).plot(kind="bar",figsize=(18,5))
plt.title('Ca x Target')
plt.xlabel('ca')
plt.ylabel('N')
plt.show()
lr = LogisticRegression(class_weight = 'balanced', solver = 'liblinear',penalty="l2")
lr.fit(x_train,y_train)

y_prob = lr.predict_proba(x_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(10,8))
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

print("acc:",accuracy_score(y_test,y_pred))
print("recall:",recall_score(y_test,y_pred))
#Transform the data in float and print the number of features
x_train = x_train.astype(float) 
x_test = x_test.astype(float) 


y_train = to_categorical(y_train,2) 
y_test =  to_categorical(y_test,2) 

x_train.shape, x_test.shape, y_train.shape , y_test.shape
#Starting Neural network
model = Sequential()


#First hidden
model.add(Dense(5 #number of neurals in the first hidden
                ,activation = 'relu' 
                ,input_shape = (23,) #Number of features that my model will receive
                ))



#out hidden
model.add(Dense(2 #number of class
                ,activation = 'softmax' #its will show me the probrably in the each class
                ))


#summary the model
model.summary()

#Compile the first Neural Network
model.compile(
                loss='categorical_crossentropy' 
               ,optimizer='adam' 
               ,metrics=['accuracy'] 
)

history = model.fit(x_train,y_train
         ,epochs=100
         ,batch_size =32
         ,verbose = 1
         ,validation_data=(x_test,y_test)
         )
#Predict the x_test
p = model.predict(x_test)
p = (p > 0.5)
print('ACC: %.3f%%' % (accuracy_score(y_test, p)*100))
print('---------')
print(classification_report(y_test, p))
plt.subplots(figsize=(13, 8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.subplots(figsize=(13, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()