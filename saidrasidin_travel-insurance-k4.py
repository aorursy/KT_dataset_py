import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df_ti = pd.read_csv("../input/travel-insurance/travel insurance.csv")
df_ti.head()
df_ti.info()
clean_ti = df_ti.copy()
clean_ti.info()
#assign to category to save memory usage from 5+mb to 2+mb
categorical_col = ['Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Destination','Gender']
# clean_ti.drop(['Gender'], axis=1, inplace=True)
clean_ti['Gender'] = clean_ti['Gender'].fillna("not_disclosed")
clean_ti[categorical_col] = clean_ti[categorical_col].astype('category') 
clean_ti.info()
#look unique value for each columns
for col in categorical_col:
    uniq = len(clean_ti[col].unique())
    print(f'{col} :{uniq} Categories')
from sklearn.preprocessing import LabelEncoder
# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = clean_ti['Claim']
labels = le.fit_transform(clean_ti['Claim'])
clean_ti.describe()
clean_ti.hist(figsize=(20,10), grid = False, layout=(3,2), bins = 10);
clean_ti[clean_ti["Duration"] <0]
clean_ti[clean_ti["Age"] >100]
clean_ti.loc[clean_ti['Duration'] < 0, 'Duration'] = 49.317
clean_ti.loc[clean_ti['Age'] > 100, 'Age'] = 39.97
clean_ti.describe()
#numerical columns
test_ti = clean_ti.copy()
test_ti['Claim2'] = labels

plt.title("Pearson Correlation for Numerical Feature")
sns.heatmap(test_ti.corr(), annot=True)
test_ti.corr()
"""
source :https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
        https://www.kaggle.com/ayangupta/predict-the-claim
"""
import scipy.stats as ss
import numpy as np

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

categorical=['Agency', 'Agency Type', 'Distribution Channel', 'Product Name',  'Destination','Gender','Claim']
cramers=pd.DataFrame({i:[cramers_v(clean_ti[i],clean_ti[j]) for j in categorical] for i in categorical})
cramers['column']=[i for i in categorical if i not in ['memberid']]
cramers.set_index('column',inplace=True)

#categorical correlation heatmap
plt.figure(figsize=(10,7))
plt.title("Cramer's V Chi-Squared")
sns.heatmap(cramers,annot=True)
plt.show()
product_claim = pd.crosstab(clean_ti['Product Name'],clean_ti['Claim'],margins=True)
product_claim.drop(index=['All'],inplace=True)

plt.figure(figsize=(10, 7))
sns.barplot(product_claim.index, product_claim.Yes.values)
plt.xticks(rotation=90)
plt.title("Claim:Yes Per Product Name")
plt.show()
plt.figure(figsize=(10, 7))
sns.barplot(product_claim.index, product_claim.No.values)
plt.xticks(rotation=90)
plt.title("Claim:No Per Product Name")
plt.show()
#target columns
sns.countplot(clean_ti['Claim'])
plt.title("Target Label Distribution")
plt.grid(axis='y')
plt.show()
#reducing Target No due to severe imbalance
random_no = clean_ti[clean_ti['Claim']=='No'].sample(frac=1)
n_to_drop = len(random_no) - 10000

clean_reduce = clean_ti.drop(axis=0, index=random_no.index[:n_to_drop])

#target columns
sns.countplot(clean_reduce['Claim'])
plt.title("Target Label Distribution After Reducing")
plt.grid(axis='y')
plt.show()
from sklearn.model_selection import train_test_split
from collections import Counter


#split label and features
#One Hot Encoding for categorical data
X = clean_reduce.drop(columns=['Claim'])
X = pd.get_dummies(X, columns=categorical_col).values
# y = clean_ti['Claim'].replace(labels).values
y = clean_reduce['Claim'].replace({'No':0, 'Yes':1}).values
print(f'Datasets Features Size {X.shape}')

#X, y without Oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
print('Traininng shape %s' % Counter(y_train))
print('Testing shape %s' % Counter(y_test))
from imblearn.over_sampling import SMOTE

#with SMOTE
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_smote))
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def model_check(models, X_train, y_train):
    for name, model in models.items():
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1)
        print(f'{name} F1 score : {np.mean(score)}')

models = {'random_forest':RandomForestClassifier(), 
          'logistic_reg':LogisticRegression(), 
          'XGB':XGBClassifier(), 
          'GB':GradientBoostingClassifier()}

print("Without SMOTE")
model_check(models, X_train, y_train)
print()

print("With SMOTE")
model_check(models, X_smote, y_smote)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

model = RandomForestClassifier(n_jobs=-1, verbose=1)
param_dist = {'n_estimators':[300, 400, 500, 600], 'max_depth':[5,6,7,8]}

random = RandomizedSearchCV(model, param_dist, random_state=0, scoring='f1', n_jobs=-1, cv=3, verbose=1)
search = random.fit(X_smote, y_smote)
print('BEST PARAM', search.best_params_)
pd.DataFrame(search.cv_results_).sort_values(by='rank_test_score')
from sklearn.metrics import f1_score
model = RandomForestClassifier(n_estimators=400, max_depth=8)
score = cross_val_score(model,  X_smote, y_smote, cv=5, scoring='f1', n_jobs=-1)

print(f'Model F1 score : {np.mean(score)}')
model.fit(X_smote, y_smote)
from sklearn.metrics import plot_confusion_matrix, classification_report, f1_score

#using test_set
plot_confusion_matrix(model, X_test, y_test)
print(classification_report(y_test, model.predict(X_test)))
print(f1_score(y_test, model.predict(X_test)))
# import joblib

# joblib.dump(model, 'model_insurance_RF.pkl') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import keras.backend as K
#splitting for validation in training process
X_smote_train, X_val, y_smote_train, y_val = train_test_split(X_smote, y_smote, 
                                                              test_size=0.2, 
                                                              random_state=0)

#Standardization is very useful for deeplearning model to learn
scaler = StandardScaler()
X_smote_scaled = scaler.fit_transform(X_smote_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
batch_size = 1024

#calling dataset
def data_to_tensor(X, y, batch_size, shuffle=True):
  ds = tf.data.Dataset.from_tensor_slices((X, y))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(X))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

#datasets per batch
train_ds = data_to_tensor(X_smote_train, y_smote_train, batch_size=batch_size)
val_ds = data_to_tensor(X_val, y_val, batch_size=batch_size)

#model
def Model_ANN():
    model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_smote_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(.3),
            layers.Dense(1, activation='sigmoid')])
    
    optim =tf.keras.optimizers.Adam(learning_rate=1e-3) 

    model.compile(optimizer=optim,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['AUC'])
    
    return model

model_ann = Model_ANN()
model_ann.summary()
# def get_f1(y_true, y_pred): #taken from old keras source code
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     recall = true_positives / (possible_positives + K.epsilon())
#     f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
#     return f1_val
tf.keras.utils.plot_model(model_ann, show_shapes=True, rankdir="TB")
EPOCHS = 1000

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', 
                                                 factor=0.1, patience=10, 
                                                 verbose=0, mode='auto',
                                                 min_delta=0.0001)

history = model_ann.fit(train_ds, epochs=EPOCHS, 
                  validation_data=val_ds, 
                  callbacks=[early_stopping], verbose=1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], linestyle="--", label='Val Loss')
plt.legend()
plt.title("Training Loss")
plt.show()
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], linestyle="--", label='Val AUC')
plt.legend()
plt.title("Training AUC")
plt.show()
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_predict = model_ann.predict(X_test_scaled)>0.5

sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt="d")
print(classification_report(y_test, y_predict))