import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning) # ignoring Future Warning to clean up notebook
sns.set_style('whitegrid')



# to print summary of cross validation score

def score_summary(scores):

    print ('cv_scores: ',scores)

    print ('mean: ',np.mean(scores))

    print ('std. deviation: ',np.std(scores))



# to save visualisations as png images

def save_figure(fig_id):

    path = os.path.join('iris_images_' + fig_id + '.png')

    plt.savefig(path, dpi=300, format='png')
df = pd.read_csv('../input/iris/Iris.csv')
df.head()
df.info()
df.describe()
df['Species'].value_counts()
sns.pairplot(df, height=4, hue = 'Species')

save_figure('1')
plt.figure(figsize=(10,6))

sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')

save_figure('2')
plt.figure(figsize=(10,6))

sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species')

save_figure('3')
# raw correlation between features

df.corr()
# plotting heatmap to visualise correlation among features

plt.figure(figsize=(10,6))

ax = sns.heatmap(df.corr(), annot=True, cmap='Greens')

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)  # to correct glitch in seaborn that clips the matrix from top and bottom

save_figure('4')
df = df.drop('Id', axis=1)



X = df.drop('Species', axis=1)

y = df['Species']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Species'])
from sklearn.preprocessing import MinMaxScaler 



scaler = MinMaxScaler()

scaler.fit_transform(X_train)

scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



log_reg_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)

log_reg_clf.fit(X_train, y_train)

log_reg_clf_scores = cross_val_score(log_reg_clf, X_train, y_train, cv=5)

score_summary(log_reg_clf_scores)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier()

sgd_clf.fit(X_train, y_train)

sgd_clf_scores = cross_val_score(sgd_clf, X_train, y_train, cv=5)

score_summary(sgd_clf_scores)
from sklearn.svm import SVC

svc_clf = SVC(gamma='auto')

svc_clf.fit(X_train, y_train)

svc_clf_scores = cross_val_score(svc_clf, X_train, y_train, cv=5)

score_summary(svc_clf_scores)
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)

dt_clf_scores = cross_val_score(dt_clf, X_train, y_train, cv=5)

score_summary(dt_clf_scores)
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100)

rf_clf.fit(X_train, y_train)

rf_clf_scores = cross_val_score(rf_clf, X_train, y_train, cv=5)

score_summary(rf_clf_scores)
from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(n_estimators=100, max_samples=0.8)

bag_clf.fit(X_train, y_train)

bag_clf_scores = cross_val_score(bag_clf, X_train, y_train, cv=5)

score_summary(bag_clf_scores)
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()

gb_clf.fit(X_train, y_train)

gb_clf_scores = cross_val_score(gb_clf, X_train, y_train, cv=5)

score_summary(gb_clf_scores)
from xgboost import XGBClassifier

xgbc_clf  = XGBClassifier(n_estimators=500, objective='multi:softmax')

xgbc_clf.fit(X_train, y_train)

xgbc_clf_scores = cross_val_score(xgbc_clf, X_train, y_train, cv=5)

score_summary(xgbc_clf_scores)
from sklearn.model_selection import GridSearchCV



param_grid = {

    'C':[8,9,10, 11, 12, 15 ],

    'gamma':[ .01,.02, 0.05, .001]

}



grid = GridSearchCV(svc_clf, param_grid, cv=30)

grid.fit(X_train, y_train)
grid.best_params_
best_model = grid.best_estimator_
from sklearn.metrics import classification_report, confusion_matrix

best_pred = best_model.predict(X_test)

print(confusion_matrix(y_test, best_pred))

print(classification_report(y_test, best_pred))
# changing categorical feature (species) to numerical for ANN 

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

encoder = LabelEncoder()

encoder.fit(y_train)

encoded_y_train = encoder.transform(y_train)

dummy_y_train = np_utils.to_categorical(encoded_y_train)



encoder.fit(y_test)

encoded_y_test = encoder.transform(y_test)

dummy_y_test = np_utils.to_categorical(encoded_y_test)
dummy_y_train.shape
dummy_y_test.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.activations import elu, relu, sigmoid, softmax



dl_model = Sequential()



dl_model.add(Dense(units=4, activation='elu'))

dl_model.add(Dense(units=8, activation='elu'))

dl_model.add(Dense(units=3, activation='softmax'))



dl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
dl_model.fit(x=X_train.values, y=dummy_y_train, epochs=1000, verbose=0)
dl_model_loss = pd.DataFrame(dl_model.history.history)

dl_model_loss.plot(figsize=(10,6))
from sklearn.metrics import classification_report, confusion_matrix

dl_pred = dl_model.predict(X_test.values)

dl_pred.shape
confusion_matrix(np.argmax(dummy_y_test,axis=1), np.argmax(dl_pred,axis=1))
print(classification_report(np.argmax(dummy_y_test,axis=1), np.argmax(dl_pred,axis=1)))
from sklearn.pipeline import Pipeline



some_data = [[3.1, 3.1, 1.4, 0.8]]



pipeline = Pipeline([

    ('min_max_scaler', MinMaxScaler()),

    ('svc', grid.best_estimator_)

])



model = pipeline.fit(X_train, y_train)

pipeline.predict(some_data)
import joblib

from keras.models import load_model



joblib.dump(model, 'iris_model.pkl')

#loaded_model = joblib.load('iris_model.pkl')



dl_model.save('iris_dl_model.h5')

#loaded_dl_model = load_model('iris_dl_model.h5')
clfs_pipeline = Pipeline([

    ('normalizer', MinMaxScaler()),

    ('clf', LogisticRegression())

])



clfs = [LogisticRegression(), SGDClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(),

       BaggingClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for classifier in clfs:

    clfs_pipeline.set_params(clf=classifier)

    clfs_score = cross_val_score(clfs_pipeline, X_train, y_train, cv=5)

    print('--------------------------------------------------------------------')

    print(str(classifier))

    score_summary(clfs_score)