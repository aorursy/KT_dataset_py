import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



import seaborn as sns

sns.set_style("darkgrid")

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sns.palplot(sns.color_palette(flatui))
df = pd.read_csv('../input/Dataset_spine.csv', na_filter = True, skip_blank_lines = True)



# Naming Columns

df.columns = ['Pelvic Incidence', 'Pelvic Tilt', 'Lumbar Lordosis Angle', 'Sacral Slope', 'Pelvic Radius',

              'Degree Spondylolisthesis', 'Pelvic Slope', 'Direct Tilt', 'Thoracic Slope', 'Cervical Tilt',

              'Sacral Angle', 'Scoliosis Slope','Target', '13']

df.drop('13', axis = 1, inplace = True)

fig, ax = plt.subplots(figsize=(15,8), ncols=4, nrows=3)



left   =  0.125  # the left side of the subplots of the figure

right  =  0.9    # the right side of the subplots of the figure

bottom =  0.1    # the bottom of the subplots of the figure

top    =  0.9    # the top of the subplots of the figure

wspace =  .5     # the amount of width reserved for blank space between subplots

hspace =  1.1    # the amount of height reserved for white space between subplots



plt.subplots_adjust(

    left    =  left, 

    bottom  =  bottom, 

    right   =  right, 

    top     =  top, 

    wspace  =  wspace, 

    hspace  =  hspace

)



y_title_margin = 1.2



plt.suptitle("Distribution of Values - Normal vs Abnormal", y = 1.09, fontsize=15)



sns.violinplot(x = 'Target', y  = 'Pelvic Incidence', data = df, ax=ax[0][0], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Pelvic Tilt', data = df, ax=ax[0][1], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Lumbar Lordosis Angle', data = df, ax=ax[0][2], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Sacral Slope', data = df, ax=ax[0][3], palette = flatui)



# second row

sns.violinplot(x = 'Target', y  = 'Pelvic Radius', data = df, ax=ax[1][0], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Degree Spondylolisthesis', data = df, ax=ax[1][1], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Pelvic Slope', data = df, ax=ax[1][2], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Direct Tilt', data = df, ax=ax[1][3], palette = flatui)



# third row

sns.violinplot(x = 'Target', y  = 'Thoracic Slope', data = df, ax=ax[2][0], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Cervical Tilt', data = df, ax=ax[2][1], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Sacral Angle', data = df, ax=ax[2][2], palette = flatui)



sns.violinplot(x = 'Target', y  = 'Scoliosis Slope', data = df, ax=ax[2][3], palette = flatui)
# y

y = df['Target']

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

y = label.fit_transform(y)





# X

dfx = df.drop(['Target'], axis = 1)

X = dfx



# Splitting into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.decomposition import PCA, KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# FEATURE SELECTION

selection = SelectKBest(k=1)



#FEATURE EXTRACTION

pca = PCA(n_components = 2)

k_pca = KernelPCA(n_components = 2)



# FEATURE UNION (FEATURE SELECTION + FEATURE EXTRACTION)

estimators = [('sel', selection),

              ('pca', pca),

              ('k_pca', k_pca)]  



combined = FeatureUnion(estimators)



X_features = combined.fit(X, y).transform(X)
log_regression = LogisticRegression()

pipeline = Pipeline([("features", combined), ("log", log_regression)])
components = [1,2,3,4,5]

original = [1,2,3,4]

Cs = np.logspace(-4, 4, 3)

param_grid = dict(features__pca__n_components=components,

                  features__k_pca__n_components=components,

                  features__sel__k=original,

                  log__C=Cs)



grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)

grid_search.fit(X, y)
print("Best score: %0.3f" % grid_search.best_score_)

print("Best parameters set:")

best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
# FEATURE SELECTION

selection = SelectKBest(k=3)



#FEATURE EXTRACTION

pca = PCA(n_components = 5)

k_pca = KernelPCA(n_components = 1)



# FEATURE UNION (FEATURE SELECTION + FEATURE EXTRACTION)

estimators = [('sel', selection),

              ('pca', pca),

              ('k_pca', k_pca)]  

combined = FeatureUnion(estimators)



X_features = combined.fit(X, y).transform(X)
from sklearn.svm import SVC

svc = SVC(kernel = 'rbf', random_state = 0)



pipeline2 = Pipeline([("features", combined), ("svc", svc)])
Cs = np.logspace(-4, 4, 3)

kernels = ['rbf','poly']

param_grid = dict(svc__C=Cs,

                 svc__kernel=kernels)



grid_search = GridSearchCV(pipeline2, param_grid=param_grid, verbose=10)

grid_search.fit(X, y)
print("Best score: %0.3f" % grid_search.best_score_)

print("Best parameters set:")

best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
log_regression = LogisticRegression(C=10000)

final_model = Pipeline([("features", combined), ("log", log_regression )])
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = final_model, X = X_train, y = y_train, cv = 10)



avg_acc = accuracies.mean()

std_acc = accuracies.std()



print ("avg_acc: {} \nstd_acc: {}".format(avg_acc,std_acc))
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

import keras.backend as K



K.clear_session()



model = Sequential()

model.add(Dense(24, input_dim = 12, activation='relu'))

model.add(Dense(6, activation ='relu'))

model.add(Dense(1, activation ='sigmoid'))

model.compile(SGD(lr=0.5),'binary_crossentropy',metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs = 1000)
y_pred = model.predict(X_test)

y_class_pred = y_pred > 0.5
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,y_class_pred)

acc