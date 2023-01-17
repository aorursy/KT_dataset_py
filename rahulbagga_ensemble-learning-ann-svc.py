import math

import numpy as np

import pandas as pd

import logging

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.model_selection import KFold

from sklearn.naive_bayes import GaussianNB, MultinomialNB,  BernoulliNB

from sklearn.metrics import accuracy_score, log_loss,jaccard_similarity_score

from sklearn.preprocessing import normalize

from sklearn import svm

from sklearn.model_selection import GridSearchCV

#xboost and heamy

import xgboost as xgb

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from heamy.dataset import Dataset

from heamy.estimator import Classifier

from heamy.pipeline import ModelsPipeline

#Keras modules

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import to_categorical

from keras.datasets import mnist

from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

from keras.utils import np_utils
DATA_DIR = "../input"

SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)

TEST_FILE = "{0}/test.csv".format(DATA_DIR)

id_test = pd.read_csv('../input/test.csv').Id
def plot_distplots(train_path, colum_num_from=1, colum_num_to=11):

    train = pd.read_csv(train_path)

    _ = plt.figure(figsize=(20, 20))

    i = 0

    for feature in train.columns[colum_num_from:colum_num_to]:

        i += 1

        plt.subplot(5, 5, i)

        sns.distplot(train[train.Cover_Type == 1][feature], hist=False, label='1')

        sns.distplot(train[train.Cover_Type == 2][feature], hist=False, label='2')

        sns.distplot(train[train.Cover_Type == 3][feature], hist=False, label='3')

        sns.distplot(train[train.Cover_Type == 4][feature], hist=False, label='4')

        sns.distplot(train[train.Cover_Type == 5][feature], hist=False, label='5')

        sns.distplot(train[train.Cover_Type == 6][feature], hist=False, label='6')

        sns.distplot(train[train.Cover_Type == 7][feature], hist=False, label='7')
plot_distplots(TRAIN_FILE)
CACHE = False

NFOLDS = 5

SEED = 1337



#cross_entropy loss

#-(yt log(yp) + (1 - yt) log(1 - yp))

METRIC = log_loss



ID = 'Id'

TARGET = 'Cover_Type'



#set precision to 5 decimal places

np.set_printoptions(precision=5)

np.set_printoptions(suppress=True)



#seed value is set

np.random.seed(SEED)

logging.basicConfig(level=logging.WARNING)
def add_feats(df):

    #Hydrology - Fire Points

    df['HF1'] = (df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points'])

    df['HF2'] = (df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])

    

    #Hydrology - Roadways

    df['HR1'] = (df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])

    df['HR2'] = (df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])

    

    #Firepoints - Roadways

    df['FR1'] = (df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])

    df['FR2'] = (df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])

    

    #Elevation & Vertical_Distance_To_Hydrology

    df['EV1'] = (df['Elevation'] + df['Vertical_Distance_To_Hydrology'])

    df['EV2'] = (df['Elevation'] - df['Vertical_Distance_To_Hydrology'])

    

    #Mean of created features

    df['Mean_HF1'] = df.HF1 / 2

    df['Mean_HF2'] = df.HF2 / 2

    df['Mean_HR1'] = df.HR1 / 2

    df['Mean_HR2'] = df.HR2 / 2

    df['Mean_FR1'] = df.FR1 / 2

    df['Mean_FR2'] = df.FR2 / 2

    df['Mean_EV1'] = df.EV1 / 2

    df['Mean_EV2'] = df.EV2 / 2    

    

    #Oblique Distance

    df['Elevation_Vertical'] = df['Elevation'] + df['Vertical_Distance_To_Hydrology']    

    df['Neg_Elevation_Vertical'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']

    

    #Given the horizontal & vertical distance to hydrology, 

    #it will be more intuitive to obtain the euclidean distance: sqrt{(verticaldistance)^2 + (horizontaldistance)^2}    

    df['slope_hyd_sqrt'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5

    

    #remove infinite value if any

    df['slope_hyd_sqrt'] = df.slope_hyd_sqrt.map(lambda x: 0 if np.isinf(x) else x)

    

    df['slope_hyd2'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)

    df['slope_hyd2'] = df.slope_hyd2.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    

    #Mean distance to Amenities 

    df['Mean_Amenities'] = (df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 

    

    #Mean Distance to Fire and Water 

    df['Mean_Fire_Hyd1'] = (df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2

    df['Mean_Fire_Hyd2'] = (df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Roadways) / 2

    

    #Shadiness

    df['Shadiness_morn_noon'] = df.Hillshade_9am / (df.Hillshade_Noon+1)

    df['Shadiness_noon_3pm'] = df.Hillshade_Noon / (df.Hillshade_3pm+1)

    df['Shadiness_morn_3'] = df.Hillshade_9am / (df.Hillshade_3pm+1)

    df['Shadiness_morn_avg'] = (df.Hillshade_9am + df.Hillshade_Noon)/2

    df['Shadiness_afternoon'] = (df.Hillshade_Noon + df.Hillshade_3pm)/2

    df['Shadiness_mean_hillshade'] =  (df['Hillshade_9am']  + df['Hillshade_Noon'] + df['Hillshade_3pm'] ) / 3    

    

    #Shade Difference

    df["Hillshade-9_Noon_diff"] = df["Hillshade_9am"] - df["Hillshade_Noon"]

    df["Hillshade-noon_3pm_diff"] = df["Hillshade_Noon"] - df["Hillshade_3pm"]

    df["Hillshade-9am_3pm_diff"] = df["Hillshade_9am"] - df["Hillshade_3pm"]



    # Mountain Trees

    df["Slope*Elevation"] = df["Slope"] * df["Elevation"]

    # Only some trees can grow on steep montain

    

    ### More features

    df['Neg_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])

    df['Neg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])

    df['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])

    

    df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])/2

    df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])/2

    df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])/2   

    df["Vertical_Distance_To_Hydrology"] = abs(df['Vertical_Distance_To_Hydrology'])

    

    df['Neg_Elev_Hyd'] = df.Elevation-df.Horizontal_Distance_To_Hydrology*0.2

    

    # Bin Features

    bin_defs = [

        # col name, bin size, new name

        ('Elevation', 200, 'Binned_Elevation'), # Elevation is different in train vs. test!?

        ('Aspect', 45, 'Binned_Aspect'),

        ('Slope', 6, 'Binned_Slope'),

        ('Horizontal_Distance_To_Hydrology', 140, 'Binned_Horizontal_Distance_To_Hydrology'),

        ('Horizontal_Distance_To_Roadways', 712, 'Binned_Horizontal_Distance_To_Roadways'),

        ('Hillshade_9am', 32, 'Binned_Hillshade_9am'),

        ('Hillshade_Noon', 32, 'Binned_Hillshade_Noon'),

        ('Hillshade_3pm', 32, 'Binned_Hillshade_3pm'),

        ('Horizontal_Distance_To_Fire_Points', 717, 'Binned_Horizontal_Distance_To_Fire_Points')

    ]

    

    for col_name, bin_size, new_name in bin_defs:

        df[new_name] = np.floor(df[col_name] / bin_size)

        

    print('Total number of features : %d' % (df.shape)[1])

    return df
def load_and_process_dataset():

    train = pd.read_csv(TRAIN_FILE)

    test = pd.read_csv(TEST_FILE)



    # XGB needs labels starting with 0!

    # now 7 become 6, 6 become 5 and so on ..

    y_train = train[TARGET].ravel() - 1

    

    classes = train.Cover_Type.unique()

    num_classes = len(classes)

    print("There are {0} classes: {1} ".format(num_classes, classes))        



    train.drop([ID, TARGET], axis=1, inplace=True)

    test.drop([ID], axis=1, inplace=True)

    

    train = add_feats(train)    

    test = add_feats(test)    

    

    cols_to_normalize = [ 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

                       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

                       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 

                       'Horizontal_Distance_To_Fire_Points', 

                       'Shadiness_morn_noon', 'Shadiness_noon_3pm', 'Shadiness_morn_3',

                       'Shadiness_morn_avg',

                       'Shadiness_afternoon', 

                       'Shadiness_mean_hillshade',

                       'HF1', 'HF2', 

                       'HR1', 'HR2', 

                       'FR1', 'FR2'

                       ]



    train[cols_to_normalize] = normalize(train[cols_to_normalize])

    test[cols_to_normalize] = normalize(test[cols_to_normalize])



    # elevation was found to have very different distributions on test and training sets

    # lets just drop it for now to see if we can implememnt a more robust classifier!

    train = train.drop('Elevation', axis=1)

    test = test.drop('Elevation', axis=1)    

    

    x_train = train.values

    x_test = test.values



    return {'X_train': x_train, 'X_test': x_test, 'y_train': y_train}
dataset = Dataset(preprocessor=load_and_process_dataset, use_cache=True)
#Multiple Classifiers are used for multiple results



#RandomForestClassifier

rf_params = {'n_estimators': 200, 'criterion': 'entropy', 'random_state': 0}

rf = Classifier(dataset=dataset, estimator=RandomForestClassifier, 

                use_cache=False, parameters=rf_params, name='rf')



#RandomForestClassifier

rf1_params = {'n_estimators': 200, 'criterion': 'gini', 'random_state': 0}

rf1 = Classifier(dataset=dataset, estimator=RandomForestClassifier, 

                 use_cache=False, parameters=rf1_params,name='rf1')



#ExtraTreesClassifier

et_params = {'n_estimators': 200, 'criterion': 'entropy', 'random_state': 0}

et = Classifier(dataset=dataset, estimator=ExtraTreesClassifier, 

                use_cache=False, parameters=et_params,name='et')



#ExtraTreesClassifier

et1_params = {'n_estimators': 200, 'criterion': 'gini', 'random_state': 0}

et1 = Classifier(dataset=dataset, use_cache=False, estimator=ExtraTreesClassifier,

                 parameters=et1_params,name='et1')



#LGBMClassifier

lgb_params = {'n_estimators': 200, 'learning_rate':0.1}

lgbc = Classifier(dataset=dataset, estimator=LGBMClassifier, 

                  use_cache=False, parameters=lgb_params,name='lgbc')



#LogisticRegression

logr_params = {'solver' : 'liblinear', 'multi_class' : 'ovr', 'C': 1, 'random_state': 0}

logr = Classifier(dataset=dataset, estimator=LogisticRegression, 

                  use_cache=False, parameters=logr_params,name='logr')



#Naive Bayes

gnb = Classifier(dataset=dataset,estimator=GaussianNB, use_cache=False, name='gnb')
def xgb_classifier(X_train, y_train, X_test, y_test=None):

    xg_params = {'seed': 0,

                'colsample_bytree': 0.7,

                'silent': 1,

                'subsample': 0.7,

                'learning_rate': 0.1,

                'objective': 'multi:softprob',   

                'num_class': 7,

                'max_depth': 4,

                'min_child_weight': 1,

                'eval_metric': 'mlogloss',

                'nrounds': 200}

    

    X_train = xgb.DMatrix(X_train, label=y_train)

    model = xgb.train(xg_params, X_train, xg_params['nrounds'])

    return model.predict(xgb.DMatrix(X_test))



xgb_first = Classifier(estimator=xgb_classifier, dataset=dataset, use_cache=CACHE, name='xgb_classifier')
pipeline = ModelsPipeline(rf, et, et1, lgbc, logr, gnb, xgb_first)

stack_ds = pipeline.stack(k=NFOLDS,seed=SEED)
stack_ds.X_train.head()
stack_ds.X_test.head()
print("Shape of out-of-fold predictions:", "X shape: ", stack_ds.X_train.shape, "y shape: ", stack_ds.y_train.shape)
X_train_outfold = stack_ds.X_train.values

X_test_outfold = stack_ds.X_test.values

X = X_train_outfold

y_train_sv = stack_ds.y_train + 1

y = y_train_sv
# Train LogisticRegression on stacked data (second stage)

lr = LogisticRegression

lr_params = {'C': 5, 'random_state' : SEED, 'solver' : 'liblinear', 'multi_class' : 'ovr',}

stacker = Classifier(dataset=stack_ds, estimator=lr, use_cache=False, parameters=lr_params)
preds_proba = stacker.predict()

# Note: labels starting with 0 in xgboost, therefore adding +1!

predictions = np.round(np.argmax(preds_proba, axis=1)).astype(int) + 1



submission = pd.read_csv(SUBMISSION_FILE)

submission[TARGET] = predictions

submission.to_csv('Stage_2_1_logregr_out_of_fold.csv', index=None)
Cs = [0.001, 0.01, 0.1, 1, 10]

gammas = [0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5, verbose=True)

#grid_search.fit(X, y)

#grid_search.best_params_

#best_svc = grid_search.best_estimator_
best_params_svc = {'C': 10, 'gamma': 0.01}
best_svc = svm.SVC(**best_params_svc)

best_svc.fit(X, y)
preds_svc = best_svc.predict(X_test_outfold)

sub_svc = pd.DataFrame({"Id": id_test.values,"Cover_Type": preds_svc})

sub_svc.to_csv("Stage_2_2_svc_out_of_fold.csv", index=False)
y_train_nn = np_utils.to_categorical(stack_ds.y_train + 1)
model = Sequential()

model.add(Dense(1024, input_dim=49, kernel_initializer='uniform', activation='selu'))

model.add(Dense(512, kernel_initializer='uniform', activation='softplus'))

model.add(Dense(256, kernel_initializer='uniform', activation='elu'))

model.add(Dense(128, kernel_initializer='uniform', activation='selu'))

model.add(Dense(64, kernel_initializer='uniform', activation='softplus'))

model.add(Dense(32, kernel_initializer='uniform', activation='elu'))

model.add(Dense(16, kernel_initializer='uniform', activation='softplus'))

model.add(Dense(8, kernel_initializer='uniform', activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Fit model

model.fit(X, y_train_nn, epochs=10, batch_size=32)
preds_nn = model.predict(X_test_outfold)

sub_nn = pd.DataFrame({"Id": id_test.values,"Cover_Type": np.argmax(preds_nn,axis=1)})

sub_nn.to_csv("Stage_2_3_ann_out_of_fold.csv", index=False)