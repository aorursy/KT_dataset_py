import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords

import time

import spacy

from scipy.misc import imread

from sklearn.utils import resample

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot

train_df = pd.read_csv('../input/kidney_disease.csv')

df = pd.read_csv('../input/kidney_disease.csv')

start_time = time.time()

end = time.time()

print("Time taken in reading the input files is {}.".format(end - start_time))

train_df.head()

print("Number of rows in train dataset {} ".format(train_df.shape[0]))



train_df.describe()



# Data Preprocessing



df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})

df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})

df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df.rename(columns={'classification':'class'},inplace=True)

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good

df['appet'] = df['appet'].replace(to_replace='no',value=0)

df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

#df.drop('id',axis=1,inplace=True)



from sklearn.model_selection import train_test_split

sns.countplot(x = 'age', hue = 'appet', data = df)

for i in ['rc','wc','pcv']:

    df[i] = df[i].str.extract('(\d+)').astype(float)

for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:

    df[i].fillna(df[i].mean(),inplace=True)

df = df.dropna(axis=0)

df['class'].value_counts()

df.head()

df.info()

values = df.values

#Data Augmentation

n_iterations = 10000

n_size = int(len(df) * 0.50)

print(n_size)

# run bootstrap

stats = list()

for i in range(n_iterations):

# prepare train and test sets

    train = resample(values, n_samples=n_size)

    test = np.array([x for x in values if x.tolist() not in train.tolist()])

# fit model

model = DecisionTreeClassifier()

model.fit(train[:,:-1], train[:,-1])

# evaluate model

predictions = model.predict(test[:,:-1])

score = accuracy_score(test[:,-1], predictions)



plt.figure(figsize=(12,3))

features = train.columns.values.tolist()

importance = clf_best.feature_importances_.tolist()

feature_series = pd.Series(data=importance,index=features)

feature_series.plot.bar()

plt.title('Feature Importance')
#another algorithm compile 2 and 3 below

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords

import time

import spacy

from scipy.misc import imread

from sklearn.utils import resample

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot

train_df = pd.read_csv('../input/kidney_disease.csv')

df = pd.read_csv('../input/kidney_disease.csv')

start_time = time.time()

end = time.time()

print("Time taken in reading the input files is {}.".format(end - start_time))

train_df.head()

print("Number of rows in train dataset {} ".format(train_df.shape[0]))



train_df.describe()



# Data Preprocessing



df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})

df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})

df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df.rename(columns={'classification':'class'},inplace=True)

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good

df['appet'] = df['appet'].replace(to_replace='no',value=0)

df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

#df.drop('id',axis=1,inplace=True)



from sklearn.model_selection import train_test_split

sns.countplot(x = 'age', hue = 'appet', data = df)

for i in ['rc','wc','pcv']:

    df[i] = df[i].str.extract('(\d+)').astype(float)

for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:

    df[i].fillna(df[i].mean(),inplace=True)

df = df.dropna(axis=0)

df['class'].value_counts()

df.head()

df.info()

values = df.values

#Run this next to first for MLP

#Binary Classifier

# Import libraries for data wrangling, preprocessing and visualization

import numpy 

#import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# Importing libraries for building the neural network

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



# Select the columns to use for prediction in the neural network

prediction_var = ['sg','al','su','bgr','sc','pot','pcv','wc','rc','dm']

X = df[prediction_var].values

Y = df['class'].values

# Diagnosis values are strings. Changing them into numerical values using LabelEncoder.

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

# Baseline model for the neural network. We choose a hidden layer of 10 neurons. The lesser number of neurons helps to eliminate the redundancies in the data and select the more important features.

def create_baseline():

    # create model

    model = Sequential()

    model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Evaluate model using standardized dataset. 

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Results:%.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import learning_curve

train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

prediction_var = ['sg','al','su','bgr','sc','pot','pcv','wc','rc','dm']

X = df[prediction_var].values

y = df['class'].values

train_sizes, train_scores, validation_scores = learning_curve(estimator = LinearRegression(),X=X,y=y,train_sizes = train_sizes,cv = 5,scoring = 'neg_mean_squared_error')

print('Training scores:\n\n', train_scores)

print('\n', '-' * 70) # separator to make the output easy to read

print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = -train_scores.mean(axis = 1)

validation_scores_mean = -validation_scores.mean(axis = 1)

print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))

print('\n', '-' * 20) # separator

print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

import matplotlib.pyplot as plt



plt.style.use('seaborn')

plt.plot(train_sizes, train_scores_mean, label = 'Training error')

plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('MSE', fontsize = 14)

plt.xlabel('Training set size', fontsize = 14)

plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)

plt.legend()

plt.ylim(0.0,1)
#Run this next to first for MLP

#Binary Classifier

# Import libraries for data wrangling, preprocessing and visualization

import numpy 

#import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# Importing libraries for building the neural network

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

# Read data file

df = pd.read_csv('../input/kidney_disease.csv')

start_time = time.time()

end = time.time()

print("Time taken in reading the input files is {}.".format(end - start_time))

train_df.head()

print("Number of rows in train dataset {} ".format(train_df.shape[0]))



train_df.describe()



# Data Preprocessing



df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})

df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})

df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df.rename(columns={'classification':'class'},inplace=True)

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good

df['appet'] = df['appet'].replace(to_replace='no',value=0)

df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

#df.drop('id',axis=1,inplace=True)



#from sklearn.model_selection import train_test_split

#sns.countplot(x = 'age', hue = 'appet', data = df)

for i in ['rc','wc','pcv']:

    df[i] = df[i].str.extract('(\d+)').astype(float)

for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:

    df[i].fillna(df[i].mean(),inplace=True)

df = df.dropna(axis=0)

df['class'].value_counts()

df.head()

df.info()

values = df.values

df = pd.read_csv('../input/kidney_disease.csv')

start_time = time.time()

end = time.time()

print("Time taken in reading the input files is {}.".format(end - start_time))

train_df.head()

print("Number of rows in train dataset {} ".format(train_df.shape[0]))



train_df.describe()



# Data Preprocessing



df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})

df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})

df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df.rename(columns={'classification':'class'},inplace=True)

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good

df['appet'] = df['appet'].replace(to_replace='no',value=0)

df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

#df.drop('id',axis=1,inplace=True)



from sklearn.model_selection import train_test_split

sns.countplot(x = 'age', hue = 'appet', data = df)

for i in ['rc','wc','pcv']:

    df[i] = df[i].str.extract('(\d+)').astype(float)

for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:

    df[i].fillna(df[i].mean(),inplace=True)

df = df.dropna(axis=0)

df['class'].value_counts()

df.head()

df.info()

values = df.values

#data = pd.read_csv('../input/kidney_disease.csv', header=0)

#seed = 5

#numpy.random.seed(seed)

# Column Unnamed : 32 holds only null values, so it is of no use to us. We simply drop that column.

#data.drop("Unnamed: 32",axis=1,inplace=True)

#data.drop("id", axis=1, inplace=True)

# Select the columns to use for prediction in the neural network

prediction_var = ['sg','al','su','bgr','sc','pot','pcv','wc','rc','dm']

X = df[prediction_var].values

Y = df['class'].values

# Diagnosis values are strings. Changing them into numerical values using LabelEncoder.

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

# Baseline model for the neural network. We choose a hidden layer of 10 neurons. The lesser number of neurons helps to eliminate the redundancies in the data and select the more important features.

def create_baseline():

    # create model

    model = Sequential()

    model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Evaluate model using standardized dataset. 

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Results:%.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

import warnings

warnings.filterwarnings("ignore")



# Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier



# Regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor



# Modelling Helpers :

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate



# Preprocessing :

from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder



# Metrics :

# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report



print("Setup complete...")

labels = df.columns[:]

X = df[labels]

print(X)

y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



#model = XGBClassifier()



#print("Accuracy: %.2f%%" % (accuracy * 100.0))

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RF', RandomForestRegressor()))

models.append(('XGB', XGBClassifier()))

models.append(('GBR', GradientBoostingRegressor()))

models.append(('ABR', AdaBoostRegressor()))



for name, model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print(name,accuracy)

    
#print(score)

stats.append(score)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.95

p = ((1.0-alpha)/2.0) * 100

lower = max(0.0, numpy.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upper = min(1.0, numpy.percentile(stats, p))

print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
print(train)

print(test)

print(train.shape)

print(test.shape)
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score

from imblearn.over_sampling import SMOTE

#loans = pd.read_csv('../lending-club-data.csv.zip')

#loans.iloc[0]

#loans.bad_loans.value_counts()

#loans = loans[~loans.payment_inc_ratio.isnull()]

#model_variables = ['grade', 'home_ownership','emp_length_num', 'sub_grade','short_emp',

          #  'dti', 'term', 'purpose', 'int_rate', 'last_delinq_none', 'last_major_derog_none',

           # 'revol_util', 'total_rec_late_fee', 'payment_inc_ratio', 'bad_loans']



#loans_data_relevent = loans[model_variables]

#loans_relevant_enconded = pd.get_dummies(loans_data_relevent)

#training_features, test_features, \

#training_target, test_target, = train_test_split(loans_relevant_enconded.drop(['bad_loans'], axis=1),

 #                                              loans_relevant_enconded['bad_loans'],

  #                                             test_size = .1,

   #                                            random_state=12)

sm = SMOTE(random_state=12, ratio = 1.0)

x_res, y_res = sm.fit_sample(training_features, training_target)

#print training_target.value_counts(), np.bincount(y_res)

x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(x_res,

                                                    y_res,

                                                    test_size = .1,

                                                    random_state=12)

#oversampling

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")



# Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier



# Regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor



# Modelling Helpers :

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate



# Preprocessing :

from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder



# Metrics :

# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report



print("Setup complete...")





df = pd.read_csv('../input/kidney_disease.csv')



target_count = df.classification.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



target_count.plot(kind='bar', title='Count (target)');

# Data Preprocessing



df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})

df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})

df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df.rename(columns={'classification':'class'},inplace=True)

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good

df['appet'] = df['appet'].replace(to_replace='no',value=0)

df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

#df.drop('id',axis=1,inplace=True)



from sklearn.model_selection import train_test_split

sns.countplot(x = 'age', hue = 'appet', data = df)

for i in ['rc','wc','pcv']:

    df[i] = df[i].str.extract('(\d+)').astype(float)

for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:

    df[i].fillna(df[i].mean(),inplace=True)

df = df.dropna(axis=0)

df['class'].value_counts()

df.head()

df.info()

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Remove 'id' and 'target' columns

labels = df.columns[1:]

print(labels)

X = df[labels]

print(X)

y = df['class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=1)

print("Training Dataset size:",X_train)

#model = XGBClassifier()

#model.fit(X_train, y_train)

#y_pred = model.predict(X_test)



#accuracy = accuracy_score(y_test, y_pred)

#print("Accuracy: %.2f%%" % (accuracy * 100.0))

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RF', RandomForestRegressor()))

models.append(('XGB', XGBClassifier()))

models.append(('GBR', GradientBoostingRegressor()))

models.append(('ABR', AdaBoostRegressor()))



for name, model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print(name,accuracy)
#run with feature which reduce the classifier performane

#model = XGBClassifier()

#model.fit(X_train[['age']], y_train)

#y_pred = model.predict(X_test[['age']])



#accuracy = accuracy_score(y_test, y_pred)

#print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
#count_class_0, count_class_1 = df.classification.value_counts()

count_class_0=target_count[0]

count_class_1=target_count[1]

df_class_0 = df[df['class'] == 0]

df_class_1 = df[df['class'] == 1]

print(df_class_1.count())

# Divide by class

df_class_1_over = df_class_1.sample(count_class_0, replace=True)

#print(df_class_1_over)

df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print(df_test_over)

print(df_test_over[df_test_over['class'] == 0])

#print(df_test_over[df_test_over['class'] == 1])

print('Random over-sampling:')

#print(df_test_over.class.value())

#print(df.info())

#print(df_test_over.class.value_counts())



df_test_over['class'].value_counts().plot(kind='bar', title='Count (target)');
import warnings

warnings.filterwarnings("ignore")



# Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier



# Regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor



# Modelling Helpers :

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate



# Preprocessing :

from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder



# Metrics :

# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report



print("Setup complete...")

method_names = []

method_scores = []

labels = df.columns[:]

X = df[labels]

print(X)

y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



#model = XGBClassifier()



#print("Accuracy: %.2f%%" % (accuracy * 100.0))

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RF', RandomForestRegressor()))

models.append(('XGB', XGBClassifier()))

models.append(('GBR', GradientBoostingRegressor()))

models.append(('ABR', AdaBoostRegressor()))



for name, model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print(name,accuracy)

    method_names.append(name)

    method_scores.append(accuracy)

    

    #print(name,accuracy_score(y_test, pred))

    

   # model.fit(X_train, y_train)

    #if model='XGB':

        #big_X = x_train[feature_columns_to_use].append(x_train[feature_columns_to_use])

        #big_X_imputed = DataFrameImputer().fit_transform(big_X)

    #pred = model.predict(y_test).astype(int)

    #print(name, accuracy_score(y_test, pred))
plt.figure(figsize=(15,10))

plt.ylim([0.20,1])

plt.bar(method_names,method_scores,width=0.5)

plt.xlabel('Method Name')

plt.ylabel('Method Accuracy')
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X = pca.fit_transform(X)



plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X, y)



print(X_ros.shape[0] - X.shape[0], 'new random picked points')



plot_2d_space(X_ros, y_ros, 'Random over-sampling')

from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_sample(X, y)



plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
#RNN

model = Sequential()

model.add(Dense(100, activation='relu', input_dim=26))

model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='nadam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])  

print(X_train.shape)

model.fit(X_train,y_train,verbose=1,shuffle=True, nb_epoch=3,batch_size=100,validation_split=0.4)



# Read test data and evaluate results





score = model.evaluate(X_test,y_test, batch_size=16)

print("LOSS")

print(score[0])

print("precision")

print(score[1])


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

encoder = LabelEncoder()

encoder.fit(y)

encoded_Y = encoder.transform(y)

def create_baseline():

    # create model

    model = Sequential()

    model.add(Dense(5, input_dim=10, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# The GRU architecture

regressorGRU = Sequential()

# First GRU layer with Dropout regularisation

regressorGRU.add(GRU(units=26, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))

# Second GRU layer

regressorGRU.add(GRU(units=26, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))

# Third GRU layer

regressorGRU.add(GRU(units=26, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))

regressorGRU.add(Dropout(0.2))

# Fourth GRU layer

regressorGRU.add(GRU(units=26, activation='tanh'))

regressorGRU.add(Dropout(0.2))

# The output layer

regressorGRU.add(Dense(units=1))

# Compiling the RNN

regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')

# Fitting to the training set

regressorGRU.fit(X_train,y_train,epochs=50,batch_size=150)
# using a different dataset with 8000 records

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



def auc_scorer(clf, X, y, model): # Helper function to plot the ROC curve

    if model=='RF':

        fpr, tpr, _ = roc_curve(y, clf.predict_proba(X)[:,1])

    elif model=='SVM':

        fpr, tpr, _ = roc_curve(y, clf.decision_function(X))

    roc_auc = auc(fpr, tpr)



    plt.figure()    # Plot the ROC curve

    plt.plot(fpr, tpr, label='ROC curve from '+model+' model (area = %0.3f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend(loc="lower right")

    plt.show()



    return fpr,tpr,roc_auc



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))


import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords

import time

import spacy

from scipy.misc import imread

train_df = pd.read_csv('../input/kidney_disease.csv')

df = pd.read_csv('../input/kidney_disease.csv')

start_time = time.time()

end = time.time()

print("Time taken in reading the input files is {}.".format(end - start_time))

train_df.head()

print("Number of rows in train dataset {} ".format(train_df.shape[0]))



train_df.describe()



# Map text to 1/0 and do some cleaning

df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})

df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})

df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df.rename(columns={'classification':'class'},inplace=True)

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good

df['appet'] = df['appet'].replace(to_replace='no',value=0)

df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

df.drop('id',axis=1,inplace=True)

corr = df.corr()

plt.figure(figsize = (15,15))

sns.heatmap(data = corr, annot = True, square = True, cbar = True)

# Further cleaning

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.grid()

plt.title("Number of Missing Values")

plt.savefig('missing.png')

df.head()

df.info()
from sklearn.model_selection import train_test_split

sns.countplot(x = 'age', hue = 'appet', data = df)

for i in ['rc','wc','pcv']:

    df[i] = df[i].str.extract('(\d+)').astype(float)

for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:

    df[i].fillna(df[i].mean(),inplace=True)



df2 = df.dropna(axis=0)

df2['class'].value_counts()
corr_df = df2.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_df, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlations between different predictors')

plt.show()
x_train, x_test, y_train, y_test = train_test_split(df2.iloc[:,:-1], df2['class'], 

                                                    test_size = 0.50, random_state=44,

                                                   stratify= df2['class'] )

print(x_train.shape)

print(x_test.shape)

#df['rbc'] = pd.to_numeric(df['rbc'],errors='coerce')

#df['wc'] = pd.to_numeric(df['wc'],errors='coerce')

#df['pc'] = pd.to_numeric(df['pc'],errors='coerce')

#df['pcc'] = pd.to_numeric(df['pcc'],errors='coerce')

#df['ba'] = pd.to_numeric(df['ba'],errors='coerce')

#df['htn'] = pd.to_numeric(df['htn'],errors='coerce')

#df['pcv'] = pd.to_numeric(df['pcv'],errors='coerce')

#df['rc'] = pd.to_numeric(df['rc'],errors='coerce')

#df['cad'] = pd.to_numeric(df['cad'],errors='coerce')

#df['dm'] = pd.to_numeric(df['dm'],errors='coerce')

#df.dtypes

y_train.value_counts()

#df2 = df.dropna(axis=0)

#no_na = df2.index.tolist()

#some_na = df.drop(no_na).apply(lambda x: pd.to_numeric(x,errors='coerce'))

#some_na = some_na.fillna(0) # Fill up all Nan by zero.



#x_test = some_na.iloc[:,:-1]

#y_test = some_na['class']

#y_true = y_test

#lr_pred = clf_best.predict(x_test)

#print(classification_report(y_true, lr_pred))



#confusion = confusion_matrix(y_test, lr_pred)

#print('Confusion Matrix:')

#print(confusion)



#print('Accuracy: %3f' % accuracy_score(y_true, lr_pred))

# Determine the false positive and true positive rates

#fpr,tpr,roc_auc = auc_scorer(clf_best, x_test, y_test, 'RF')
import xgboost as xgb

model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,max_depth=8, n_estimators=50)

model.fit(x_train, y_train)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score



method_names = []

method_scores = []

log_reg = LogisticRegression()

log_reg.fit(x_train,y_train) #Fitting

print("Logistic Regression Classification Test Accuracy {}".format(log_reg.score(x_test,y_test)))

method_names.append("Logistic Reg.")

method_scores.append(log_reg.score(x_test,y_test))



#Confusion Matrix

y_pred = log_reg.predict(x_test)

conf_mat = confusion_matrix(y_test,y_pred)

#Visualization Confusion Matrix

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Predicted Values")

plt.ylabel("True Values")

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score



dec_tree = DecisionTreeClassifier()

dec_tree.fit(x_train,y_train)

print("Decision Tree Classification Score: ",dec_tree.score(x_test,y_test))

method_names.append("Decision Tree")

method_scores.append(dec_tree.score(x_test,y_test))



#Confusion Matrix

y_pred = dec_tree.predict(x_test)

conf_mat = confusion_matrix(y_test,y_pred)

#Visualization Confusion Matrix

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Predicted Values")

plt.ylabel("True Values")

plt.show()
plt.figure(figsize=(15,10))

plt.ylim([0.85,1])

plt.bar(method_names,method_scores,width=0.5)

plt.xlabel('Method Name')

plt.ylabel('Method Score')
tuned_parameters = [{'n_estimators':[7,8,9,10,11,12,13,14,15,16],'max_depth':[2,3,4,5,6,None],

                     'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[42]}]

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10,scoring='f1')

clf.fit(x_train, y_train)





print("Detailed classification report:")

y_true, lr_pred = y_test, clf.predict(x_test)

print(classification_report(y_true, lr_pred))



confusion = confusion_matrix(y_test, lr_pred)

print('Confusion Matrix:')

print(confusion)



# Determine the false positive and true positive rates

#fpr,tpr,roc_auc = auc_scorer(clf, x_test, y_test, 'RF')



print('Best parameters:')

print(clf.best_params_)

clf_best = clf.best_estimator_

plt.figure(figsize=(12,3))

features = x_test.columns.values.tolist()

importance = clf_best.feature_importances_.tolist()

feature_series = pd.Series(data=importance,index=features)

feature_series.plot.bar()

plt.title('Feature Importance')
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(x_train,y_train) #Fitting

print("Logistic Regression Classification Test Accuracy {}".format(log_reg.score(x_test,y_test)))

method_names.append("Logistic Reg.")

method_scores.append(log_reg.score(x_test,y_test))



#Confusion Matrix

y_pred = log_reg.predict(x_test)

conf_mat = confusion_matrix(y_test,y_pred)

#Visualization Confusion Matrix

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Predicted Values")

plt.ylabel("True Values")

plt.show()
import warnings

warnings.filterwarnings("ignore")



# Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier



# Regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor



# Modelling Helpers :

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate



# Preprocessing :

from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder



# Metrics :

# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report



print("Setup complete...")

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RF', RandomForestRegressor()))

models.append(('XGB', XGBClassifier()))

models.append(('GBR', GradientBoostingRegressor()))

models.append(('ABR', AdaBoostRegressor()))



for name, model in models:

    model.fit(x_train, y_train)

    #if model='XGB':

        #big_X = x_train[feature_columns_to_use].append(x_train[feature_columns_to_use])

        #big_X_imputed = DataFrameImputer().fit_transform(big_X)

    pred = model.predict(x_test).astype(int)

    print(name, accuracy_score(y_test, pred))
from sklearn.preprocessing import LabelEncoder



sns.heatmap(df.isnull(),cmap="viridis",cbar=False,yticklabels=False)

x_train, xtest, y_train, y_test = train_test_split(df.iloc[:,:-1], df['classification'], 

                                                    test_size = 0.33, random_state=44,

                                                   stratify= df['classification'] )









    
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.title("Number of Missing Values")

plt.savefig('missing_updated.png')
from sklearn.preprocessing import LabelEncoder

import glob

import keras as k

import numpy as np

import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import matplotlib.pyplot as plt



#columns_to_retain = ["sg", "al", "sc", "hemo","pcv", "wbcc", "rbcc", "htn", "classification"]

#olumns_to_retain = df.columns

#df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)

    # now drop the rows with na values

#df = df.dropna(axis=0)



#for column in df.columns:

   # if df[column].dtype == np.number:

   #     continue

    #df[column] = LabelEncoder().fit_transform(df[column])



#y = df.classification.values

#x = df.drop(["classification"],axis=1)

#from sklearn.preprocessing import LabelEncoder



#for i in ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification']:

   # df[i] = df[i].map({'3+': 3})

 #   df[i] = LabelEncoder().fit_transform(df[i])



#df2 = df.dropna(axis=0)

#df2['class'].value_counts()

#x_train = df.drop(['id','classification'],axis=1)

#y_train = df['classification']

#x_train, x_test, y_train, y_test = train_test_split(df2.iloc[:,:-1], df2['classification'], 

 #                                                   test_size = 0.33, random_state=44,

  #                                                 stratify= df2['classification'] )

#x_scaler = MinMaxScaler()

#x_scaler.fit(x)

#column_names = x.columns

#x[column_names] = x_scaler.transform(x)



#x_train,  x_test, y_train, y_test = train_test_split(x, y, test_size=5, shuffle=True)



#optimizer = k.optimizers.Adam()

#checkpoint = ModelCheckpoint("ckd.best.model", monitor="loss",

                                # mode="min", save_best_only=True, verbose=0)

#model = k.models.Sequential()

#model.add(Dense(256, input_dim=len(x.columns),kernel_initializer=k.initializers.random_normal(seed=13), activation="relu"))

#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)





import warnings

warnings.filterwarnings("ignore")



# Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier



# Regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor



# Modelling Helpers :

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate



# Preprocessing :

from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder



# Metrics :

# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report



print("Setup complete...")

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RF', RandomForestRegressor()))

models.append(('XGB', XGBClassifier()))

models.append(('GBR', GradientBoostingRegressor()))

models.append(('ABR', AdaBoostRegressor()))



for name, model in models:

    model.fit(x_train, y_train)

    pred = model.predict(x_test).astype(int)

    print(name, accuracy_score(y_test, pred))


corr_df = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_df, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlations between different predictors')

plt.show()
svc = SVC()



params = {

    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],

    'kernel':['linear', 'rbf'],

    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]

}



clf = GridSearchCV(svc, param_grid = params, scoring = 'accuracy', cv = 10, verbose = 2)



clf.fit(x_train, y_train)

clf.best_params_

model = Sequential()

model.add(Dense(100,input_dim=X.shape[1],activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dense(25,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X,Y,epochs=100,batch_size=40,validation_split=.2,verbose=2)

model.summary()
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
scores = model.evaluate(X,Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(X_train.shape)

print(X_test.shape)
y_train.value_counts()
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



def auc_scorer(clf, X, y, model): # Helper function to plot the ROC curve

    if model=='RF':

        fpr, tpr, _ = roc_curve(y, clf.predict_proba(X)[:,1])

    elif model=='SVM':

        fpr, tpr, _ = roc_curve(y, clf.decision_function(X))

    roc_auc = auc(fpr, tpr)



    plt.figure()    # Plot the ROC curve

    plt.plot(fpr, tpr, label='ROC curve from '+model+' model (area = %0.3f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend(loc="lower right")

    plt.show()



    return fpr,tpr,roc_auc



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))
tuned_parameters = [{'n_estimators':[7,8,9,10,11,12,13,14,15,16],'max_depth':[2,3,4,5,6,None],

                     'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[42]}]

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10,scoring='f1')

clf.fit(x_train, y_train)



print("Detailed classification report:")

y_true, lr_pred = y_test, clf.predict(x_test)

print(classification_report(y_true, lr_pred))



confusion = confusion_matrix(y_test, lr_pred)

print('Confusion Matrix:')

print(confusion)



# Determine the false positive and true positive rates

fpr,tpr,roc_auc = auc_scorer(clf, x_test, y_test, 'RF')



print('Best parameters:')

print(clf.best_params_)

clf_best = clf.best_estimator_

plt.figure(figsize=(12,3))

features = x_test.columns.values.tolist()

importance = clf_best.feature_importances_.tolist()

feature_series = pd.Series(data=importance,index=features)

feature_series.plot.bar()

plt.title('Feature Importance')
list_to_fill = x_test.columns[feature_series>0.03]

print(list_to_fill)
import numpy as np

import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import EarlyStopping

from keras import optimizers

from keras import regularizers

BATCH_SIZE = 40

EPOCHS = 50

LEARNING_RATE = 0.001

DATASET_SIZE = 400

model = Sequential()

print(x_test.columns[feature_series>0.03].shape[0])

model.add(Dense(100,input_dim=x_test.columns[feature_series>0.03].shape[0],activation='relu'))

model.add(BatchNormalization())

model.add(Dense(50,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(25,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(1,activation='sigmoid'))

model.add(BatchNormalization())



model.add(Dense(1))



adam = optimizers.adam(lr=LEARNING_RATE)

model.compile(loss='mse', optimizer=adam, metrics=['mae'])

print('Dataset size: %s' % DATASET_SIZE)

print('Epochs: %s' % EPOCHS)

print('Learning rate: %s' % LEARNING_RATE)

print('Batch size: %s' % BATCH_SIZE)

print('Input dimension: %s' % x_test.shape[1])

print('Features used: %s' % x_test.columns[feature_series>0.03])
model.summary()
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from sklearn.preprocessing import LabelEncoder

import time

from keras import metrics

max_len = 24

def get_cnn_model_v1():   

    model = Sequential()

    # we start off with an efficient embedding layer which maps

    # our vocab indices into embedding_dims dimensions

    # 1000 is num_max

    model.add(Embedding(400,

                        20,

                        input_length=max_len))

    model.add(Dropout(0.2))

    model.add(Conv1D(64,

                     3,

                     padding='valid',

                     activation='relu',

                     strides=1))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(256))

    model.add(Dropout(0.2))

    model.add(Activation('relu'))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['acc',metrics.binary_accuracy])

    return model





def check_model(model,x_train,y_train):

    model.fit(X,y,batch_size=40,epochs=10,verbose=1,validation_split=0.5)



m = get_cnn_model_v1()

check_model(m,x_train, y_train)

 
