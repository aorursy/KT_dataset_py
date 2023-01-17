import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import matplotlib.style as style

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df_test = pd.read_csv('Test.csv')

df_train = pd.read_csv('Train.csv')
df_test.describe()
df_train.describe()
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass', data = df_train)
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass',hue = 'program_type', data = df_train, palette = 'rainbow')
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass',hue = 'program_duration', data = df_train,palette = 'rainbow')
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass',hue = 'test_type', data = df_train,palette = 'rainbow')
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass',hue = 'difficulty_level', data = df_train,palette = 'rainbow')
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass',hue = 'education', data = df_train,palette = 'rainbow')
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass',hue = 'gender', data = df_train,palette = 'rainbow')
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass',hue = 'city_tier', data = df_train,palette = 'rainbow')
sns.set_style('whitegrid')

sns.countplot(x= 'is_pass',hue = 'is_handicapped', data = df_train,palette = 'rainbow')
sns.boxplot(df_train['age'])

print('outliers')
TRN = df_train.copy()
TRN['gender'] = pd.get_dummies(TRN['gender'], drop_first = True)

TRN['is_handicapped'] = pd.get_dummies(TRN['is_handicapped'], drop_first = True)

TRN['test_type'] = pd.get_dummies(TRN['test_type'], drop_first = True)
style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (12,8))

mask = np.zeros_like(TRN.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(TRN.corr(), 

            cmap=sns.diverging_palette(20, 220, n=200), 

            mask = mask, 

            annot=True, 

            center = 0, 

           );

plt.title("Heatmap of all the Features", fontsize = 30);
df_test.isnull().sum()
df_test.duplicated().sum()
df_train.isnull().sum()
df_train.duplicated().sum()
print(df_test[['age','trainee_engagement_rating']].mode())

print(df_train[['age','trainee_engagement_rating']].mode())
df_test['age'].fillna(45.0, inplace=True)

df_test['trainee_engagement_rating'].fillna(1.0, inplace=True)

print(df_test.isnull().sum())
df_train['age'].fillna(45.0, inplace=True)

df_train['trainee_engagement_rating'].fillna(1.0, inplace=True)

print(df_train.isnull().sum())
## Saving the target values in "y_train". 

y = df_train['is_pass'].reset_index(drop=True)
# getting a copy of train

previous_train = df_train.copy()

previous_test = df_test.copy()
df = df_test.copy()

df = df.append(df_train, sort = False)

df
df.isnull().sum()
df.duplicated().sum()
df.info()
df.describe()
df.drop(['program_duration'],axis=1, inplace=True)

df.drop(['gender'],axis=1, inplace=True)

df.drop(['education'],axis=1, inplace=True)

df.drop(['is_handicapped'],axis=1, inplace=True)

df.drop(['id'],axis=1, inplace=True)

df.drop(['test_id'],axis=1, inplace=True)

df.drop(['trainee_id'],axis=1, inplace=True)

df.drop(['age'],axis=1, inplace=True)

df.drop(['total_programs_enrolled'],axis=1, inplace=True)
## Dropping the target variable. 

df.drop(['is_pass'], axis = 1, inplace = True)
final_features = pd.get_dummies(df,drop_first = True)

final_features.shape
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()



num_vars = ['city_tier','trainee_engagement_rating']



final_features[num_vars] = sc.fit_transform(final_features[num_vars])
X = final_features.iloc[:len(y), :]



X_sub = final_features.iloc[len(y):, :]
## Train test s

from sklearn.model_selection import train_test_split

## Train test split follows this distinguished code pattern and helps creating train and test set to build machine learning. 

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .3, random_state = 0)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
## importing necessary models.

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error



## Call in the LinearRegression object

lin_reg = LinearRegression(normalize=True, n_jobs=-1)

## fit train and test data. 

lin_reg.fit(X_train, y_train)

## Predict test data. 

y_pred = lin_reg.predict(X_test)
print ('%.2f'%mean_squared_error(y_test, y_pred))
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

lin_reg = LinearRegression()

cv = KFold(shuffle=True, random_state=2, n_splits=10)

scores = cross_val_score(lin_reg, X,y,cv = cv, scoring = 'neg_mean_absolute_error')
print ('%.8f'%scores.mean())
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('GBC',GradientBoostingClassifier()))

models.append(('RFC', RandomForestClassifier()))
result = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=51)

    croresults = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    result.append(croresults)

    output = "%s: %f (%f)" % (name, croresults.mean(), croresults.std())

    print(output)
LR = LogisticRegression()

LR.fit(X_train, y_train)

predictions = LR.predict(X_test)
print(accuracy_score(y_test, predictions))
predictions
confusion = metrics.confusion_matrix(y_test, predictions)

print(confusion)

#[row, column]

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]
# use float to perform true division, not integer division

print((TP + TN) / float(TP + TN + FP + FN))

print(metrics.accuracy_score(y_test, predictions))
classification_error = (FP + FN) / float(TP + TN + FP + FN)



print(classification_error)

print(1 - metrics.accuracy_score(y_test, predictions))
sensitivity = TP / float(FN + TP)



print(sensitivity)

print(metrics.recall_score(y_test, predictions))
predictions
LR.predict_proba(X_test)[0:10, 1]




# store the predicted probabilities for class 1

y_pred_prob = LR.predict_proba(X_test)[:, 1]



from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)



plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('Gradient Boosting Classifier ROC Curve')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

plt.show()
# define a function that accepts a threshold and prints sensitivity and specificity

def evaluate_threshold(threshold):

    print('Sensitivity:', tpr[thresholds > threshold][-1])

    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
evaluate_threshold(0.5)
evaluate_threshold(0.3)
# IMPORTANT: first argument is true values, second argument is predicted probabilities

print(metrics.roc_auc_score(y_test, y_pred_prob))
# calculate cross-validated AUC

#from sklearn.cross_validation import cross_val_score

cross_val_score(LR, X, y, cv=10, scoring='roc_auc').mean()
df_test.drop(['program_duration'],axis=1, inplace=True)

df_test.drop(['gender'],axis=1, inplace=True)

df_test.drop(['education'],axis=1, inplace=True)

df_test.drop(['is_handicapped'],axis=1, inplace=True)

df_test.drop(['id'],axis=1, inplace=True)

df_test.drop(['test_id'],axis=1, inplace=True)

df_test.drop(['trainee_id'],axis=1, inplace=True)

df_test.drop(['age'],axis=1, inplace=True)

df_test.drop(['total_programs_enrolled'],axis=1, inplace=True)
final_features = pd.get_dummies(df_test,drop_first = True)

final_features.shape
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()



num_vars = ['trainee_engagement_rating']



final_features[num_vars] = sc.fit_transform(final_features[num_vars])
pred_test = LR.predict(final_features)

pred_test = pd.DataFrame(pred_test)

pred_test = pred_test.rename({0: 'is_pass'}, axis=1)
pred_test.is_pass.value_counts()
finalsubmission = pd.merge(previous_test, pred_test, left_index=True, right_index=True)

finalsubmission.head()
finalsubmission.to_csv ('finalsubmission.csv', index = False, header=True)