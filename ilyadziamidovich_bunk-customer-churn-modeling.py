import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')

data.head()
del data['RowNumber'], data['CustomerId'], data['Surname']
gender_dummies = pd.get_dummies(data['Gender'])

country_dummies = pd.get_dummies(data['Geography'])

data = pd.concat([data, gender_dummies, country_dummies], axis=1)
del data['Gender'], data['Geography']
data.head()
data.shape
data.isna().sum()
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split





y = data['Exited']

x = data.drop(['Exited'], axis = 1)



def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):

    np.random.seed(seed)

    perm = np.random.permutation(df.index)

    m = len(df.index)

    train_end = int(train_percent * m)

    validate_end = int(validate_percent * m) + train_end

    train = df.iloc[perm[:train_end]]

    validate = df.iloc[perm[train_end:validate_end]]

    test = df.iloc[perm[validate_end:]]

    return train, validate, test



data_train, data_valid, data_test = train_validate_test_split(data, seed=12345)



x_train = data_train.drop(['Exited'], axis = 1)

y_train = data_train['Exited']



x_valid = data_valid.drop(['Exited'], axis = 1)

y_valid = data_valid['Exited']



x_test = data_test.drop(['Exited'], axis = 1)

y_test = data_test['Exited']
model = DecisionTreeClassifier(random_state=12345)

model.fit(x_train, y_train)
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt



def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
result = model.predict(x_valid)

print("F1 Score {0}".format(f1_score(y_valid, result)))

probs = model.predict_proba(x_valid)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

fpr, tpr, thresholds = roc_curve(y_valid, probs)

plot_roc_curve(fpr, tpr)
data['Exited'].value_counts()
7963/2037
from sklearn.linear_model import LogisticRegression



linear_model = LogisticRegression(random_state=12345)

linear_model.fit(x_train, y_train)

result = linear_model.predict(x_valid)



print("F1 Score {0}".format(f1_score(y_valid, result)))

probs = linear_model.predict_proba(x_valid)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

fpr, tpr, thresholds = roc_curve(y_valid, probs)

plot_roc_curve(fpr, tpr)
from sklearn.ensemble import RandomForestClassifier



forest_model = RandomForestClassifier(random_state=12345)

forest_model.fit(x_train, y_train)

result = forest_model.predict(x_valid)



print("F1 Score {0}".format(f1_score(y_valid, result)))

probs = forest_model.predict_proba(x_valid)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

fpr, tpr, thresholds = roc_curve(y_valid, probs)

plot_roc_curve(fpr, tpr)
best_estim = None

best_depth = None

best_f1_score = None

for estimators in range (5, 121, 10):

    for depth in range (5, 26, 5):

        forest_model = RandomForestClassifier(random_state=12345, max_depth=depth, n_estimators=estimators)

        forest_model.fit(x_train, y_train)

        result = forest_model.predict(x_valid)

        score = f1_score(result, y_valid)

        if best_f1_score is None:

            best_f1_score = score

            best_estim = estimators

            best_depth = depth

        elif best_f1_score < score:

            best_f1_score = score

            best_estim = estimators

            best_depth = depth

        print("{0} estim, {1} depth, {2} score".format(estimators, depth, score))
print("Estimators - {0}, Depth - {1}, F1 Score - {2}".format(best_estim, best_depth, best_f1_score))
forest_model = RandomForestClassifier(random_state=12345, class_weight='balanced', n_estimators=best_estim, max_depth=best_depth)

forest_model.fit(x_train, y_train)

result = forest_model.predict(x_valid)

print("F1 Score {0}".format(f1_score(y_valid, result)))

probs = forest_model.predict_proba(x_valid)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

fpr, tpr, thresholds = roc_curve(y_valid, probs)

plot_roc_curve(fpr, tpr)
def upsample(features, target, repeat):

    target_zeros = target[target == 0]

    target_ones = target[target == 1]

    

    features_zeros = features[target == 0]

    features_ones = features[target == 1]

    

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)

    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)



    return features_upsampled, target_upsampled



x_upsampled, y_upsampled = upsample(x_train, y_train, 4)
forest_model = RandomForestClassifier(random_state=12345, n_estimators=best_estim, max_depth=best_depth,class_weight='balanced')

forest_model.fit(x_upsampled, y_upsampled)

result = forest_model.predict(x_valid)

print("F1 Score {0}".format(f1_score(y_valid, result)))

probs = forest_model.predict_proba(x_valid)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

fpr, tpr, thresholds = roc_curve(y_valid, probs)

plot_roc_curve(fpr, tpr)
forest_model = RandomForestClassifier(random_state=12345, n_estimators=best_estim, max_depth=best_depth)

forest_model.fit(x_upsampled, y_upsampled)

result = forest_model.predict(x_valid)

print("F1 Score {0}".format(f1_score(y_valid, result)))

probs = forest_model.predict_proba(x_valid)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

fpr, tpr, thresholds = roc_curve(y_valid, probs)

plot_roc_curve(fpr, tpr)
from sklearn.utils import shuffle



def downsample(features, target, fraction):

    features_zeros = features[target == 0]

    features_ones = features[target == 1]

    target_zeros = target[target == 0]

    target_ones = target[target == 1]



    features_downsampled = pd.concat(

        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])

    target_downsampled = pd.concat(

        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])

    

    features_downsampled, target_downsampled = shuffle(

        features_downsampled, target_downsampled, random_state=12345)

    

    return features_downsampled, target_downsampled



x_downsampled, y_downsampled = downsample(x_train, y_train, 0.25)
forest_model = RandomForestClassifier(random_state=12345, n_estimators=best_estim, max_depth=best_depth, class_weight='balanced')

forest_model.fit(x_downsampled, y_downsampled)

result = forest_model.predict(x_valid)

print("F1 Score {0}".format(f1_score(y_valid, result)))

probs = forest_model.predict_proba(x_valid)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

fpr, tpr, thresholds = roc_curve(y_valid, probs)

plot_roc_curve(fpr, tpr)
forest_model = RandomForestClassifier(random_state=12345, n_estimators=best_estim, max_depth=best_depth)

forest_model.fit(x_downsampled, y_downsampled)

result = forest_model.predict(x_valid)

print("F1 Score {0}".format(f1_score(y_valid, result)))

probs = forest_model.predict_proba(x_valid)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

fpr, tpr, thresholds = roc_curve(y_valid, probs)

plot_roc_curve(fpr, tpr)
x_upsampled, y_upsampled = upsample(x_train, y_train, 4)



forest_model = RandomForestClassifier(random_state=12345, n_estimators=best_estim, max_depth=best_depth)

forest_model.fit(x_upsampled, y_upsampled)

result = forest_model.predict(x_test)

f1_score(result, y_test)
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt



def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()



probs = forest_model.predict_proba(x_test)

probs = probs[:, 1]



print("AUC Score {0}".format(roc_auc_score(y_test, probs)))

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)