import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

df.drop(['id'], axis=1, inplace=True)

print(df.shape)

df.head()
# The whole freaking column is null, lol. Screw it, drop it

print(df.isna().sum())

df.drop(['Unnamed: 32'], axis=1, inplace=True)

df.head()
%matplotlib inline

df = pd.get_dummies(df)

df.head()
fig = plt.figure(figsize=(32, 32))

corr = df.corr().round(3)

sns.heatmap(corr, annot=True)
print(corr)

# Just coz i couldn't see in the heatmap, lol
def plot_this(df, plot_type='boxplot'):

    fig = plt.figure(figsize=(32, 32))

    for i in range(df.shape[1]):

        plt.subplot(4, 8, i + 1)

        if plot_type=='boxplot':

            sns.boxplot(df.iloc[:, i])

        elif plot_type == 'distplot':

            sns.distplot(df.iloc[:, i], color=(0.2 + (i+1)/50.0, 0.0 + (i+1)/100.0, 1.0 - (i+1)/50.0, 0.3))

    plt.show()
# Just checking the boxplot for each features to determine the IQR's, to help us choosing the right scaler model

plot_this(df)
# Plotting the distribution

plot_this(df, plot_type='distplot')
# The dataset looks pretty much normalised, hence we only have to focus on bringin our data within the IGR range,

# we can do so using the RobustScaler in sklean.preprocessing

# Before that let's seperate the features and labels

train = df.copy()

X = train.drop(['diagnosis_B', 'diagnosis_M'], axis=1)

y = train[['diagnosis_B', 'diagnosis_M']]
# Robust Scaling the data to bring it to the IQR range

# IDK whytf RobustScaler wasn't working, so i decided i'll use my own RobustScaler

median_values = X.median().values

std_values = X.std().values



X = (X - median_values) / std_values

X.head()
# Still no use

plot_this(X)
# Let's not waste time and continue with model selection part, first let's implement without info gain, info gain

# is not necessary since almost all of the data features have a pretty good correlation with the output labels

def train_model(model, x_t, y_t, x_v, y_v):

    model.fit(x_t, y_t)

    print('Training Score', model.score(x_t, y_t))

    print('Validation Score', model.score(x_v, y_v))

    print('f1_score', f1_score(model.predict(x_v), y_v))

    print('Classification Report\n', classification_report(y_v, model.predict(x_v)))

    print('Confusion Matrix\n', confusion_matrix(y_v, model.predict(x_v)))

    

    return model
# converting 2 categorical labels to a single one, so that the standard models can fit.

from sklearn.model_selection import train_test_split

Y = []

for ele in y.values:

    if ele[0] == 1:

        Y.append(0)

    elif ele[1] == 1:

        Y.append(1)

Y = np.array(Y)

print(Y)
x_t, x_v, y_t, y_v = train_test_split(X.values, Y, test_size=0.25, random_state=42)
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier



lgs = LogisticRegression(solver='lbfgs')

knn = KNeighborsClassifier()

xgb = XGBClassifier(n_estimators=500, max_depth=5,learning_rate=0.1,scale_pos_weight=1.4266790777602751)

dtc = DecisionTreeClassifier()



print('\nLogistic Regression\n')

lgs = train_model(lgs, x_t, y_t, x_v, y_v)



print('\nKNN\n')

knn = train_model(knn, x_t, y_t, x_v, y_v)



print('\nXGBoost\n')

xgb = train_model(xgb, x_t, y_t, x_v, y_v)



print('\nDecision Trees\n')

dtc = train_model(dtc, x_t, y_t, x_v, y_v)
# Looks like we got our results and it's pretty clear that Logistic Regression performs the best, with the highest

# Accuracy, f1_score and Recall(which plays the most crucial role, since we are trying reduce the number of 

# True Negatives).