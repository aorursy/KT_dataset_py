# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error

%matplotlib inline
# importing data

data = pd.read_csv('../input/fifa19/data.csv')
data.head()
data.columns
data.info()
# assigning dataframe to 'df' and droping unnecessary columns 

df = data.drop(columns=['Unnamed: 0', 'ID', 'Photo', 'Flag', 

                     'Club Logo', 'Real Face', 'Jersey Number', 

                     'Loaned From', 'Contract Valid Until', 'Release Clause'], axis=1)
df.head()
df.columns
# looking how distributed are null values

plt.figure(figsize=(12,8))

sns.heatmap(df[['Overall', 'Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes']].isnull(), cbar=False, cmap='viridis', yticklabels=False)

plt.show()
# assign 'df' to name 'properties' and droping null values

properties = df[['Overall', 'Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes']]

properties.dropna(inplace=True)

properties['Crossing'].isnull().sum()
plt.figure(figsize=(12,8))

sns.heatmap(properties[['Overall', 'Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes']].isnull(), cbar=False, cmap='viridis', yticklabels=False)

plt.show()
properties.describe()
# when I have clean dataframe with all important properties of a player I can start investigate 

# which of them is most important for overall performance.
# from correlation matrix taking just overall performance column and sort it

properties.corr()['Overall'].sort_values(ascending=False)
# as we can see in the graph below there are plenty of properties that correlate. 

# In order to avoid collinearity we have to exclude one from correlating pair 

# (except when high (>.8) correlation appears with target feature this case 'Overall')

plt.figure(figsize=(12,8))

sns.heatmap(properties.corr(), cmap='viridis')

plt.show()
# after excluding collinear properties I got these features

# for those of whom this and previous steps are confusing I suggest reading about collinearity 

regModel = properties[['Overall', 'Strength', 'Stamina', 'Jumping', 'Composure', 'Reactions', 'ShortPassing', 'GKKicking']]
regModel.corr()["Overall"].sort_values(ascending=False).head(12)



# here I explain features:

# Reactions: measures how quickly a player responds to a situation happening around him. 

# It has nothing to do with the player’s speed.





# Composure: this attribute determines at what distance the player 

# with the ball starts feeling the pressure from the opponent. 

# This then affects the chances of the player making an error when he shoots, 

# passes and crosses.
# last time checking to avoid collinearity

plt.figure(figsize=(8,6))

sns.heatmap(regModel.corr(), cmap='viridis', annot=True)

plt.show()
# assign x and y

X = regModel[['Strength', 'Stamina', 'Jumping', 'Composure', 'Reactions', 'ShortPassing', 'GKKicking']]

y = regModel['Overall']
from sklearn.model_selection import train_test_split
# spliting variables into train and test, setting test_size and random state

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
# instantiate LinearRegression model and assign it to 'lm'

lm = LinearRegression()
# fitting my training data to the model

lm.fit(X_train, y_train)
# here I'm creating dataframe from my model coefficients

coefs = pd.DataFrame(lm.coef_, X_train.columns, columns=["Coefficients"])
print(f'rSquared: {round(lm.score(X_train, y_train), 3)}')

coefs

# rSquared is a metric which describe how good is your model.

# coefficients interpretation: if you hold other features fixed and increase 'Reactions'

# in one unit you get increase in 'Overall' by 0.379
# in order to get predicions I input 'X_test' values to the model's predict method 

# and assign to the variable 'predictions'

predictions = lm.predict(X_test)
# plot actual vs. predicted values

plt.rcParams.update({'font.size': 12})

plt.title('Actual vs. Predicted')

plt.xlabel('Actual Values')

plt.ylabel('Predicted Values')

plt.scatter(y_test, predictions)

plt.show()
from sklearn import metrics
# here I use a metric (Root Mean Squared Error) which represents difference between real and predicted values.

# this difference is expressed in the same units as predicted value (in this case 'Overall')

# other way to test your model is to plot residuals distribution. If it visually seems normally distributed and mean around 0 

# it indicates that your model is the right decision for this data

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Root Mean Squared Error: {round(rmse, 3)}")

plt.title('Residuals')

sns.distplot((y_test-predictions),bins=50)

plt.show()
# creating separate dataframe with "Position" column

features = df[['Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes', 'Position']]
# cheching for null values

features.isnull().sum()
# droping null values

features.dropna(inplace=True)
features.head()
# checking how many unique positions I have

features.Position.nunique()
# function which changes position from goolkeeper to 1, defender to 2, midfielder to 3, striker to 4.

def simplePosition(col):

    if (col == 'GK'):

        return 1

    elif ((col == 'RB') | (col == 'LB') | (col == 'CB') | (col == 'LCB') | (col == 'RCB') | (col == 'RWB') | (col == 'LWB') ):

        return 2

    elif ((col == 'LDM') | (col == 'CDM') | (col == 'RDM') | (col == 'LM') | (col == 'LCM') | 

          (col == 'CM') | (col == 'RCM') | (col == 'RM') | (col == 'LAM') | (col == 'CAM') | 

          (col == 'RAM') | (col == 'LW') | (col == 'RW')):

        return 3

    elif ((col == 'RS') | (col == 'ST') | (col == 'LS') | (col == 'CF') | (col == 'LF') | (col == 'RF')):

        return 4

    else:

        return 'error'
# applying that funcion to position column

features["Position"] = features.Position.apply(simplePosition)
features.Position.unique()
features.head()
from sklearn.preprocessing import StandardScaler
# instantiating StandardScaler object

scaler = StandardScaler()
# fitting data to the scaler object except position column

scaler.fit(features.drop('Position', axis=1))
# perform actual scaling

scaled_fetures = scaler.transform(features.drop('Position', axis=1))
# and now we have dataframe with scaled features

df_features = pd.DataFrame(scaled_fetures, columns=features.columns[:-1])
from sklearn.model_selection import train_test_split
# assign scaled features dataframe and position column to the varibles.

feat = df_features

targ = features.Position

# spliting data into train and test

xTrain, xTest, yTrain, yTest = train_test_split(feat, targ, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xTrain, yTrain)
pred = knn.predict(xTest)
from sklearn.metrics import classification_report, confusion_matrix
# confusion matrix and classification report explains how good our classification algorith performs

print(confusion_matrix(yTest, pred))

print('\n')

print(classification_report(yTest, pred))
confMatrix = confusion_matrix(yTest, pred)
# this function plots good looking confusion matrix, accuracy and error rates. 

def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')





    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
# here is same, but a bit better looking confusion matrix 

plot_confusion_matrix(cm = confMatrix, normalize = False, 

                      target_names = ['Goolkeeper', 'Defender', 'Midfielder', 'Striker'],

                      title= "Confusion Matrix K=1")
# here I am looping through same classification algorithm with different n_neighbors values (it takes some time)

error_rate = []

for i in range(1,30):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(xTrain, yTrain)

    pred_i = knn.predict(xTest)

    error_rate.append(np.mean(pred_i != yTest))
# error rate for different number neighbors (K)

# as we can see around 8 or 9 neighbors error rate reach plateau

plt.plot(range(1,30),error_rate,color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# rerun classification algorith with n_neighbors where error rate was smallest

# it is always better to choose odd number of neighbors

Knn = KNeighborsClassifier(n_neighbors=9)

Knn.fit(xTrain, yTrain)

pred_9 = Knn.predict(xTest)

print(confusion_matrix(yTest, pred_9))

print('\n')

print(classification_report(yTest, pred_9))
conf9Matrix = confusion_matrix(yTest, pred_9)
# model prediction improved from ~86% to ~90%

plot_confusion_matrix(cm = conf9Matrix, normalize = False, 

                      target_names = ['Goolkeeper', 'Defender', 'Midfielder', 'Striker'],

                      title = "Confusion Matrix with K=9")