import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
# Read in data which was created by PreProcessing.py
X_df = pd.read_csv("../input/puntanalytics/x.csv")
y_df = pd.read_csv("../input/puntanalytics/y.csv", names=["target"])
#define variables for diff in orientation and direction. Using cos function so that diff between 360 degrees and 0 degrees is the same

X_df['o_diff'] = abs(np.cos(np.radians(X_df.o - X_df.o_partner)))
X_df['dir_diff'] = abs(np.cos(np.radians(X_df.dir - X_df.dir_partner)))
X_df['self_diff'] = abs(np.cos(np.radians(X_df.o - X_df.dir)))
X_df['partner_diff'] = abs(np.cos(np.radians(X_df.o_partner - X_df.dir_partner)))
# Convert Start_Time variable to a categorical variable that denotes the hour of game start
X_df['Start_Time'] = pd.to_datetime(X_df['Start_Time'])
X_df['start_hour'] = X_df.Start_Time.apply(lambda x: x.hour)
# Coerce boolean to integer to create a dummy variable for high vs. low playing temperature
X_df["high_temp"] = (X_df["Temperature"] >= X_df["Temperature"].median())

# Coerce boolean to integer to create a dummy variable for mid/late week
X_df["late_week"] = (X_df["Week"] >= X_df["Week"].median())

# Coerce boolean to integer to create a dummy variable for middle of field
maxY = 53.3
X_df["average_y"] = X_df[["y","y_partner"]].mean(axis = 1)
X_df["middle_field"] = (X_df.average_y > maxY * 0.25) & (X_df.average_y < maxY * 0.75)
# Remove unneeded columns
X_df = X_df.drop(["Season_Year","GameKey","PlayID","GSISID","Time","Event",
                  "GamePlayKey","GamePlayTimeKey","GSISID_partner","Game_Date",
                  "PlayPlayerPartnerID", "Start_Time","Temperature","Week", "y",
                 "y_partner", "average_y"], axis=1)
# Create boolean vectors to indicate any missing values
for column in X_df.columns:
    if X_df[column].isnull().sum() > 0:
        col_str = str(column) + "_isnull"
        X_df[col_str] = X_df[column].isnull()
# Grouping all non-sunday games together
X_df.loc[X_df.Game_Day != "Sunday", "Game_Day"] = "Not_Sunday"

# Create dummies for categorical features
for column in ["Season_Type","Game_Day"]:
    # Create dummy variable columns, drop one dummy column to avoid multicollinearity, and remove the original, non-binary column
    X_df = pd.concat([X_df.drop(column, axis=1), pd.get_dummies(X_df[column], drop_first=True)], axis=1)
    
X_df = pd.concat([X_df.drop("start_hour", axis=1), pd.get_dummies(X_df["start_hour"], drop_first=True, prefix='hour')], axis=1)
#normalizing data from 0 to 1 to see most important components in logistic regression
X_df = X_df.astype(float)
X_df = X_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# Split into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df.values.ravel(), test_size=0.2, random_state = 53)
# Instantiate logistic regression object and fit to training data
logreg = LogisticRegression(random_state = 53, solver='lbfgs').fit(X_train,y_train)
# Training and Test set accuracy of logistic regression model
trainset_acc = logreg.score(X_train,y_train)
testset_acc = logreg.score(X_test,y_test)
print('logreg training set accuracy: {:.3f}'.format(trainset_acc))
print('logreg testing set accuracy: {:.3f}'.format(testset_acc))
# Five-fold cross-validation accuracy of the logistic regression model on the training set
scores = cross_val_score(logreg, X_train, y_train, cv=5)
print('logreg mean cv accuracy: {:.3f}'.format(np.mean(scores)))
logRegCoef = sorted(list(zip(X_train.columns, logreg.coef_[0])), key=lambda x: x[1])

for i in range(len(logRegCoef)):
    #formatting print for easier reading
    print(logRegCoef[i][0], ":", ' '*(25 - len(logRegCoef[i][0]) - int(logRegCoef[i][1] < 0)), "{:.4f}".format(logRegCoef[i][1]))
