from numpy import genfromtxt

EEG = genfromtxt("../input/EEG data.csv", delimiter=",")
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



def run_experiment(removal):

    X = np.delete(EEG, list(set(removal+[14])), axis=1)

    y = EEG[:, [14]]

    y = y.reshape(12811, )

    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.33, random_state=42

        )

    rfc = RandomForestClassifier(n_estimators=5)

    rfc.fit(X_train, y_train)

    print(rfc.score(X_test, y_test))
removal = [2,3,4,5,6,7,8,9,10,11,12,13]

run_experiment(removal)
removal = [1,2,3,4,5,6,7,8,9,10,11,12,13]

run_experiment(removal)
removal = [0,2,3,4,5,6,7,8,9,10,11,12,13]

run_experiment(removal)