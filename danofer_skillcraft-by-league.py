import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import pandas as pd

df = pd.read_csv('../input/SkillCraft.csv')

print(df.shape)

df.head()
y = df.LeagueIndex.astype(int)

X = df.drop(["LeagueIndex","GameID"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define classifiers

classifiers = [ 

    GradientBoostingClassifier(n_estimators=250,max_depth=5),

    RandomForestClassifier(n_estimators=300,max_depth=8),

    KNeighborsClassifier(25)

]
# target_names = list(set(y)) # technically, league names.



for classifier in classifiers:

    print (classifier.__class__.__name__)

    start = time.time()

    classifier.fit(X_train, y_train)

    print ("  -> Training time:", time.time() - start)

    preds = classifier.predict(X_test)

    print()

    print(classification_report(y_test, preds)) # , target_names=target_names