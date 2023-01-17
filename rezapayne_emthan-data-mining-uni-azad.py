from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats as st


df = pd.read_csv('../input/database/Diabetes_Diagnosis.csv')

df.describe()
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import mean_absolute_error
import pandas as pd

col = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness',  'bmi', 'diab_pred', 'age']

X_train, X_test, Y_train, Y_test = train_test_split(
    df[col], 
    df['diabetes'],
    train_size=0.85,
    random_state=0
)


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 4, random_state = 0)

clf.fit(X_train, Y_train)

clf.predict(X_test)

val_predictions = clf.predict(X_test)

correct = 0
for i in val_predictions:

    if i == Y_test.values[0][i]:
        correct += 1

        
print(correct , ' correct guess out of ', val_predictions.size)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20), dpi=300)

tree.plot_tree(
    clf,
    filled = True,
    feature_names = col, 
    class_names=['False', 'True'],
)
fig.savefig('imagename.png')
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats as st


df = pd.read_csv('../input/data-mining-bacheha/Diabetes_Diagnosis.csv')


a = 'age'
b = 'bmi'
c = 'insulin'
d = 'glucose_conc'
t = 'diabetes'


aa = df[a]
bb = df[b]
cc = df[c]
dd = df[d]
tt = df[t]


ax1 = df[tt].plot(kind='scatter', x=a, y=b, color='red', alpha=0.5, figsize=(20,10))
df[tt==False].plot(kind='scatter', x=a, y=b, color='green', alpha=0.5, figsize=(20 ,10), ax=ax1)


plt.legend(labels=['True', 'False'])
plt.xlabel(c, size=18)
plt.ylabel(d, size=18)
plt.show()

