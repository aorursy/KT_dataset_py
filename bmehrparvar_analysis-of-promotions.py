import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import cross_validation, tree

from IPython.display import Image



data = pd.read_csv('../input/HR_comma_sep.csv')

import matplotlib.pyplot as plt
promoted = data[data.promotion_last_5years == 1]

promotedNot = data[data.promotion_last_5years == 0]



sns.kdeplot(promotedNot.satisfaction_level, color = 'r')

sns.kdeplot(promoted.satisfaction_level, color = 'g')

plt.legend(['Not promoted','Promoted'])

plt.title('Satisfaction level')
pro_left = len(promoted[promoted.left==1])/len(promoted)

pro_leftN = len(promoted[promoted.left==0])/len(promoted)

proN_left = len(promotedNot[promotedNot.left==1])/len(promotedNot)

proN_leftN = len(promotedNot[promotedNot.left==0])/len(promotedNot)



mydata = np.array([[0,0,proN_leftN] ,[0,1,proN_left],[1,0,pro_leftN],[1,1,pro_left]])

mydataframe = pd.DataFrame(mydata, columns=['promoted', 'left', 'percentage'])

mydataframe

sns.barplot(x='promoted', y='percentage', hue='left',data=mydataframe)
X = data[['last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']]

y = data[['promotion_last_5years']]



shuffle = cross_validation.ShuffleSplit(len(X), n_iter=10, test_size=0.3, random_state=0)

model = tree.DecisionTreeClassifier(max_depth=10)

scores = cross_validation.cross_val_score(model, X, y, cv=shuffle)

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

clf = tree.DecisionTreeClassifier()



clf = clf.fit(X, y)

with open("tree.png", 'w') as f:

     f = tree.export_graphviz(clf, out_file=f)

Image('tree.png')