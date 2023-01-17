import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_curve, auc
turnover = pd.read_csv("../input/turnover.csv")

turnover.head()
# converting salary into categories

turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(['low', 'medium', 'high']).cat.codes

turnover.head()
# converting department into dummies

department = pd.get_dummies(turnover["sales"])

department.head()
turnover = turnover.drop(["sales"], axis=1)

turnover.head()
# creating correlation matrix

turnover.corr()
# plotting the correlation matrix

# as seaborn is based on matplotlib, we need to use plt.show() to see the plot

sns.heatmap(turnover.corr())

plt.show()
turnover.info()
turnover.describe()
# joining the departments

turnover = turnover.join(department)

turnover.head()
# the percentage of leavers

turnover['left'].value_counts()/len(turnover)*100
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
X = turnover.drop(['left'], axis=1)

y = turnover['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



dt.fit(X_train, y_train)

dt.score(X_train, y_train)*100
pred = dt.predict(X_test)

dt.score(X_test, y_test)*100
# Export our trained model as a .dot file

with open("tree.dot", 'w') as f:

     f = export_graphviz(dt, out_file=f,

                         feature_names = list(X),

                         impurity = True, rounded = True, filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree.dot','-o','tree.png'])



img = Image.open("tree.png")

draw = ImageDraw.Draw(img)

font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)



img.save('sample-out.png')

PImage("sample-out.png")
confusion_matrix(y_test, pred)
fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)

auc(fpr, tpr)