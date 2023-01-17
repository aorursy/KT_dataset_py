import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

df=pd.read_csv("../input/winequality-red.csv")

df.head()
# Getting the consolidated information about the dataset

df.info()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

sns.set_color_codes("pastel")

%matplotlib inline



sns.barplot(df.quality,df.alcohol)
#relation of features with other features



plt.style.use('ggplot')

fig=plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),annot=True)
#import various ML algorithms to be used from the library



from sklearn.svm import SVC,NuSVC

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

classification_algos_name = ["SVC", "NuSVC", "KNeighborsClassifier", "GaussianNB", "MultinomialNB", "SGDClassifier", "LogisticRegression", "DecisionTreeClassifier",

                            "ExtraTreeClassifier", "QuadraticDiscriminantAnalysis", "LinearDiscriminantAnalysis", "RandomForestClassifier", "AdaBoostClassifier",

                            "GradientBoostingClassifier", "XGBClassifier"]

classification_algos=[SVC(),

                      NuSVC(nu=0.285),

                      KNeighborsClassifier(),

                      GaussianNB(),

                      MultinomialNB(),

                      SGDClassifier(),

                      LogisticRegression(),

                      DecisionTreeClassifier(),

                      ExtraTreeClassifier(),

                      QuadraticDiscriminantAnalysis(),

                      LinearDiscriminantAnalysis(),

                      RandomForestClassifier(),

                      AdaBoostClassifier(),

                      GradientBoostingClassifier(),

                      XGBClassifier()]
df.isnull().sum()
#Converting discreate values of the quality column into categorial values



bins=(2,6.5,8)

category=["bad","good"]

df["quality"]=pd.cut(df["quality"],bins=bins, labels=category)
le=LabelEncoder()

df["quality"]=le.fit_transform(df["quality"])

df["quality"].unique()
df["quality"].value_counts()
x_train, y_test, x_train_target, y_test_target = train_test_split(df.drop("quality", axis=1), df["quality"], test_size = 0.25, random_state = 1)

print(x_train.shape, " ",y_test_target.shape)
accuracy_score_list = []

for mod in classification_algos:

    model = mod

    model.fit(x_train, x_train_target)

    pred = model.predict(y_test)

    accuracy_score_list.append(accuracy_score(y_test_target,pred))

for idx,i in enumerate(accuracy_score_list):

    print(classification_algos_name[idx]," ",i)
from bokeh.io import output_notebook

data = pd.DataFrame({"algorithms": classification_algos_name, "accuracy_score": accuracy_score_list})

data['color'] = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724','#30678D','#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']

output_notebook()
from bokeh.io import show, output_file

from bokeh.models import ColumnDataSource, FactorRange

from bokeh.plotting import figure

from bokeh.palettes import Spectral6

from bokeh.transform import factor_cmap



source = ColumnDataSource(data=data)



p = figure(x_range=data['algorithms'],

           y_range=(0,1),

           plot_width = 800,

           plot_height = 600,

           title = "Comparison",

           tools="hover",

           tooltips="@algorithms: @accuracy_score")

p.vbar(x='algorithms', top='accuracy_score',color= 'color',

       width=0.95, source=source)



p.xgrid.grid_line_color = None

p.xaxis.major_label_orientation = 120

output_file('comparison.html')

show(p)