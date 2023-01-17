# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/mobile-price-classification/train.csv')
data.head()

data.info()   
"""
    id -> ID
    battery_power -> Total energy a battery can store in one time measured in mAh -numeric
    blue -> Has bluetooth or not -categorical nominal
    clock_speed -> speed at which microprocessor executes instructions -numeric
    dual_sim -> Has dual sim support or not -categorical nominal
    fc -> Front Camera mega pixels -numeric
    four_g -> Has 4G or not -categorical nominal
    int_memory -> Internal Memory in Gigabytes -numeric
    m_dep -> Mobile Depth in cm -numeric
    mobile_wt -> Weight of mobile phone -numeric
    n_cores -> Number of cores of processor -numeric
    pc -> Primary Camera mega pixels -numeric
    px_height -> Pixel Resolution Height -numeric
    px_width -> Pixel Resolution Width -numeric
    ram -> Random Access Memory in Megabytes -numeric
    sc_h -> Screen Height of mobile in cm -numeric
    sc_w -> Screen Width of mobile in cm -numeric
    talk_time -> longest time that a single battery charge will last when you are -numeric
    three_g -> Has 3G or not -categorical nominal
    touch_screen -> Has touch screen or not -categorical nominal
    wifi -> Has wifi or not -categorical nominal
    price_range -> This is the target variable with value of 0(low cost), 1(medium cost),
                            2(high cost) and 3(very high cost). -categorical ordinal
"""
data.isnull().sum() #checking null values, there is no null values.
data.describe() #show basic statistical details
sns.pairplot(data) #Plot pairwise relationships in a dataset.
#visualize the correlation
plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True,fmt=".0%")
plt.show() 

#my aim is predicting the price_range and because of that i will use most correlated features with price_range: battery_power, 
#px_height, px-width and ram. RAM has the biggest impact on price range.
sns.lmplot('ram', 'battery_power', data, hue='price_range', fit_reg=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()

#We can say that battery and ram has no relation together, but if ram is higher than 3100 it belongs most probably high price.
data['price_range'].value_counts().plot.pie() 
#in every category(0(low cost), 1(medium cost),
#                            2(high cost) and 3(very high cost)), there are almost same amount of sample in dataset.
#Not stack in one class.
sns.barplot(x="dual_sim", y='ram', hue="price_range", data=data, color='red') 
#As seen in the figure, having dual sim support or not has almost no impact on price

plt.bar(data['price_range'].values, data['ram'].values, color='blue')
bars = ('0(low cost)', '1(medium cost)','2(high cost)','3(very high cost)')
plt.title('RAM effect on Price Range')
plt.xlabel('Price Range')
plt.ylabel('Ram')
y_pos = np.arange(len(bars))
plt.xticks(y_pos, bars)

features = ['battery_power', 'px_height', 'px_width', 'ram'] #4 most correlated features with price
X = data[features]
y = data['price_range'] #target is predicting price range
X.head()

#I prefer supervised learning: classification models, because my target is categorical (class).

from sklearn.model_selection import train_test_split

# 80% Train, 10% Validation, %10 Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#Train -> Train the model (Train our dataset)
#Test -> Evalute the accuracy of model
#Validation -> Evaluate a given model
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
models = {
    'GaussianNB': GaussianNB(),
    'LogisticRegression': LogisticRegression(),
    'DecisionTree' :DecisionTreeClassifier(criterion='gini', splitter='random'),
    'DecisionTree2' :DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth= 14, min_samples_leaf = 2, min_samples_split= 2),
    'DecisionTree3' :DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=27,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=6,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='random'),
    'KNN' : KNeighborsClassifier(),
    'KNN2' : KNeighborsClassifier(n_neighbors=9, algorithm='auto'), #best model
    'RandomForestClassifer' : RandomForestClassifier(criterion='entropy'),
    'RandomForestClassifer2' : RandomForestClassifier()
}

for m in models:
  model = models[m]
  model.fit(X_train, y_train)
  score = model.score(X_valid, y_valid)
  print(f'{m} validation score => {score}')
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth': range(1,20), 
              'min_samples_split': range(2,10), 
              'min_samples_leaf': range(2, 10),
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random']}

gcv = GridSearchCV(DecisionTreeClassifier(), parameters).fit(X_train, y_train)
gcv.best_estimator_
gcv.best_params_
gcv.best_estimator_.score(X_valid, y_valid)


#another way

parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}]
clf = KNeighborsClassifier()
Grid1 = GridSearchCV(clf, parameters, cv=4)
Grid1.fit(X_train, y_train)

Grid1.best_estimator_

scores = Grid1.cv_results_
scores['mean_test_score']

modelS = KNeighborsClassifier(n_neighbors=9, algorithm='auto')
modelS.fit(X_train, y_train)

validation_score = modelS.score(X_valid, y_valid)
print(f'Validation score of trained model: {validation_score}')

test_score = modelS.score(X_test, y_test)
print(f'Test score of trained model: {test_score}')
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

y_predictions = modelS.predict(X_test)

conf_matrix = confusion_matrix(y_predictions, y_test)
print(f'Accuracy: {accuracy_score(y_predictions, y_test)}')
print(f'Confussion matrix: \n{conf_matrix}\n')

sns.heatmap(conf_matrix, annot=True)