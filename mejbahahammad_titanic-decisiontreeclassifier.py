import warnings

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing



from sklearn.model_selection import StratifiedKFold

import statsmodels.api as sm

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier







# currently its available as part of mlxtend and not sklearn

from mlxtend.classifier import EnsembleVoteClassifier



from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.model_selection import train_test_split





warnings.filterwarnings('ignore')



from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt





from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn import metrics

from sklearn.preprocessing import StandardScaler



%matplotlib inline
datasets = pd.read_csv('../input/titanic/train.csv')
datasets.head()
datasets['Sex'] = np.where(datasets['Sex'] == 'male', 0, datasets['Sex'])

datasets['Sex'] = np.where(datasets['Sex'] == 'female', 1, datasets['Sex'])

datasets.fillna(0, inplace=True)
X = datasets.drop(['Survived', 'Name', 'Cabin', 'Embarked', 'Ticket'], axis = 1)

y = datasets['Survived']
std_scale = preprocessing.StandardScaler().fit(X)

X_std = std_scale.transform(X)



minmax_scale = preprocessing.MinMaxScaler().fit(X)

X_minmax = minmax_scale.transform(X)
print('Mean before standardization: Pclass = {:.1f}, Sex = {:.1f}, Age = {:.1f}, SibSp = {:.1f}, Parch = {:.1f}, Fare = {:.1f}'

      .format(X.iloc[:,0].mean(), X.iloc[:,1].mean(), X.iloc[:,2].mean(), X.iloc[:,3].mean(), X.iloc[:,4].mean(), X.iloc[:,5].mean()))

print('\n')



print('Standard Deviation before standardization: Pclass = {:.1f}, Sex = {:.1f}, Age = {:.1f}, SibSp = {:.1f}, Parch = {:.1f}, Fare = {:.1f}'

      .format(X.iloc[:,0].std(), X.iloc[:,1].std(), X.iloc[:,2].std(), X.iloc[:,3].std(), X.iloc[:,4].std(), X.iloc[:,5].std()))
print('Mean After standardization: Pclass = {:.1f}, Sex = {:.1f}, Age = {:.1f}, SibSp = {:.1f}, Parch = {:.1f}, Fare = {:.1f}'

      .format(X_std[:,0].mean(), X_std[:,1].mean(), X_std[:,2].mean(), X_std[:,3].mean(), X_std[:,4].mean(), X_std[:,5].mean()))



print('\n')

print('Standard Deviation After standardization: Pclass = {:.1f}, Sex = {:.1f}, Age = {:.1f}, SibSp = {:.1f}, Parch = {:.1f}, Fare = {:.1f}'

      .format(X_std[:,0].std(), X_std[:,1].std(), X_std[:,2].std(), X_std[:,3].std(), X_std[:,4].std(), X_std[:,5].std()))
plt.figure(figsize = (10, 8))

datasets.boxplot()     # plot boxplot  

plt.title("Bar Plot", fontsize=16)

plt.tight_layout()

plt.show()
datasets.boxplot(by="Survived", figsize=(12, 6))
datasets.groupby(by = "Survived").mean().plot(kind="bar")



plt.title('Suervived vs Measurements')

plt.ylabel('mean measurement(cm)')

plt.xticks(rotation=0)  # manage the xticks rotation

plt.grid(True)



# Use bbox_to_anchor option to place the legend outside plot area to be tidy 

plt.legend(loc="upper left", bbox_to_anchor=(1,1))
sc = StandardScaler()

sc.fit(X)

X = sc.transform(X)



model = KMeans(n_clusters=3, random_state=11)

model.fit(X)

print(model.labels_)
datasets['PredSurvived'] =  np.choose(model.labels_, [1, 0, 2]).astype(np.int64)



print("Accuracy :", metrics.accuracy_score(datasets.Survived, datasets.PredSurvived))

print("Classification report :", metrics.classification_report(datasets.Survived, datasets.PredSurvived))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
DTC = DecisionTreeClassifier()

DTC.fit(X, y)
test_data = pd.read_csv('../input/titanic/test.csv')
test_data = test_data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
test_data['Sex'] = np.where(test_data['Sex'] == 'male', 0, test_data['Sex'])

test_data['Sex'] = np.where(test_data['Sex'] == 'female', 1, test_data['Sex'])

test_data.fillna(0, inplace=True)
predict = DTC.predict(test_data)
output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                       'Survived': predict})



output.to_csv('TitanicFinalSubmission.csv', index=False)