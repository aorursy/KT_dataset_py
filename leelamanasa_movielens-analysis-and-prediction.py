# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
ratings_col = ['UserID','MovieID','Rating','Timestamp']

ratings_data=pd.read_table("../input/Data science with Python 1/ratings.dat" , sep = "::", header = None,names = ratings_col,engine='python')

ratings_data.head()
users_col = ['UserID','Genere','Age','Occupation','Zip-code']

users_data=pd.read_table("../input/Data science with Python 1/users.dat" , sep = "::", header = None,names = users_col,engine='python')

users_data
movies_col = ['MovieID','Title','Genres']

movies_data=pd.read_table("../input/Data science with Python 1/movies.dat" , sep = "::", header = None,names = movies_col,engine='python')

movies_data.head()
Master_Data1 = pd.merge(movies_data, ratings_data[['UserID','MovieID','Rating']], how='right')

Master_Data1
Master_Dataf= Master_Data2.drop(['Genres'],axis=1)

Master_Data2.head()
Master_Data2 = pd.merge(Master_Data1, users_data[['UserID','Genere','Age','Occupation']], how='inner')

Master_Data2
Master_Dataf.Age.plot(kind="kde")
Master_Dataf.Age.plot.hist(bins=25)

plt.title("Distribution of user's ages")

plt.ylabel('count of users')

plt.xlabel('Age')
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

Master_Dataf['Age_Group'] = pd.cut(Master_Dataf.Age, range(0, 81, 10), right=False, labels=labels)

Master_Dataf[['Age', 'Age_Group']].drop_duplicates()[:10]
groupby_title = Master_Dataf.groupby('Title')

groupby_rating = Master_Dataf.groupby('Rating')

groupby_uid = Master_Dataf.groupby('UserID')
top_25_Movies = groupby_title.size().sort_values(ascending=False).to_frame().reset_index().head(25)

top_25_Movies.style.background_gradient(cmap="Greens")
toy_Stotry_data = groupby_title.get_group("Toy Story (1995)")

toy_Stotry_data
user_rating = toy_Stotry_data.groupby('UserID')['Rating']

user_rating.head()
plt.figure(figsize=(10,10))

plt.scatter(toy_Stotry_data['MovieID'],toy_Stotry_data['Rating'])

plt.title('Plot showing  the user rating of the movie “Toy Story”')

plt.show()
toy_Stotry_data[['Title','Age_Group']]
userid_2696 = groupby_uid.get_group(2696)

userid_2696[['UserID','Rating']].style.background_gradient(cmap="Reds")
list_geners=Master_Data2['Genere']
new_geners=[]

for i in list_geners:

    if i == "F":

        new_geners.append(1)

    else:

        new_geners.append(0)
f = Master_Data2.copy()

df = f.join(f.pop('Genres').str.get_dummies('|'))

df.head()
Master_Data2['new_geners']=new_geners

Master_Data2
predict_rating = Master_Data2[Master_Data2['MovieID']==1]

predict_rating
predict_rating.corr()
Master_Data_500=Master_Data2[0:1500:]

Master_Data_500
x_features = Master_Data_500[['MovieID','Age','Occupation','new_geners']].to_numpy()
y_features = Master_Data_500[['Rating']].to_numpy().reshape(-1,1)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
x_features_train,x_features_test,y_features_train,y_features_test = train_test_split(x_features,y_features,test_size=0.33,random_state = 42)
#Linear Regression

sc = StandardScaler()

logreg = LogisticRegression()

x_features_train=sc.fit_transform(x_features_train)

x_features_test=sc.transform(x_features_test)

logreg.fit(x_features_train,y_features_train)

Y_pred = logreg.predict(x_features_test)

acc_log = round(logreg.score(x_features_train, y_features_train) * 100, 2)

acc_log
from sklearn import preprocessing



#SVM

svc = SVC()

x_features_train= preprocessing.scale(x_features_train)

svc.fit(x_features_train, y_features_train)

Y_pred = svc.predict(x_features_test)

acc_svc = round(svc.score(x_features_train, y_features_train) * 100, 2)

acc_svc
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_features_train, y_features_train)

Y_pred = random_forest.predict(x_features_test)

random_forest.score(x_features_train, y_features_train)

acc_random_forest = round(random_forest.score(x_features_train, y_features_train) * 100, 2)

acc_random_forest
# K Nearest Neighbors Classifier



knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(x_features_train, y_features_train)

Y_pred = knn.predict(x_features_test)

acc_knn = round(knn.score(x_features_train, y_features_train) * 100, 2)

acc_knn
# Perceptron



perceptron = Perceptron()

perceptron.fit(x_features_train, y_features_train)

Y_pred = perceptron.predict(x_features_test)

acc_perceptron = round(perceptron.score(x_features_train, y_features_train) * 100, 2)

acc_perceptron
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(x_features_train, y_features_train)

Y_pred = gaussian.predict(x_features_test)

acc_gaussian = round(gaussian.score(x_features_train, y_features_train) * 100, 2)

acc_gaussian
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(x_features_train, y_features_train)

Y_pred = linear_svc.predict(x_features_test)

acc_linear_svc = round(linear_svc.score(x_features_train, y_features_train) * 100, 2)

acc_linear_svc
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_features_train, y_features_train)

Y_pred = decision_tree.predict(x_features_test)

acc_decision_tree = round(decision_tree.score(x_features_train, y_features_train) * 100, 2)

acc_decision_tree
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(x_features_train, y_features_train)

Y_pred = sgd.predict(x_features_test)

acc_sgd = round(sgd.score(x_features_train, y_features_train) * 100, 2)

acc_sgd
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron',

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron,

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
c=sns.catplot(x='Model',y='Score', kind='bar',data=models)

c.set_xticklabels(rotation=90)