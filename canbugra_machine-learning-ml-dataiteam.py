# linear regression

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/heart-disease-uci/heart.csv',sep = ',')

df_new = df.head(20)



compareScore = []



x = df_new.age.values.reshape(-1,1)

y = df_new.trestbps.values.reshape(-1,1)



plt.scatter(x,y)

plt.xlabel('age')

plt.ylabel('trestbps')

plt.show()



from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()

linear_reg.fit(x,y)



b0 = linear_reg.intercept_

b1 = linear_reg.coef_

print("b0 is ",b0)

print("b1 is ",b1)



array = np.array([0,100,200,300,400,500,600,700,800,900,1000]).reshape(-1,1)

y_head = linear_reg.predict(array)

y_head

array

plt.plot(array,y_head, color='red')

plt.xlabel('age')

plt.ylabel('trestbps')
# multiple linear regression

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/heart-disease-uci/heart.csv',sep=',')

df_new = df.head(10)



x = df_new.iloc[:,[3,4]].values

y = df_new.age.values.reshape(-1,1)



from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(x,y)



b0 = linear_reg.intercept_

b1 = linear_reg.coef_



print("b0 is ",b0)

print("b1 is ",b1)



x = np.array([[10,20],[100,200]])

y_head = linear_reg.predict(x)

plt.plot(x,y_head)

plt.xlabel('trestbps and chol')

plt.ylabel('age')
# polynomial linear regression

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv('../input/heart-disease-uci/heart.csv',sep=',')

df_new = df.head(10)



x = df_new.cp.values.reshape(-1,1)

y = df_new.chol.values.reshape(-1,1)



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



polynomial_reg = PolynomialFeatures(degree=4)

x_new = polynomial_reg.fit_transform(x)



linear_model = LinearRegression()

linear_model.fit(x_new,y)



y_new = linear_model.predict(x_new)

plt.plot(x_new,y_new,color='red')

plt.xlabel('cp')

plt.ylabel('chol')
# random forest

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df = df.head(10)



x = df.age.values.reshape(-1,1)

y = df.trestbps.values.reshape(-1,1)



from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators = 100,random_state=42)

rf.fit(x,y)



x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = rf.predict(x_)



plt.scatter(x,y,color='red')

plt.plot(x_,y_head,color='green')

plt.xlabel('age')

plt.ylabel('trestbps')

# Decision Tree



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df_new = df.head(10)



x = df_new.age.values.reshape(-1,1)

y = df_new.trestbps.values.reshape(-1,1)



from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)



x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = tree_reg.predict(x_)





plt.scatter(x,y,color='red')

plt.plot(x_,y_head,color='green')

plt.xlabel('age')

plt.ylabel('trestbps')

#logistic regression

import pandas as pd

import numpy as np

import matplotlib.pyplot as plot



data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.drop(["cp","sex","fbs","restecg"],axis=1,inplace = True)

y = data.age.values

print("Mean of ages: ",np.mean(y))

data.age = [1 if each > np.mean(y) else 0 for each in data.age]

x_data = data.drop("age",axis = 1)



x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values



from sklearn.model_selection  import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)



from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



lr.fit(x_train.T,y_train.T)

print("test accuracy: ",format(lr.score(x_test.T,y_test.T)))
# logistic regression 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.drop(["sex","cp","fbs","restecg"],axis=1,inplace=True)

y = data.age.values

print("Mean of ages: ",np.mean(y))

data.age = [1 if each < np.mean(y) else 0 for each in data.age.values]

y = data.age.values

x = data.drop("age",axis = 1)



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)



from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train,y_train)



print("Test accuracy: ",format(lr.score(x_test,y_test)))



lrScore = lr.score(x_test, y_test) * 100

compareScore.append(lrScore)
# knn(K-Nearest Neighbors)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df.drop(['sex','cp','fbs','restecg','exang','oldpeak','slope','ca','thal','thalach'],axis = 1,inplace = True)

target_1 = df[df.target == 1]

target_0 = df[df.target == 0]

plt.scatter(target_1.trestbps,target_1.chol,color="red",label="target_1")

plt.scatter(target_0.trestbps,target_0.chol,color="green",label="target_0")

plt.show()
y = df.target.values

x_data = df.drop(['target','age'],axis=1)



#normalization

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=24)



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

print('{} nn score: {}'.format(3,knn.score(x_train,y_train)))

knnScore = knn.score(x_test, y_test) * 100

compareScore.append(knnScore)



# confusion matrix

y_pred = knn.predict(x_test)

y_true = y_test

                     

from sklearn.metrics import confusion_matrix  



cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

# Support Vector Machine(SVM)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/heart-disease-uci/heart.csv')

a = df['chol']

df.chol = [1 if each > np.mean(a) else 0 for each in df.chol]



chol_1 = df[df.chol == 1]

chol_0 = df[df.chol == 0]



sns.countplot(x = "chol", data = df)

df.loc[:,'chol'].value_counts()



plt.scatter(chol_1.age,chol_1.trestbps,label='over_chol',color="red")

plt.scatter(chol_0.age,chol_0.trestbps,label='lowel_chol',color="green")

plt.xlabel('age')

plt.ylabel('trestbps')
y = df.chol.values.reshape(-1,1)

x_data = df.drop('chol',axis=1)

#x_data = df.iloc[:,[0,1,2,3,5,6]]

x_data

# normalization

x = ((x_data - np.min(x_data)) / (np.max(x_data)- np.min(x_data))).values



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)



from sklearn.svm import SVC



svm = SVC(random_state = 1)

svm.fit(x_train,y_train)



print("print accuracy of svm algo: ",svm.score(x_test,y_test))



svmScore = svm.score(x_test, y_test) * 100

compareScore.append(svmScore)

# confusion matrix

y_pred = svm.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_true")

plt.ylabel("y_pred")

plt.show()

 
# naive bayes

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df.age = [1 if each > np.mean(df.age) else 0 for each in df.age]



age_old = df[df.age == 1]

age_young = df[df.age == 0]

plt.scatter(age_old.trestbps,age_old.chol,label = "older",color="red")

plt.scatter(age_young.trestbps,age_young.chol,label="younger",color="green")

plt.xlabel('trestbps')

plt.ylabel('chol')

plt.show()
y = df.age.values.reshape(-1,1)

x_data = df.drop('age',axis=1)



x = ((x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))).values



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=24)



from sklearn.naive_bayes import GaussianNB



nb = GaussianNB() 

nb.fit(x_train,y_train)

print("naive_bayes accuracy: ",nb.score(x_test,y_test))



nbScore = nb.score(x_test, y_test) * 100

compareScore.append(nbScore)

# confusion matrix

y_pred = nb.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_true")

plt.ylabel("y_head")

plt.show()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



algoList = ["LogisticRegression", "KNN", "SVM", "NaiveBayes"]

comparison = {"Models" : algoList, "Accuracy" : compareScore}

dfComparison = pd.DataFrame(comparison)



newIndex = (dfComparison.Accuracy.sort_values(ascending = False)).index.values

sorted_dfComparison = dfComparison.reindex(newIndex)





data = [go.Bar(

               x = sorted_dfComparison.Models,

               y = sorted_dfComparison.Accuracy,

               name = "Scores of Models",

               marker = dict(color = "rgba(116,173,209,0.8)",

                             line=dict(color='rgb(0,0,0)',width=1.0)))]



layout = go.Layout(xaxis= dict(title= 'Models',ticklen= 5,zeroline= False))



fig = go.Figure(data = data, layout = layout)



iplot(fig)