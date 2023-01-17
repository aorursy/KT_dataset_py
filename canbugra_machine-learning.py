import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

compareScore = []

df = pd.read_csv('../input/falldata/falldeteciton.csv')

df.info()
data = df.set_index("ACTIVITY")

df_A1 = data.drop([0,2,3,4,5],axis=0)

df_A1.head()
# linear regression

x = df_A1.BP.values.reshape(-1,1)

y = df_A1.HR.values.reshape(-1,1)

plt.scatter(x,y)

plt.xlabel("BP")

plt.ylabel("HR")

plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)



# y = m*x+ b

b = lr.intercept_

m = lr.coef_

print("y =",m,"*x + ",b)
# multiple linear regression

df = pd.read_csv('../input/falldata/falldeteciton.csv')

data = df.set_index('ACTIVITY')

df_A1 = data.drop([0,2,3,4,5],axis=0)

df_A1.head()
x = df_A1.iloc[:,[4,5]].values

y = df_A1.CIRCLUATION.values.reshape(-1,1)



from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(x,y)



# y = m*x + b

b = lr.intercept_

m = lr.coef_



print("y = ",m,"*x + ",b)
#polynomial linear regression

df = pd.read_csv('../input/falldata/falldeteciton.csv')

data = df.set_index('ACTIVITY')

x = data.BP.values.reshape(-1,1)

y = data.CIRCLUATION.values.reshape(-1,1)



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



polynomial_reg = PolynomialFeatures(degree=4)

x_new = polynomial_reg.fit_transform(x)



linear_reg = LinearRegression()

linear_reg.fit(x_new,y)



y_new = linear_reg.predict(x_new)

plt.plot(x_new,y_new,color="red")

#random forest

df = pd.read_csv('../input/falldata/falldeteciton.csv')

data = df.set_index('ACTIVITY')

df_A1 = data.drop([0,2,3,4,5],axis=0)

x = df_A1.BP.values.reshape(-1,1)

y = df_A1.HR.values.reshape(-1,1)



from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(x,y)



x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_new = rf.predict(x_).reshape(-1,1)



plt.scatter(x,y,color="red")

plt.plot(x_,y_new,color="green")

# Decision Tree

df = pd.read_csv('../input/falldata/falldeteciton.csv')

data = df.set_index('ACTIVITY')

df_A1 = data.drop([0,2,3,4,5],axis=0)



x = df_A1.HR.values.reshape(-1,1)

y = df_A1.CIRCLUATION.values.reshape(-1,1)



from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)



x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_new = tree_reg.predict(x_).reshape(-1,1)



plt.scatter(x,y,color="red")

plt.plot(x_,y_new,color="green")



#logistic regression

df = pd.read_csv('../input/falldata/falldeteciton.csv')

data = df.set_index('ACTIVITY')

df_A1 = data.drop([0,2,3,4,5],axis = 0)

y = df_A1.BP.values

print("Mean of SL: ",np.mean(y))

df_A1.BP = [1 if each > np.mean(y) else 0 for each in df_A1.BP]

y = df_A1.BP.values

x_data = df_A1.drop("SL",axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size=0.3,random_state = 42)



from sklearn.linear_model import LogisticRegression

   

lr = LogisticRegression()

lr.fit(x_train,y_train)

     

print("Test accuracy: ",format(lr.score(x_test,y_test)))

lrScore = lr.score(x_test, y_test) * 100

compareScore.append(lrScore)



# confusion matrix

y_true = y_test

y_pred = lr.predict(x_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_true")

plt.ylabel("y_pred")

plt.show()
# k-NN(k Nearest Neighbor)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv('../input/falldata/falldeteciton.csv')

df.drop(['ACTIVITY','TIME','SL','EEG'],axis=1,inplace=True)

y_data = df.HR.values

mean = np.mean(y_data)

y = [1 if each > mean else 0 for each in y_data] 

x_data = df.drop('HR',axis=1)





#Normalization

x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

print("{} nn score: {}".format(3,knn.score(x_test,y_test)))

knnScore = knn.score(x_test, y_test) * 100

compareScore.append(knnScore)

# confusion matrix

y_true = y_test

y_pred = knn.predict(x_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_true")

plt.ylabel("y_pred")

plt.show()
# SVM(Support Vector Machine)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/falldata/falldeteciton.csv')

a = df['CIRCLUATION']

df.CIRCLUATION = [1 if each > np.mean(a) else 0 for each in df.CIRCLUATION]



cir_1 = df[df.CIRCLUATION == 1]

cir_0 = df[df.CIRCLUATION == 0]



sns.countplot(x='CIRCLUATION',data = df)

df.loc[:,'CIRCLUATION'].value_counts()
plt.scatter(cir_1.BP,cir_1.HR,label='High_Circluation',color="red")

plt.scatter(cir_0.BP,cir_0.HR,label='Low_Circluation',color="green")

plt.xlabel("BP")

plt.ylabel("HR")

plt.show()
y = df.CIRCLUATION.values.reshape(-1,1)

x_data = df.drop('CIRCLUATION',axis=1)



x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)



from sklearn.svm import SVC

svm = SVC(random_state=1)

svm.fit(x_train,y_train)



print("print accuracy SVM: ",svm.score(x_test,y_test))

svmScore = svm.score(x_test, y_test) * 100

compareScore.append(svmScore)

# Confusion Matrix



y_pred = svm.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot = True, linewidths = 0.5, linecolor="red", fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

# naive-bayes

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/falldata/falldeteciton.csv')

df.TIME = [1 if each > np.mean(df.TIME) else 0 for each in df.TIME]



time_less = df[df.TIME == 0]

time_more = df[df.TIME == 1]



sns.countplot(x = 'TIME', data = df)

df.loc[:,'TIME'].value_counts

plt.scatter(time_less.BP,time_less.HR,label = "less_time",color = "red")

plt.scatter(time_more.BP,time_more.HR,label = "more_time",color = "green")

plt.xlabel("BP")

plt.ylabel("HR")

plt.show()

y = df.TIME.values.reshape(-1,1)

x_data = df.drop('TIME',axis = 1)



x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 42)



from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()



nb.fit(x_train,y_train)



print("accuracy naive bayes: ",nb.score(x_test,y_test))

nbScore = nb.score(x_test, y_test) * 100

compareScore.append(nbScore)

# Confusion Matrix



y_pred = nb.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



# %% cm visualization

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True, linewidths = 0.5,linecolor="red",fmt = ".0f", ax=ax)

plt.xlabel("y_predict")

plt.ylabel("y_true")

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
