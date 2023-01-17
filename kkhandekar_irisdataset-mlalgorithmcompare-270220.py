#Importing Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/iris-dataset/iris.csv',header='infer')
data.shape
#Checking for missing / null values

data.isna().sum()
data.head()
# --- Finding Correlation ---

data_feature = pd.DataFrame(data,columns=['sepal_length','sepal_width','petal_length','petal_width'])





#plotting the correlation

corr = data_feature.corr(method='pearson')

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='YlGnBu', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(data_feature.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(data_feature.columns)

ax.set_yticklabels(data_feature.columns)

plt.show()
#Sepal Width vs Sepal Length

sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(10,10))

ax = sns.swarmplot (x='sepal_length', y='sepal_width', data=data, hue = 'species')

plt.title('Sepal Width vs Sepal Length')

plt.ylabel('Sepal Width')

plt.xlabel('Sepal Length')
#Petal Length vs Petal Width

sns.set(style="darkgrid")

fig = plt.figure()

fig = sns.relplot(x="petal_length", y="petal_width", hue="species", data=data, kind="scatter", legend="full", height=10,aspect=1,palette="ch:r=-.5,l=.75")

fig.fig.set_size_inches(10,10)

fig.set_titles("Petal Length vs Petal Width")

fig.set_xlabels("Petal Length")

fig.set_ylabels("Petal Width")

plt.show()
#Pair Plot

sns.pairplot(data, hue="species")
# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(6, 15))



sns.set_color_codes("pastel")

sns.barplot(x="sepal_length", y="species", data=data,

            label="Sepal Length", color="b")



sns.set_color_codes("muted")

sns.barplot(x="sepal_width", y="species", data=data,

            label="Sepal Width", color="b")



sns.set_color_codes("deep")

sns.barplot(x="petal_length", y="species", data=data,

            label="Petal Length", color="r")



sns.set_color_codes("dark")

sns.barplot(x="petal_width", y="species", data=data,

            label="Petal Width", color="r")









# Add a legend and informative axis label

ax.legend(ncol=2, loc="upper right", frameon=True)

ax.set(xlim=(0, 24), ylabel="",

       xlabel="Sepal&Petal Dimensions")

sns.despine(left=True, bottom=True)
# --- Importing ML libraries ---

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



#Metrics Libraries

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



#ML Classifier Algorithm Libraries

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

data.columns
#Feature & Target Selection

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

target = ['species']



# Feature& Target  Dataset

X = data[features]

y = data[target]
#Dataset Split  [train = 90%, test = 10%]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0) 



#Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# -- Building Model List --

models = []

models.append(('CART', DecisionTreeClassifier()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RFC', RandomForestClassifier()))
# -- Model Evaluation --

model_results = []

model_names = []



for name, model in models:

    kfold = KFold(n_splits=10, random_state=None, shuffle=False)

    cross_val_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    model_results.append(cross_val_results)

    model_names.append(name)

    print(name, ":--", "Mean Accuracy =", '{:.2%}'.format(cross_val_results.mean()), 

                       "Standard Deviation Accuracy =", '{:.2%}'.format(cross_val_results.std())

         )

         

    
model_names
#Visualisation - Algorithm Compare



fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(model_results)

ax.set_xticklabels(model_names)

plt.show()
#Instantiating SVC Model

svc = SVC()



#Instantiating LDA Model

lda = LinearDiscriminantAnalysis()
#Training the model

svc.fit(X_train, y_train)  #SVC 

lda.fit(X_train, y_train) #LDA
#Converting the X_test to DataFrame

test_df = pd.DataFrame(X_test,columns=features)
test_df.head()
# Making Predictions

pred_svc = svc.predict(X_test)

pred_lda = lda.predict(X_test)



#Appending the predictions to the test - dataset

test_df['svc_prediction'] = pred_svc

test_df['lda_prediction'] = pred_lda
# -- Calculating Metrics

print("Accuracy Score - SVC Model: ",'{:.2%}'.format(accuracy_score(y_test,pred_svc)) )

print("Accuracy Score - LDA Model: ",'{:.2%}'.format(accuracy_score(y_test,pred_lda)) )
test_df.head(15)