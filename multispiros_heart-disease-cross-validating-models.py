import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.preprocessing import MinMaxScaler,normalize 

from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings("ignore")
heart =pd.read_csv("../input/heart.csv")
heart.shape
heart.isnull().sum()
heart.describe()
plt.figure(figsize=(20,8))

plt.title("Correlation Heatmap")

sns.heatmap(heart.corr(),annot=True,cmap="Reds",fmt="f",cbar=True)
menheart = heart[(heart.sex == 1)]

womanheart = heart[(heart.sex == 0)]

mensum = len(menheart)

womansum = len(womanheart)

plt.figure(figsize=(4,8))

plt.bar(['MALE'],[mensum],color='g',label='MALE',width=0.5)

plt.bar(['FEMALE'],[womansum],color='r',label='FEMALE',width=0.5)

plt.title("Genders total in dataset")

plt.legend()

plt.show()
pd.crosstab(heart.sex,heart.target).plot(kind="bar",figsize=(15,6),color=['r','b' ],alpha = 0.5)

plt.title('Patients total by gender')

plt.xlabel('Gender (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["no heart disease","heart disease"])

plt.ylabel('Total')

plt.show()
mantrue=heart[(heart.sex==1) & (heart.target==1)]

womantrue=heart[(heart.sex==0) & (heart.target==1)]

plt.rcParams["figure.figsize"] = (15,6)

plt.scatter((mantrue.age),(mantrue.chol),color="b",label="MALE")

plt.scatter((womantrue.age),(womantrue.chol),color="r",label="FEMALE")

plt.axhline(y=200,color="g")

plt.ylabel('Cholesterol levels')

plt.xlabel('Age')

plt.title("Cholesterol levels(mg/dl) by age on patients with heart disease.")

plt.legend()

plt.plot()
hearttrue = heart[(heart.target == 1)]

heartfalse = heart[(heart.target == 0)]

sumtrue = len(hearttrue)

sumfalse = len(heartfalse)

plt.figure(figsize=(4,8))

plt.title("Total for target=1 and target=0")

plt.bar(['true'],[sumtrue],color="b",alpha=0.5,label="heart disease",width=0.5)

plt.bar(['false'],[sumfalse],color="r",alpha=0.5,label='no heart disease',width=0.5) 

plt.legend()

plt.ylabel('Total')

plt.show()
pd.crosstab(heart.cp,heart.target).plot(kind="bar",figsize=(15,6),color=['g','b'],)

plt.title('Total target=1 and target=0 by pain type')

plt.xlabel('Pain type, 0 = nopain , 1 = pain1 , 2 = pain2 , 3 = pain3')

plt.ylabel('Total')

plt.legend(("no heart disease","heart disease"))

plt.xticks(rotation=1 )

plt.show()
a = pd.get_dummies(heart['cp'], prefix = "cp")

b = pd.get_dummies(heart['thal'], prefix = "thal")

c = pd.get_dummies(heart['slope'], prefix = "slope")

frames = [heart, a, b, c]

heart = pd.concat(frames, axis = 1)

heart.iloc[0:4,:]
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

X = heart.drop(['target'],axis =1)

y = heart['target']

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']

print(featureScores.nlargest(30,'Score')) 
nlargest = featureScores.nlargest(13,'Score')

largest = nlargest['Specs'].values

large = heart[largest]

target = heart['target']

heart = pd.concat([large,target],axis=1)

heart.head(2)
scaler = MinMaxScaler(feature_range=(0, 1)) 

knnheart = pd.DataFrame(scaler.fit_transform(heart), columns=heart.columns, index=heart.index)

knnheart.iloc[1:5,:]
temp = 0

tempi = 0

df = pd.DataFrame(columns = ["k",'score'])

for i in range (1,11): 

 model=KNeighborsClassifier(n_neighbors=i)

 cv_r2_scores_rf = cross_val_score(model,knnheart.drop('target', 1),knnheart['target'], cv=10)

 mean = np.mean(cv_r2_scores_rf)

 df.loc[i] = i

 df.loc[i,'score'] = mean

 if  mean > temp:

  temp = mean

  tempi = i 

print ("biggest success:",temp,", k =",tempi)
plt.figure(figsize=(16,8))

sns.barplot(y=df.score,x=df.k)

plt.title("k means score(Blue line shows the biggest score)")

plt.axhline(0.84789766407119,0,5)
plt.figure(figsize=(8,5))

model = KNeighborsClassifier(n_neighbors=7)

crossmodel = cross_val_predict(model,knnheart.drop('target', 1),knnheart['target'],cv=10)

conf_mat = confusion_matrix(heart['target'],crossmodel)

plt.title("knn confusion matrix")

sns.heatmap(conf_mat,annot=True,cmap="Blues",fmt="d",cbar=False)
scaler = MinMaxScaler(feature_range=(0, 1)) 

svmheart = pd.DataFrame(scaler.fit_transform(heart), columns=heart.columns, index=heart.index)

svmheart.iloc[1:4,:]
model=SVC(kernel='linear')

cv_r2_scores_rf = cross_val_score(model,svmheart.drop('target', 1),svmheart['target'], cv=10)

for row in cv_r2_scores_rf:

 print(row)

print("Mean 10-Fold R Squared: {}".format(np.mean(cv_r2_scores_rf)))
plt.figure(figsize=(8,5))

crossmodel = cross_val_predict(model,svmheart.drop('target', 1),svmheart['target'],cv=10)

conf_mat = confusion_matrix(heart['target'],crossmodel)

plt.title("svm confusion matrix")

sns.heatmap(conf_mat,annot=True,cmap="Blues",fmt="d",cbar=False)
model=GaussianNB()

cv_r2_scores_rf = cross_val_score(model,heart.drop('target', 1),heart['target'], cv=10,scoring='accuracy')

for row in cv_r2_scores_rf:

 print(row)

print("Mean 10-Fold R Squared: {}".format(np.mean(cv_r2_scores_rf)))
plt.figure(figsize=(8,5))

crossmodel = cross_val_predict(model,heart.drop('target', 1),heart['target'],cv=10)

conf_mat = confusion_matrix(heart['target'],crossmodel)

plt.title("Naive bayes confusion matrix")

sns.heatmap(conf_mat,annot=True,cmap="Blues",fmt="d",cbar=False)
model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)

cv_r2_scores_rf = cross_val_score(model, heart.drop('target',1), heart['target'], cv=10,scoring='accuracy')

for row in cv_r2_scores_rf:

 print(row)

print("Mean 10-Fold R Squared: {}".format(np.mean(cv_r2_scores_rf)))
plt.figure(figsize=(8,5))

crossmodel = cross_val_predict(model,heart.drop('target', 1),heart['target'],cv=10)

conf_mat = confusion_matrix(heart['target'],crossmodel)

plt.title("Decicion tree confusion matrix")

sns.heatmap(conf_mat,annot=True,cmap="Blues",fmt="d",cbar=False)
methods = ["KNN", "SVM", "Naive Bayes","Random Forest"]

accuracy = [84.78,83.37,81.76,83.37]

colors = ["blue", "orange", "pink", "green"]



sns.set_style("whitegrid")

plt.figure(figsize=(10,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=methods, y=accuracy, palette=colors)

plt.show()