# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Import Dataset

data = pd.read_csv("../input/Admission_Predict.csv")

data.shape
data.head(2)
data.columns.values
data.drop('Serial No.', axis=1, inplace=True)
data.rename({'Chance of Admit ': 'Chance of Admit', 'LOR ':'LOR'}, axis=1, inplace=True)
#Let's see top 10 observation row and column wise

data.head(10)
# Let's see the detail information of dataset

data.info()
## General statistics of the data

data.describe()
## Correlation coeffecients heatmap

sns.heatmap(data.corr(), annot=True).set_title('Correlation Factors Heat Map', color='black', size='20')
# Isolating GRE Score data

GRE = pd.DataFrame(data['GRE Score'])

GRE.describe()
# # Probability Distribution

sns.distplot(GRE).set_title('Probability Distribution for GRE Test Scores', size='20')

plt.show()
# Correlation Coeffecients for GRE Score Test

GRE_CORR = pd.DataFrame(data.corr()['GRE Score'])

GRE_CORR.drop('GRE Score', axis=0, inplace=True)

GRE_CORR.rename({'GRE Score': 'GRE Correlation Coeff'}, axis=1, inplace=True)

GRE_CORR
# Isolating and describing TOEFL Score

TOEFL = pd.DataFrame(data['TOEFL Score'], columns=['TOEFL Score'])

TOEFL.describe()
# Probability distribution for TOEFL Scores

sns.distplot(TOEFL).set_title('Probability Distribution for TOEFL Scores', size='20')

plt.show()
# Isolating and describing the CGPA

CGPA = pd.DataFrame(data['CGPA'], columns=['CGPA'])

CGPA.describe()
sns.distplot(CGPA).set_title('Probability Distribution Plot for CGPA', size='20')

plt.show()
RES_Count = data.groupby(['Research']).count()

RES_Count = RES_Count['GRE Score']

RES_Count = pd.DataFrame(RES_Count)

RES_Count.rename({'GRE Score': 'Count'}, axis=1, inplace=True)

RES_Count.rename({0: 'No Research', 1:'Research'}, axis=0, inplace=True)

plt.pie(x=RES_Count['Count'], labels=RES_Count.index, autopct='%1.1f%%')

plt.title('Research', pad=5, size=30)

plt.show()
# Isolating and describing 

University_Rating = data.groupby(['University Rating']).count()

University_Rating = University_Rating['GRE Score']

University_Rating = pd.DataFrame(University_Rating)

University_Rating.rename({'GRE Score': 'Count'}, inplace=True, axis=1)

University_Rating
# Barplot for the distribution of the University Rating

sns.barplot(University_Rating.index, University_Rating['Count']).set_title('University Rating', size='20')

plt.show()
#Isolating and describing

SOP = pd.DataFrame(data.groupby(['SOP']).count()['GRE Score'])

SOP.rename({'GRE Score':'Count'}, axis=1, inplace=True)

SOP
# Barplot for SOP 

sns.barplot(SOP.index, SOP['Count']).set_title('Statement of Purpose', size='20')

plt.show()
LOR = pd.DataFrame(data.groupby(['LOR']).count()['GRE Score'])

LOR.rename({'GRE Score':'Count'}, axis=1, inplace=True)

LOR
# Distribution of the LOR

sns.barplot(LOR.index, LOR['Count']).set_title('Letter of Recommendation', size='20')

plt.show()
data['Chance of Admit']

sns.distplot(data['Chance of Admit']).set_title('Probability Distribution of Chance of Admit', size='20')

plt.show()
data.describe()['Chance of Admit']
COA_corr = pd.DataFrame(data.corr()['Chance of Admit'])

COA_corr.rename({'Chance of Admit': 'Correlation Coeffecient'}, axis=1, inplace=True)

COA_corr.drop('Chance of Admit', inplace=True)

COA_corr.sort_values(['Correlation Coeffecient'], ascending=False, inplace=True)

COA_corr_x = COA_corr.index

COA_corr_y = COA_corr['Correlation Coeffecient']

sns.barplot(y=COA_corr_x,x=COA_corr_y).set_title('Chance of Admit Correlation Coeffecients', size='20')

plt.show()
COA_corr
X = data.drop(['Chance of Admit'], axis=1)

y = data['Chance of Admit']
#Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X[['CGPA','GRE Score', 'TOEFL Score']] = scaler.fit_transform(X[['CGPA','GRE Score', 'TOEFL Score']])
#Splitting

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
#### Linear Regression (All Features)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
pd.DataFrame({"Actual": y_test, "Predict": y_test}).head()
from sklearn.metrics import r2_score, mean_squared_error

lr_r2 = r2_score(y_test, y_pred)

lr_mse = mean_squared_error(y_test, y_pred)

lr_rmse = np.sqrt(lr_mse)

print('Linear Regression R2 Score: {0} \nLinear Regression MSE: {1}, \nLinear Regression RMSE:{2}'.format(lr_r2, lr_mse, lr_rmse))
sns.set(rc={'figure.figsize':(12.7,8.27)})

sns.distplot((y_test - y_pred))

plt.title('Linear Regression (All Features) Residuals', fontdict={'fontsize':20}, pad=20)

plt.show()
sns.set(rc={'figure.figsize':(12.7,8.27)})

# sns.(y_test, y_pred)

sns.scatterplot(y_test, y_pred)

plt.show()
X_selected = X[['CGPA', 'GRE Score', 'TOEFL Score']]

X_sel_train, X_sel_test, y_train, y_test = train_test_split(X_selected, y, random_state=101)
lr_sel = LinearRegression()

lr_sel.fit(X_sel_train, y_train)

lr_sel_predictions = lr_sel.predict(X_sel_test)
lr_sel_r2 = r2_score(y_test, lr_sel_predictions)

lr_sel_mse = mean_squared_error(y_test, lr_sel_predictions)

lr_sel_rmse = np.sqrt(lr_sel_mse)

print('Linear Regression R2 Score: {0} \nLinear Regression MSE: {1}, \nLinear Regression RMSE:{2}'.format(lr_sel_r2, lr_sel_mse, lr_sel_rmse))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 100, random_state = 101)

rfr.fit(X_train,y_train)

y_head_rfr = rfr.predict(X_test) 
from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test, y_head_rfr))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 101)

dtr.fit(X_train,y_train)

y_head_dtr = dtr.predict(X_test) 
from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_head_dtr))
y = np.array([r2_score(y_test,y_pred),r2_score(y_test,y_head_rfr),r2_score(y_test,y_head_dtr)])

x = ["LinearRegression","RandomForestReg.","DecisionTreeReg."]

plt.bar(x,y)

plt.title("Comparison of Regression Algorithms")

plt.xlabel("Regressor")

plt.ylabel("r2_score")

plt.show()
red = plt.scatter(np.arange(0,80,5),y_pred[0:80:5],color = "red")

green = plt.scatter(np.arange(0,80,5),y_head_rfr[0:80:5],color = "green")

blue = plt.scatter(np.arange(0,80,5),y_head_dtr[0:80:5],color = "blue")

black = plt.scatter(np.arange(0,80,5),y_test[0:80:5],color = "black")

plt.title("Comparison of Regression Algorithms")

plt.xlabel("Index of Candidate")

plt.ylabel("Chance of Admit")

plt.legend((red,green,blue,black),('LR', 'RFR', 'DTR', 'REAL'))

plt.show()
data["Chance of Admit"].plot(kind = 'hist',bins = 200,figsize = (6,6))

plt.title("Chance of Admit")

plt.xlabel("Chance of Admit")

plt.ylabel("Frequency")

plt.show()
# reading the dataset

df = pd.read_csv("../input/Admission_Predict.csv")

df.shape
# it may be needed in the future.

serialNo = df["Serial No."].values

df.drop(["Serial No."],axis=1,inplace = True)



df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
X = df.drop(["Chance of Admit"],axis=1)

y = df["Chance of Admit"].values
# separating train (80%) and test (%20) sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 101)
# normalization

from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))

X_train[X_train.columns] = scalerX.fit_transform(X_train[X_train.columns])

X_test[X_test.columns] = scalerX.transform(X_test[X_test.columns])
y_train_01 = [1 if each > 0.8 else 0 for each in y_train]

y_test_01  = [1 if each > 0.8 else 0 for each in y_test]



# list to array

y_train_01 = np.array(y_train_01)

y_test_01 = np.array(y_test_01)
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()

logr.fit(X_train,y_train_01)
y_predlogr = logr.predict(X_test)
from sklearn import metrics

from sklearn.metrics import accuracy_score

print("Accuracy Score:", accuracy_score(y_predlogr, y_test_01))
# confusion matrix

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test_01,y_predlogr))

cm_lrc = confusion_matrix(y_test_01,y_predlogr)

# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29
# cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_lrc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01, y_predlogr))

print("recall_score: ", recall_score(y_test_01, y_predlogr))



from sklearn.metrics import f1_score

print("f1_score: ",f1_score(y_test_01, y_predlogr))
cm_lrc_train = confusion_matrix(y_train_01,logr.predict(X_train))

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_lrc_train,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.title("Test for Train Dataset")

plt.show()
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(X_train,y_train_01)

y_pred_svm = svm.predict(X_test)

print("score: ", svm.score(X_test,y_test_01))
# confusion matrix

from sklearn.metrics import confusion_matrix

cm_svm = confusion_matrix(y_test_01,y_pred_svm)

# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29

cm_svm
# cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_svm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01, y_pred_svm))

print("recall_score: ", recall_score(y_test_01,y_pred_svm))



from sklearn.metrics import f1_score

print("f1_score: ",f1_score(y_test_01, y_pred_svm))
cm_svm_train = confusion_matrix(y_train_01, svm.predict(X_train))

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_svm_train, annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.title("Test for Train Dataset")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train_01)

y_pred_nb = nb.predict(X_test)

print("score: ", nb.score(X_test,y_test_01))
# confusion matrix

from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_test_01, y_pred_nb)

# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29

cm_nb
# cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_nb,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01, y_pred_nb))

print("recall_score: ", recall_score(y_test_01,y_pred_nb))



from sklearn.metrics import f1_score

print("f1_score: ",f1_score(y_test_01, y_pred_nb))
cm_nb_train = confusion_matrix(y_train_01,nb.predict(X_train))

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_nb_train,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.title("Test for Train Dataset")

plt.show()
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train,y_train_01)

y_pred_dtc = dtc.predict(X_test)

print("score: ", dtc.score(X_test,y_test_01))
# confusion matrix

from sklearn.metrics import confusion_matrix

cm_dtc = confusion_matrix(y_test_01, y_pred_dtc)

# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29

cm_dtc
# cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_dtc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01, y_pred_dtc))

print("recall_score: ", recall_score(y_test_01, y_pred_dtc))



from sklearn.metrics import f1_score

print("f1_score: ",f1_score(y_test_01, y_pred_dtc))
cm_dtc_train = confusion_matrix(y_train_01,dtc.predict(X_train))

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_dtc_train,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.title("Test for Train Dataset")

plt.show()
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100,random_state = 1)

rfc.fit(X_train,y_train_01)



y_pred_rfc = rfc.predict(X_test)



print("score: ", rfc.score(X_test, y_test_01))
# confusion matrix

from sklearn.metrics import confusion_matrix

cm_rfc = confusion_matrix(y_test_01, y_pred_rfc)

# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29

cm_rfc
# cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_rfc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01, y_pred_rfc))

print("recall_score: ", recall_score(y_test_01, y_pred_rfc))



from sklearn.metrics import f1_score

print("f1_score: ",f1_score(y_test_01, y_pred_rfc))
cm_rfc_train = confusion_matrix(y_train_01, rfc.predict(X_train))

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_rfc_train,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.title("Test for Train Dataset")

plt.show()
from sklearn.neighbors import KNeighborsClassifier



# finding k value

scores = []

for each in range(1,50):

    knn_n = KNeighborsClassifier(n_neighbors = each)

    knn_n.fit(X_train, y_train_01)

    scores.append(knn_n.score(X_test, y_test_01))

    

plt.plot(range(1,50),scores)

plt.xlabel("k")

plt.ylabel("accuracy")

plt.show()
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(X_train, y_train_01)



y_pred_knn = knn.predict(X_test)

print("score of 3 :",knn.score(X_test,y_test_01))
# confusion matrix

from sklearn.metrics import confusion_matrix

cm_knn = confusion_matrix(y_test_01, y_pred_knn)

# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29

cm_knn
# cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_knn,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01, y_pred_knn))

print("recall_score: ", recall_score(y_test_01, y_pred_knn))



from sklearn.metrics import f1_score

print("f1_score: ",f1_score(y_test_01, y_pred_knn))
cm_knn_train = confusion_matrix(y_train_01,knn.predict(X_train))

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_knn_train,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.title("Test for Train Dataset")

plt.show()
y = np.array([logr.score(X_test, y_test_01), svm.score(X_test, y_test_01), nb.score(X_test, y_test_01), dtc.score(X_test,y_test_01), rfc.score(X_test, y_test_01), knn.score(X_test, y_test_01)])

#x = ["LogisticRegression","SVM","GaussianNB","DecisionTreeClassifier","RandomForestClassifier","KNeighborsClassifier"]

x = ["LogisticReg.", "SVM", "GNB", "Dec.Tree", "Ran.Forest", "KNN"]



plt.bar(x,y)

plt.title("Comparison of Classification Algorithms")

plt.xlabel("Classfication")

plt.ylabel("Score")

plt.show()
data = pd.read_csv("../input/Admission_Predict.csv")

data.shape
data.columns
data = data.rename(columns = {'Chance of Admit ':'ChanceOfAdmit'})

serial = data["Serial No."]

data.drop(["Serial No."],axis=1,inplace = True)
data = (data - np.min(data))/(np.max(data)-np.min(data))

X = data.drop(["ChanceOfAdmit"],axis=1)

y = data.ChanceOfAdmit
# for data visualization

from sklearn.decomposition import PCA

pca = PCA(n_components = 1, whiten= True )  # whitten = normalize

pca.fit(X)

x_pca = pca.transform(X)

x_pca = x_pca.reshape(400,)

dictionary = {"x":x_pca,"y":y}

data1 = pd.DataFrame(dictionary)

print("data:")

print(data1.head())

print("\ndata:")

print(data.head())
data["Serial No."] = serial

from sklearn.cluster import KMeans

wcss = []

for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)

plt.xlabel("k values")

plt.ylabel("WCSS")

plt.show()



kmeans = KMeans(n_clusters=3)

clusters_knn = kmeans.fit_predict(X)



data["label_kmeans"] = clusters_knn





plt.scatter(data[data.label_kmeans == 0 ]["Serial No."], data[data.label_kmeans == 0].ChanceOfAdmit,color = "red")

plt.scatter(data[data.label_kmeans == 1 ]["Serial No."], data[data.label_kmeans == 1].ChanceOfAdmit,color = "blue")

plt.scatter(data[data.label_kmeans == 2 ]["Serial No."], data[data.label_kmeans == 2].ChanceOfAdmit,color = "green")

plt.title("K-means Clustering")

plt.xlabel("Candidates")

plt.ylabel("Chance of Admit")

plt.show()



data["label_kmeans"] = clusters_knn

plt.scatter(data1.x[data.label_kmeans == 0 ],data1[data.label_kmeans == 0].y,color = "red")

plt.scatter(data1.x[data.label_kmeans == 1 ],data1[data.label_kmeans == 1].y,color = "blue")

plt.scatter(data1.x[data.label_kmeans == 2 ],data1[data.label_kmeans == 2].y,color = "green")

plt.title("K-means Clustering")

plt.xlabel("X")

plt.ylabel("Chance of Admit")

plt.show()
data["Serial No."] = serial



from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(X, method="ward")

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()



from sklearn.cluster import AgglomerativeClustering

hiyerartical_cluster = AgglomerativeClustering(n_clusters = 3, affinity= "euclidean", linkage = "ward")

clusters_hiyerartical = hiyerartical_cluster.fit_predict(X)



data["label_hiyerartical"] = clusters_hiyerartical



plt.scatter(data[data.label_hiyerartical == 0 ]["Serial No."],data[data.label_hiyerartical == 0].ChanceOfAdmit,color = "red")

plt.scatter(data[data.label_hiyerartical == 1 ]["Serial No."],data[data.label_hiyerartical == 1].ChanceOfAdmit,color = "blue")

plt.scatter(data[data.label_hiyerartical == 2 ]["Serial No."],data[data.label_hiyerartical == 2].ChanceOfAdmit,color = "green")

plt.title("Hierarchical Clustering")

plt.xlabel("Candidates")

plt.ylabel("Chance of Admit")

plt.show()



plt.scatter(data1[data.label_hiyerartical == 0 ].x,data1.y[data.label_hiyerartical == 0],color = "red")

plt.scatter(data1[data.label_hiyerartical == 1 ].x,data1.y[data.label_hiyerartical == 1],color = "blue")

plt.scatter(data1[data.label_hiyerartical == 2 ].x,data1.y[data.label_hiyerartical == 2],color = "green")

plt.title("Hierarchical Clustering")

plt.xlabel("X")

plt.ylabel("Chance of Admit")

plt.show()
fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
df = pd.read_csv('../input/Admission_Predict.csv')

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})



newDF = pd.DataFrame()

newDF["GRE Score"] = df["GRE Score"]

newDF["TOEFL Score"] = df["TOEFL Score"]

newDF["CGPA"] = df["CGPA"]

newDF["Chance of Admit"] = df["Chance of Admit"]



x_new = df.drop(["Chance of Admit"],axis=1)

y_new = df["Chance of Admit"].values



# separating train (80%) and test (%20) sets

from sklearn.model_selection import train_test_split

x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_new, y_new, test_size = 0.20, random_state = 42)



# normalization

from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))

x_train_new[x_train_new.columns] = scalerX.fit_transform(x_train_new[x_train_new.columns])

x_test_new[x_test_new.columns] = scalerX.transform(x_test_new[x_test_new.columns])



from sklearn.linear_model import LinearRegression

lr_new = LinearRegression()

lr_new.fit(x_train_new, y_train_new)

y_head_lr_new = lr_new.predict(x_test_new)



from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test_new, y_head_lr_new))