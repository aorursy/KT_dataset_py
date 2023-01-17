import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = sns.load_dataset("iris")
display(data)
val = data.columns.to_list()

plt.figure(figsize=(45,15))
for i in range(0,4):
    plt.subplot(1,4,i+1)
    plt.hist(data.iloc[:,i])
    plt.title("{}".format(val[i]),fontsize=40,color="white")
print("Variable distribution")
null_value = data.isnull().sum().to_frame(name="Null Value")
null_value
def encode(x):
    if "setosa" in x:
        return 0
    elif "versicolor" in x:
        return 1
    else:
        return 2
species_encoded = data.species.apply(lambda x : encode(x)).to_frame(name="species_encoded")
process = pd.concat([data,species_encoded],axis=1)
processed_data = process.drop(columns=["species"])
display(processed_data)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
scaler = MinMaxScaler()

X = processed_data.iloc[:,:-1]



y = processed_data.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

display(X)
display(y)
model1 = DecisionTreeClassifier()
model1.fit(X_train,y_train)
model1_pred = model1.predict(X_test)
Accuracy1 = accuracy_score(y_test,model1_pred)


print("DecisionTreeClassifier accuracy score : {}".format(Accuracy1))
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
cluster1 = KMeans(n_clusters=3)
cluster1_pred = cluster1.fit_predict(X)

score1 = adjusted_rand_score(y,cluster1_pred)
print("Rand score of KMeans : {}".format(score1))
cluster2 = DBSCAN(eps = 1.5, min_samples=50)
cluster2_pred = cluster2.fit_predict(X)
score2 = adjusted_rand_score(y,cluster2_pred)
print("Rand score of DBSCAN : {}".format(score2))
cluster3 = AgglomerativeClustering(n_clusters=3,linkage="average")
cluster3_pred = cluster3.fit_predict(X)
score3 = adjusted_rand_score(y,cluster3_pred)
print("Rand score od AgglomerativeClustering : {}".format(score3))
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

linkage_matrix = linkage(X,"average")
plt.figure(figsize=(16,16))

dendrogram(linkage_matrix)
plt.show()
