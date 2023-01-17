import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
print(os.listdir("../input"))


data = pd.read_csv("../input/data.csv")
data = data.drop(['Unnamed: 32','id'],axis = 1)





print(data.shape)
print
print(data.columns)
data.head()
data.diagnosis.unique()


m = plt.hist(data[data["diagnosis"] == "M"].radius_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].radius_mean,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()
frequent_malignant_radius_mean = m[0].max()
index_frequent_malignant_radius_mean = list(m[0]).index(frequent_malignant_radius_mean)
most_frequent_malignant_radius_mean = m[1][index_frequent_malignant_radius_mean]
print("Most frequent malignant radius mean is: ",most_frequent_malignant_radius_mean)
data_bening = data[data["diagnosis"] == "B"]
data_malignant = data[data["diagnosis"] == "M"]
desc = data_bening.radius_mean.describe()
Q1 = desc[4]
Q3 = desc[6]
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")
print("Outliers: ",data_bening[(data_bening.radius_mean < lower_bound) | (data_bening.radius_mean > upper_bound)].radius_mean.values)
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['radius_mean', 'texture_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()
plt.hist(data_bening.radius_mean,bins=50,fc=(0,1,0,0.5),label='Bening',normed = True,cumulative = True)
sorted_data = np.sort(data_bening.radius_mean)
y = np.arange(len(sorted_data))/float(len(sorted_data)-1)
plt.plot(sorted_data,y,color='red')
plt.title('CDF of bening tumor radius mean')
plt.show()
plt.figure(figsize = (15,10))
sns.jointplot(data.radius_mean,data.area_mean,kind="regg")
plt.show()


# Also we can look relationship between more than 2 distribution
sns.set(style = "white")
df = data.loc[:,["radius_mean","area_mean","fractal_dimension_se"]]
g = sns.PairGrid(df,diag_sharey = False,)
g.map_lower(sns.kdeplot,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,lw =3)
plt.show()


f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()
ranked_data = data.rank()
spearman_corr = ranked_data.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
print("Spearman's correlation: ")
print(spearman_corr)
statistic, p_value = stats.ttest_rel(data.radius_mean,data.area_mean)
print('p-value: ',p_value)
df = pd.read_csv("../input/data.csv",header = 0)
df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
features_mean=list(df.columns[1:11])
# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()
data = pd.read_csv("../input/data.csv")
data = data.drop(['Unnamed: 32','id'],axis = 1)
data['diagnosis'].replace(to_replace='M', value = 1, inplace=True)
data['diagnosis'].replace(to_replace='B', value = 0, inplace=True)
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']
from sklearn import ensemble, linear_model, svm, neighbors, gaussian_process, naive_bayes, tree 
from sklearn.model_selection import cross_val_score
scoreFrame = pd.DataFrame(columns = ['Algorithm Name', 'Average', 'Standard Deviation'])
algList=[
    #linear
    linear_model.Ridge(),
    linear_model.SGDClassifier(),
    #Neighbors
    neighbors.KNeighborsClassifier(),
    #SVM
    svm.SVC(),
    #Gaussian Process
    gaussian_process.GaussianProcessClassifier(),
    #Naive Bayes
    naive_bayes.GaussianNB(),
    #Tree
    tree.DecisionTreeClassifier(),
    #Ensemble
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.AdaBoostClassifier()
]
for alg in algList:
    scores = cross_val_score(alg, X, y, cv = 10)
    algName = alg.__class__.__name__
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    scoreFrame.loc[len(scoreFrame)] = [algName, scoreAverage, scoreSTD]
    print(algName, "is done.")

scoreFrame.sort_values('Average', ascending=False)

svmFrame = pd.DataFrame(columns = ['Gamma', 'Average', 'Standard Deviation'])

for g in range(1,1000):
    g = g/1000000
    alg = svm.SVC(gamma=g)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmFrame.loc[len(svmFrame)] = [g, scoreAverage, scoreSTD]
  
svmFrame.sort_values('Average', ascending=False).head(10)
svmFrame = pd.DataFrame(columns = ['Kernel', 'Average', 'Standard Deviation'])
kernelList = [ 'linear', 'poly', 'rbf', 'sigmoid']
for k in kernelList:
    alg = svm.SVC(gamma=0.000148, kernel=k)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmFrame.loc[len(svmFrame)] = [k, scoreAverage, scoreSTD]
    
svmFrame.sort_values('Average', ascending=False).head(10)
svmFrame = pd.DataFrame(columns = ['Degrees', 'Average', 'Standard Deviation'])

for d in range(1,5):
    alg = svm.SVC(gamma=0.000148, kernel='poly', degree=d)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmFrame.loc[len(svmFrame)] = [d, scoreAverage, scoreSTD]
   
svmFrame.sort_values('Average', ascending=False).head(10)

data = pd.read_csv('../input/data.csv')
del data['Unnamed: 32']
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

classifier = tf.keras.Sequential()

classifier.add(layers.Dense(16, activation='relu', input_shape=(30,)))
classifier.add(layers.Dropout(0.1))

classifier.add(layers.Dense(16, activation='relu', input_shape=(30,)))
classifier.add(layers.Dropout(0.1))
classifier.add(layers.Dense(1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=150)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
sns.heatmap(cm,annot=True)
plt.savefig('h.png')
