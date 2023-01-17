# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
data.info()
data.describe()
colorList = ["red" if i=="Abnormal" else "green" for i in data["class"]]

pd.plotting.scatter_matrix(data.loc[:,data.columns!="class"],

                          c=colorList,

                           figsize=(15,15),

                           diagonal="hist",

                           alpha=0.5,

                           s=200,

                           marker="*",

                           edgecolor="black"

                          )

plt.show()
g=sns.factorplot(x="class",size=6,data=data,kind="count")

g.add_legend()

g.set_ylabels("Rates class of trouble")

plt.show()

print(data["class"].value_counts())
g=sns.FacetGrid(data,col="class",height=4)

g.map(plt.hist,"pelvic_incidence",bins=25)

g.add_legend()

plt.show()
g=sns.FacetGrid(data,col="class",height=4)

g.map(plt.hist,"pelvic_tilt numeric",bins=25)

g.add_legend()

plt.show()
data.head()
g=sns.FacetGrid(data,row="class",height=4)

g.map(plt.hist,"lumbar_lordosis_angle",bins=25)

g.add_legend()

plt.show()
g=sns.FacetGrid(data,row="class",height=4)

g.map(sns.pointplot,"lumbar_lordosis_angle",bins=25)

g.add_legend()

plt.show()
g=sns.FacetGrid(data,row="class",height=5)

g.map(sns.pointplot,"pelvic_incidence","pelvic_tilt numeric")

g.add_legend()

plt.show()
A=data[data["class"]=="Abnormal"]

N=data[data["class"]=="Normal"]
plt.scatter(A["pelvic_incidence"],A["pelvic_tilt numeric"],color="red",label="Abnormal")

plt.scatter(N["pelvic_incidence"],N["pelvic_tilt numeric"],color="green",label="Normal")

plt.xlabel("pelvic_incidence")

plt.ylabel("pelvic_tilt numeric")

plt.legend()

plt.show()
g=sns.FacetGrid(data,row="class",height=5)

g.map(sns.pointplot,"sacral_slope")

g.add_legend()

plt.show()
g=sns.FacetGrid(data,row="class",height=5)

g.map(plt.hist,"sacral_slope",bins=25)

g.add_legend()

plt.show()
g=sns.FacetGrid(data,row="class",height=5)

g.add_legend()

g.map(plt.hist,"pelvic_radius",bins=5)

plt.show()
g=sns.FacetGrid(data,row="class",height=5)

g.add_legend()

g.map(sns.pointplot,"pelvic_radius")

plt.show()
g=sns.FacetGrid(data,row="class",height=6)

g.map(plt.hist,"degree_spondylolisthesis",bins=35)

plt.show()
g=sns.FacetGrid(data,row="class",height=6)

g.map(sns.pointplot,"degree_spondylolisthesis")

plt.show()
plt.scatter(A["pelvic_radius"],A["pelvic_incidence"],color="red",label="Abnormal")

plt.scatter(N["pelvic_radius"],N["pelvic_incidence"],color="green",label="Normal")

plt.xlabel("pelvic_radius")

plt.ylabel("pelvic_incidence")

plt.legend()

plt.show()
plt.scatter(A["pelvic_tilt numeric"],A["lumbar_lordosis_angle"],color="red",label="Abnormal")

plt.scatter(N["pelvic_tilt numeric"],N["lumbar_lordosis_angle"],color="green",label="Normal")

plt.xlabel("pelvic_tilt numeric")

plt.ylabel("lumbar_lordosis_angle")

plt.legend()

plt.show()
plt.scatter(A["sacral_slope"],A["pelvic_incidence"],color="red",label="Abnormal")

plt.scatter(N["sacral_slope"],N["pelvic_incidence"],color="green",label="Normal")

plt.xlabel("sacral_slope")

plt.ylabel("pelvic_incidence")

plt.legend()

plt.show()
plt.scatter(A["pelvic_radius"],A["sacral_slope"],color="red",label="Abnormal")

plt.scatter(N["pelvic_radius"],N["sacral_slope"],color="green",label="Normal")

plt.xlabel("pelvic_radius")

plt.ylabel("sacral_slope")

plt.legend()

plt.show()
plt.scatter(A["lumbar_lordosis_angle"],A["pelvic_radius"],color="red",label="Abnormal")

plt.scatter(N["lumbar_lordosis_angle"],N["pelvic_radius"],color="green",label="Normal")

plt.xlabel("lumbar_lordosis_angle")

plt.ylabel("pelvic_radius")

plt.legend()

plt.show()
plt.scatter(A["pelvic_tilt numeric"],A["degree_spondylolisthesis"],color="red",label="Abnormal")

plt.scatter(N["pelvic_tilt numeric"],N["degree_spondylolisthesis"],color="green",label="Normal")

plt.xlabel("pelvic_tilt numeric")

plt.ylabel("degree_spondylolisthesis")

plt.legend()

plt.show()
plt.scatter(A["lumbar_lordosis_angle"],A["degree_spondylolisthesis"],color="red",label="Abnormal")

plt.scatter(N["lumbar_lordosis_angle"],N["degree_spondylolisthesis"],color="green",label="Normal")

plt.xlabel("lumbar_lordosis_angle")

plt.ylabel("degree_spondylolisthesis")

plt.legend()

plt.show()
plt.scatter(A["pelvic_tilt numeric"],A["degree_spondylolisthesis"],color="red",label="Abnormal")

plt.scatter(N["pelvic_tilt numeric"],N["degree_spondylolisthesis"],color="green",label="Normal")

plt.xlabel("pelvic_tilt numeric	")

plt.ylabel("degree_spondylolisthesis")

plt.legend()

plt.show()
data.head()
sns.heatmap(data.loc[:,data.columns!="class"].corr(),annot=True)

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=3)

x,y=data.loc[:,data.columns!="class"],data.loc[:,"class"]

knn.fit(x,y)

prediction=knn.predict(x)

print("Prediction : {}".format(prediction))
from sklearn.model_selection import train_test_split

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.3,random_state=1)

knn=KNeighborsClassifier(n_neighbors=3)

x,y=data.loc[:,data.columns!="class"],data.loc[:,"class"]

knn.fit(xTrain,yTrain)

prediction=knn.predict(xTest)

print("R^2 score is (k=3) = {}".format(knn.score(xTest,yTest)))
from sklearn.metrics import confusion_matrix

import seaborn as sns



y_pred=knn.predict(x_test)

y_true=y_test

cm1=confusion_matrix(y_pred,y_true)





f,ax=plt.subplots(figsize=(6,6))

sns.heatmap(cm1,annot=True,color="red",fmt="0.5f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Table of erros about predict")

plt.show()
neighbours = np.arange(1,25)

trainAccuracy=[]

testAccuracy=[]



for i,k in enumerate(neighbours):

    # k between 1-25

    knn = KNeighborsClassifier(n_neighbors=k)

    # train(fit) data

    knn.fit(xTrain,yTrain)

    # append accuracy values to relevant places

    trainAccuracy.append(knn.score(xTrain,yTrain))

    testAccuracy.append(knn.score(xTest,yTest))



plt.figure(figsize=(15,8))

plt.plot(neighbours, testAccuracy, label = 'Testing Accuracy')

plt.plot(neighbours, trainAccuracy, label = 'Training Accuracy')

plt.legend()

plt.title("Values and Accuracy")

plt.xticks(neighbours)

plt.show()
data1=data[data["class"]=="Abnormal"]

x=data1.loc[:,"pelvic_incidence"].values.reshape(-1,1)

y=np.array(data1.loc[:,"sacral_slope"]).reshape(-1,1)

# I showed two way to find x and y,you can use what you want

plt.figure(figsize=(15,8))

plt.scatter(x=x,y=y)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.title("This scatter belong to abnormal patients")

plt.show()
from sklearn.linear_model import LinearRegression



linearReg=LinearRegression()

linearReg.fit(x,y)



yHead = linearReg.predict(x)



plt.figure(figsize=(15,8))

plt.scatter(x=x,y=y)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.title("This scatter belong to abnormal patient")

plt.plot(x,yHead,color="red")

plt.show()

print("R^2 score is {}".format(linearReg.score(x,y)))
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



plt.figure(figsize=(15,8))

plt.scatter(x,y)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")



# prediction section

lr=LinearRegression()

lr.fit(x,y)

# visualize

yHead = lr.predict(x)

plt.plot(x,yHead,color="green",label="linear")

plt.legend()

plt.show()

# visualize 2



pl = PolynomialFeatures(degree=2)

xpl=pl.fit_transform(x)  # use as both implement fith(train) data and save as variable

lr2 = LinearRegression()

lr2.fit(xpl,y)

#  

plt.figure(figsize=(15,8))

yHead2 = lr2.predict(xpl)

plt.scatter(x,y)

plt.plot(x,yHead2,color="red",label="polynomal")

plt.legend()

plt.show()
from sklearn.tree import DecisionTreeRegressor

treeReg = DecisionTreeRegressor()

treeReg.fit(x,y)

yHead = treeReg.predict(x)

# visualize

plt.figure(figsize=(15,8))

plt.scatter(x,y,color="blue")

plt.plot(x,yHead,color="red")

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.title("This scatter belong to abnormal patient")

plt.show()
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

yHead=treeReg.predict(x_)



plt.figure(figsize=(15,8))

plt.scatter(x,y,color="blue")

plt.plot(x_,yHead,color="red")

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.title("This scatter belong to abnormal patient")

plt.show()
from sklearn.ensemble import RandomForestRegressor



rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x,y)

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

yHead=rf.predict(x_)

# visualize

plt.figure(figsize=(15,8))

plt.scatter(x,y,color="blue")

plt.plot(x_,yHead,color="green")

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.title("This scatter belong to abnormal patient")

plt.show()
data["class"]=[1 if i=="Normal" else 0 for i in data["class"]]
data["class"]
y=data["class"].values

#x_data=data.loc[:,data.columns!="class"].values

x_data=data.drop(["class"],axis=1)

#x_data

x= (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x  # normalze
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)

x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T

print("\nLater...\n")

print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
def initialize_weights_and_bias(dimension):

    w=np.full((dimension,1),0.01)

    b=0.0

    return w,b
def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
sigmoid(0)
def forward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z) # probabilistic 0-1

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    return cost 
def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(w,b,x_test):

    z=sigmoid(np.dot(w.T,x_test)+b)

    y_prediction=np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            y_prediction[0,i]=0

        else:

            y_prediction[0,i]=1

    return y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):

    dimension=x_train.shape[0]

    w,b=initialize_weights_and_bias(dimension)

    parameters,gradients,cost_list=update(w,b,x_train,y_train,learning_rate,num_iterations)

    y_prediction_train=predict(parameters["weight"],parameters["bias"],x_train)

    

    print("train accuracy : {}".format(100-np.mean(np.abs(y_prediction_train-y_train))*100))

    
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=0.01,num_iterations=150)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(random_state=42,max_iter=150)



print("Train accuracy = {}".format(logreg.fit(x_train.T,y_train.T).score(x_test.T,y_test.T)))
data=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
y=data["class"].values

x_data=data.drop(["class"],axis=1).values
# normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.svm import SVC



svm = SVC(random_state=1)

svm.fit(x_train,y_train)



print("Output which accuracy of svm algorithm = {}".format(svm.score(x_test,y_test)))
A=data[data["class"]=="Abnormal"]

N=data[data["class"]=="Normal"]
plt.scatter(A["pelvic_tilt numeric"],A["sacral_slope"],color="red",label="Abnormal")

plt.scatter(N["pelvic_tilt numeric"],N["sacral_slope"],color="green",label="Normal")

plt.xlabel("pelvic_tilt")

plt.ylabel("sacral_slope")

plt.legend()

plt.show()
data.head()
from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()

nb.fit(x_train,y_train)



print("print accuracy of naive bayes algorithm: ",nb.score(x_test,y_test))
data.head()

plt.scatter(data["pelvic_radius"],data["degree_spondylolisthesis"])

plt.xlabel("pelvic_radius")

plt.ylabel("degree_spondylolisthesis")

plt.show()
data2 = data.loc[:,["pelvic_radius","degree_spondylolisthesis"]]



from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=2) # it will occur 2 clusters

kmeans.fit(data2)

labels = kmeans.predict(data2)

labels # normal or abnormal

plt.scatter(data.pelvic_radius,data.degree_spondylolisthesis,c=labels)

plt.xlabel("pelvic_radius")

plt.ylabel("degree_spondylolisthesis")

plt.show()
df = pd.DataFrame({ "labels" : labels , "class" : data["class"]})



ct = pd.crosstab(df["labels"],df["class"])

print(ct)
clusterList = np.empty(10)



for i in range(1,10):

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(data2)

    clusterList[i] = kmeans.inertia_



plt.plot(range(0,10),clusterList)    

plt.xlabel('Number of cluster')

plt.ylabel('Distances')

plt.show()

# it seems that 3 is most suitable value for kmeans's cluster value.Reason why elbow shape start to occur in 3
data3 = data.drop('class',axis = 1)

data3
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline



scaler = StandardScaler()

kmeans = KMeans(n_clusters=2)

pipe = make_pipeline(scaler,kmeans)

pipe.fit(data3)

labels = pipe.predict(data3)



df = pd.DataFrame({'labels':labels,"class":data['class']})



ct = pd.crosstab(df['labels'],df['class'])



print(ct)
from scipy.cluster.hierarchy import linkage,dendrogram



merg = linkage(data3.iloc[100:120,:],method="single")

dendrogram(merg,leaf_rotation=90,leaf_font_size=10)

plt.show()
from sklearn.manifold import TSNE



color_list = ["red" if i == "Abnormal" else "green" for i in data.loc[:,"class"]]



model = TSNE(learning_rate=100)

transform = model.fit_transform(data2)

transform



x= transform[:,0]

y= transform[:,1]



plt.scatter(x,y,c=color_list)

plt.xlabel("pelvic_radius")

plt.ylabel("degree_spondylolisthesis")

plt.show()
from sklearn.decomposition import PCA



model = PCA()

model.fit(data3)

transform = model.transform(data3)

transform

print('Principle components: ',model.components_)
plt.bar(range(model.n_components_),model.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.show()
pca = PCA(n_components=2)

pca.fit(data3)



transform = pca.transform(data3)

#transform



x= transform[:,0]

y= transform[:,1]





plt.scatter(x,y,c=color_list)

plt.show()
