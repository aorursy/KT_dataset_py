# Importing the libraries

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier







iris= load_iris()

iris
# splitting the data in to train , test

xtrain,xtest,ytrain,ytest= train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
## pipeline Creation

# 1) Applying data preprocessing for standardscaler

# 2) Dimension Reduction by PCA

# 3) Applying the Classifier/algorithms.

# creating the pipeline for Logistic Regression model



pipeline_1= Pipeline([("sacler1", StandardScaler()),

                        ("pca1" ,PCA(n_components=2)),

                        ("lr_classifier", LogisticRegression(random_state=0))])
# creating the pipeline for DecisionTree model



pipeline_2= Pipeline([("sacler2", StandardScaler()),

                        ("pca2" ,PCA(n_components=2)),

                        ("dt_classifier", DecisionTreeClassifier())])
# creating the pipeline for RandomForest model



pipeline_3= Pipeline([("sacler3", StandardScaler()),

                        ("pca3" ,PCA(n_components=2)),

                        ("rf_classifier", RandomForestClassifier())])
# creating all the pipelines in a list

pipelines= [pipeline_1,pipeline_2,pipeline_3]

pipelines
# initializing the values 

best_accuracy=0.0

best_classifier= 0

best_pipeline=""
#Dictionary of pipelines and classifiers



pipe_dict= {0: "LogisticRegression", 1: "DecisionTreeClassifier", 2: "RandomForestClassifier"}

pipe_dict
# Training the models of all pipelines



for pipe in pipelines:

    pipe.fit(xtrain,ytrain)

    #print(pipe)
# finding the accuracy

for i,model in enumerate(pipelines):

    print("{} Test_Accuracy is : {} ".format(pipe_dict[i], model.score(xtest,ytest)))

   
# finding the best of three models



for i , model in enumerate(pipelines):

    if model.score(xtest,ytest)>best_accuracy :

        best_accuracy= model.score(xtest,ytest)

        #best_pipeline= model

        best_classifier=i

print("Best classifier is :{}".format(pipe_dict[best_classifier]), "and best_accuracy is", best_accuracy)

        

   

    