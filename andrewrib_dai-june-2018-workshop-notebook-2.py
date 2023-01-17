# General Libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

# Scikit-Learn Libraries
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split,learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model,svm,tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Keras Libraries 
from keras.models import Model
from keras.layers import Input, Dense, Dropout

# Interactive Widgets
from ipywidgets import interact_manual
from IPython.display import display

# Print out the folders where our datasets live. 
print("Datasets: {0}".format(os.listdir("../input")))
# Load Test Data
testDF = pd.read_csv("../input/titanic/test.csv")

# Load Training Data
titanicDF = pd.read_csv("../input/titanic/train.csv")
titanicDF.head(10)
nRows = 3
nCols = 3

plt.figure(figsize=(6*3,6*3))
plt.subplot(nRows,nCols,1)

titanicDF["Survived"].value_counts().plot.pie(autopct="%.2f%%")

plt.subplot(nRows,nCols,2)
titanicDF["Sex"].value_counts().plot.pie(autopct="%.2f%%")

plt.subplot(nRows,nCols,3)
titanicDF["Pclass"].value_counts().plot.pie()

plt.subplot(nRows,nCols,4)
titanicDF["SibSp"].value_counts().plot.pie()

plt.subplot(nRows,nCols,5)
titanicDF["Parch"].value_counts().plot.pie()

plt.subplot(nRows,nCols,6)
titanicDF["Embarked"].value_counts().plot.pie()

plt.subplot(nRows,nCols,7)
plt.title("Age Histogram")
titanicDF["Age"].hist()

plt.subplot(nRows,nCols,9)
plt.title("Fare Histogram")
titanicDF["Fare"].hist()

plt.show()
# Plot
def plotChilds(ageThresh=10):
    df = titanicDF["Age"].dropna()
    childFeat = df.map(lambda x: x<ageThresh)
    
    plt.figure(figsize=(12,12))
    
    plt.subplot(221)
    plt.title("Size of Age Group")
    childFeat.value_counts().plot.pie(autopct="%.2f%%")
    
    
    newDF = titanicDF.assign(isChild=df.map(lambda x: x<ageThresh))
    res = newDF.groupby("isChild")["Survived"]
    adultC,childC = res.count()
    adultS,childS = res.sum()
    
    plt.subplot(222)
    plt.title("Age Group Survival")
    plt.pie([childC-childS,childS],labels=["Died","Survived"],autopct="%.2f%%")
    
    plt.subplot(223)
    plt.title("Age Group Gender Distribution")
    newDF.groupby("isChild")["Sex"].value_counts()[1].plot.pie(autopct="%.2f%%")
    
    plt.subplot(224)
    plt.title("Age Group Class Distribution")
    newDF.groupby("isChild")["Pclass"].value_counts()[1].plot.pie(autopct="%.2f%%")
    
    plt.show()
    display(newDF.head())
    
interact_manual(plotChilds,ageThresh=(0,100))
# Our prepprocessing function which is applied to every row of the target dataframe. 
def preprocessRow(row):
    # Process Categorical Variables - One-Hot-Encoding
    sex      = [0,0]
    embarked = [0,0,0]
    pclass   = [0,0,0]
    
    if row["Sex"] == "male":
        sex = [0,1]
    elif row["Sex"] == "female":
        sex = [1,0]
    
    if row["Embarked"] == "S":
        embarked = [0,0,1]
    elif row["Embarked"] == "C":
        embarked = [0,1,0]
    elif row["Embarked"] == "Q":
        embarked = [1,0,0]
    
    if row["Pclass"] == 1:
        pclass   = [0,0,1]
    elif row["Pclass"] == 2:
        pclass   = [0,1,0]
    elif row["Pclass"] == 3:
        pclass   = [1,0,0]
    
    return pclass+sex+[row["Age"],row["SibSp"],row["Parch"],row["Fare"]]+embarked

# Labels for the feature columns. 
featureLabels = ["3 Class","2 Class","1 Class","Female","Male","Age","SibSp",
                 "Parch","Fare","Q Embarked","C Embarked","S Embarked"]

# Fill Missing Values
titanicDF = titanicDF.fillna(0).sample(frac=1)

# Preprocess Data
titanicMat = np.stack(titanicDF.apply(preprocessRow,axis=1).values)

# View what the training vectors look like. 
tmp = pd.DataFrame(titanicMat)
tmp.columns = featureLabels
tmp.head()
# Size of validation set. 
splitSize = 0.2

titanic_X, titanic_y = [titanicMat, titanicDF["Survived"].values]
titanic_train_x, titanic_validation_x, titanic_train_y , titanic_validation_y = train_test_split(titanic_X,titanic_y, test_size=splitSize)
def meanBaseline(df):
    nRows = df["Survived"].shape[0]
    mean = df["Survived"].sum()/nRows
    return log_loss(df["Survived"],np.full(nRows,mean))

print("Mean Baseline: {0}".format(meanBaseline(titanicDF)))
def genderBaseline(df):
    nRows = df["Survived"].shape[0]
    pred = df["Sex"].map(lambda x: 1 if x == "female" else 0)
    res = df["Survived"]==pred
    return res.sum()/res.count()

print("Gender Baseline: {0}".format(genderBaseline(titanicDF)))
# Put your code here. 
models = {"Linear Regression":linear_model.LinearRegression(),"Logistic Regression":linear_model.LogisticRegression(),
         "Ridge Regression":linear_model.Ridge(),"Lasso Regression":linear_model.Lasso(),
         "Bayesian Ridge Regression":linear_model.BayesianRidge(),"Perceptron":linear_model.Perceptron(max_iter=1000),
         "Support Vector Machine":svm.SVC(gamma="auto"),"Gaussian Naive Bayes":GaussianNB(),"Decision Tree":tree.DecisionTreeClassifier(),
         "Random Forest":RandomForestClassifier(),"AdaBoost":AdaBoostClassifier(),
         "Gradient Boosting":GradientBoostingClassifier()}
kFolds = 40

def scoringFN(model,X,y):
    pred = model.predict(X)
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    return np.sum(y == pred)/y.shape[0]

for mod in models:
    model = models[mod]
    cross_val = cross_val_score(model, titanic_X, titanic_y,cv=kFolds,scoring= scoringFN).mean()
    print("{0:30} {1} Fold Cross-Validation Accuracy: {2:7f}".format(mod,kFolds,cross_val))
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(20,12))
    plt.title(title,size=30)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples",size=15)
    plt.ylabel("Score",size=15)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Gradient Boosting Classifier Parameters (Partial List)
learning_rate     = 0.1
n_estimators      = 100
max_depth         = 3
min_samples_split = 2
subsample         = 1

# Define the gradient boosting classifier. 
gradBoost = GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators,
                                       max_depth=max_depth,min_samples_split=min_samples_split,subsample=subsample)

# Plot the learning curve of the gradient boosting classifier. 
plot_learning_curve(gradBoost,"Gradient Boosting Learning Curve",titanic_X,titanic_y,cv=5).show()

# Fit model on the full dataset. 
gradBoost.fit(titanic_X,titanic_y)
# Network Definition.
# Here we use the keras functional api.
inputs = Input(shape=(titanic_train_x.shape[1],),name="input")
x = Dense(20,activation="sigmoid")(inputs)
x = Dense(20,activation="sigmoid")(x)
x = Dense(20,activation="sigmoid")(x)
out = Dense(1,activation="sigmoid", name="output")(x)

# Instantiate the network.
simpleModel = Model(inputs=inputs, outputs=out)

# Compile the network. 
simpleModel.compile(optimizer="adam",loss="binary_crossentropy", metrics=['acc'])

# Pretty print the details of the network. 
simpleModel.summary()
hist = simpleModel.fit(titanic_train_x, titanic_train_y,validation_data=(titanic_validation_x,titanic_validation_y), batch_size=30,epochs=30, verbose=1)
def learningCurves(hist):
    histAcc_train = hist.history['acc']
    histLoss_train = hist.history['loss']
    histAcc_validation = hist.history['val_acc']
    histLoss_validation = hist.history['val_loss']
    maxValAcc = np.max(histAcc_validation)
    minValLoss = np.min(histLoss_validation)

    plt.figure(figsize=(12,12))
    epochs = len(histAcc_train)
    plt.plot(range(epochs),np.full(epochs,meanBaseline(titanicDF)),label="Unbiased Estimator", color="red")

    plt.plot(range(epochs),histLoss_train, label="Training Loss", color="#acc6ef")
    plt.plot(range(epochs),histAcc_train, label="Training Accuracy", color = "#005ff9" )

    plt.plot(range(epochs),histLoss_validation, label="Validation Loss", color="#a7e295")
    plt.plot(range(epochs),histAcc_validation, label="Validation Accuracy",color="#3ddd0d")

    plt.scatter(np.argmax(histAcc_validation),maxValAcc,zorder=10,color="green")
    plt.scatter(np.argmin(histLoss_validation),minValLoss,zorder=10,color="green")

    plt.xlabel('Epochs',fontsize=14)
    plt.title("Learning Curves",fontsize=20)

    plt.legend()
    plt.show()

    print("Max validation accuracy: {0}".format(maxValAcc))
    print("Minimum validation loss: {0}".format(minValLoss))
    
learningCurves(hist)
#### Training Hyperparameters ####
batch_size = 300
epochs = 1000

#### Model Hyperparameters  ####
nLayers = 3
layerSize = 80
dropoutPercent = 0.87# Regularization 

# Possible loss fuctions: https://keras.io/losses/
# mean_squared_error, mean_absolute_error,mean_absolute_percentage_error,mean_squared_logarithmic_error
# squared_hinge, hinge, categorical_hinge, logcosh, kullback_leibler_divergence, poisson, cosine_proximity
lossFn = 'binary_crossentropy'

# Possible optimizers: https://keras.io/optimizers/
# SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
optimizer = 'adam'

# Possible Activation Functions: https://keras.io/activations/
# elu, selu, softplus, softsign, relu, tanh, hard_sigmoid, linear
# Possible Advanced Activations: https://keras.io/layers/advanced-activations/
# LeakyReLU, PReLU, ELU, ThresholdedReLU
activationFn = 'sigmoid'
# Model Architecture 
def makeModel(inputShape,nLayers,layerSize,dropoutPercent,lossFn,optimizer):
    inputs = Input(shape=(inputShape,),name="input")
    x = None 
    
    for layer in range(nLayers):
        if x == None:
            x = inputs

        x = Dense(layerSize, activation=activationFn,name="fc"+str(layer))(x)
        x = Dropout(dropoutPercent,name="fc_dropout_"+str(layer))(x)

    out = Dense(1,activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizer,
                  loss=lossFn,
                  metrics=['acc'])
    
    return model

modelMain = makeModel(titanic_train_x.shape[1],nLayers,layerSize,dropoutPercent,lossFn,optimizer)
modelMain.summary()
hist = modelMain.fit(titanic_train_x, titanic_train_y,validation_data=(titanic_validation_x,titanic_validation_y), batch_size=batch_size,epochs=epochs, verbose=0)
learningCurves(hist)
 # Cross-Validation Parameter 
kFolds = 3

kfold = StratifiedKFold(n_splits=kFolds, shuffle=True)
means = []
stds = []
lossesLs = []
accuracyLs = []

runningLoss = []
runningAccuracy = []

# Train on k-folds of the data. 
for train, test in kfold.split(titanic_X, titanic_y):
    
    # Create new instance of our model. 
    model = makeModel(titanic_X.shape[1],nLayers,layerSize,dropoutPercent,lossFn,optimizer)
    
    # Train the model on this kfold. 
    model.fit(titanic_X[train], titanic_y[train],batch_size=batch_size,epochs=epochs, verbose=0)

    # Evaluate the model
    loss,acc = model.evaluate(titanic_X[test], titanic_y[test], verbose=0)
    
    # Log Cross-Validation Data
    lossesLs.append(loss)
    accuracyLs.append(acc)
    mean = np.mean(lossesLs)
    std = np.std(lossesLs)
    
    accuracyMean = np.mean(accuracyLs)
    accuracyStd = np.std(accuracyLs)
    
    runningLoss.append(mean)
    runningAccuracy.append(accuracyMean)
    
    print("Loss: %.2f%% (+/- %.2f%%) | Accuracy: %.2f%% (+/- %.2f%%)" % (mean*100,std,accuracyMean*100,accuracyStd))

plt.show()
# The thrshold function which assigns a class of 0 or 1 based on the sigmoid output of the network. 
def thresholdFn(x):
    if(x < 0.5):
        return 0
    else:
        return 1

pred = modelMain.predict(np.stack(testDF.apply(preprocessRow,axis=1)))
    
# Save the predictions to a CSV file in the format suitable for the competition. 
data_to_submit = pd.DataFrame.from_items([
    ('PassengerId',testDF["PassengerId"]),
    ('Survived', pd.Series(np.hstack(pred)).map(thresholdFn))])

data_to_submit.to_csv('neuralNet.csv', index = False)