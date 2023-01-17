import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import warnings
import random
from scipy import stats
warnings.filterwarnings("ignore")

#set random seed
random.seed(123)
np.random.seed(123)
#load all tha datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/gender_submission.csv")
print(train.shape)
#make not of the length of train set, so that it can be used to split it in the future again
len_train = train.shape[0]

#join train and test data to make common operations on both of them
full_data = pd.concat(objs=[train,test]).reset_index(drop=True)
#lets see that the whole data looks like
full_data.describe()
##important to know the datatypes of all the columns tbefore making any operations on them.
#this is is useful to know before trying to impute null values
full_data.dtypes
#the no of null values present in the dataset. Age and Cabin have the most while Embarked
#and Fare have little. These values have to be imputed, because the dataset as a whole is
#not too big and we cannot afford to simply remove records with null values. 
full_data.isnull().sum()
#Embarked and Fare are fairly straight forward. Embarked has only 2 values missing, hence
#replaced with most frequent value since its a categorical feature. Fare is numerical feature
# and has only one missing. It is replaced with mean of the feature.
full_data.Embarked.fillna(full_data.Embarked.value_counts().index[0], inplace=True)
full_data.Fare.fillna(full_data.Fare.mean(),inplace=True)
full_data.isnull().sum()

#Cabin is tricky. It has 1014 null values. Before imputing all the values, we need to understand
#what cabin means to the survival. There are lots of records without cabin information, it 
#could mean that they did not have a cabin also. So people with cabin could be easily more
#influential and wealthy. That makes their survival chances higher. So instead of imputing we can
#just replace cabin with binary value. '0' is no cabin and '1' if person had a cabin.
#Thus we extract the needed information and as well as impute the column in one step. 
full_data.Cabin = pd.Series([0 if pd.isnull(i) else 1 for i in full_data.Cabin])
full_data = full_data.infer_objects()
full_data.isnull().sum()
#Age again is a numerical value and can be replaced with mean of the feature
full_data.Age = full_data.Age.fillna(full_data.Age.mean())
full_data.isnull().sum()
#It is easier to think of age as a factor for survival. Younger people have more chance
#of survival. But isntead of having it as a mere number, it makes sense to convert them
#to buckets, so that people with similar ages are in same bucket and hence have same chances
#of survival. This is needed because, a person of age 31 has more or less same chance of survival
#and agility as a person of age 36. But leaving it as number, would mean the later would have
#an advantage. We bucket the age feature into 8 bins(0-10, 10-20...70-80) and assign them digits
# like '0', '1', etc.
full_data['Age'] = pd.cut(full_data['Age'], 8, labels=['0', '1', '2', '3', '4', '5', '6', '7'])
#no of family members could hamper the chance of survival. More family members missing in the 
#chaos, more time you need to get to safety. So makes sense to join the two columns siblings and
#parents together into another column Family_Count.
full_data["Family_Count"] = full_data["SibSp"]+full_data["Parch"]
full_data.head()
#drop the columns that may not be needed. SOme columns could still be used to extract more
#info like the title from Name, but lets keep it simple for now.
full_data.drop(columns=['PassengerId','SibSp','Parch','Ticket'],inplace=True)
full_data.head()
#Encode the sex column. 0 for male and 1 for female
full_data["Sex"] = full_data["Sex"].map({"male": 0, "female":1})
full_data.head()
#The column Name at first may look unnecessary. But it containes the titles of each person
#For example, person with title Master or Miss may have a greater chance of survival, since
#women and children are evacuated first.First we will extract the titles alone from the names.
full_data["Title_Name"] = pd.Series([name.split(",")[1].split(".")[0].strip() for name in full_data["Name"]])
full_data.drop(columns=['Name'],inplace=True)
full_data.head()
#The distribzution of different titles and their corresponding survival values. We can see
#that titles like Miss, Mrs, Master have greater survival rates and title with Mr has 
#the least.
full_data.groupby('Title_Name')['Survived'].value_counts().unstack().fillna(0)
#Previously we have seen the presence of other titles apart from usual four -> Mr, Miss, Mrs
# & Master. These titles are very low in count if accounted for individually. SO makes sense
#to combine into one value so that the no. of distince value reduces. We will term other titles
# as 'Other'. Followed by the distribution after the replacements. 
list_title = ['Mr', 'Mrs', 'Miss', 'Master']
full_data.Title_Name = full_data.Title_Name.apply(lambda x: x if x in list_title else "Other")
full_data.groupby('Title_Name')['Survived'].value_counts().unstack().fillna(0)
#one hot encode the categorical columns.
full_data = pd.get_dummies(full_data, columns = ["Age","Embarked","Pclass", "Title_Name"],drop_first=True)
full_data.head()
#seperate the target variable from train dataset
target = np.array(full_data.Survived.values)
target = target[:len_train-1]
target = target.astype(int)
full_data.drop(columns=['Survived'],inplace=True)
full_data.head()
#just converting to np array both train and test sets
train_x = np.array(full_data[:len_train-1])
test_x = np.array(full_data[len_train:])
print(len(train_x))
print(len(test_x))
#scaling is need in this case because, there various features with different units of
#measurements. Example, Age and Fare are independent, but still Fare is usually in 1000s
# and age is between 0 1nd 100 only. THis makes Fare dominate the training process for
#no reason. While tree based classifiers are not hampered by this, other gradient based
# ones like MLP are not. So we scale the date and use it for tree based as well as 
#gradient based algorithms. Besides PCA needs the data to be scaled.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=False)
scaler.fit(train_x)
scaler.transform(train_x)
scaler.transform(test_x)
#PCA is perfomed usually for dimenionality reduction. But in our case, it is just used to
#remove any multi-collinearity if present. PCA reduces dimensions and removes multi-collinearity.
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)
#MLP, SGD, RF, KNN classifiers are used. These models are first built and added to a list
#for the prupose of grid search. Here mlp is added
classifiers =[]
mlp = MLPClassifier(hidden_layer_sizes=(100,2), random_state=123, verbose=1, batch_size=64, 
                    max_iter=50,early_stopping = True)
classifiers.append(mlp)
#Here sgd classifier is added. It has loss as 'log' which makes it a logistic regression model.
sgd_reg = SGDClassifier(max_iter=100, verbose = 1, early_stopping=True, n_iter_no_change=5,
                       eta0=0.001, learning_rate='adaptive',random_state=123,n_jobs=-1,
                       tol=1e-3,penalty='l2',loss='log')
classifiers.append(sgd_reg)
#Here rf classifier is added.
rf = RandomForestClassifier(n_estimators=100,verbose=1, random_state=123,oob_score=True,
                           n_jobs=-1)
classifiers.append(rf)
#Here knn classifier is added.
knn = KNeighborsClassifier(n_jobs=-1, algorithm='auto')
classifiers.append(knn)
#Parameters dict needed for the above added models are added to a list called parameters
#in order. The differnt possible values for hyper parameters for each model is taken as parameters
# and grid search will tell us which was the best combination for each model.
parameters=[]
parameters_mlp = {'max_iter': [30,40,50], 'alpha': 10.0 ** -np.arange(1, 4), 
              'hidden_layer_sizes':[(100,2), (80,2), (60,2), (40,2)], 
                  'random_state':[0,1,123,42],
                 'learning_rate' : ['constant','adaptive'],
                 'learning_rate_init':[0.001,0.003,0.005,0.0009]}
parameters.append(parameters_mlp)
parameters_sgd = {"max_iter": [50,80,100,120,150],
              "eta0": [0.001,0.003,0.005,0.0009],
              "random_state": [0,1,123,42],
              "penalty": ['l2','l1']}
parameters.append(parameters_sgd)
parameters_rf = {"max_features": [2,3,4],
               "random_state":[0,1,123,42],
              "n_estimators" :[100,300,50,200]}
parameters.append(parameters_rf)
parameters_knn = {"n_neighbors": [3,5,7,9],
                 "weights": ['uniform','distance'],
                 "p":[1,2,3]}
parameters.append(parameters_knn)
#Here grid search is performed for each model. THe best estimator combination for each model
#is stored in a list best_models.
from tqdm import tqdm
def gridSearchCV(models,params,count):
    best_models=[]
    for i in tqdm(range(0,count)):
        model_grid = GridSearchCV(models[i], parameters[i], n_jobs=-1, verbose=1, cv=5)
        model_grid.fit(train_x,target)
        best_models.append(model_grid.best_estimator_)
    return best_models

best_model_list = gridSearchCV(classifiers,parameters,4)
#Voting classifier is given the best models of mlp,sgd and rf. It fits all three models with
#given data. Eventually predictions are made through a voting system. THe voting chosen here
#is soft, which means it takes the argmax of the sums of predicted probabilities. THis ensemble
#method helps to make predictions more reliable. 
vot_clf = VotingClassifier(estimators=[('mlp', best_model_list[0]), ('sgd', best_model_list[1]), 
                                       ('rf', best_model_list[2]), ('knn', best_model_list[3])], 
                           voting='soft',n_jobs=-1)
vot_clf.fit(train_x,target)
#the list of estimators trained on by this voting classifier. It is always good practice,
#to choose odd no of estimators.
print(vot_clf.estimators_)
#make submission
y_pred=vot_clf.predict(test_x).astype(int)
sub = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})
sub.to_csv('Submission3.csv',index=False)