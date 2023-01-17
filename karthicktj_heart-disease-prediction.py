

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import cm
from IPython.display import Image 


data = pd.read_csv("../input/heart-disease-uci/heart.csv")
data.head()
data.tail()  #last Five rows
data.info() #about the Dataset
print("Shape of the Data:{}".format(data.shape)) #shape of the data
#check any missing value in the data set
data.isnull().sum()
for feature in data.columns:
    print("{} in Unique Values:{}".format(feature ,data[feature].nunique()))
plt.figure(figsize = (10,10))
sns.distplot(data.age,kde = True)
fig ,ax = plt.subplots(1,2,figsize =(20,10))
sns.countplot(x="sex" ,data = data ,palette = "Set3" ,ax = ax[0])
ax[0].set_title("sex (Male =1 ,,Female =0)")
labels = ["Male" ,"Female"]
colors = ["lightskyblue" ,"gold"]
ax[1].set_title("sex (Male =1 ,,Female =0)")
data.sex.value_counts().plot.pie(explode = [0.1,0] ,autopct="%1.1f%%" ,shadow = True,ax=ax[1] , labels = labels , colors = colors )
ax[1].legend(labels ,loc = "upper right")
plt.show()

fig ,ax = plt.subplots(1,2,figsize =(20,10))
sns.countplot(x="cp" ,data = data ,palette = "Accent" ,ax = ax[0])
ax[0].set_title("Cheast Pain(cp)")
ax[1].set_title("CheastPain(cp)")
colors = ["lightskyblue" ,"gold","red","blue"]
data.cp.value_counts().plot.pie(explode = [0.1,0,0,0] ,autopct="%1.1f%%" ,shadow = True,ax=ax[1],colors = colors)
labels = ["0","1","2","3"]
ax[1].legend(labels ,loc = "lower right")

plt.show()
fig,ax =plt.subplots(1,2,figsize=(20,10))
sns.countplot(x="cp" ,hue = "sex" ,orient = 'h' ,data = data ,saturation = 0.9,palette = "ocean",ax=ax[0])
ax[0].set_title("Relationship Between Cp(chest Pain) and Sex(Male:1 , Female :0) ")
sns.countplot(x = "sex" , data = data ,hue = "target" ,palette = "Set2" ,ax = ax[1])
ax[1].set_title("Heart Disease affected by sex(Male:1 ,Female:0)")
labels = ["0-Not affected ","1-affted"]
ax[1].legend(labels ,loc = "upper right")
plt.show()
fig ,ax = plt.subplots(1,2,figsize=(20,10))
sns.kdeplot(data.trestbps ,ax =ax[1])
ax[0].set_title("trestbps(resting blood pressure")
data.hist(column = "trestbps",bins = 10 ,ax = ax[0])
ax[1].set_title("trestbps(resting blood pressure)")
plt.show()
fig ,ax = plt.subplots(1,2,figsize =(20,10))
sns.countplot(x="target" ,data = data ,palette = "Set2" ,ax = ax[0])
ax[0].set_title("Target (1= Affected by Heart diseases ,0=Not Affected by Heart diseases)")
labels = ["Affected by Heart diseases" ,"Not Affected by Heart diseases"]
colors = ["lightskyblue" ,"gold"]
ax[1].set_title("Target (1= Affected by Heart diseases ,0=Not Affected by Heart diseases)")
data.target.value_counts().plot.pie(explode = [0.1,0] ,autopct="%1.1f%%" ,shadow = True,ax=ax[1] , labels = labels , colors = colors )
ax[1].legend(labels ,loc = "upper right")
plt.show()
plt.figure(figsize=(10,5))
sns.kdeplot(data.chol,shade = True)
plt.title("Serum cholestoral in mg/dl")
plt.show()
fig ,ax = plt.subplots(1,2,figsize =(20,10))
sns.countplot(x="fbs" ,data = data ,palette = "Pastel1" ,ax = ax[0])
ax[0].set_title("Fasting blood sugar &gt; 120 mg/dl) (1 = True; 0 = False)")
labels = ["1-True" , "0-False"]
colors = ["lightskyblue" ,"red"]
ax[1].set_title("Fasting blood sugar &gt; 120 mg/dl) (1 = True; 0 = False)")
data.fbs.value_counts().plot.pie(explode = [0.1,0] ,autopct="%1.1f%%" ,shadow = True,ax=ax[1] , labels = labels , colors = colors )
ax[1].legend(labels ,loc = "upper right")
plt.show()
fig ,ax = plt.subplots(1,2,figsize =(20,10))
sns.countplot(x="restecg" ,data = data ,palette = "Pastel2" ,ax = ax[0])
ax[0].set_title("Resting electrocardiographic results")
colors = ["gold" ,"red" ,"lightskyblue"]
ax[1].set_title("Resting electrocardiographic results")
labels =[0,1,2]
data.restecg.value_counts().plot.pie(explode = [0,0,0] ,autopct="%1.1f%%" ,shadow = True,ax=ax[1] , labels = labels , colors = colors )
ax[1].legend(labels ,loc = "upper right")
plt.show()
plt.figure(figsize=(10,5))
sns.violinplot(x="target", y ="thalach" ,hue ="sex" ,data = data,palette = "Set3")
plt.title("Maximum heart rate affeted person heart Diseases (1-Affected ,0-Not Affected)")
labels = ["1-Male" ,"0-Female"]
plt.legend(labels ,loc = "upper right")
plt.show()
plt.figure(figsize =(10,5))
sns.boxplot(x="exang" ,y = "age" , data = data ,palette = "Reds")
plt.title("Relationship Between Exang and Age")
plt.show()

data.groupby("target")["oldpeak"].mean()
fig ,ax = plt.subplots(1,3,figsize = (10,5))
sns.boxplot(x = "oldpeak", data = data,ax=ax[0])
ax[0].set_title("Oldpeak")
sns.kdeplot(data.oldpeak ,ax = ax[1] )
ax[1].set_title("oldpeak")
sns.scatterplot(x= "age" , y = "oldpeak" ,data = data ,ax=ax[2])
ax[2].set_title("Relationship between Age and old peak")
plt.show()
def target(data ,feature):
    x = data.groupby(feature)["target"].sum()
    print("Feature {}:{}".format(feature ,x))
columns =["slope" ,"ca" ,"thal"]
for var in columns:
    target(data , var)
fig ,ax = plt.subplots(1,2,figsize =(20,10))
sns.countplot(x="slope" ,data = data ,palette = "Pastel1",hue = "target" ,ax = ax[0] )
ax[0].set_title("The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)")
labels = [ "0: upsloping", "1: flat", "2: downsloping"]
colors = ["lightskyblue" ,"red" ,"gold"]
ax[1].set_title("The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)")
data.slope.value_counts().plot.pie(explode = [0.1,0,0] ,autopct="%1.1f%%" ,shadow = True,ax=ax[1] , labels = labels , colors = colors )
ax[1].legend(labels ,loc = "upper right")
plt.show()
fig ,ax = plt.subplots(1,2,figsize =(20,10))
sns.countplot(x="ca" ,data = data ,palette = "Pastel2",hue = "target" ,ax = ax[0] )
ax[0].set_title("Number of Major Vessels Affectd by Heart Diseases")
ax[0].set_xlabel("Ca Major Vessels")
labels = ["0","1","2","3" ,"4"]
colors = ["lightskyblue" ,"red" ,"gold" ,"blue" ,"green"]
ax[1].set_title("Number of Major Vessels ")
data.ca.value_counts().plot.pie(explode = [0.1,0,0,0,0] ,autopct="%1.1f%%" ,shadow = True,ax=ax[1] , labels = labels , colors = colors )
ax[1].legend(labels ,loc = "upper right")
plt.show()

fig ,ax = plt.subplots(1,2,figsize =(20,10))
sns.countplot(x = "thal" ,data = data ,palette = "Set1",hue = "target" ,ax = ax[0] )
ax[0].set_title("Thalassemia A Blood Disorder Affectd by Heart Diseases")
ax[0].set_xlabel("thal - Thalassemia")
labels = ["0" ,"1" ,"2" ,"3"]
colors = ["red" ,"gold" ,"blue","lightskyblue"]
ax[1].set_title("Thalassemia A Blood Disorder ")
data.thal.value_counts().plot.pie(explode = [0,0,0 ,0.1] ,autopct="%1.1f%%" ,shadow = True,ax=ax[1] , labels = labels , colors = colors )
ax[1].legend(labels ,loc = "upper right")
plt.show()
data.columns
df = ["trestbps","chol"]
fig = plt.figure(figsize=(15,5))
for i ,var in zip(range(1,3),df):
    ax = fig.add_subplot(1,3,i)
    sns.boxplot(data[var] ,ax= ax ,palette = "Set3")
    plt.xlabel(var)
    plt.title(var)
plt.show()

    

IQR = data["trestbps"].quantile(0.75) - data["trestbps"].quantile(0.25)
upper_boundary = data["trestbps"].quantile(0.75) + (1.5 * IQR)
Lower_boundary = data["trestbps"].quantile(0.25) - (1.5 * IQR)
print("IQR: {} upperBoundary:{} and Lowerboundary:{}".format(IQR , upper_boundary ,Lower_boundary))
#chol feature
IQR = data["chol"].quantile(0.75) - data["trestbps"].quantile(0.25)
upper_boundary = data["chol"].quantile(0.75) + (1.5 * IQR)
Lower_boundary = data["chol"].quantile(0.25) - (1.5 * IQR)
print("IQR: {} upperBoundary:{} and Lowerboundary:{}".format(IQR , upper_boundary ,Lower_boundary))
data.loc[data["trestbps"] > 170 ,"trestbps" ] = 170
data.loc[data["chol"]>400 , "chol"] = 400
plt.figure(figsize=(10,10))
corr = data.corr()
sns.heatmap(corr ,annot = True ,cmap = "Pastel1")
plt.title("Correlation Between All Feature")
plt.show()
df = data.copy()
data['sex'] = data['sex'].astype('object')
data['cp'] = data['cp'].astype('object')
data['fbs'] = data['fbs'].astype('object')
data['restecg'] = data['restecg'].astype('object')
data['exang'] = data['exang'].astype('object')
data['slope'] = data['slope'].astype('object')
data['thal'] = data['thal'].astype('object')
data.dtypes
data.head()
data = pd.get_dummies(data ,drop_first = True)
data.shape
x = data.drop(columns ="target" ,axis = 1)
y = data["target"]
print(x.columns)
y.head()
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score ,recall_score ,roc_auc_score
x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size =0.2 , random_state = 0)
print("Shape of Training  and  Testing")
print(x_train.shape ,x_test.shape)
from sklearn.preprocessing import MinMaxScaler
x_train_std = MinMaxScaler().fit_transform(x_train)
x_test_std = MinMaxScaler().fit_transform(x_test)
model = {"LG":LogisticRegression() ,"RF":RandomForestClassifier() ,"DT": DecisionTreeClassifier() ,"svc":SVC()}

def create_modle(model ,x_train ,y_train ,x_test,y_test): 
    model_score_train = {}
    model_score_test = {}
    cnn = {}
    for name,model in model.items():
        np.random.seed(42)
        model.fit(x_train ,y_train) #fit model
        pred = model.predict(x_test)
        model_score_train[name] = model.score(x_train ,y_train)
        model_score_test[name] = model.score(x_test ,y_test)
        cnn[name] = confusion_matrix(y_test ,pred)
        
    return model_score_train,model_score_test,cnn
    
training_score = create_modle(model , x_train ,y_train ,x_test ,y_test)
train ,test,cnn = training_score
cnn
train
test
data1 = pd.DataFrame({"Train_score":train ,"Testing_score":test })
data1.head()
plt.figure(figsize=(10,20))
data1.T.plot.bar(width = 0.9)
plt.title("Model Comparison of Training  and Testing Score")
plt.show()
fig  = plt.figure(figsize=(10,5))
a =1
for key,name in cnn.items():
    ax = fig.add_subplot(2,2,a)
    plt.title(key)
    sns.heatmap(name ,annot =True , ax =ax ,cmap = "Set3")
    a = a+1
plt.show()
training_score_std = create_modle(model , x_train_std ,y_train ,x_test_std ,y_test)
train_std ,test_std,cnn_std = training_score_std
print("Training Score of After Normalization :{}".format(train_std))
print("Testing Score of After Normalization :{}".format(test_std))
fig  = plt.figure(figsize=(10,5))
a =1
for key,name in cnn_std.items():
    ax = fig.add_subplot(2,2,a)
    plt.title(key)
    sns.heatmap(name ,annot =True , ax =ax ,cmap = "Set3")
    a = a+1
plt.show()
param = {"penalty": ["l1","l2" ,"elasticnet" ,"none"],    #Regularization paramater
         "C" : [0.1,0.001,0.1,1.0,1.5 ,3.0],             #strength of regularization
         "solver":["newton-cg", "lbfgs", "liblinear"],
         "multi_class":['auto', 'ovr', "multinomial"],
         "max_iter" :[10,20,30,50,100]
        }
LG_H = LogisticRegression()
Grid_lg = GridSearchCV(LG_H , param_grid = param ,cv = 5 , scoring = 'accuracy' ,n_jobs = -1).fit(x_train_std,y_train)
Grid_lg.best_params_
Grid_best_est = Grid_lg.best_estimator_
LG_Gr_pred = Grid_best_est.predict(x_test_std)
Lg_accuracy = accuracy_score(y_test , LG_Gr_pred)
Lg_precision = precision_score(y_test , LG_Gr_pred)
Lg_recall = recall_score(y_test ,LG_Gr_pred)
Lg_conf = confusion_matrix(y_test , LG_Gr_pred)
print("Accuracy_score of Logistic Regression:{}".format(round(Lg_accuracy * 100)))
print("RecallScore(Positive Prediction,Low False Negative Rate):{}".format(round(Lg_recall*100)))
print("PrecisionScore(Low false Postitive rate):{}".format(round(Lg_precision *100)))
sns.heatmap(Lg_conf ,annot = True , cmap = "Pastel1")
plt.title("Confusion_matrix on LogisticRegression")
plt.show()
kernel = ['linear', 'poly', 'rbf', 'sigmoid']  # what type of algorithm is used
C = [0.001,0.005,0.01,0.05, 0.1, 0.5, 1, 5, 10, 50,100,500,1000] 
gamma = [0.001, 0.01, 0.1, 0.5, 1]     #this parem used for Rbf
degree =[1,2,3,4]                     #used for polmonial algorithm
param = {'kernel':kernel , #
         "gamma":gamma,
         "degree": degree,
            "C" :C}


svc = SVC()
svc_grid = GridSearchCV(svc , param_grid = param ,scoring = "accuracy" ,n_jobs = -1 ,verbose = 2 ,cv = 10).fit(x_train_std,y_train)
svc_grid.best_params_
svc_est = svc_grid.best_estimator_
svc_Gr_pred = Grid_best_est.predict(x_test_std)
svc_accuracy = accuracy_score(y_test , svc_Gr_pred)
svc_precision = precision_score(y_test , svc_Gr_pred)
svc_recall = recall_score(y_test ,svc_Gr_pred)
svc_conf = confusion_matrix(y_test , svc_Gr_pred)
print("Accuracy_score of Support Vector Classifier:{}".format(round(svc_accuracy * 100)))
print("RecallScore(Positive Prediction,Low False Negative Rate):{}".format(round(svc_recall*100)))
print("PrecisionScore(Low false Postitive rate):{}".format(round(svc_precision *100)))
sns.heatmap(Lg_conf ,annot = True , cmap = "Blues")
plt.title("Confusion_matrix on Support Vector Classifier")
plt.show()

neighbour = [i for i in range(1,30)]
print(neighbour ,end = " ")
train_score,test_score =[] ,[]
for n in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = n).fit(x_train ,y_train)
    train_score.append(knn.score(x_train ,y_train))
    test_score.append(knn.score(x_test ,y_test))
test_score
plt.figure(figsize = (10,10))
plt.plot(train_score)
plt.plot(test_score)
plt.title("Training and Testing Accuracy")
plt.xlabel("Kneighbors")
plt.ylabel("accuracy")
plt.xticks(range(1,21))
plt.legend(labels=["Train_score" ,"Test_score"] ,loc = "upper right")
plt.show()
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
parem ={"criterion" : ['gini', 'entropy'] ,
       "splitter":['best', 'random'] ,
        "max_depth" : [int(x) for x in np.linspace(1, 1000,500)],
        'min_samples_split': [2, 5, 10,14],
        'min_samples_leaf' :[1, 2, 4,6,8]
       }
DT = DecisionTreeClassifier()
Randomized = RandomizedSearchCV(estimator=DT,param_distributions=parem,n_iter=100,cv=5,verbose=2,
                               random_state=100,n_jobs=-1)
Randomized.fit(x_train ,y_train)
Randomized.best_params_
Best_estimater = Randomized.best_estimator_
pred = Best_estimater.predict(x_test)
y_pred = Best_estimater.predict_proba(x_test)[:, 1]
Confusion_Dt = confusion_matrix(y_test ,pred)
print("Accuracy_score for Decision Tree:{}".format(accuracy_score(y_test ,pred)))
sns.heatmap(Confusion_Dt , annot = True ,cmap = "Set3")
plt.title("Confusion_matrix for Decision Tree")
plt.show()
from sklearn.metrics import roc_curve
False_positive_rate, True_pos_rate, thresholds = roc_curve(y_test, y_pred)

fig, ax = plt.subplots()
plt.plot(False_positive_rate ,True_pos_rate)
plt.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC  classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
DT = DecisionTreeClassifier()
parem ={"criterion" : ['gini', 'entropy'] ,
       "splitter":['best', 'random'] ,
        "max_depth" : [int(x) for x in np.linspace(1, 100,50)],
        'min_samples_split': [2, 5, 10,14],
        'min_samples_leaf' :[1, 2, 4,6,8]
       }
GS_T = GridSearchCV(estimator = DT , param_grid = parem , scoring = "accuracy" ,n_jobs = -1 ,cv = 5  ).fit(x_train ,y_train)
GS_T.best_params_
GSVT = GS_T.best_estimator_
dpred = GSVT.predict(x_test)
d_con = confusion_matrix(y_test ,dpred)
d_precision = precision_score(y_test ,dpred)
d_Recall = recall_score(y_test ,dpred)
d_accuracy = accuracy_score(y_test , dpred) 
print("Accuracy_score of Decision Tree Classifier:{}".format(round(d_accuracy * 100)))
print("RecallScore(Positive Prediction,Low False Negative Rate):{}".format(round(d_Recall*100)))
print("PrecisionScore(Low false Postitive rate):{}".format(round(d_precision *100)))
sns.heatmap(d_con ,annot = True , cmap = "Blues")
plt.title("Confusion_matrix on Decision_tree")
plt.show()

n_estimators = [int(x) for x in np.linspace(start = 2, stop = 2000, num = 1000)]
max_features = ['auto', 'sqrt','log2']
max_depth = [int(x) for x in np.linspace(1, 1000,500)]
min_samples_split = [2, 5, 10,14,15]
min_samples_leaf = [1, 2, 4,6,8,10,12]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
RF=RandomForestClassifier()
Rf_tuning=RandomizedSearchCV(estimator=RF,param_distributions=random_grid,n_iter=100,cv=5,verbose=2,
                               random_state=100,n_jobs=-1)
Rf_tuning.fit(x_train,y_train)
Rf_tuning.best_params_
rf = Rf_tuning.best_estimator_
RF_pred = rf.predict(x_test)
Confusion_Dt = confusion_matrix(y_test ,RF_pred)
pre = precision_score(y_test ,RF_pred)
Recall = recall_score(y_test ,RF_pred)
print("Accuracy_score for Random Forest Classifier:{}".format(accuracy_score(y_test ,RF_pred)))
print("RecallScore(Positive Prediction,Low False Negative Rate):{}".format(round(Recall*100)))
print("PrecisionScore(Low false Postitive rate):{}".format(round(pre *100)))
sns.heatmap(Confusion_Dt , annot = True ,cmap = "Set3")
plt.title("Confusion_matrix for RandomForestClassifier")
plt.show()