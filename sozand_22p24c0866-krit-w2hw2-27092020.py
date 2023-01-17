# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings
#Read data

dt=pd.read_csv('../input/titanic/train.csv')
dt.head()
dt.info()
#check number of servior

dt.groupby(['Survived'])['PassengerId'].count()
#check % of servior

dt.groupby(['Survived']).count().apply(lambda x: 100 * x / float(x.sum()))['PassengerId']
#check % of servior per passager class [Pclass]

dead_per_class = dt.groupby(['Survived','Pclass']).count().apply(lambda x: 100 * x / float(x.sum()))['PassengerId'][0]

survived_per_clase = dt.groupby(['Survived','Pclass']).count().apply(lambda x: 100 * x / float(x.sum()))['PassengerId'][1]

f,ax=plt.subplots(1,2,figsize=(18,8))

dead_per_class.plot.bar(ax=ax[0])

ax[0].set_title('Dead vs class')

ax[0].set_ylabel('% of dead passenger')

survived_per_clase.plot.bar(ax=ax[1])

ax[1].set_title('Survived vs class')

ax[1].set_ylabel('% of survived passenger')

plt.show()
#check % of servior per Sex

dead_per_class = dt.groupby(['Survived','Sex']).count().apply(lambda x: 100 * x / float(x.sum()))['PassengerId'][0]

survived_per_clase = dt.groupby(['Survived','Sex']).count().apply(lambda x: 100 * x / float(x.sum()))['PassengerId'][1]

f,ax=plt.subplots(1,2,figsize=(18,8))

dead_per_class.plot.bar(ax=ax[0])

ax[0].set_title('Dead vs sex')

ax[0].set_ylabel('% of dead passenger')

survived_per_clase.plot.bar(ax=ax[1])

ax[1].set_title('Survived vs sex')

ax[1].set_ylabel('% of survived passenger')

plt.show()
print('คนที่แก่สุด:',dt['Age'].max(),'ปี')

print('คนที่เด็กสุด:',dt['Age'].min(),'ปี')

print('อายุเฉลี่ยผู้โดยสาร:',dt['Age'].mean(),'ปี')
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=dt,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=dt,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
#Check null

dt.isnull().sum()
#หาคำนำหน้าชื่อ

dt['Name_Title']=0

for i in dt:

    dt['Name_Title']=dt.Name.str.extract('([A-Za-z]+)\.')

dt
pd.crosstab(dt.Name_Title,dt.Sex).T
#จัดกลุ่มคำนำหน้าชื่อใหม่

dt['Name_Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

dt
#จำนวนคนในแต่ละกลุ่มที่แบ่งตามคำนำหน้าชื่อ

dt.groupby(['Name_Title'])['PassengerId'].count()
#หาค่าเฉลี่ยอายุของคนแต่ละกลุ่มแบ่งตามคำนำหน้าชื่อ

dt.groupby(['Name_Title'])['Age'].mean()
# Fill missing data [Age]

dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Mr'),'Age']=33

dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Mrs'),'Age']=36

dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Master'),'Age']=5

dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Miss'),'Age']=22

dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Other'),'Age']=46
#check % of servior per Sex



f,ax=plt.subplots(1,2,figsize=(18,8))

dead = dt[dt['Survived'] == 0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

survived = dt[dt['Survived'] == 1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')

plt.show()
#Check null

dt.isnull().sum()
dt.groupby(['Embarked'])['PassengerId'].count()
# Fill missing data [Embarked]

dt['Embarked'].fillna('S',inplace=True)
#Check null

dt.isnull().sum()
pd.crosstab(dt.SibSp,dt.Survived)
pd.crosstab(dt.Parch,dt.Survived)
#Cabin

dt.groupby(['Cabin']).count()['PassengerId']
dt.Cabin.isnull().sum()
#Fare

dt['SFare']=pd.qcut(dt['Fare'],4)

dt.groupby(['SFare'])['Survived'].mean().to_frame()
#Feature engineering

###################################################

#Age [Make 4 bins]

dt['Age_bin']=0

dt.loc[dt['Age']<=16,'Age_bin']=0

dt.loc[(dt['Age']>16)&(dt['Age']<=32),'Age_bin']=1

dt.loc[(dt['Age']>32)&(dt['Age']<=48),'Age_bin']=2

dt.loc[(dt['Age']>48)&(dt['Age']<=64),'Age_bin']=3

dt.loc[dt['Age']>64,'Age_bin']=4

#Family and Alone

#Family = SibSp + Parch

#If Family == 0 Then Alone = 1

dt['Family']=0

#Family

dt['Family']=dt['Parch']+dt['SibSp']

dt['Alone']=0

#Alone

dt.loc[dt.Family==0,'Alone']=1

#Fare [Make 4 bins]

dt['Fare_Range']=0

dt.loc[dt['Fare']<=7.91,'Fare_Range']=0

dt.loc[(dt['Fare']>7.91)&(dt['Fare']<=14.454),'Fare_Range']=1

dt.loc[(dt['Fare']>14.454)&(dt['Fare']<=31),'Fare_Range']=2

dt.loc[(dt['Fare']>31)&(dt['Fare']<=513),'Fare_Range']=3

#Sex

dt['Sex'].replace(['male','female'],[0,1],inplace=True)

#Embarked

dt['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

#Name_Title

dt['Name_Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
dt
#Drop unwanted columns

dt.drop(['Name','Age','Ticket','Fare','SFare','Cabin','PassengerId'],axis=1,inplace=True)
dt
from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score #for confusion matrix

import tensorflow as tf

import numpy as np

import pandas as pd

import time

from sklearn.preprocessing import StandardScaler
#k-fold cross validation

def kfolds(dt,k=5):

  datasets={}

  dataset_Length = len(dt)

  for i in range(k):

    datasets[i] = dt[i:dataset_Length:k]

  return datasets
#Neural network : MLP

def sigmoid(x):

    r = (1 / (1 + np.exp(-x)))

    return np.round(r)

    #return r

def relu(x):

    return tf.where(x>=0,x,0)

class MLP():

    def __init__(self, neurons=[1,100,100,1],activation=[relu,relu,None],lr=0.0001, epochs=1000):

        self.W=[]

        self.activation = activation

        self.lr = lr

        self.epochs = epochs

        self.neurons = neurons

        

        #Initialize W and b

        for i in range(1,len(neurons)):

            self.W.append(tf.Variable(np.random.randn(neurons[i-1],neurons[i]),dtype=tf.float32)) #W

            self.W.append(tf.Variable(np.random.randn(neurons[i]),dtype=tf.float32)) #b

    

    def __call__(self,x):

        for i in range(0,len(self.W),2):

            #print(f'x : {x.shape} {type(x)}, w : {self.W[i].shape } {type(self.W[i])}, b : {self.W[i+1].shape}')

            #print(str(i))

            x = x @ self.W[i]+self.W[i+1]

            if self.activation[i//2] is not None:

                x = self.activation[i//2](x)

        return x

    def __name__(self):

        name = ''

        for i in range(len(self.neurons)):

            if name == '':

                name = '[' + str(self.neurons[i])

            else:

                name = name + ',' + str(self.neurons[i])

        name = name + ']'

        print(f'Custom_MLP : {name}')

        

    def predict(self,x):

        X = tf.Variable(x.to_numpy(),dtype=tf.float32)

        return pd.Series(np.array(self(X)).reshape(-1))

    

    def fit(self,x,y):

        bce = tf.keras.losses.BinaryCrossentropy()

        X = tf.Variable(x.to_numpy(),dtype=tf.float32)

        Y = tf.Variable(y.to_numpy(),dtype=tf.float32)

        for epoch in range(self.epochs):

            with tf.GradientTape() as t:

                #loss = tf.reduce_mean((self(X)-Y)**2)

                loss = bce(Y, self(X))

            dW = t.gradient(loss,self.W)

            for i,w in enumerate(self.W):

                w.assign_sub(self.lr*self.W[i])

            #if epoch % 200 == 0 :

            #    print(f'epoch : {epoch}; loss : {loss.numpy()}')

                

    def score(self,X,y, normalize=True, sample_weight=None):

        y_pred = self.predict(X)

        y_true = y

        #print(y_pred.shape)

        #print(y_true.shape)

        return accuracy_score(y_true,y_pred, sample_weight=sample_weight)

                
def feature_engineering(dt):

    #Age [Make 4 bins]

    dt['Age_bin']=0

    dt.loc[dt['Age']<=16,'Age_bin']=0

    dt.loc[(dt['Age']>16)&(dt['Age']<=32),'Age_bin']=1

    dt.loc[(dt['Age']>32)&(dt['Age']<=48),'Age_bin']=2

    dt.loc[(dt['Age']>48)&(dt['Age']<=64),'Age_bin']=3

    dt.loc[dt['Age']>64,'Age_bin']=4

    #Family and Alone

    #Family = SibSp + Parch

    #If Family == 0 Then Alone = 1

    dt['Family']=0

    #Family

    dt['Family']=dt['Parch']+dt['SibSp']

    dt['Alone']=0

    #Alone

    dt.loc[dt.Family==0,'Alone']=1

    #Fare [Make 4 bins]

    dt['Fare_Range']=0

    dt.loc[dt['Fare']<=7.91,'Fare_Range']=0

    dt.loc[(dt['Fare']>7.91)&(dt['Fare']<=14.454),'Fare_Range']=1

    dt.loc[(dt['Fare']>14.454)&(dt['Fare']<=31),'Fare_Range']=2

    dt.loc[(dt['Fare']>31)&(dt['Fare']<=513),'Fare_Range']=3

    #Sex

    dt['Sex'].replace(['male','female'],[0,1],inplace=True)

    #Embarked

    dt['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

    #Name_Title

    dt['Name_Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

    dt.drop(['Name','Age','Ticket','Fare','PassengerId'],axis=1,inplace=True)

    return dt
def preprocess_data(dt):

    #Fill Age

    dt['Name_Title']=0

    for i in dt:

        dt['Name_Title']=dt.Name.str.extract('([A-Za-z]+)\.')

    dt['Name_Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

    dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Mr'),'Age']=33

    dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Mrs'),'Age']=36

    dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Master'),'Age']=5

    dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Miss'),'Age']=22

    dt.loc[(dt.Age.isnull())&(dt.Name_Title=='Other'),'Age']=46



    # Fill Embarked

    dt['Embarked'].fillna('S',inplace=True)

    

    #Drop cabin

    dt.drop(['Cabin'],axis=1,inplace=True)



    dataset = feature_engineering(dt)



    return dataset
class collected_metrics():

    def __init__(self,clf_name,foldCV):

        #Classifier name

        self.clf_name = clf_name

        self.folds = foldCV

        #Survived

        self.f_s_tp = [0] * self.folds

        self.f_s_tn = [0] * self.folds

        self.f_s_fp = [0] * self.folds

        self.f_s_fn = [0] * self.folds

        self.f_s_precision = [0] * self.folds

        self.f_s_recall = [0] * self.folds

        self.f_s_f1 = [0] * self.folds

        #Dead

        self.f_d_tp = [0] * self.folds

        self.f_d_tn = [0] * self.folds

        self.f_d_fp = [0] * self.folds

        self.f_d_fn = [0] * self.folds

        self.f_d_precision = [0] * self.folds

        self.f_d_recall = [0] * self.folds

        self.f_d_f1 = [0] * self.folds

        #Average

        self.f_f1_avg = [0] * self.folds

    def save_metrics(self,fold,y_test,y_pred):

        #print(f'flod : {fold}')

        confusion_m = confusion_matrix(y_test, y_pred)

        #Survived

        self.f_s_tn[fold], self.f_s_fp[fold], self.f_s_fn[fold], self.f_s_tp[fold] = confusion_m.ravel()

        self.f_s_precision[fold] = self.f_s_tp[fold]/(self.f_s_tp[fold] + self.f_s_fp[fold])

        self.f_s_recall[fold] = self.f_s_tp[fold]/(self.f_s_tp[fold] + self.f_s_fn[fold])

        self.f_s_f1[fold] = 2*(self.f_s_precision[fold] * self.f_s_recall[fold])/(self.f_s_precision[fold] + self.f_s_recall[fold])

        #Dead

        self.f_d_tn[fold], self.f_d_fp[fold], self.f_d_fn[fold], self.f_d_tp[fold] = confusion_m.ravel()

        self.f_d_precision[fold] = self.f_d_tp[fold]/(self.f_d_tp[fold] + self.f_d_fp[fold])

        self.f_d_recall[fold] = self.f_d_tp[fold]/(self.f_d_tp[fold] + self.f_d_fn[fold])

        self.f_d_f1[fold] = 2*(self.f_d_precision[fold] * self.f_d_recall[fold])/(self.f_d_precision[fold] + self.f_d_recall[fold])

        #F1 Average

        self.f_f1_avg[fold] = (self.f_s_f1[fold] + self.f_d_f1[fold])/2

    def export_metrics(self):

        #Survived

        f_s_tp=np.array(self.f_s_tp).reshape(-1,1)

        f_s_tn=np.array(self.f_s_tn).reshape(-1,1)

        f_s_fp=np.array(self.f_s_fp).reshape(-1,1)

        f_s_fn=np.array(self.f_s_fn).reshape(-1,1)

        f_s_precision=np.array(self.f_s_precision).reshape(-1,1)

        f_s_recall=np.array(self.f_s_recall).reshape(-1,1)

        f_s_f1=np.array(self.f_s_f1).reshape(-1,1)

        #Dead

        f_d_tp=np.array(self.f_d_tp).reshape(-1,1)

        f_d_tn=np.array(self.f_d_tn).reshape(-1,1)

        f_d_fp=np.array(self.f_d_fp).reshape(-1,1)

        f_d_fn=np.array(self.f_d_fn).reshape(-1,1)

        f_d_precision=np.array(self.f_d_precision).reshape(-1,1)

        f_d_recall=np.array(self.f_d_recall).reshape(-1,1)

        f_d_f1=np.array(self.f_d_f1).reshape(-1,1)

        #Avg

        f_f1_avg=np.array(self.f_f1_avg).reshape(-1,1)

        survived = np.hstack((f_s_tp,f_s_tn,f_s_fp,f_s_fn,f_s_precision,f_s_recall,f_s_f1,f_f1_avg))

        survived_export = pd.DataFrame(survived,index= [i for i in range(survived.shape[0])], columns=['TP_1','TN_1','FP_1','FN_1','Precision_1','Recall_1','F-Measure_1','F1-Measure_avg'])

        dead = np.hstack((f_d_tp,f_d_tn,f_d_fp,f_d_fn,f_d_precision,f_d_recall,f_d_f1,f_f1_avg))

        dead_export = pd.DataFrame(dead,index= [i for i in range(dead.shape[0])], columns=['TP_0','TN_0','FP_0','FN_0','Precision_0','Recall_0','F-Measure_0','F1-Measure_avg'])

        total = np.hstack((f_s_tp,f_d_tp,f_s_tn,f_d_tn,f_s_fp,f_d_fp,f_s_fn,f_d_fn,f_s_precision,f_d_precision,f_s_recall,f_d_recall,f_s_f1,f_d_f1,f_f1_avg))

        total_export = pd.DataFrame(total,index= [i for i in range(total.shape[0])], columns=['TP_1','TP_0','TN_1','TN_0','FP_1','FP_0'

                                                                                              ,'FN_1','FN_0','Precision_1','Precision_0','Recall_1','Recall_0'

                                                                                              ,'F1-Measure_1','F1-Measure_0','F1-Measure_avg'])

        return self.clf_name,survived_export,dead_export,total_export

        

        
def train_model(clf,X_train,y_train,X_test,y_test,i,metrics,verb=False):

    #print(f'x :{X_train.shape},y :{y_train.shape}')

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    #r = y_pred

    #print(f'type: {type(r)}, Shape: {r.shape}')

    #e = pd.Series(r)

    #print(f'type: {type(e)}, Shape: {e.shape}')

    #print(f'y_shape: {y_pred.shape}, Type: {type(y_pred)}')

    #print(f'Y_pred_type: {type(y_pred)}, y_test_type: {type(y_test)}')

    metrics.save_metrics(i,y_test,y_pred)

    if verb:

        print('**************************************')

        print('at Fold :{}'.format(str(i)))

        print('**************************************')

        print('Train score : {}'.format(clf.score(X_train,y_train)))

        print('Test score : {}'.format(clf.score(X_test,y_test)))

        print('Root Mean Square Error : {}'.format(mean_squared_error(y_test,y_pred)**0.5))

    return clf,y_pred,clf.score(X_train,y_train),clf.score(X_test,y_test),mean_squared_error(y_test,y_pred)**0.5
#Pipeline

def pipeline(verb = False):

    #Define parameters##########################

    folds = 5

    ############################################



    #Read plant1 dataset 

    df = pd.read_csv('../input/titanic/train.csv') 



    #Create dataset

    dataset = preprocess_data(df)



    #Create k=Flod dataset

    folds_dataset = kfolds(dataset,k=folds)

    



    #Define models##############################

    Decision_Tree=DecisionTreeClassifier()



    Naive_bayes=GaussianNB()

    

    classifier_mlp = MLPClassifier(solver='sgd', alpha=1e-5, activation = 'logistic',

                    hidden_layer_sizes=(30,), max_iter=500, learning_rate = 'adaptive',

                    random_state=42)



    neurons = [10,64,32,8,1]

    ac = [relu,relu,relu,sigmoid]

    NN = MLP(neurons=neurons,activation = ac,lr=0.0015,epochs=600)

    

    clfs = [Decision_Tree,Naive_bayes,NN,classifier_mlp]

    ############################################

    

    

    #Lists For Metric###########################

    train_scores = []

    test_scores = []

    rmse_scores = []

    compare_scores = []

    compare_metrics = []

    ############################################



    for clf in clfs:

        if verb :

            print('==============================================================')

            print('==============================================================')

            print(f'Model : {type(clf).__name__}')

            print('==============================================================')

            print('==============================================================')

            

        clsName = type(clf).__name__

        if(clsName == 'MLP'):

            clsName = "Custom_MLP"

        cm = collected_metrics(clsName,folds)



        #k fold cross validation training

        for fold in range(folds):

            train_data = pd.DataFrame()

            for j in range(folds):

                if j==fold:

                    pass

                else:

                    train_data = pd.concat([train_data,folds_dataset[j]])



            test_data = folds_dataset[fold]



            #Prepare training data

            X_train = train_data.drop(['Survived'],axis=1)

            y_train = train_data['Survived']

            if verb :

                print(f'Training data size : {X_train.shape}')



            #Prepare testing data

            X_test = test_data.drop(['Survived'],axis=1)

            y_test = test_data['Survived']

            if verb :

                print(f'Testing data size : {X_test.shape}')



            model,y_pred,train_score,test_score,rmse_score = train_model(clf,X_train,y_train,X_test,y_test,fold,cm,verb=verb)

            train_scores.append(train_score)

            test_scores.append(test_score)

            rmse_scores.append(rmse_score)

            if verb :

                print('**************************************')

                print(f'Average Train score: {np.mean(train_scores)}')

                print(f'Average Test score: {np.mean(test_scores)}')

                print(f'Average RMSE score: {np.mean(rmse_scores)}')

        compare_metrics.append([cm.export_metrics()])

        compare_scores.append([clsName,np.mean(train_scores),np.mean(test_scores),np.mean(rmse_scores)]) #,np.mean(train_scores),np.mean(test_scores)

    return compare_scores, compare_metrics
c_s, c_m = pipeline()
pd.DataFrame(c_s,columns=['Predictor','Training score','Testing score','RMSE score'])
#5folds

total_m = []

for i in c_m:

    print(i[0][0])#clf_Name

    print('#######################################################################################')

    print("Class = 1 | Survivor's metrics")

    print('------------------------------')

    print(i[0][1])

    print('#######################################################################################')

    print("Class = 0 | Dead metrics")

    print('------------------------------')

    print(i[0][2])

    print('#######################################################################################')

    print("Total metrics")

    print('------------------------------')

    print(i[0][3])

    print('\n')

    print('\n')

    total_m.append([i[0][0],i[0][3]])

    
def summary(result,cls):

    decimal_places = 4

    print(result[cls][0])

    print(f"Recall [class = 1| Survived] : {round(result[cls][1]['Recall_1'].agg('mean'),decimal_places)}")

    print(f"Recall [class = 0| Dead] : {round(result[cls][1]['Recall_0'].agg('mean'),decimal_places)}")

    print(f"Precision [class = 1| Survived] : {round(result[cls][1]['Precision_1'].agg('mean'),decimal_places)}")

    print(f"Precision [class = 0| Dead] : {round(result[cls][1]['Precision_0'].agg('mean'),decimal_places)}")

    print(f"F-Measure [class = 1| Survived] : {round(result[cls][1]['F1-Measure_1'].agg('mean'),decimal_places)}")

    print(f"F-Measure [class = 0| Dead] : {round(result[cls][1]['F1-Measure_0'].agg('mean'),decimal_places)}")

    print(f"Average F-Measure ของทั้งชุดข้อมูล : {round(result[cls][1]['F1-Measure_avg'].agg('mean'),decimal_places)}")

    print('\n')

    print(result[cls][1].head(10))
#Decision Tree

summary(total_m,0)
#Naive bayes

summary(total_m,1)
#Custom implemented MLP

summary(total_m,2)
#MLPClassifier from sklearn

summary(total_m,3)