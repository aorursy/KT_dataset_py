!pip install sidetable
##Standard Libraries for data handling

import numpy as np

import pandas as pd



##Standard Visualisation Libraries

import matplotlib.pyplot as plt

import seaborn as sns

from mlxtend.plotting import plot_confusion_matrix as pcm

import plotly.graph_objects as go



## Finding the area under the curve

from scipy import integrate



##For data summarize

import sidetable



##Libraries for Preprocesing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



##Libraries for Model building

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



##Library for Model performance

from sklearn.metrics import confusion_matrix, accuracy_score,plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve

from sklearn.metrics import average_precision_score,recall_score



## Libraries for Deep Models

from keras.models import Sequential

from keras.layers import Dense



##Setting up environment

import os

import pylab
pylab.rc('figure', figsize=(10,7))



SMALL_SIZE = 8

MEDIUM_SIZE = 10

BIGGER_SIZE = 12



plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes

plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title

plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels

plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv('/kaggle/input/Telco_Churn.csv')
dataset.head()
sns.heatmap(dataset.isna(),cbar=False,yticklabels=False);
dataset.info()
dataset["TotalCharges"]=[i.strip() for i in dataset["TotalCharges"]]

dataset["TotalCharges"]=pd.to_numeric(dataset["TotalCharges"], downcast="float")
dataset.isna().sum()
Null_values=dataset["TotalCharges"].isna()

dataset[Null_values]
dataset=dataset.fillna(0)
dataset['Churn'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8));

plt.title("% of Attrition")
fig, ax =plt.subplots(nrows=2,ncols=2,figsize=(14,10))

sns.countplot('MultipleLines',hue='Churn',data=dataset,color='darkblue',ax=ax[0,0])

sns.countplot('PhoneService',hue='Churn',data=dataset,color='firebrick',ax=ax[0,1])

sns.countplot('InternetService',hue='Churn',data=dataset,color='goldenrod',ax=ax[1,0])

sns.countplot('OnlineSecurity',hue='Churn',data=dataset,color='darkslategray',ax=ax[1,1])

plt.show()
fig, ax =plt.subplots(nrows=2,ncols=2,figsize=(14,10))

sns.countplot('OnlineBackup',hue='Churn',data=dataset,color='aqua',ax=ax[0,0],)

sns.countplot('DeviceProtection',hue='Churn',data=dataset,color='crimson',ax=ax[0,1])

sns.countplot('TechSupport',hue='Churn',data=dataset,color='indigo',ax=ax[1,0])

sns.countplot('StreamingTV',hue='Churn',data=dataset,color='saddlebrown',ax=ax[1,1])

plt.show()
fig, ax =plt.subplots(nrows=2,ncols=2,figsize=(14,10))

sns.countplot('StreamingMovies',hue='Churn',data=dataset,color='salmon',ax=ax[0,0])

sns.countplot('PaperlessBilling',hue='Churn',data=dataset,color='forestgreen',ax=ax[0,1])

sns.countplot('Contract',hue='Churn',data=dataset,color='forestgreen',ax=ax[1,0])

sns.countplot('PaymentMethod',hue='Churn',data=dataset,color='forestgreen',ax=ax[1,1])

plt.xticks(rotation=15)

plt.show()
dataset.stb.freq(['TechSupport','Churn'],style="{:.2%}")
dataset.stb.freq(['InternetService','Churn'],style="{:.2%}")
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,7))

dataset[dataset['Churn']=='Yes']['TechSupport'].value_counts().plot.pie(explode=[0.05,0.0,0.0],autopct='%1.1f%%',shadow=True,ax=ax[0])

dataset[dataset['Churn']=='Yes']['InternetService'].value_counts().plot.pie(explode=[0.05,0.0,0.0],autopct='%1.1f%%',shadow=True,ax=ax[1]);
def col_trans(col):

    X=[1 if i=='Yes' else 0 for i in col]

    return X



make_dollar = lambda x: "${:,.2f}".format(x)

def highlight_max(s):

    is_max = s == s.max()

    return ['background-color: orangered' if v else '' for v in is_max]







def highlight_min(s):

    is_min = s == s.min()

    return ['background-color: palegreen' if v else '' for v in is_min]

dataset['Churn']=col_trans(dataset['Churn'])



demographics=dataset.groupby(['gender', 'SeniorCitizen']).agg({'customerID': ['count'],

                                                     'Churn':['sum'],

                                                       'MonthlyCharges': ['mean']}).reset_index()
demographics['Index']=(demographics['Churn']['sum']/demographics['customerID']['count'])

demographics['% Mix']=(demographics['customerID']['count']/np.sum(demographics['customerID']['count']))

demographics["MonthlyCharges"]=demographics["MonthlyCharges"]["mean"].apply(make_dollar)
demographics.columns=["Gender","SeniorCitizen","CustomerCount[A]","ChurnCount[B]","Avg. Monthly Rev","% Churn([A]/[B])","% Mix([A]/Sum[A])"]

demographics["% Mix([A]/Sum[A])"] = pd.Series(["{0:.2f}%".format(val * 100) for val in demographics["% Mix([A]/Sum[A])"]], index = demographics.index)

demographics['% Churn([A]/[B])'] = pd.Series(["{0:.2f}%".format(val * 100) for val in demographics['% Churn([A]/[B])']], index = demographics.index)

demographics.style.apply(highlight_max,subset=["Avg. Monthly Rev",'% Churn([A]/[B])'])
dataset['tenure_bins'] = pd.cut(x=dataset['tenure'], bins=[0,12, 24, 36,48,60,72])

plt.figure(figsize=(12,9))

sns.barplot(dataset['tenure_bins'],dataset['MonthlyCharges'],hue=dataset['Churn'],palette='Wistia')

plt.title('Avg. Monthly charge by Tenure (in months)',fontsize=14)

plt.xlabel('Tenure (in months)',fontsize=14)

plt.ylabel('Monthly Charges',fontsize=14)

plt.show()
plt.figure(figsize=(12,9))

sns.barplot(dataset['tenure_bins'],dataset['TotalCharges'],hue=dataset['Churn'],palette='Wistia')

plt.title('Avg. Monthly charge by Tenure (in months)',fontsize=14)

plt.xlabel('Tenure (in months)',fontsize=14)

plt.ylabel('Total Charges',fontsize=14)

plt.show()
churn_by_tenure=dataset.groupby(['tenure_bins']).agg({'customerID': ['count'],

                                                     'Churn':['sum'],

                                                       }).reset_index()

churn_by_tenure.columns=['Tenure (in months)','# Customers','# Churns']

churn_by_tenure['Index']=churn_by_tenure['# Churns']/churn_by_tenure['# Customers']

churn_by_tenure['Index']=pd.Series(["{0:.2f}%".format(val * 100) for val in churn_by_tenure["Index"]], index = churn_by_tenure.index)

churn_by_tenure
sns.scatterplot(x="MonthlyCharges",y="TotalCharges",hue="tenure_bins",data=dataset,palette='tab20')

plt.title('Distribution of Total Charges with Monthly Charges by Tenure')

plt.show()
#dataset.drop('tenure_bins',axis=1,inplace=True)

col_count=pd.DataFrame({"col_name":dataset.nunique().index,

              "Unique_Val":dataset.nunique()}).reset_index(drop=True)

def col_cat(col):  ##To differentiate the column types

    x=[]

    for i in col:

        if i ==2:

            x.append('Binary')

        elif (i>2) & (i<7):

            x.append('Categorical')

        else:

            x.append('Continuous')

    return x

        



col_count['Type']=col_cat(col_count["Unique_Val"])



col_count
continuous=list(col_count[col_count["Type"]=='Continuous']['col_name'])

binary=list(col_count[col_count["Type"]=='Binary']['col_name'])

categorical=list(col_count[col_count["Type"]=='Categorical']['col_name'])

binary.pop(binary.index('Churn'))

continuous.pop(continuous.index('customerID'))
le=LabelEncoder()

for i in binary:

    dataset[i]=le.fit_transform(dataset[i])
sns.heatmap(dataset.corr(),annot=True,cmap='viridis');
X=dataset.drop('tenure_bins',axis=1)

y=X.iloc[:,-1]

X=X.iloc[:,1:-1]
categorical_ind=[i for i,j in enumerate(X.columns) if j in categorical]



X=X.values

y=y.values
# Let's check shape of X and y



print("Dimension of X vector:",X.shape)

print("Dimension of y labels:",y.shape)
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),list(categorical_ind))],remainder='passthrough')

X=np.array(ct.fit_transform(X))  ##Moves the dummy columns to the begining
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
def capcurve(y_values, y_preds_proba): ##Cap Curve for model performance

    num_pos_obs = np.sum(y_values)

    num_count = len(y_values)

    rate_pos_obs = float(num_pos_obs) / float(num_count)

    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})

    xx = np.arange(num_count) / float(num_count - 1)

    



    y_cap = np.c_[y_values,y_preds_proba]

    y_cap_df_s = pd.DataFrame(data=y_cap)

    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index( drop=True)



    #print(y_cap_df_s.head(20))



    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)

    yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0



    percent = 0.5

    row_index = int(np.trunc(num_count * percent))



    val_y1 = yy[row_index]

    val_y2 = yy[row_index+1]

    if val_y1 == val_y2:

        val = val_y1*1.0

    else:

        val_x1 = xx[row_index]

        val_x2 = xx[row_index+1]

        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)



    sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1

    sigma_model = integrate.simps(yy,xx)

    sigma_random = integrate.simps(xx,xx)



    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)

    #ar_label = 'ar value = %s' % ar_value

    val=np.round(val,2)



    fig, ax = plt.subplots(nrows = 1, ncols = 1,figsize=(10,7))

    ax.plot(ideal['x'],ideal['y'], color='C0', label='Perfect Model',lw=2,marker='o')

    ax.plot(xx,yy, color='red', label='Our Model')

    #ax.scatter(xx,yy, color='red')

    ax.plot(xx,xx, color='blue', label='Random Model',lw=2)

    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=2)

    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=2, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')



    plt.xlim(0, 1.02)

    plt.ylim(0, 1.25)

    plt.title(f"CAP Curve - a_r value ={ar_value:.2f}")

    plt.xlabel('% of the data')

    plt.ylabel('% of Positive obs (Churn=1)')

    plt.legend()

    plt.show()

    return val
def classifier_Logistic(X_train,y_train,X_test,y_test):  

    classifier = LogisticRegression(random_state = 0)

    classifier.fit(X_train, y_train)

    y_pred_prob=classifier.predict_proba(X_test)

    y_pred = classifier.predict(X_test)

    val_accuracy=accuracy_score(y_test,y_pred)

    cm=confusion_matrix(y_test,y_pred)

    

    plt.style.use('seaborn')

    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    

    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])

    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])

    ax[0].set_title("ROC Curve")

    ax[1].set_title("Precision vs Recall Curve")

    val=capcurve(y_test,y_pred_prob[:,1])

    precision=average_precision_score(y_test,y_pred)

    recall=recall_score(y_test,y_pred)

    #plt.style.use('default')

    #plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues,normalize='true')

    pcm(cm,colorbar=True,show_normed=True)

    plt.title('Confusion Matrix')

    plt.show()

    score={"accuracy":val_accuracy,

           "con_mat":cm,

           "y_pred":y_pred,

           "y_pred_prob":y_pred_prob,

           "classifier":classifier,

           "CAC":val,

          "precision":precision,

          "recall":recall}

    print(f"Validation Accuracy of the Logistic Regression model is {val_accuracy:.2f}%")

    return score
def classifier_SVC(X_train,y_train,X_test,y_test,kernel='linear'):  

    classifier = SVC(kernel=kernel,random_state = 0,probability=True)

    classifier.fit(X_train, y_train)

    y_pred_prob=classifier.predict_proba(X_test)

    y_pred = classifier.predict(X_test)

    val_accuracy=accuracy_score(y_test,y_pred)

    cm=confusion_matrix(y_test,y_pred)

    

    plt.style.use('seaborn')

    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    

    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])

    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])

    ax[0].set_title("ROC Curve")

    ax[1].set_title("Precision vs Recall Curve")

    val=capcurve(y_test,y_pred_prob[:,1])

    precision=average_precision_score(y_test,y_pred)

    recall=recall_score(y_test,y_pred)

    #plt.style.use('default')

    #plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues,normalize='true')

    pcm(cm,colorbar=True,show_normed=True)

    plt.title('Confusion Matrix')

    plt.show()

    score={"accuracy":val_accuracy,

           "con_mat":cm,

           "y_pred":y_pred,

           "y_pred_prob":y_pred_prob,

           "classifier":classifier,

           "CAC":val,

          "precision":precision,

          "recall":recall}



    print(f"Validation Accuracy of the Support Vector model is {val_accuracy:.2f}%")

    return score
def classifier_KNN(X_train,y_train,X_test,y_test,n_neighbors=5):  

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors,p=2,metric='minkowski')

    classifier.fit(X_train, y_train)

    y_pred_prob=classifier.predict_proba(X_test)

    y_pred = classifier.predict(X_test)

    val_accuracy=accuracy_score(y_test,y_pred)

    cm=confusion_matrix(y_test,y_pred)

    

    plt.style.use('seaborn')

    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    

    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])

    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])

    ax[0].set_title("ROC Curve")

    ax[1].set_title("Precision vs Recall Curve")

    val=capcurve(y_test,y_pred_prob[:,1])

    precision=average_precision_score(y_test,y_pred)

    recall=recall_score(y_test,y_pred)

    #plt.style.use('default')

    #plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues,normalize='true')

    pcm(cm,colorbar=True,show_normed=True)

    plt.title('Confusion Matrix')

    plt.show()

    score={"accuracy":val_accuracy,

           "con_mat":cm,

           "y_pred":y_pred,

           "y_pred_prob":y_pred_prob,

           "classifier":classifier,

           "CAC":val,

          "precision":precision,

          "recall":recall}

    

    print(f"Validation Accuracy of the KNN model is {val_accuracy:.2f}%")

    return score
def classifier_Tree(X_train,y_train,X_test,y_test,criterion='entropy'):  

    classifier = DecisionTreeClassifier(criterion=criterion)

    classifier.fit(X_train, y_train)

    y_pred_prob=classifier.predict_proba(X_test)

    y_pred = classifier.predict(X_test)

    val_accuracy=accuracy_score(y_test,y_pred)

    cm=confusion_matrix(y_test,y_pred)

    

    plt.style.use('seaborn')

    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    

    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])

    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])

    ax[0].set_title("ROC Curve")

    ax[1].set_title("Precision vs Recall Curve")

    val=capcurve(y_test,y_pred_prob[:,1])

    precision=average_precision_score(y_test,y_pred)

    recall=recall_score(y_test,y_pred)

    #plt.style.use('default')

    #plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues,normalize='true')

    pcm(cm,colorbar=True,show_normed=True)

    plt.title('Confusion Matrix')

    plt.show()

    score={"accuracy":val_accuracy,

           "con_mat":cm,

           "y_pred":y_pred,

           "y_pred_prob":y_pred_prob,

           "classifier":classifier,

           "CAC":val,

          "precision":precision,

          "recall":recall}



    print(f"Validation Accuracy of Decision Tree the model is {val_accuracy:.2f}%")

    return score
def classifier_RF(X_train,y_train,X_test,y_test,n_estimators=10,criterion='entropy'):  

    classifier = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion)

    classifier.fit(X_train, y_train)

    y_pred_prob=classifier.predict_proba(X_test)

    y_pred = classifier.predict(X_test)

    val_accuracy=accuracy_score(y_test,y_pred)

    cm=confusion_matrix(y_test,y_pred)

    

    plt.style.use('seaborn')

    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    

    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])

    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])

    ax[0].set_title("ROC Curve")

    ax[1].set_title("Precision vs Recall Curve")

    val=capcurve(y_test,y_pred_prob[:,1])

    precision=average_precision_score(y_test,y_pred)

    recall=recall_score(y_test,y_pred)

    #plt.style.use('default')

    #plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues,normalize='true')

    pcm(cm,colorbar=True,show_normed=True)

    plt.title('Confusion Matrix')

    plt.show()

    score={"accuracy":val_accuracy,

           "con_mat":cm,

           "y_pred":y_pred,

           "y_pred_prob":y_pred_prob,

           "classifier":classifier,

           "CAC":val,

          "precision":precision,

          "recall":recall}

    



    print(f"Validation Accuracy of the Random Forest model is {val_accuracy:.2f}%")

    return score
def classifier_NaiveB(X_train,y_train,X_test,y_test):  

    classifier = GaussianNB()

    classifier.fit(X_train, y_train)

    y_pred_prob=classifier.predict_proba(X_test)

    y_pred = classifier.predict(X_test)

    val_accuracy=accuracy_score(y_test,y_pred)

    cm=confusion_matrix(y_test,y_pred)

    

    plt.style.use('seaborn')

    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    

    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])

    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])

    ax[0].set_title("ROC Curve")

    ax[1].set_title("Precision vs Recall Curve")

    val=capcurve(y_test,y_pred_prob[:,1])

    precision=average_precision_score(y_test,y_pred)

    recall=recall_score(y_test,y_pred)

    #plt.style.use('default')

    #plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues,normalize='true')

    pcm(cm,colorbar=True,show_normed=True)

    plt.title('Confusion Matrix')

    plt.show()

    score={"accuracy":val_accuracy,

           "con_mat":cm,

           "y_pred":y_pred,

           "y_pred_prob":y_pred_prob,

           "classifier":classifier,

           "CAC":val,

          "precision":precision,

          "recall":recall}



    print(f"Validation Accuracy of the Naive Bayers model is {val_accuracy:.2f}%")

    return score
def classifier_ANN(X_train,y_train,X_test,y_test,epochs=100,batch_size=32,optimizer='adam',loss='binary_crossentropy'): 

    classifier=Sequential()



    #Creating the input layer and first hidden layer

    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=40))  ## Check all the other params



    #adding 2nd hidden Layer

    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))





    #adding 3rd hidden Layer

    #classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))



    #Adding the output layer

    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))



    #Compiling the ANN

    classifier.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])



    #Fitting ANN to training set

    classifier.fit(X_train,y_train,batch_size=batch_size,epochs=epochs)

    y_pred_prob=classifier.predict(X_test)

    y_pred=(y_pred_prob>0.5).astype('int')

    val_accuracy=accuracy_score(y_test,y_pred)

    cm=confusion_matrix(y_test,y_pred)

    

    

    

    plt.style.use('seaborn')

    #fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    

    #plot_roc_curve(classifier,X_test,y_test,ax=ax[0])

    #plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])

    #ax[0].set_title("ROC Curve")

    #ax[1].set_title("Precision vs Recall Curve")

    val=capcurve(y_test,y_pred_prob)

    precision=average_precision_score(y_test,y_pred)

    recall=recall_score(y_test,y_pred)

    #plt.style.use('default')

    #plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues,normalize='true')

    pcm(cm,colorbar=True,show_normed=True)

    plt.title('Confusion Matrix')

    plt.show()

    score={"accuracy":val_accuracy,

           "con_mat":cm,

           "y_pred":y_pred,

           "y_pred_prob":y_pred_prob,

           "classifier":classifier,

           "CAC":val,

          "precision":precision,

          "recall":recall}



    

    

    #score=(val_accuracy,cm,y_pred,y_pred_prob)

    print(f"Validation Accuracy of the ANN model is {val_accuracy:.2f}%")

    

    return score
Score_Log=classifier_Logistic(X_train,y_train,X_test,y_test)
Score_SVCL=classifier_SVC(X_train,y_train,X_test,y_test)
Score_SVCR=classifier_SVC(X_train,y_train,X_test,y_test,kernel='rbf')
Score_KNN=classifier_KNN(X_train,y_train,X_test,y_test)
Score_Tree=classifier_Tree(X_train,y_train,X_test,y_test)
Score_RF=classifier_RF(X_train,y_train,X_test,y_test)
Score_NaiveB=classifier_NaiveB(X_train,y_train,X_test,y_test)
Score_ANN=classifier_ANN(X_train,y_train,X_test,y_test,epochs=10)
metrics={}

for metric in Score_Log.keys():

    metrics[metric]=[Score_Log[metric],Score_NaiveB[metric],Score_KNN[metric],Score_SVCL[metric],Score_SVCR[metric],Score_Tree[metric],Score_RF[metric],Score_KNN[metric]]
modelnames=['Log Reg',

            'Naive Bayes',

            'KNN','SVM','Kernel','Tree','Random Forest','Artificial NN']



metrics['ModelNames']=modelnames

model_summary=pd.DataFrame(metrics)
model_summary=model_summary[['ModelNames','accuracy','precision','recall','CAC','con_mat']]

model_summary.columns=['Model Names','Accuracy','Precision','Recall','CAC Score','Confusion Mat']

model_summary=model_summary.round(3)
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(14,10))



sns.barplot(model_summary['Model Names'],model_summary['Accuracy'],ax=ax[0,0])



sns.barplot(model_summary['Model Names'],model_summary['Precision'],ax=ax[0,1])



sns.barplot(model_summary['Model Names'],model_summary['Recall'],ax=ax[1,0])



sns.barplot(model_summary['Model Names'],model_summary['CAC Score'],ax=ax[1,1])



plt.tight_layout()
fig = go.Figure(data=[go.Table(

    header=dict(values=list(model_summary.columns),

                fill_color='paleturquoise',

                align='center'),

    cells=dict(values=[model_summary['Model Names'], model_summary['Accuracy'], model_summary['Precision'], model_summary['Recall'],model_summary['CAC Score'],model_summary['Confusion Mat']],

               fill_color='lavender',

               align='left'))

])



fig.show()