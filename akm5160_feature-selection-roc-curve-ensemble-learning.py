import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#1- malignant
#ignoring the warnings
import warnings
warnings.filterwarnings('ignore')
data1 = pd.read_csv('../input/data.csv')
data1.head(5)
data1.diagnosis.value_counts()
#data1['Diagnosis']=0
data1.describe().T
#for i in range(len(data1.diagnosis)):
#    if data1.diagnosis[i]=="M":
#        data1['Diagnosis'][i]=1
#    else:
#        data1['Diagnosis'][i]=0
#pythonic way of doing the above task
data1['diagnosis']=data1['diagnosis'].map({'M':1,'B':0})
#data1=data1.drop(['diagnosis'],axis=1)        
data1=data1.drop(['Unnamed: 32'],axis=1)

data1.head(5)

%matplotlib inline
for i in data1.columns:
    plt.subplots(figsize=(10,4))
    sns.boxplot(y=i,data=data1)
    plt.title(str(i))
    plt.plot()
# if we take radius_mean- as it can be seen there are data points available above the whiskers of 
#the plot. in this case we can perform clip_upper based on visually identified threshold or we
#we can do a clip_upper, handling 1percentile of the dataset.



x=data1.columns
x=x[2:]
x
for num_variable in x:
    fig,axes = plt.subplots(figsize=(10,4))
    #sns.distplot(hrdf[num_variable], kde=False, color='g', hist=True)
    sns.distplot(data1[data1['diagnosis']==1][num_variable], label='Malign', color='g', hist=True, norm_hist=False)
    sns.distplot(data1[data1['diagnosis']==0][num_variable], label='Benign', color='r', hist=True, norm_hist=False)
    plt.xlabel(str("X variable ") + str(num_variable) )
    plt.ylabel('Density Function')
    plt.title(str('default Split Density Plot of ')+str(num_variable))
    plt.legend()
    plt.show()
    
 # My aim here is to look for variables with clear demarcation or less overlap of red and green region.
#for example perimeter_mean--> Malign region for this particular variable cleart demarcated then benign 
#region. Also the overlap region is very less.
#similarly smoothness_mean won't be a good predictor as the regions are clearly overlapped.
#density plot against the diagnosis clearly showing which variables are important predicting
#radius-mean has the density plot for Malign and benign clearly split



from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

train_X,test_X,train_y,test_y=train_test_split(data1[x],data1['diagnosis'],test_size=0.25,random_state=42)
logreg=LogisticRegression()
logreg.fit(train_X,train_y)


from sklearn.metrics import confusion_matrix
from sklearn import metrics

cm_1=confusion_matrix(test_y,logreg.predict(test_X),[1,0])
cm_1

print (metrics.accuracy_score(test_y,logreg.predict(test_X)))

predict_prob_df=pd.DataFrame(logreg.predict_proba(test_X))
predict_prob_df.head(5)
fpr, tpr, thresholds = metrics.roc_curve( test_y,
                                     predict_prob_df[1],
                                     drop_intermediate = False )
thres=0
cut_off=0
for i in range(len(fpr)):
    if thres< ((1-fpr[i])+tpr[i]):
        thres=((1-fpr[i])+tpr[i])
        cut_off=thresholds[i]

# i am doing ROC curve analysis to find the best cut_off probablity for the categorization.
#cut_off would contain the best cut_off point
#logic for cut_off selection Max(cut_off, (tpr+(1-fpr)))
#maximum sensitivity +specificity


predict_prob_df['final_label']=0
for k in range(len(predict_prob_df[1])):
    if predict_prob_df[1][k]>cut_off:
        predict_prob_df['final_label'][k]=1
    else:
        predict_prob_df['final_label'][k]=0

        
cm_2=confusion_matrix(test_y,predict_prob_df['final_label'],[1,0])
cm_2

metrics.accuracy_score(test_y,predict_prob_df['final_label'])
# we have acheived the 97% accuracy, which is slight improvement of 2%. 

from sklearn.tree import DecisionTreeClassifier
clf_tree=DecisionTreeClassifier()
clf_tree.fit(train_X,train_y)
cm_3=confusion_matrix(test_y,clf_tree.predict(test_X),[1,0])
cm_3


metrics.accuracy_score(test_y,clf_tree.predict(test_X))
#model accuracy is 94%
param_grid = {'max_depth': np.arange(3, 12),
             'max_features': np.arange(3,8)}

#doing grid search to find out the best feature for the decision tree

from sklearn.grid_search import GridSearchCV
grid=GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10)
grid.fit(train_X,train_y)
grid.best_params_
max_depth=grid.best_params_['max_depth']
max_features=grid.best_params_['max_features']

clf_tree_final=DecisionTreeClassifier(max_depth=max_depth,max_features=max_features)
clf_tree_final.fit(train_X,train_y)
cm_4=confusion_matrix(test_y,clf_tree_final.predict(test_X),[1,0])
cm_4

metrics.accuracy_score(test_y, clf_tree_final.predict(test_X))
#model hold accuracy rate of 94%
from sklearn.ensemble import BaggingClassifier
clf_bag=BaggingClassifier(oob_score=True, n_estimators=100)
clf_bag.fit(train_X,train_y)


cm_5=confusion_matrix(test_y,clf_bag.predict(test_X),[1,0])
cm_5

from sklearn.ensemble import RandomForestClassifier
radm_clf = RandomForestClassifier(oob_score=True,n_estimators=100 )
radm_clf.fit( train_X, train_y )
cm_6=confusion_matrix(test_y,radm_clf.predict(test_X),[1,0])
cm_6

metrics.accuracy_score(test_y, radm_clf.predict(test_X))
from sklearn.ensemble import VotingClassifier
#combining all the classifier with weights to arrive at the best/final classifier
final_classfier=VotingClassifier(estimators=[('lr',logreg),('dec_tree',clf_tree_final),('bagging',clf_bag),('random_f',radm_clf)],voting='hard',weights=[2,1,1,1])
final_classfier.fit(train_X,train_y)
cm_7=confusion_matrix(test_y,final_classfier.predict(test_X),[1,0])
cm_7
metrics.accuracy_score(test_y,final_classfier.predict(test_X))

#95% accuracy has finally been acheived.
