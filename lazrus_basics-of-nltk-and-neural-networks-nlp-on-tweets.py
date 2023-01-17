#Libraries
import numpy as np
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms

#Data
train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
ss = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
#Missing Value
print('Number of Missing Values in Target feature: {}'.format(train.target.isnull().sum()))
#Distribution
canv, axs = plt.subplots(1,2,figsize=(22,8))
color = ['darkgreen','darkslategrey']

plt.sca(axs[0])
plt.pie(train.groupby('target').count()['id'],explode=(0.1,0),startangle=120,colors=color,
    textprops={'fontsize':15},labels=['Not Disaster (57%)', 'Disaster (43%)'])

plt.sca(axs[1])
bars = plt.bar([0,0.5],train.groupby('target').count()['id'],width=0.3,color=color)
plt.xticks([0,0.5],['Not Disaster','Disaster'])
plt.tick_params(axis='both',labelsize=15,size=0,labelleft=False)

for sp in plt.gca().spines.values():
    sp.set_visible(False)
    
for bar,val in zip(bars,train.groupby('target').count()['id']):
    plt.text(bar.get_x()+0.113,bar.get_height()-250,val,color='w',fontdict={'fontsize':18,'fontweight':'bold'})

canv.suptitle('Target Value Distribution in Training Data',fontsize=18);
#Train Data
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na==0].index).sort_values(ascending=False)

pd.DataFrame({'Train Missing Ratio' :train_na}).head(3)
#Test Data
test_na = (test.isnull().sum() / len(test)) * 100
test_na = test_na.drop(test_na[test_na==0].index).sort_values(ascending=False)

pd.DataFrame({'Test Missing Ratio' :test_na}).head(3)
#Visualizing Mssing Values
title = 'Train'
data = [train_na,test_na]
canv, axs = plt.subplots(1,2)
canv.set_size_inches(18,5)
for ax, dat in zip(axs,data):
    plt.sca(ax)
    sns.barplot(x=dat.index, y=dat,dodge=False)  
    plt.xlabel('Features', fontsize=15,labelpad=10)
    plt.ylabel('Percent of missing values', fontsize=15,labelpad=13)
    plt.title('Percent missing data by feature in {} Data'.format(title), fontsize=15,pad=20)
    plt.tick_params(axis='both',labelsize=12)
    
    sp = plt.gca().spines
    sp['top'].set_visible(False)
    sp['right'].set_visible(False)
    
    title = 'Test'
#Filling Missing Data
for df in [train,test]:
    for col in ['keyword','location']:
        df[col].fillna('None',inplace=True)
#Function
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
#Target Value
y = train.target
X = train.text

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5,ngram_range=(1,5)).fit(X_train)

#train
X_train_vect = tfidf.transform(X_train)

#test
X_test_vect = tfidf.transform(X_test)
X_train_vect = add_feature(X_train_vect,X_train.apply(lambda x : len(str(x).split())))
X_test_vect = add_feature(X_test_vect,X_test.apply(lambda x : len(str(x).split())))
X_train_vect = add_feature(X_train_vect,X_train.apply(lambda x : len(set(str(x).split()))))
X_test_vect = add_feature(X_test_vect,X_test.apply(lambda x : len(set(str(x).split()))))
X_train_vect = add_feature(X_train_vect,X_train.apply(lambda x : len(str(x))))
X_test_vect = add_feature(X_test_vect,X_test.apply(lambda x : len(str(x))))
X_train_vect = add_feature(X_train_vect,X_train.apply(lambda x : x.count('#')))
X_test_vect = add_feature(X_test_vect,X_test.apply(lambda x : x.count('#')))
X_train_vect = add_feature(X_train_vect,X_train.apply(lambda x : x.count('@')))
X_test_vect = add_feature(X_test_vect,X_test.apply(lambda x : x.count('@')))
X_train_vect = add_feature(X_train_vect,X_train.apply(lambda x : x.count('http')))
X_test_vect = add_feature(X_test_vect,X_test.apply(lambda x : x.count('http')))
X_train_vect = add_feature(X_train_vect,X_train.str.count(r'[\\/!?,\.:=<>^-]'))
X_test_vect = add_feature(X_test_vect,X_test.str.count(r'[\\/!?,\.:=<>^-]'))
X_train_vect = add_feature(X_train_vect,X_train.str.count(r'\d'))
X_test_vect = add_feature(X_test_vect,X_test.str.count(r'\d'))
# Number of Features
print('The Number of Features in the Processed Data: {}'.format(X_train_vect.shape[-1]))
# Lets look at some of the Vacabulary from the Tfidf
print(tfidf.get_feature_names()[350:420])
#Metric and Model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer,accuracy_score,roc_auc_score

#Creating Log Loss Scorer
LogLoss = make_scorer(f1_score, greater_is_better=True, needs_proba=True)

#Two Classifier
mlp1 = MLPClassifier(max_iter=200,verbose=False,solver='sgd',activation='relu',learning_rate='adaptive')
mlp2 = MLPClassifier(max_iter=200,verbose=False,solver='adam',activation='relu',learning_rate='adaptive')

#Parameters for tunning
parameter_space = {
      'hidden_layer_sizes': [(50,50,50), (50,100,50),(100,100,100),(100,100),(500,500)],
      'alpha': [0.0001, 0.05],
  }
#Tunning the First Classifier
clf1 = GridSearchCV(mlp1, parameter_space,verbose=2,cv=3,scoring='f1')
clf1.fit(X_train_vect,y_train)
#Tunning the Second Classifier
clf2 = GridSearchCV(mlp2, parameter_space,verbose=2,cv=3,scoring='f1')
clf2.fit(X_train_vect,y_train)
#Results from the first GridSearch
pd.DataFrame(data=clf1.cv_results_,columns=['param_hidden_layer_sizes','param_alpha','mean_test_score','std_test_score']).sort_values('mean_test_score',
                                                                                                     ascending=False).reset_index(drop=True).iloc[:10]
#Results from the Second GridSearch
pd.DataFrame(data=clf2.cv_results_,columns=['param_hidden_layer_sizes','param_alpha','mean_test_score','std_test_score']).sort_values('mean_test_score',
                                                                                                     ascending=False).reset_index(drop=True).iloc[:10]
# So we can see that the best esitmator has the following parameters
prm = {'alpha':0.05,'hidden_layer_sizes':(100,100,100),'max_iter':200,'solver':'adam','activation':'relu','learning_rate':'adaptive'}
print('The Best Parameters are:\n')
for p,c in zip(prm.items(),range(1,7)):
    print('{}. {} = {}'.format(c,p[0],p[1]))

#Final F1 Score of test data
print('\n\nAnd the Accuracy of the Final Model on Test Data: {:.3f}'.format(accuracy_score(y_test,clf2.predict(X_test_vect))))
print('\n\nAnd the F1 score of the Final Model on Test Data: {:.3f}'.format(f1_score(y_test,clf2.predict(X_test_vect))))