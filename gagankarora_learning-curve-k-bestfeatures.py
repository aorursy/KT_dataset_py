import pandas as pd

data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
# first few rows
data.head()
# columns
data.info()
catDf = pd.DataFrame(columns=['Categorical Features','Distinct Values','Distinct Count'])
data['Attrition'].nunique()
i=0
for col in data.select_dtypes(['object']).columns.tolist():
    catDf.loc[i] = [col,data[col].unique(),data[col].nunique()]
    i=i+1
catDf
%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins=50,figsize=(20,15))
plt.show()
# stata
data.describe()

plt.figure(figsize=(80,80))
i=1

for col in data.select_dtypes('int64').columns.tolist():
    plt.subplot(7,4,i)
    x1 = list(data[data['Attrition'] == 'Yes'][col])
    x2 = list(data[data['Attrition'] == 'No'][col])
    colors = ['#E69F00', '#56B4E9']
    names = ['Attrition-Yes(1)','Attrition-No(0)']
    plt.hist([x1,x2],bins=50,label=names,color=colors)
    plt.legend(fontsize=28)
    plt.xlabel(col,fontsize=28)
    plt.ylabel('# of attrition',fontsize=28)
    plt.tick_params(labelsize=26)
    i = i+1
    
# mapping the target variable to numeric 
import numpy as np

data['Attrition'] = np.where(data['Attrition'] == 'Yes',1,0)
data['Attrition'].head(3)
from sklearn.model_selection import train_test_split

train,test = train_test_split(data,test_size=0.2,random_state=7)
X_train = train.drop(['Attrition'],axis=1)
y_train = train['Attrition']
# Ref: https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
X_train.head()
print("Top 10 Correlations pairs")
print(get_top_abs_correlations(X_train.select_dtypes(exclude=['object']), 10))
# We will use the below statement, lets wrap it under feature engineering class
#data = data.drop(['DailyRate','EmployeeCount','StandardHours'],axis=1)

#include Object
cat_attributes = X_train.select_dtypes(include=['object']).columns.tolist()
#exclude Object
num_attributes = X_train.select_dtypes(exclude=['object']).columns.tolist()

#We also observed we can drop columns such as Daily rate, Employee Count and standard hours
num_attributes.remove('DailyRate')
num_attributes.remove('EmployeeCount')
num_attributes.remove('StandardHours')
num_attributes.remove('MonthlyIncome')
#num_attributes.append('monthlyIncomePerUnitAge')

print('cat_attributes: ',cat_attributes)
print('num_attributes: ',num_attributes)
len(num_attributes)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

mergeCols_pipeline = ColumnTransformer([
         #('imputer' ,Imputer(strategy="median")), we don't need in this case there is no NA
        #('attrAdder',CombinedAttributesAdder(),num_attributes),
       ('scaler',MinMaxScaler(),num_attributes),
       ("cat", OneHotEncoder(), cat_attributes)
    ])

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectKBest, chi2

classifiers = {
    'Nearest Neighbors' : KNeighborsClassifier(3),
    'Linear SVM'        :SVC(kernel="linear", C=0.025),
    'RBF SVM'           :SVC(gamma=2, C=1),
    'Gaussian Process'  :GaussianProcessClassifier(1.0 * RBF(1.0)),
    'Decision Tree'     :DecisionTreeClassifier(max_depth=5),
    'Random Forest'     :RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'Neural Net'        :MLPClassifier(alpha=1),
    'AdaBoost'          :AdaBoostClassifier(),
    'Naive Bayes'       :GaussianNB(),
    'QDA'               :QuadraticDiscriminantAnalysis()
}


mergeColWithSelectKbest_pipeline = Pipeline([('prep',mergeCols_pipeline),
                          ('featSelection',SelectKBest(chi2,k=10))
                        ])

X_train_prep = mergeColWithSelectKbest_pipeline.named_steps['prep'].fit_transform(X_train)

X_train_featureSel = mergeColWithSelectKbest_pipeline.named_steps['featSelection'].fit_transform(X_train_prep,y_train)

#ref: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

plt.figure(figsize=(40,30))
i=1


for key in classifiers:
    
    train_sizes, train_scores, test_scores = learning_curve(classifiers[key], 
                                                        X_train_featureSel, 
                                                        y_train,
                                                        # Number of folds in cross-validation
                                                        cv=7,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.10, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.subplot(4,3,i)
    
    
    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve " + key)
    plt.xlabel("Training Set Size",fontsize=30), plt.ylabel("Accuracy Score",fontsize=30), plt.legend(loc="best",fontsize=24)
    plt.tight_layout()
    
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tick_params(axis='both', which='minor', labelsize=25)
    
    i=i+1
    
    #plt.show()
# How many features we have 
np.shape(X_train_prep)[1]
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

df_k_acc = pd.DataFrame(columns=['K','Accuracy'])
i=1
for i in range(1,np.shape(X_train_prep)[1]+1):
    
    mergeColWithSelectKbest_pipeline = Pipeline([('prep',mergeCols_pipeline),
                          ('featSelection',SelectKBest(chi2,k=i))
                        ])

    X_train_prep = mergeColWithSelectKbest_pipeline.named_steps['prep'].fit_transform(X_train)

    X_train_featureSel = mergeColWithSelectKbest_pipeline.named_steps['featSelection'].fit_transform(X_train_prep,y_train)
    
    regressor = AdaBoostClassifier()
    # fit
    regressor.fit(X_train_featureSel,y_train)
    # predict 
    train_predictions = regressor.predict(X_train_featureSel)
    
    
    df_k_acc.loc[i] = [i,accuracy_score(y_train, train_predictions)]


#Plot the accuracy v/s K - number of features 
plt.plot(df_k_acc['K'],df_k_acc['Accuracy'])
plt.xlabel('K[Number of features]')
plt.ylabel('Accuracy')
from sklearn.model_selection import GridSearchCV

#list of dictionary 
param_grid = [ {'n_estimators'  : [10,100,500,1000],
                'learning_rate' : [0.2,0.4,0.6,0.8,1,1.2,1.4],
                'algorithm'     : ['SAMME', 'SAMME.R']
               }
             ]

classifier = AdaBoostClassifier()

grid_search = GridSearchCV(classifier, param_grid, cv=5,
                           scoring='accuracy', return_train_score=True)

grid_search.fit(X_train_prep, y_train)
grid_search.best_params_
X_test = test.drop(['Attrition'],axis=1)
y_test = test['Attrition']

X_test_prep = mergeCols_pipeline.transform(X_test)

test_predictions = grid_search.best_estimator_.predict(X_test_prep)
accuracy_score(y_test, test_predictions)
