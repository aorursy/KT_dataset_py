import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
## Read file
raw_data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
print('Shape of the google playstore data',raw_data.shape)
raw_data.head()
total = raw_data.isnull().sum().sort_values(ascending=False)
percent = 100*(raw_data.isnull().sum()/raw_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent (%)'])

print(missing_data)
raw_data.dropna(inplace=True)
print('Shape of the google playstore data',raw_data.shape)
duplicate_sum = raw_data.duplicated().sum()
percentage = 100 * (duplicate_sum / len(raw_data.index))

print('The dataset contains {} duplicate row that represent {}% of overall dataset '.format(duplicate_sum, percentage))
raw_data=raw_data.drop_duplicates()
print('Shape of the google playstore data',raw_data.shape)
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt

%matplotlib inline
data_visual = raw_data.copy()
data_visual.describe(include='all')
plt.figure(figsize=(10,10))
g = sns.countplot(y="Category",data=data_visual, palette = "Set2")
plt.title('Total apps of each Category',size = 20)
data_visual['Installs'] = data_visual['Installs'].str.replace(r'\D','').astype(int)

plt.figure(figsize=(10,10))
g = sns.barplot(x="Installs", y="Category", data=data_visual, capsize=.3)
plt.title('Installations in each Category',size = 20)
data_visual[data_visual.Installs == 1000000000].head()
from sklearn import preprocessing

def encode_feature(data_feature):
    le = preprocessing.LabelEncoder()
    new_label = le.fit_transform(data_feature)
    return new_label
def preprocessor_data(data):
    data.drop(labels = ['App', 'Current Ver','Android Ver'], axis = 1, inplace = True)
    
    data['Last Updated']= pd.to_datetime(data['Last Updated'], format='%B %d, %Y')
    data['Installs'] = data['Installs'].str.replace(r'\D','').astype(int)
    data['Price']=data['Price'].str.replace('$','').astype(float)

    data['Reviews']=data['Reviews'].astype(int)
    data['Rating']=data['Rating'].astype(float)

    data['Type'] = encode_feature(data['Type'])
    data['Category'] = encode_feature(data['Category'])
    data['Content Rating'] = encode_feature(data['Content Rating'])
    
    data['Size'].replace('Varies with device', np.nan, inplace = True ) 
    data['Size']= (data['Size'].replace(r'[kM]+$', '', regex=True).astype(float) * \
             data['Size'].str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))
    data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'),inplace = True)
    data['Size'] =(data['Size']-data['Size'].min())/(data['Size'].max()-data['Size'].min())

    data2 = data['Genres'].str.get_dummies(sep=';').rename(lambda x: 'Genres_' + x, axis='columns')
    data = pd.concat([data,data2],axis=1)
    data.drop(labels = ['Genres'], axis = 1, inplace = True)
    
    data.loc[data['Installs'] < 5000, 'Installs'] = 0 
    data.loc[(data['Installs'] >= 5000) & (data['Installs'] < 1000000), 'Installs'] = 1 
    data.loc[data['Installs'] >= 1000000, 'Installs'] = 2
    
    data = data.sort_values(by=['Last Updated'])
    data.drop(labels = ['Last Updated'], axis = 1, inplace = True)

    return data
#dropping of unrelated and unnecessary items
preprocessed_data = preprocessor_data(raw_data.copy())
print('Shape after preprocessed_data data',preprocessed_data.shape)
preprocessed_data.tail()
preprocessed_data['Installs'].value_counts().sort_index()
from sklearn.model_selection import train_test_split 

train_label = preprocessed_data['Installs']
train_data = preprocessed_data.drop('Installs', axis=1)

# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.25, shuffle=False)

print('X_train Shape : {}, y_train Shape : {}'.format(X_train.shape, y_train.shape))
print('X_test  Shape : {}, y_test  Shape : {}'.format(X_test.shape,  y_test.shape ))
from sklearn.metrics import f1_score

def performance_metric(y_true, y_predict):
    score = f1_score(y_true, y_predict, average='micro')
    
    return score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 2, 0, 0, 1] # 3 labels out of 6 are correct (0%)
y_pred_bad = [1, 2, 1, 2, 0, 1] # 0 labels out of 6 are correct (50%)

right_score =  performance_metric(y_true, y_true)
score =  performance_metric(y_true, y_pred)
bad_score =  performance_metric(y_true, y_pred_bad)  


print('bad score = ({}), best score = ({}), random score = ({}) '.format(bad_score, right_score, score))
from time import time

def evaluate_classifier(cls, X_train, X_test, y_train, y_test):
    
    start_train = time() # Get start time
    cls.fit(X_train,y_train)
    end_train = time() # Get end time

    start_test = time() # Get start time
    y_train_pred = cls.predict(X_train)
    y_test_pred = cls.predict(X_test)
    end_test = time() # Get end time

    y_train_score =  performance_metric(y_train, y_train_pred)
    y_test_score =  performance_metric(y_test, y_test_pred)

    train_time = end_train - start_train
    test_time = end_test - start_test
    
    print('- Classifier[{}]\n- Training f1-score = ({:.4f}) in {:.4f}s.\n- Testing f1-score = ({:.4f}) in {:.4f}s.'.format(cls.__class__.__name__,y_train_score, train_time, y_test_score, test_time))
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(multi_class='ovr', solver='liblinear')
evaluate_classifier(clf_lr, X_train, X_test, y_train, y_test)
from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

evaluate_classifier(clf_gnb, X_train, X_test, y_train, y_test)
from sklearn.tree import DecisionTreeClassifier

clf_dtc = DecisionTreeClassifier(random_state=333)

evaluate_classifier(clf_dtc, X_train, X_test, y_train, y_test)
from sklearn.svm import SVC

clf_svc = SVC(random_state=333, gamma='auto', kernel='rbf')
evaluate_classifier(clf_svc, X_train, X_test, y_train, y_test)
from sklearn.ensemble import GradientBoostingClassifier

clf_gbc = GradientBoostingClassifier(random_state=333)

evaluate_classifier(clf_gbc, X_train, X_test, y_train, y_test)
clf_scores_plt = pd.DataFrame({'Classifier':['LogisticRegression', 'GaussianNB', 'DecisionTree', 'SVC', 'GradientBoosting', 'LogisticRegression', 'GaussianNB', 'DecisionTree', 'SVC', 'GradientBoosting'],
'Score':['Train Score' ,'Train Score' ,'Train Score' ,'Train Score' ,'Train Score' ,'Test Score' ,'Test Score' ,'Test Score' ,'Test Score' ,'Test Score'],
'Values':[0.4203, 0.7375, 1.0000, 0.9359, 0.9253, 0.7385, 0.8339, 0.9032, 0.8969, 0.9383]})

plt.figure(figsize=(8,8))
sns.barplot(x="Classifier",y="Values", hue="Score",data=clf_scores_plt)
plt.title('F1-Score For Each Classifier',size = 20)
clf_time_plt = pd.DataFrame({'Classifier':['LogisticRegression', 'GaussianNB', 'DecisionTree', 'SVC', 'GradientBoosting', 'LogisticRegression', 'GaussianNB', 'DecisionTree', 'SVC', 'GradientBoosting'],
'Time':['Train Time', 'Train Time', 'Train Time', 'Train Time', 'Train Time', 'Test Time', 'Test Time','Test Time','Test Time','Test Time'],
'Seconds':[0.0602, 0.0118, 0.0718, 6.4302, 3.3559, 0.0050, 0.0121, 0.0064, 3.9732, 0.0442]})

plt.figure(figsize=(8,8))
ax = sns.barplot(x="Classifier",y="Seconds", hue="Time",data=clf_time_plt)
plt.title('Train/Test Time For Each Classifier',size = 20)
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer

def coss_validate_clf(cls, X_train, y_train):
    fone_scorer = make_scorer(f1_score, average='micro')
    scores = cross_validate(cls, X_train, y_train, cv=10,scoring=fone_scorer, return_train_score=True)
    print('--------------Classifier [%s] ----------------'  % (cls.__class__.__name__))
    print("- Train score: %0.2f (+/- %0.2f)" % (scores['train_score'].mean(), scores['train_score'].std() * 2))
    print("- Test score: %0.2f (+/- %0.2f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))
coss_validate_clf(clf_lr, X_train, y_train)

coss_validate_clf(clf_gnb, X_train, y_train)
coss_validate_clf(clf_dtc, X_train, y_train)
coss_validate_clf(clf_svc, X_train, y_train)
coss_validate_clf(clf_gbc, X_train, y_train)
clf_cv_plt = pd.DataFrame({'Classifier': ['LogisticRegression', 'GaussianNB', 'DecisionTree', 'SVC', 'GradientBoosting', 'LogisticRegression', 'GaussianNB', 'DecisionTree', 'SVC', 'GradientBoosting'],
'Score':['Train Score' ,'Train Score' ,'Train Score' ,'Train Score' ,'Train Score' ,'Test Score' ,'Test Score' ,'Test Score' ,'Test Score' ,'Test Score'],
'Avr. Score':[0.55, 0.74, 1.00, 0.94, 0.93, 0.54, 0.74, 0.86, 0.83, 0.90]})

plt.figure(figsize=(8,8))
ax = sns.barplot(x="Classifier",y="Avr. Score", hue="Score",data=clf_cv_plt)
plt.title('Average F1-Score For Cross-Validation For Each Classifier',size = 20)
from sklearn.decomposition import PCA

def pca_init(pca_comp):
    pca = PCA(n_components=pca_comp)
    # TODO: Apply PCA by fitting the good data with the same number of dimensions as features
    pca.fit(X_train)

    x_train_pca = pca.transform(X_train)
    x_test_pca = pca.transform(X_test)
    print('-----------------[PCA Component=',pca_comp,']-----------------', )
    print('x_train_pca Shape : {}, y_train Shape : {}'.format(x_train_pca.shape, y_train.shape))
    print('x_test_pca  Shape : {}, y_test  Shape : {}'.format(x_test_pca.shape,  y_test.shape ))
    
    return pca,x_train_pca, x_test_pca
pca_componnents = [20,25,30,35,40,45,50,55]

for pca_comp in pca_componnents:
    pca, x_train_pca, x_test_pca = pca_init(pca_comp)
    evaluate_classifier(clf_dtc, x_train_pca, x_test_pca, y_train, y_test)
    evaluate_classifier(clf_gbc, x_train_pca, x_test_pca, y_train, y_test)
clf_pca_plt = pd.DataFrame({'PCA-Component': [20,20,25,25,30,30,35,35,40,40,45,45,50,50,55,55],
'Classifier':['DecisionTree' ,'GradientBoosting' ,'DecisionTree' ,'GradientBoosting' ,'DecisionTree' ,'GradientBoosting' ,'DecisionTree' ,'GradientBoosting' ,'DecisionTree' ,'GradientBoosting' ,'DecisionTree' ,'GradientBoosting' ,'DecisionTree' ,'GradientBoosting' ,'DecisionTree' ,'GradientBoosting'],
'Score':[0.9118, 0.9383, 0.9127, 0.9392, 0.9100, 0.9370, 0.9032, 0.9383, 0.9050, 0.9370, 0.9014, 0.9379, 0.8992, 0.9370, 0.8983, 0.9374]})

plt.figure(figsize=(8,8))
ax = sns.barplot(x="PCA-Component",y="Score", hue="Classifier",data=clf_pca_plt)
plt.title('Prediction F1-Score For Selected Classifiers Over Number Of Components',size = 20)
best_comp_num = 25
pca,x_train_pca, x_test_pca = pca_init(best_comp_num)
evaluate_classifier(clf_gbc, x_train_pca, x_test_pca, y_train, y_test)
import random

sample_num = 8
row_index = []
for x in range(sample_num):
  row_index.append(random.randint(1,8887)) 

print(row_index)
sample_data = raw_data.iloc[row_index]
sample_data.head(n=sample_num)
row_id = list(sample_data.index) 
print(row_id)
sample_preprocessed = preprocessed_data.loc[row_id]
sample_preprocessed.head(n=sample_num)
sample_result_actual = sample_preprocessed['Installs']
sample_preprocessed.drop(labels=['Installs'], axis=1, inplace=True)
sample_pca = pca.transform(sample_preprocessed)
sample_result = clf_gbc.predict(sample_pca)
print(sample_result)
performance_metric(sample_result,sample_result_actual)