import os
os.listdir("../input/data")
import pandas as pd
final_sort = pd.read_pickle('../input/data/final_sort.p')
print("Count of each value:\n", final_sort['Score'].value_counts())

def time_split(dataframe, n):
#Sample 100k Points to apply on the model
    final_sample = dataframe.head(n)

#Time based splitting in such a way that test data contains more recent data
    df_train = final_sample.head(int(len(final_sample) * 0.7))
    df_test = final_sample.tail(int(len(final_sample) * 0.3))
    return(final_sample, df_train, df_test)
#Calling function to split data
final_sample, df_train, df_test = time_split(final_sort, 100000)
print(len(df_train))
print(len(df_test))
df_test['Score'].value_counts()
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
final_counts = CountVectorizer()
final_fit = final_counts.fit(df_train['Cleaned_text'].values)

def vector_bow(lst):
    final_sparse = final_counts.transform(lst)
    return(final_sparse)

x_train_bow = vector_bow(df_train['Cleaned_text'].values)
x_test_bow = vector_bow(df_test['Cleaned_text'].values)

print("Shape of train data:", x_train_bow.shape)
print("Shape of test data:", x_test_bow.shape)
#Load class label pickle files
import numpy as np
y_train_npz = np.load('../input/data/y_train.npz')
y_test_npz = np.load('../input/data/y_test.npz')

y_train = y_train_npz['y_train']
y_test = y_test_npz['y_test']
print(type(y_test))
print("Shape of train class label:", y_train.shape)
print("Shape of test class label:", y_test.shape)
from sklearn.preprocessing import StandardScaler
#Standardizing data with mean and variance of train data
stand_func = StandardScaler(with_mean = False)
x_train_fit = stand_func.fit(x_train_bow)

def standard_data(train, test):
    x_train_stand = stand_func.transform(train)
    x_test_stand = stand_func.transform(test)
    x_train = x_train_stand
    x_test = x_test_stand
    return(x_train, x_test)

#Calling Standard_data function
x_train_bowV, x_test_bowV = standard_data(x_train_bow, x_test_bow)
print("Shape of train data:", x_train_bowV.shape)
print("Shape of train class label:", y_train.shape)
print("Shape of test data:", x_test_bowV.shape)
print("Shape of test class label:", y_test.shape)
from sklearn.model_selection import TimeSeriesSplit
def timeseriessplit(train):
    tscv = TimeSeriesSplit(n_splits=3)
    my_cv = list(TimeSeriesSplit(n_splits=3).split(train))
    return(my_cv)
my_cv = timeseriessplit(x_train_bowV)
%%time
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
def randomsearch(x_train, x_test):
           
#Hyper Parameter
    max_depth = [int(x) for x in np.linspace(10,100,num=10)]
    max_depth.append(None)
    tuned_parameters = {'n_estimators': [int(x) for x in np.linspace(10,110,num=10)],
                        'max_depth': max_depth,
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'min_samples_split': [5,10,15],
                        'min_samples_leaf': [3,5,7]}

#Using RandomizedSearchCV
    model = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1), tuned_parameters, scoring = 'f1', cv=my_cv, n_iter = 50, n_jobs = -1)
    model.fit(x_train, y_train)
#     f1_scores = [x[1] for x in model.grid_scores_]
#     dic = [x[0] for x in model.grid_scores_]
# print(dic)
    a = model.cv_results_['mean_test_score']
    b = model.cv_results_['mean_train_score']
    maxd_val = model.cv_results_['param_max_depth']
    est_val = model.cv_results_['param_n_estimators']
    max_feat = model.cv_results_['param_max_features']
    max_split = model.cv_results_['param_min_samples_split']
    max_leaf = model.cv_results_['param_min_samples_leaf']
    abls = [(i,j,k,l,m)  for i,j,k,l,m in zip(maxd_val,est_val, max_feat, max_split, max_leaf)]
#Plot of Train and Test scores
    plt.xlabel('Hyper Parameters')
    plt.ylabel('Score')
    plt.xticks(np.arange(len(est_val)), abls)
    plt.plot(a,label='CrossValidation Score')
    plt.plot(b,label='Train Score')
    plt.legend()
    plt.title('Train Vs CrossValidation Scores')
    plt.show()
                  
    print("Classifier with Best parameters:\n ", model.best_estimator_)
    print('*' * 100)
    print("Best Cross validation Score with optimal parameters:\n ", model.best_score_)
    print('*' * 100)
    dic_params = model.best_params_
    print("Best Parameters are:\n", dic_params)
#Calculate metrics
    print("f1 score obtained on Test data is ", model.score(x_test, y_test))
    return(dic_params)

dic_params = randomsearch(x_train_bowV, x_test_bowV)
def params(dic_params):
    optimal_est = dic_params['n_estimators']
    optimal_depth = dic_params['max_depth']
    optimal_features = dic_params['max_features']
    optimal_sample_split = dic_params['min_samples_split']
    optimal_sample_leaf = dic_params['min_samples_leaf']
    return(optimal_est, optimal_depth, optimal_features, optimal_sample_split, optimal_sample_leaf)
optimal_est, optimal_depth, optimal_features, optimal_sample_split, optimal_sample_leaf = params(dic_params)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(test_y, predict_y):
    labels = [0,1]
    C = confusion_matrix(test_y, predict_y)
    print("-"*20, "Confusion matrix", "-"*20)
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
from tqdm import tqdm
import numpy as np

def dt_metric(x_train, x_test,optimal_est, optimal_depth, optimal_features, optimal_sample_split, optimal_sample_leaf):
    
    clf_dt = RandomForestClassifier(n_estimators = optimal_est, max_depth = optimal_depth, max_features = optimal_features, min_samples_split = optimal_sample_split, min_samples_leaf = optimal_sample_leaf, n_jobs = -1)

#Fitting the model
    clf_dt.fit(x_train, y_train)
    
# predict the response
    pred = clf_dt.predict(x_test)
# Calculate Confusion matrix
    plot_confusion_matrix(y_test, pred)
# Calculate Precision Score
    pre = precision_score(y_test, pred, pos_label = 1, average = 'binary') *100
    pre = np.round(pre,2)
    print('Precision for %d is %f' % (optimal_est, pre))
    print('*' * 100)
# Calculate Recall Score
    rec = recall_score(y_test, pred, pos_label = 1, average = 'binary') * 100
    rec = np.round(rec,2)
    print('Recall Score for %d is %f' % (optimal_est, rec))
    print('*' * 100)
# Calculate f1 Score
    f1 = f1_score(y_test, pred, pos_label = 1, average = 'binary') * 100
    f1 = np.round(f1,2)
    print('f1 Score for %d is %f' % (optimal_est, f1))
    print('*' * 100)     
# evaluate accuracy
    acc = (accuracy_score(y_test, pred) * 100)
    acc = np.round(acc,2)
    print('\nThe accuracy of the Random ForestDecision tree for optimal_estimators = %d is %f%%' % (optimal_est, acc))
    return(acc, pre, rec, f1, clf_dt)

#Calling dt_metric function
acc, pre, rec, f1, clf_dt = dt_metric(x_train_bowV, x_test_bowV,optimal_est, optimal_depth, optimal_features, optimal_sample_split, optimal_sample_leaf)
from wordcloud import WordCloud
import matplotlib as mpl
%matplotlib inline

def wordcloud(feature_names, feature_imp):
    
    feature_mapping = pd.DataFrame({'Features': feature_names, 'Importances': feature_imp})
    feature_sort = feature_mapping.sort_values('Importances', ascending = False)
    imp_features = list(feature_sort['Features'])
    mpl.rcParams['font.size'] =12
    mpl.rcParams['figure.subplot.bottom'] = .1

    wordcloud = WordCloud(background_color = "white", max_words=30322, max_font_size = 30).generate(str(imp_features))
    plt.figure(figsize =(8,8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    return(imp_features)
feat = wordcloud(final_counts.get_feature_names(), clf_dt.feature_importances_)
print("Top 10 features:\n", feat[:10])
from sklearn.feature_extraction.text import TfidfVectorizer
final_counts_tfidf = TfidfVectorizer()
final_fit_tfidf = final_counts_tfidf.fit(df_train['Cleaned_text'].values)

def vector_tfidf(lst):
    final_sparse = final_counts.transform(lst)
    return(final_sparse)

x_train_tfidf = vector_tfidf(df_train['Cleaned_text'].values)
x_test_tfidf = vector_tfidf(df_test['Cleaned_text'].values)

print("Shape of train data:", x_train_tfidf.shape)
print("Shape of test data:", x_test_tfidf.shape)
#Calling Standardization function
stand_func = StandardScaler(with_mean = False)
x_train_fit = stand_func.fit(x_train_tfidf)

x_train_tfidfV, x_test_tfidfV = standard_data(x_train_tfidf, x_test_tfidf)
print("Shape of train data:", x_train_tfidfV.shape)
print("Shape of train class label:", y_train.shape)
print("Shape of test data:", x_test_tfidfV.shape)
print("Shape of test class label:", y_test.shape)
#Calling TimeSeriesSplit Function
my_cv = timeseriessplit(x_train_tfidfV)
#Calling RandomSearchCV function
dic_params_tfidf = randomsearch(x_train_tfidfV, x_test_tfidfV)
optimal_est_tfidf, optimal_depth_tfidf, optimal_features_tfidf, optimal_sample_split_tfidf, optimal_sample_leaf_tfidf = params(dic_params_tfidf)
#Calling dt_metric function
acc_tfidf, pre_tfidf, rec_tfidf, f1_tfidf, clf_dt_tfidf = dt_metric(x_train_tfidfV, x_test_tfidfV,optimal_est_tfidf, optimal_depth_tfidf, optimal_features_tfidf, optimal_sample_split_tfidf, optimal_sample_leaf_tfidf)
feat1 = wordcloud(final_counts_tfidf.get_feature_names(), clf_dt_tfidf.feature_importances_)
print("Top 10 features are:\n", feat1[:10])
#Load AvgW2V train and test pickle files
import numpy as np
x_train_npz = np.load('../input/data/x_train_w2v.npz')
x_test_npz = np.load('../input/data/x_test_w2v.npz')

x_train_w2v = x_train_npz['x_train_w2v']
x_test_w2v = x_test_npz['x_test_w2v']
print(type(x_train_w2v))
print("Shape of train AvgW2V:", x_train_w2v.shape)
print("Shape of test AvgW2V:", x_test_w2v.shape)
from sklearn.preprocessing import StandardScaler
#Standardizing data with mean and variance of train data
stand_func = StandardScaler()
x_train_fit = stand_func.fit(x_train_w2v)

#Calling Standard_data function
x_train_w2vV, x_test_w2vV = standard_data(x_train_w2v, x_test_w2v)
print("Shape of train data:", x_train_w2vV.shape)
print("Shape of train class label:", y_train.shape)
print("Shape of test data:", x_test_w2vV.shape)
print("Shape of test class label:", y_test.shape)
#Calling TimeSeriesSplit function
my_cv = timeseriessplit(x_train_w2vV)
#Calling RandomSearchCV function
dic_params_w2v = randomsearch(x_train_tfidfV, x_test_tfidfV)
optimal_est_w2v, optimal_depth_w2v, optimal_features_w2v, optimal_sample_split_w2v, optimal_sample_leaf_w2v = params(dic_params_w2v)
#Calling dt_metric function
acc_w2v, pre_w2v, rec_w2v, f1_w2v, clf_dt_w2v = dt_metric(x_train_w2vV, x_test_w2vV, optimal_est_w2v, optimal_depth_w2v, optimal_features_w2v, optimal_sample_split_w2v, optimal_sample_leaf_w2v)
#Load TFIDF AvgW2V train and test pickle files
import numpy as np
x_train_tfidf_npz = np.load('../input/data/x_train_tfidf_w2v.npz')
x_test_tfidf_npz = np.load('../input/data/x_test_tfidf_w2v.npz')

x_train_tfidf_w2v = x_train_tfidf_npz['x_train_tfidf_w2v']
x_test_tfidf_w2v = x_test_tfidf_npz['x_test_tfidf_w2v']
print(type(x_train_tfidf_w2v))
print("Shape of train tfidf AvgW2V:", x_train_tfidf_w2v.shape)
print("Shape of test tfidf AvgW2V:", x_test_tfidf_w2v.shape)
from sklearn.preprocessing import StandardScaler
#Standardizing data with mean and variance of train data
stand_func = StandardScaler()
x_train_fit = stand_func.fit(x_train_tfidf_w2v)

#Calling Standard_data function
x_train_tfidf_w2vV, x_test_tfidf_w2vV = standard_data(x_train_tfidf_w2v, x_test_tfidf_w2v)
print("Shape of train data:", x_train_tfidf_w2vV.shape)
print("Shape of train class label:", y_train.shape)
print("Shape of test data:", x_test_tfidf_w2vV.shape)
print("Shape of test class label:", y_test.shape)
#Calling TimeSeriesSplit function
my_cv = timeseriessplit(x_train_tfidf_w2vV)
#Calling RandomSearchCV function
dic_params_tfidf_w2v = randomsearch(x_train_tfidfV, x_test_tfidfV)
optimal_est_tfidf_w2v, optimal_depth_tfidf_w2v, optimal_features_tfidf_w2v, optimal_sample_split_tfidf_w2v, optimal_sample_leaf_tfidf_w2v = params(dic_params_tfidf_w2v)
#Calling dt_metric function
acc_tfidf_w2v, pre_tfidf_w2v, rec_tfidf_w2v, f1_tfidf_w2v, clf_dt_tfidf_w2v = dt_metric(x_train_tfidf_w2vV, x_test_tfidf_w2vV, optimal_est_tfidf_w2v, optimal_depth_tfidf_w2v, optimal_features_tfidf_w2v, optimal_sample_split_tfidf_w2v, optimal_sample_leaf_tfidf_w2v)
!pip install https://pypi.python.org/packages/source/P/PrettyTable/prettytable-0.7.2.tar.bz2
from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Vectorization", "Optimal No.of Base learners", "Optimal depth", "Optimal features", "Optimal_min_split", "Optimal_min_leaf","Accuracy", "Precision", "Recall", "f1 Score"]
x.add_row(["BOW", optimal_est, optimal_depth, optimal_features, optimal_sample_split, optimal_sample_leaf, acc, pre, rec, f1])
x.add_row(["TFIDF", optimal_est_tfidf, optimal_depth_tfidf, optimal_features_tfidf, optimal_sample_split_tfidf, optimal_sample_leaf_tfidf, acc_tfidf, pre_tfidf, rec_tfidf, f1_tfidf])
x.add_row(["AvgW2V", optimal_est_w2v, optimal_depth_w2v, optimal_features_w2v, optimal_sample_split_w2v, optimal_sample_leaf_w2v, acc_w2v, pre_w2v, rec_w2v, f1_w2v])
x.add_row(["TFIDF AvgW2V", optimal_est_tfidf_w2v, optimal_depth_tfidf_w2v, optimal_features_tfidf_w2v, optimal_sample_split_tfidf_w2v, optimal_sample_leaf_tfidf_w2v, acc_tfidf_w2v, pre_tfidf_w2v, rec_tfidf_w2v, f1_tfidf_w2v])
print(x)

