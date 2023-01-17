import pandas as pd
import numpy as np
import lightgbm as gbm
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report
import warnings
import time
import gc #garbage collection
pd.set_option('display.max_columns', 100)
#pd.set_option('max_colwidth',40)
warnings.filterwarnings('ignore')
#Create and wrangle columns in training or test data
def wrangle(data, resource_data):
    
    #Get year, month from submitted datetime
    data['project_submitted_datetime'] = pd.to_datetime(data['project_submitted_datetime'])
    data['Year'] = data['project_submitted_datetime'].dt.year
    data['Month'] = data['project_submitted_datetime'].dt.month
    
    #Merge with resource file
    data = data.merge(resource_data, on='id')
    
    #Find number and amount of simultaneously submitted projects with same ID; aggregate $total
    data['$total'] = data['price'] * data['quantity']
    newcol = data.groupby('id')['$total'].agg({'$aggregated':'sum', 'number_submitted':'count'})
    if 'project_is_approved' in data.columns:
        newcol['number_approved'] = data.groupby('id')['project_is_approved'].sum()
    newcol = newcol.reset_index()    
    
    #Drop duplicated ID rows, keeping only the first one; merge the new columns
    data.drop_duplicates(['id'], keep='first', inplace=True)
    data = data.merge(newcol, on='id', how='inner')    
    
    #Compute categorical project aggregated funding requests
    data['$Total_cat'] = pd.cut(data['$aggregated'], bins=[0,100,250,500,1000,16000], 
                                labels=['0-100 USD','101-250 USD','251-500 USD','501-1000 USD','>1000 USD'])
    data['#Prior_cat'] = pd.cut(data["teacher_number_of_previously_posted_projects"],bins=[-1,1,5,10,25,50,500],
                                         labels=['0-1','2-5','6-10','11-25','26-50','51+'])
    #Separate project categories and tabulate--there are maximum 3 categories (by prior analysis)
    data[['cat1','cat2','cat3']] = data['project_subject_categories'].str.split(',', 3, expand=True)
    data['cat1'] = data['cat1'].str.strip()
    data['cat2'] = data['cat2'].str.strip()
    data['cat3'] = data['cat3'].str.strip()
    #Get the number of tags that were assigned to each project
    data['#Project_categories'] = data[['cat1','cat2','cat3']].count('columns')
    data['#Project_essays'] = data[['project_essay_1','project_essay_2','project_essay_3','project_essay_4']].count('columns')
    data['essays'] = data['project_essay_1'].astype(str)+' '+data['project_essay_2'].astype(str)
    data.drop(columns=['teacher_id','project_submitted_datetime','project_title','project_essay_3',
                      'project_essay_4', 'description','quantity','price','$total'], inplace=True)
    data['teacher_prefix'].fillna('unknown', inplace=True)
    data.rename({'teacher_prefix': 'Teacher_prefix', 'school_state':'State',
                 'project_grade_category':'Grade_cat',
                 'project_subject_categories':'Subject_cat'}, axis='columns', inplace=True)
    gc.collect()
    return data
#Read Training Data, Test Data, and Resources Data
resources = pd.read_csv('../input/resources.csv', sep=',')
train_data = pd.read_csv('../input/train.csv', sep=',')
train_data = wrangle(train_data, resources)
print(train_data.shape,'\n',train_data.columns)
display(train_data.head(3))
categories = pd.DataFrame(train_data[['cat1','cat2','cat3']].stack().value_counts(), columns=['#TotalTags'])
categories['#approved'] = train_data.groupby('cat1')['project_is_approved'].sum()
categories['%approved'] = round(100 * categories['#approved']/categories['#TotalTags'], 2)
categories = categories.sort_values(by='%approved', ascending=False)

ax = categories[['#TotalTags','#approved']].sort_values(by='#TotalTags', ascending=False).plot(kind='bar', legend=True, fontsize=16, 
                                        figsize=(12,6), rot=30, title='Funding Counts by First Project Category')
ax.set_xlabel('Project Category', fontsize=14)
ax.set_ylabel('#Projects with Tag', fontsize=16)
print(categories.sum(), '\n', categories)
prior = pd.DataFrame(train_data.groupby('#Prior_cat')['project_is_approved'].count())
prior['#approved'] = train_data.groupby('#Prior_cat')['project_is_approved'].sum()
prior.rename(columns = {'project_is_approved': '#total'}, inplace=True)
prior['#not_approved'] = prior['#total'] - prior['#approved']
prior['%_approved'] = 100 * prior['#approved']/prior['#total']

#fig, axes = plt.subplots(1,1, figsize=(16,8), sharex=True)
axA = prior[['#total','#approved','#not_approved']].plot(kind='bar', figsize=(12,6), 
                            legend=True, fontsize=16, title='Number of Previously Posted Projects')
axB = prior['%_approved'].plot(kind='line', secondary_y=True, fontsize=14, color='r', legend=True, alpha=1.0)
axA.set_xlabel('Number of Prior Projects Posted by Teacher', fontsize=16)
axA.set_ylabel('Number of Submitted Projects', fontsize=16)
axB.set_ylabel('Percentage Projects Approved', fontsize=16)
axB.set_ylim(70,100)
print(train_data["teacher_number_of_previously_posted_projects"].describe()[['min','50%','mean','max']])
grades = pd.DataFrame(train_data.groupby('Grade_cat')['project_is_approved'].count())
grades['#approved'] = train_data.groupby('Grade_cat')['project_is_approved'].sum()
grades['$approved'] = round(train_data[train_data['project_is_approved']==1].groupby('Grade_cat')['$aggregated'].mean(),2)
grades.rename(columns = {'project_is_approved': '#total'}, inplace=True)
grades['#not_approved'] = grades['#total'] - grades['#approved']
grades['$not_approved'] = round(train_data[train_data['project_is_approved']==0].groupby('Grade_cat')['$aggregated'].mean(),2)
grades['%approved'] = round(100 * grades['#approved']/grades['#total'],1)
grades = grades.reindex(index=['Grades PreK-2','Grades 3-5','Grades 6-8','Grades 9-12'])
display(grades)

#Plot the dataframe
ax1 = grades[['#total', '#approved', '#not_approved']].plot(kind='bar', figsize=(12,6), rot=0, legend=True,
                                                               fontsize=14, color=['gray','g','r'], alpha=0.5,
                                                               title='Project Approval Counts by Grades')
ax2 = grades['%approved'].plot(kind='line', secondary_y=True, fontsize=14, legend=True, alpha=0.8)
ax1.set_xlabel('Grade Category', fontsize=16)
ax1.set_ylabel('Number of Projects', fontsize=16)
ax1.set_ylim(0,80000)
ax2.set_ylabel('Percentage Projects Approved', fontsize=16)
ax2.set_ylim(74,90)

ax3 = grades[['$approved', '$not_approved']].plot(kind='bar', figsize=(12,6), rot=0, color=['g','r'], alpha=0.5,
                                                     fontsize=14, title='Mean Project Amount by Grades')
ax3.set_xlabel('Grade Category', fontsize=16)
ax3.set_ylabel('Mean (Total) Project Amounts (USD)', fontsize=16)

states = pd.DataFrame(train_data.groupby('State').size(), columns=['#submitted'])
states['#approved'] = train_data.groupby('State')['project_is_approved'].sum()
states['%_approved'] = round(100 * states['#approved']/states['#submitted'], 2)
states['$_approved'] = train_data[train_data['project_is_approved']==1].groupby('State')['$aggregated'].mean()
states.sort_values(by='%_approved', ascending=True, inplace=True)
fig, axes = plt.subplots(1,1, figsize=(16,8), sharex=True)
states['%_approved'].plot(kind='line',color=['blue'], alpha=1.0, legend=True)
states['$_approved'].plot(kind='bar', color=['gray'], secondary_y=True, ylim=(0,160), alpha=0.7, linewidth=9, legend=True)
axes.set_xlabel('Projects from State', fontsize=16)
axes.set_ylim(70,90)
axes.set_ylabel('Percentage of Projects Approved', fontsize=16)
axes.right_ax.set_ylabel('Mean Approved Project Value (USD)', fontsize=16)
years = pd.DataFrame(train_data.groupby('Year').size(), columns=['#_submitted'])
years['#_approved'] = train_data.groupby('Year')['project_is_approved'].sum()
years['%_approved'] = round(100 * years['#_approved']/years['#_submitted'], 2)
years.sort_values(by='%_approved', ascending=True, inplace=True)
print(years)

month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
months = pd.DataFrame(train_data.groupby('Month').size(), columns=['#_submitted'])
months['#_approved'] = train_data.groupby('Month')['project_is_approved'].sum()
months['%_approved'] = 100 * months['#_approved']/months['#_submitted']
months.sort_index(ascending=True, inplace=True)
print(months)
axm1 = months[['#_submitted', '#_approved']].plot(kind='bar', figsize=(12,6), 
                                            rot=0, legend=True, sharex=True, alpha=0.7,
                                            fontsize=16, title='Project Approvals by Month')
axm2 = months['%_approved'].plot(secondary_y=True, legend=True, color='g', linewidth=3)
axm1.set_xlabel('Month', fontsize=16)
axm1.set_xticklabels(month_names)
axm1.set_ylabel('Number of Projects', fontsize=16)
axm2.set_ylabel('Percentage of Projects Approved', fontsize=16)
axm2.set_ylim(0,100)
essays = pd.DataFrame(train_data.groupby('#Project_essays').size(), 
                      columns=['#_completed_essay_questions'])
essays['#_approved'] = train_data.groupby('#Project_essays')['project_is_approved'].sum()
essays['%_approved'] = round(100 * essays['#_approved']/essays['#_completed_essay_questions'],1)
essays
#train_data.groupby('id')['id'].count().value_counts(ascending=False)
submissions = pd.DataFrame()
submissions['Number prior submitted per teacher'] = train_data['teacher_number_of_previously_posted_projects'].value_counts(ascending=False)
submissions['Currently submitted per teacher'] = train_data['number_submitted'].value_counts(ascending=False)
submissions
ax = submissions.plot(kind='bar', figsize=(14,6))
ax.set_xlabel('Number of Submissions by Teacher', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.set(xlim=(0,100), ylim=(0,50000))
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
prefix = pd.DataFrame(train_data.groupby('Teacher_prefix').size(), columns=['#_prefix'])
prefix['#_approved'] = train_data.groupby('Teacher_prefix')['project_is_approved'].sum()
prefix['%_approved'] = round(100 * prefix['#_approved']/prefix['#_prefix'], 1)
display(prefix)
ax = prefix[['#_prefix','#_approved']].plot(kind='bar', rot=0, figsize=(12,6))
ax.set_xlabel('Teacher Prefix', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
#Release some memory
dfs = [grades, states, years, months, essays, submissions, prefix]
del dfs
gc.collect()
#Tokenize text column features using tfidf/count vectorizer
def vect_text(data, n_resource_feat, n_essay_feat, scaling=False):
    #n_resource_feat = desired number of text features from resource descriptions
    #n_essay_feat = desired number of text features from essay quesstions
    from sklearn.feature_extraction.text import TfidfVectorizer
    #import nltk
    #stemmer = nltk.stem.PorterStemmer()
    if scaling:
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler(copy=False) #Works with sparse matrices
    t0 = time.time()
    #use_idf: False for CountVectorizer
    #min_df: minimum document frequency to be included
    #stop_words: common words that won't be included
    #ngram_range: specify (min, max) n-gram size; #features explode for larger sizes
    #norm: normalization using either 'l1', 'l2', or None (for Count)
    
    #Vectorize project resource summary using tfidf or count
    vect = TfidfVectorizer(use_idf=True, norm='l2', min_df=5, ngram_range=(1,2), 
                                   max_features=n_resource_feat, stop_words="english")
    
    #Word stemming (optional--resource intensive)
    #tokens = stemmer.stem(token for token in data['project_resource_summary'][:nrec])
    #X_resource = vect.fit_transform(tokens)
    
    X_resource = vect.fit_transform(data['project_resource_summary'])
    if scaling:
        X_resource = scaler.fit_transform(X_resource)
    resource_names = vect.get_feature_names()
    
    #Vectorize project essays
    vect = TfidfVectorizer(use_idf=True, norm='l2', min_df=5, ngram_range=(1,2), 
                                   max_features=n_essay_feat, stop_words="english")    
    X_essay = vect.fit_transform(data['essays'])
    if scaling:
        X_essay = scaler.fit_transform(X_essay)
    essay_names = vect.get_feature_names()

    #Combine into a sparse matrix
    X_vect = hstack([X_resource, X_essay], 'csr')
    feature_names = resource_names + essay_names
    print('\nText Features:')
    print('   Vectorization time:', round(time.time() - t0,1))
    print('   Shape:', X_vect.shape)
    return X_vect, feature_names
#Tokenize categorical features
#Note: not necessary if assigning categorical features in LightGBM classifier
def vect_cat(data, features, scaling=False):
    t0 = time.time()
    if scaling:
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler(copy=False) #Works with sparse matrices
        
    le = LabelEncoder()
    feat_vect = []
    for f in range(len(features)):
        feat = le.fit_transform(data[features[f]].astype(str))  
        if scaling:
            feat = scaler.fit_transform(feat.reshape(-1,1))
        feat_vect.append(csr_matrix(feat).T)
    X_vect = hstack(feat_vect, 'csr')
    print('\nCategorical Features:')
    print('   Vectorization time:', round(time.time() - t0,1))
    print('   Shape:', X_vect.shape)
    return X_vect, feat_vect
#Reduce categorical features only for training; no labels for prediction
def feat_reduce(X_vect, features, labels, filt_value):
    #filt_value = Percent desired reduction in number of categorical features
    from sklearn.feature_selection import SelectPercentile
    #Categorical feature reduction
    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X_vect, labels, test_size=0.2, random_state=0)
    select = SelectPercentile(percentile=filt_value)
    select.fit(X_train, y_train)
    feature_mask = np.array(select.get_support())
    selected_features = np.array(features)[feature_mask]
    X_vect = X_vect[:,feature_mask]
    X_train_selected = select.transform(X_train)
    print('\nCategorical Feature Reduction:')
    print('   Reduction time:', round(time.time() - t0,1))
    print('   Final Shape:', X_vect.shape)
    print('   Selected Features:', selected_features)
    return X_vect, selected_features
nrec = len(train_data) #number of records to use with random sampling
nres = 100             #Number of top Project Resource text features to use
ness = 200             #Number of top Essay text features to use
#pcat = 100              #Percentage of top categorical features to use; 100 for no feature reduction
#Input categorical features to consider using; final set depends on outcome of feature reduction
cat_features = ['Month','Grade_cat', '#Prior_cat','Subject_cat', 
                '$Total_cat', 'Teacher_prefix','#Project_categories']

#Get reduced dataset, using sampling, for development
#train_data_reduce = train_data.sample(nrec)

X_text, text_features = vect_text(train_data, nres, ness, False)  
X_cat, cat_features_sparse = vect_cat(train_data, cat_features, False)

#Feature reduction for categorical columns (optional)
#X_cat, cat_features = feat_reduce(X_cat, cat_features, labels, pcat)

#Define joint sparse matrix
X_vect = hstack([X_text, X_cat], 'csr')
#Straight-forward lightGBM classifier without Gridsearch
def classify_gbm(X_vect, labels, cat_features, text_features):

    t0 = time.time()
    eval_results = {}

    X_train, X_test, y_train, y_test = train_test_split(X_vect, labels, test_size=0.2, random_state=0)
    feature_names = text_features + cat_features

    params = {'boosting_type': 'gbdt','objective': 'binary','metric': 'auc', 'max_depth': 12,
              'num_leaves': 31,'feature_fraction': 0.85,'bagging_fraction': 0.85,'learning_rates':1,
              'bagging_freq': 5,'verbose': 0,'num_threads': 1,'lambda_l2': 2,'min_gain_to_split': 0,
              'min_data_in_leaf':50, 'num_boost_round':1000, 'early_stopping_rounds':20}  

    clfGBM = gbm.train(params,
                      gbm.Dataset(X_train, y_train),
                      valid_sets=[gbm.Dataset(X_test, y_test)],
                      categorical_feature = cat_features,
                      verbose_eval = 100,
                      evals_result = eval_results,
                      feature_name = feature_names)

    y_pred = clfGBM.predict(X_test, num_iteration=clfGBM.best_iteration)
    y_pred_class = np.digitize(y_pred, [0.75])
    #confusion = confusion_matrix(y_test, y_pred_class)

    print('\nClassification:')
    print('   Time (sec.):', round(time.time() - t0,1))
    print('   Bin Counts:', np.bincount(y_pred_class))
    print('   Best AUC:', round(roc_auc_score(y_test, y_pred), 3))
    print('   Prediction Accuracy:', round(accuracy_score(y_test, y_pred_class), 3))
    print('   F1 Score:', round(f1_score(y_test, y_pred_class), 3))
    print('   Classification Report:\n', classification_report(y_test, y_pred_class))
    gbm.plot_metric(eval_results, metric='auc', figsize=(12,6), title='AUC metric: training validation set')
    #print('Confusion Matrix Results:\n#TP:',confusion[1,1],'\n#TN:',confusion[0,0],'\n#FP:',confusion[0,1],'\n#FN:',confusion[1,0])

    return clfGBM
labels = train_data['project_is_approved']
clf = classify_gbm(X_vect, labels, cat_features, text_features)
#ALTERNATE LightGBM classifier with grid search and kfold validation
def classify_gbm_grid(X_vect, labels, cat_features, text_features):
    t0 = time.time()
    kfolds = 5
    
    X_train, X_test, y_train, y_test = train_test_split(X_vect, labels, test_size=0.3, random_state=0)
    param_grid = {'learning_rate': [1], 'num_boost_round': [1000], 'max_depth': [6], 'reg_lambda': [1]}
    clfGBM = gbm.LGBMClassifier(
          boosting_type= 'gbdt', 
          objective = 'binary', 
          max_bin = 10, 
          silent = False,
          num_leaves = 31,
          min_split_gain = 0.0,
          is_unbalance = True) #because classes are unbalanced
          #evals_result_ = eval_results)

    #Perform grid search using defined parameter grid, scoring, and #kfolds
    grid_search = GridSearchCV(clfGBM, param_grid, scoring='roc_auc', verbose=1, cv=kfolds)
    all_features = text_features + cat_features
    grid_search.fit(X_train, y_train, 
                categorical_feature=cat_features, 
                feature_name=text_features + cat_features)

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    feature_importance = model.feature_importances_
    results = pd.DataFrame(grid_search.cv_results_)
    display(results)

    print('\nClassification:')
    print('   Time (sec.):', round(time.time() - t0,1))
    print('   Test Score: {:.2f}'.format(model.score(X_test, y_test)))
    print('   Best Params: {}'.format(grid_search.best_params_))
    print('   Best Validation Score: {:.2f}'.format(grid_search.best_score_))
    print('   Best Estimator:\n{}'.format(model))
    print('\n   Classification Report:\n', classification_report(y_test, y_pred))
    
    return model
#Show most important features
features_df = pd.DataFrame(clf.feature_name(), columns=['feature'])
features_df['importance'] = clf.feature_importance()
features_df.sort_values('importance', ascending=False, inplace=True)
features_df.set_index('feature', inplace=True)

ax = features_df[:25].plot(kind='barh', figsize=(14,8), legend=False)
ax.invert_yaxis()
ax.set_xlabel('Model Importance', fontsize=16)
ax.set_ylabel('Feature', fontsize=16)
ax.set_title('25 Most Import Model Features', fontsize=16)
#Use clf model to make predictions from test data; use same cat_features as above
#del [train_data]
test_data = pd.read_csv('../input/test.csv', sep=',')
test_data = wrangle(test_data, resources)
display(test_data.head(3))

cat_features = ['Month','Grade_cat', '#Prior_cat','Subject_cat', 
                '$Total_cat', 'Teacher_prefix','#Project_categories']

X_text, text_features = vect_text(test_data, nres, ness, False)
X_cat, features = vect_cat(test_data, cat_features, False)
X_vect = hstack([X_text, X_cat], 'csr')
y_pred = clf.predict(X_vect)
my_submission = pd.DataFrame({'id': test_data["id"], 'project_is_approved': y_pred}) #y_pred[:,1]})
print(test_data.shape,'\n',test_data.columns)
print(my_submission.head().to_csv(index=False))
my_submission.to_csv('my_submission.csv', index=False)
