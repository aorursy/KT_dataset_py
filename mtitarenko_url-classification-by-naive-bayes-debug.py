%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
# --------------------------------------
DEBUG = True

print("Debug mode is: {}".format(('OFF', 'ON')[DEBUG]))

if DEBUG:
    import sklearn
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
# --------------------------------------
names=['URL','Category']
df=pd.read_csv('../input/URL Classification.csv',names=names, na_filter=False)

# --------------------------------------
if DEBUG:
    df_original = df.copy()
# --------------------------------------

df1 = df[1:2001]
df2 = df[50000:52000]
df3 = df[520000:522000]
df4 =df[535300:537300]
df5 = df[650000:652000]
df6= df[710000:712000]
df7=  df[764200:766200]
df8=  df[793080:795080]
df9=  df[839730:841730]
df10=  df[850000:852000]
df11=  df[955250:957250]
df12=  df[1013000:1015000]
df13=  df[1143000:1145000]
df14=  df[1293000:1295000]
df15=  df[1492000:1494000]
dt=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], axis=0)
df.drop(df.index[1:2000],inplace= True) # <--- why 2000, not 2001?
df.drop(df.index[50000:52000],inplace= True)
df.drop(df.index[520000:522000],inplace= True)
df.drop(df.index[535300:537300],inplace= True)
df.drop(df.index[650000:652000],inplace= True)
df.drop(df.index[710000:712000],inplace= True)
df.drop(df.index[764200:766200],inplace= True)
df.drop(df.index[793080:795080],inplace= True)
df.drop(df.index[839730:841730],inplace= True)
df.drop(df.index[850000:852000],inplace= True)
df.drop(df.index[955250:957250],inplace= True)
df.drop(df.index[1013000:1015000],inplace= True)
df.drop(df.index[1143000:1145000],inplace= True)
df.drop(df.index[1293000:1295000],inplace= True)
df.drop(df.index[1492000:1494000],inplace= True)
df.tail()
df.Category.value_counts().plot(figsize=(12,5),kind='bar',color='green');
plt.xlabel('Category')
plt.ylabel('Total Number Of Individual Category for Training')
dt.Category.value_counts().plot(figsize=(12,5),kind='bar',color='green');
plt.xlabel('Category')
plt.ylabel('Total Number Of Individual Category for Testing')
X_train=df['URL']
y_train=df['Category']
X_train.shape
X_test=dt['URL']
y_test=dt['Category']
X_test.shape
# --------------------------------------
if DEBUG:
    
    """
    Problem #1 - the dataset has: 
    1) completely duplicate rows
    2) rows with equal 'URL' value and different 'Category' values
    """
    
    def check_duplicates(df, subset=['URL'], keep='first', verbose=True):

        df_duplicates = df[df.duplicated(subset=subset, keep=keep)]
        num_duplicates = len(df_duplicates)
        
        if verbose:
            print('Duplicate rows found ({}): {}'.format(subset, num_duplicates))
        
        return num_duplicates
    
    check_duplicates(df_original, subset=None) # 22493 - completely duplicated rows
    check_duplicates(df_original) # 47754 - rows, duplicated by the 'URL' column
    
    assert len(df_original) - len(df_original['URL'].unique()) == check_duplicates(df_original, verbose=False) # correctness check
    
    print('\nSame URL, different categories:')
    print(df.loc[df['URL'] == 'http://www.worldpressphoto.org/']) # Example: 3 different labels for the same URL
# --------------------------------------
# --------------------------------------
if DEBUG:

    """
    THE FIX to Problem #1
    """
    
    # Remove duplicates:
    
    df_original_no_duplicates = df_original.drop_duplicates(subset=['URL'], keep='first')
    df_original_no_duplicates.index = np.arange(1, len(df_original_no_duplicates) + 1) # remove gaps in index caused by deleted rows
        
    # Check correctness:
    
    len_before = len(df_original)
    len_after = len(df_original_no_duplicates)
    print('Len before: {}, len after: {}'.format(len_before, len_after))
    print('Difference: {}'.format(len_before - len_after))
    
    check_duplicates(df_original_no_duplicates) # 0
# --------------------------------------
# --------------------------------------
if DEBUG:

    """
    Problem #2 - the train-test split is incorrect:
    train and test sets have a huge intersection
    """
    
    def check_intersection_df1_df2(df1, df2, col_name='URL', suffix=''):

        num_intersection = len(df1.merge(df2.drop_duplicates(subset=[col_name]), on=[col_name]))
            
        if num_intersection:
            ratio_intersection = 100. * num_intersection / len(df1) # calculated relative to df1
            print('Intersection num rows{}: {}\nIntersection ratio: {:.2f}%!'.format(suffix, num_intersection, ratio_intersection))
        else:
            print('There is no intersection{}!'.format(suffix))
            
        return num_intersection

    check_intersection_df1_df2(dt, df, suffix=' (between train and test)')
    
    """
    So, the original test set overlaps 93.44 percent with the training set along the 'URL' column.

    Therefore, the reported F1-score = 0.876 is a fake!
    The real performance is much worse!
    """
# --------------------------------------
# --------------------------------------
if DEBUG:
    
    """
    THE FIX to Problem #2.
    We have to use the correct 'drop-inplace' code to split train and test data.
    In case we want to get balanced consecutive data (although this is also the wrong way to obtain a representative model), 
    we can use the following:
    """

    idxs_cats_start = ( # idxs from the original code - to split categories for test data
        1, 
        50000, 
        520000,
        535300,
        650000,
        710000,
        764200,
        793080,
        839730,
        850000,
        955250,
        1013000,
        1143000,
        1293000,
        1492000
    )
    
    n_per_cat = 2000
    
    # Let's get new idxs: 
    #   - taking into account deleted duplicates and 
    #   - so that the new test set is as close as possible to the original:
    
    idxs_cats_start_new = []
        
    for idx in idxs_cats_start:
        url_cat_start = df_original.iloc[idx]['URL']
        idx_cat_start = df_original_no_duplicates.index[df_original_no_duplicates['URL'] == url_cat_start]
        assert len(idx_cat_start) == 1
        assert url_cat_start == df_original[idx: idx+1]['URL'].values[0]
        idxs_cats_start_new.append(idx_cat_start[0])
    
    # Create new train and test sets:
    
    df_correct = df_original_no_duplicates.copy()
    dt_correct = pd.concat([df_correct[idx:idx+n_per_cat] for idx in idxs_cats_start_new], axis=0)

    for idx in idxs_cats_start_new:
        df_correct.drop(range(idx+1, idx+n_per_cat+1), inplace=True)
        # +1 since dt_correct used slicing which starts from 0, but df_correct idxs in drop start from 1

    X_train_correct = df_correct['URL']
    y_train_correct = df_correct['Category']
    X_test_correct = dt_correct['URL']
    y_test_correct = dt_correct['Category']
        
    # Check correctness:
    
    check_intersection_df1_df2(dt_correct, df_correct, suffix=' (between corrected train and test)')

    dt_correct.Category.value_counts().plot(figsize=(12,5),kind='bar',color='blue');
    plt.xlabel('Category')
    plt.ylabel('Total Number Of Individual Category for Testing')
    
    test_cats_len = dt_correct['Category'].value_counts().tolist()
    assert len(test_cats_len) == len(df_original['Category'].unique()) and all((cat_size == n_per_cat for cat_size in test_cats_len))

    check_intersection_df1_df2(dt_correct, dt, suffix=' (between new and old test)')
    
    """
    So, we got the new test set:
        - 15 categories with 2000 samples in each
        - no intersection with the new train set
        - 96.27% intersection with the original test set 
            -> therefore, we can estimate the real performance for the suggested algorithm
    """
# --------------------------------------
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train, y_train) # <--- for what?
n_iter_search = 5
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf = RandomizedSearchCV(text_clf, parameters, n_iter = n_iter_search)
gs_clf = gs_clf.fit(X_train, y_train)
# --------------------------------------
if DEBUG:
    print('Best score: {:.3f}'.format(gs_clf.best_score_)) 
    
    # so, the best score via cross-validation is below 0.4
# --------------------------------------
y_pred=gs_clf.predict(X_test)
precision_recall_fscore_support(y_test, y_pred, average='weighted')
y_pred=gs_clf.predict(X_test) # <--- DUPLICATE
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4)) # <--- DUPLICATE
# --------------------------------------
if DEBUG:

    """
    Let's see the real performance:
    """
    
    clf_new = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    gs_clf_new = RandomizedSearchCV(clf_new, parameters, n_iter=n_iter_search)
    
    gs_clf_new.fit(X_train_correct, y_train_correct)
    
    y_pred_correct = gs_clf_new.predict(X_test_correct)
    print(classification_report(y_test_correct, y_pred_correct))
    
    """
    ... the real F1-score is just 0.21
    """
# --------------------------------------
array = confusion_matrix(y_test, y_pred)
cm=np.array(array)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cm, index = [i for i in "0123456789ABCDE"],
                  columns = [i for i in "0123456789ABCDE"])
plt.figure(figsize = (20,15))
sn.heatmap(df_cm, annot=True)
print('Naive Bayes Train Accuracy = ', accuracy_score(y_train,gs_clf.predict(X_train)))
print('Naive Bayes Test Accuracy = ', accuracy_score(y_test,gs_clf.predict(X_test))) # <--- DUPLICATE
print(gs_clf.predict(['http://www.businesstoday.net/']))
print(gs_clf.predict(['http://www.gamespot.net/']))