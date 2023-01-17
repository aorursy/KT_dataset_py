# coding: utf-8



# Init the script

# ERG2050 Group Work

# CUHK(SZ) 2016 Term 2



import warnings

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import interp

from itertools import cycle

from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve,auc

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import Lasso

import itertools

from sklearn.model_selection import StratifiedKFold





def rename_for_kaggle(default):

    default=default.rename(columns = {'default.payment.next.month':'default'})

    default=default.rename(columns = {'PAY_0':'PAY_1'})

    cols = list(default.columns[2:]) # Dismiss BAL_LIMIT, since it wasn't in my version of dataset originally.

    cols = [cols[-1]] + cols[:-1]

    return default[cols]



default = pd.read_csv("../input/UCI_Credit_Card.csv")

default = rename_for_kaggle(default)
warnings.filterwarnings('ignore')



colors = cycle(['brown','lightcoral','red','magenta','cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])



def get_model(algoname,feature,target):

    X_train = feature

    y_train = target

    return algoname.fit(X_train,y_train.values.ravel())



def algorithm(algoname,colors,train,test,pos):

    mean_tpr,lw,i =0.0, 2,1

    mean_fpr = np.linspace(0, 1, 100)

    fold_accuracy= []

    cnf_mat = 0

    skfold = StratifiedKFold(n_splits=10,shuffle = True)

    for (trainindex,testindex), color in zip(skfold.split(train, test.values.ravel()), colors):

        X_train, X_test = train.loc[trainindex], train.loc[testindex]

        y_train, y_test = test.loc[trainindex], test.loc[testindex]

        model = algoname.fit(X_train,y_train.values.ravel())

        fold_accuracy.append(model.score(X_test,y_test.values.ravel()))

        result = model.predict(X_test)

        fpr, tpr, thresholds= roc_curve(y_test.values,result,pos_label=pos)

        mean_tpr += interp(mean_fpr, fpr, tpr)

        mean_tpr[0] = 0.0

        roc_auc = auc(fpr, tpr)

        cm = confusion_matrix(y_test.values,result)

        cnf_mat +=  cm

        plt.step(fpr, tpr, lw=lw, color=color,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i+=1

    mean_tpr /= skfold.get_n_splits(train,test.values.ravel())

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    plt.step(mean_fpr, mean_tpr, color='g', linestyle='--',

             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.title("Average accuracy: {0:.3f}".format(np.asarray(fold_accuracy).mean()))

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    plt.legend(loc="lower right") 

    plt.show()

    plt.figure()

    plot_confusion_matrix(cnf_mat, classes=["0","1"],

                      title='Confusion matrix, without normalization')

    plt.show()

    return("Average accuracy: {0:.3f} (+/-{1:.3f})".format(np.asarray(fold_accuracy).mean(),

                                                           np.asarray(fold_accuracy).std()),

           "\n Confustion Matrix:",cnf_mat)



def benchmark(default):

    default_train,default_test = default.iloc[:,1:].astype(int), default.iloc[:,0].astype(int)



    # In[5]:



    print("\n Default of Credit Card Clients Data Set")

    print("\n Random Forest")

    forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                max_depth=100, max_features='auto', max_leaf_nodes=None,

                min_impurity_split=1e-07, min_samples_leaf=50,

                min_samples_split=2, min_weight_fraction_leaf=0.0,

                n_estimators=600, n_jobs=-1, oob_score=False,

                random_state=None, verbose=0, warm_start=False)

    print(algorithm(forest,colors,default_train,default_test,pos = None))

    

    #print("\n Random Forest (EXP)")

    forest_exp = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',

                max_depth=200, max_features=None, max_leaf_nodes=None,

                min_impurity_split=1e-07, min_samples_leaf=50,

                min_samples_split=2, min_weight_fraction_leaf=0.0,

                n_estimators=600, n_jobs=-1, oob_score=False,

                random_state=None, verbose=0, warm_start=False)

    #print(algorithm(forest_exp,colors,default_train,default_test,pos = None))





    # In[6]:

    print("\n Logistic")

    logistic = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,

              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,

              penalty='l1', random_state=None, solver='liblinear', tol=0.0001,

              verbose=0, warm_start=False)

    print(algorithm(logistic,colors,default_train,default_test,pos = None))





    # In[7]:

    print("\n Naive")

    naive = GaussianNB()

    print(algorithm(naive,colors,default_train,default_test,pos = None))





    # In[8]:

    print("\n KNN")

    knneigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

               metric_params=None, n_jobs=-1, n_neighbors=50, p=2,

               weights='uniform')

    print(algorithm(knneigh,colors,default_train,default_test,pos = None))





    # In[9]:

    print("\n SVM")

    svm = LinearSVC(C=1, class_weight=None, dual=False, fit_intercept=True,

         intercept_scaling=1, loss='squared_hinge', max_iter=10,

         multi_class='ovr', penalty='l1', random_state=1000, tol=0.0001,

         verbose=0)

    print(algorithm(svm,colors,default_train,default_test,pos = None))

    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

def benchmark_hard(default):

    default_train,default_test = default.iloc[:,1:].astype(int), default.iloc[:,0].astype(int)

    print("\n Default of Credit Card Clients Data Set")

    forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                max_depth=1000, max_features='auto', max_leaf_nodes=None,

                min_impurity_split=1e-07, min_samples_leaf=10,

                min_samples_split=2, min_weight_fraction_leaf=0.0,

                n_estimators=6000, n_jobs=-1, oob_score=False,

                random_state=None, verbose=0, warm_start=False)

    print(algorithm(forest,colors,default_train,default_test,pos = None))

    

def export_false_prediction(result, filename, data):

    false_index = []

    for i in range(len(result)):

        if result[i] != data.iloc[i,0]:

            false_index.append(i)

    false_pred = data.iloc[false_index,:]

    false_pred.to_csv(filename, index=False)

    print("Trainning Errors: " + str(len(false_pred)))

    

def export_negative(result, filename, data):

    negative_index = []

    for i in range(len(result)):

        if result[i] == 0:

            negative_index.append(i)

    data.iloc[negative_index,:].to_csv(filename)

    

    

def stage1(filename):

    default = pd.read_csv(filename)

    default = rename_for_kaggle(default)

    breakpoint = int(1 * len(default))

    train_features, train_target = default.iloc[:breakpoint,1:].astype(int), default.iloc[:breakpoint,0].astype(int)

    test_features, test_target = default.iloc[breakpoint:,1:].astype(int), default.iloc[breakpoint:,0].astype(int)

    forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                    max_depth=100, max_features='auto', max_leaf_nodes=None,

                    min_impurity_split=1e-07, min_samples_leaf=50,

                    min_samples_split=2, min_weight_fraction_leaf=0.0,

                    n_estimators=600, n_jobs=-1, oob_score=False,

                    random_state=None, verbose=0, warm_start=False)

    model = get_model(forest, train_features, train_target)

    # result = model.predict(test_features)

    result = model.predict(train_features)

    export_false_prediction(result, "false_negative.csv", default)

    return model



def stage2(filename_false_negative):

    default = pd.read_csv(filename_false_negative)

    default = rename_for_kaggle(default)

    train_features, train_target = default.iloc[:,1:].astype(int), default.iloc[:,0].astype(int)

    forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                    max_depth=100, max_features='auto', max_leaf_nodes=None,

                    min_impurity_split=1e-07, min_samples_leaf=50,

                    min_samples_split=2, min_weight_fraction_leaf=0.0,

                    n_estimators=600, n_jobs=-1, oob_score=False,

                    random_state=None, verbose=0, warm_start=False)

    logistic = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,

              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,

              penalty='l1', random_state=None, solver='liblinear', tol=0.0001,

              verbose=0, warm_start=False)

    naive = GaussianNB()

    model = get_model(naive, train_features, train_target)

    return model



def naive(filename):

    default = pd.read_csv(filename)

    default = rename_for_kaggle(default)

    breakpoint = int(1 * len(default))

    train_features, train_target = default.iloc[:breakpoint,1:].astype(int), default.iloc[:breakpoint,0].astype(int)

    test_features, test_target = default.iloc[breakpoint:,1:].astype(int), default.iloc[breakpoint:,0].astype(int)

    forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                    max_depth=100, max_features='auto', max_leaf_nodes=None,

                    min_impurity_split=1e-07, min_samples_leaf=50,

                    min_samples_split=2, min_weight_fraction_leaf=0.0,

                    n_estimators=600, n_jobs=-1, oob_score=False,

                    random_state=None, verbose=0, warm_start=False)

    naive_model = GaussianNB()

    model = get_model(naive_model, train_features, train_target)

    # result = model.predict(test_features)

    result = model.predict(train_features)

    export_false_prediction(result, "false_negative.csv", default)

    return model



def magic(train_file, test_file):

    model1 = stage1(train_file)

    model2 = stage2("false_negative.csv")

    # Models trainetest_ext4.csv   

    # Load Test file

    test = pd.read_csv(test_file)

    print("Test size: " + str(len(test)))

    test_features = test.iloc[:,1:].astype(int)

    test_target = test.iloc[:,0].astype(int)



    # Test

    # First Stage Fit



    print("Model 1 Score:" + str(model1.score(test_features,test_target.values.ravel())))

    print("Model 2 Score:" + str(model2.score(test_features,test_target.values.ravel())))

    # Using Model 1 to predict test set

    result1 = model1.predict(test_features)

    error_count = 0

    for i in range(len(test)):

        if result1[i] == 0 and test_target.iloc[i] == 1:

            error_count += 1

    print("Model 1 Testing False Negative: " + str(error_count))



    # Find negative predictions

    export_negative(result1, "negative.csv", test)

    negative = pd.read_csv("negative.csv")

    print("Model 2 Input Size: " + str(len(negative)))

    negative_features = negative.iloc[:,2:].astype(int)



    # Second Stage

    result2 = model2.predict(negative_features)

    print("Result2 Size: " + str(len(result2)))

    diff = 0

    diff_fn = 0

    for i in range(len(result2)):

        result_1 = result1[negative.iloc[i,1].astype(int)]

        if result_1 != result2[i]:

            result1[negative.iloc[i,1].astype(int)] = result2[i]

            diff += 1

            if result_1 != test_target.iloc[negative.iloc[i,1].astype(int)]:

                diff_fn += 1

    print("DIFF: " + str(diff) + " Correction: " + str(diff_fn))





    # Get Score

    error_count = 0

    for i in range(len(test)):

        if result1[i] != test_target.iloc[i]:

            error_count += 1

    print("Errors: " + str(error_count))

    print("Accuracy: " + str(1-(error_count/len(test))))

    error_count = 0

    for i in range(len(test)):

        if result1[i] == 0 and test_target.iloc[i] == 1:

            error_count += 1

    print("Model 1+2 Testing False Negative: " + str(error_count))



def get_model(algoname,feature,target):

    X_train = feature

    y_train = target

    return algoname.fit(X_train,y_train.values.ravel())



def test(train_file, test_file):

    model1 = stage1(train_file)

    

    # Load Test file

    test = pd.read_csv(test_file)

    print("Test size: " + str(len(test)))

    test_features = test.iloc[:,1:].astype(int)

    test_target = test.iloc[:,0].astype(int)

    

    # Test

    # First Stage Fit



    print("Model Score:" + str(model1.score(test_features,test_target.values.ravel())))

    # Using Model 1 to predict test set

    result1 = model1.predict(test_features)

    

    # Draw Confusion Matrix

    cm = confusion_matrix(test_target.values,result1)

    plot_confusion_matrix(cm, classes=["0","1"],

                      title='Confusion matrix, without normalization')

    plt.show()

    

    # Count

    error_count = 0

    for i in range(len(test)):

        if result1[i] == 0 and test_target.iloc[i] == 1:

            error_count += 1

    print("Model Testing False Negative: " + str(error_count))

    

def naive_test(train_file, test_file):

    model1 = naive(train_file)

    

    # Load Test file

    test = pd.read_csv(test_file)

    print("Test size: " + str(len(test)))

    test_features = test.iloc[:,1:].astype(int)

    test_target = test.iloc[:,0].astype(int)



    # Test

    # First Stage Fit



    print("Model Score:" + str(model1.score(test_features,test_target.values.ravel())))

    # Using Model 1 to predict test set

    result1 = model1.predict(test_features)

    

    # Draw Confusion Matrix

    cm = confusion_matrix(test_target.values,result1)

    plot_confusion_matrix(cm, classes=["0","1"],

                      title='Confusion matrix, without normalization')

    plt.show()

    

    error_count = 0

    for i in range(len(test)):

        if result1[i] == 0 and test_target.iloc[i] == 1:

            error_count += 1

    print("Model Testing False Negative: " + str(error_count))

    



print("Ready.")
%%timeit -n 1 -r 1

benchmark(default)
def BILL_regression(default):

    data = default.copy()

    pandas = pd

    from scipy import stats

    for i in range(len(data)):

        temp = pandas.DataFrame.transpose(pandas.DataFrame(data=data.iloc[i]))

        for j in range(1,7):

            temp.loc[j] = temp.iloc[0]

        temp["BILL_AMT"] = 0

        temp["BILL_DATE"] = 0

        for j in range(1,7):

            temp.at[j, "BILL_AMT"] = data.iloc[i]["BILL_AMT" + str(j)]

            temp.at[j, "BILL_DATE"] = j

        slope, intercept, r_value, p_value, std_err = stats.linregress(temp["BILL_DATE"],temp["BILL_AMT"])

        data.at[i, "BILL_SLOPE"] = slope

        data.at[i, "BILL_INCEPT"] = intercept

        data.at[i, "BILL_STDERR"] = std_err

    return data

v1 = BILL_regression(default)

v1.head()
def BILL_poly_regression(default):

    data = default.copy()

    for i in range(len(data)):

        temp = pd.DataFrame.transpose(pd.DataFrame(data=data.iloc[i]))

        for j in range(1,7):

            temp.loc[j] = temp.iloc[0]

        temp["BILL_AMT"] = 0

        temp["BILL_DATE"] = 0

        for j in range(1,7):

            temp.at[j, "BILL_AMT"] = data.iloc[i]["BILL_AMT" + str(j)]

            temp.at[j, "BILL_DATE"] = j

        result = np.polynomial.polynomial.polyfit(temp["BILL_DATE"],temp["BILL_AMT"],4)

        for j in range(len(result)):

            data.at[i, "BILL_POLY" + str(j)] = result[j]

    return data

v2 = BILL_poly_regression(v1)

v2.head()
def threshold_count(label, default, thresholds):

    data = default.copy()

    for threshold in thresholds:

        for i in range(len(data)):

            count = 0

            for j in range(1,7):

                if data.iloc[i][label + str(j)] <= threshold:

                    count += 1

            data.at[i, label + "_COUNT_" + str(threshold)] = count

    return data

v3 = threshold_count("BILL_AMT", v2, [0,20000,70000])

v3 = threshold_count("PAY_AMT", v3, [0,1000,5000])

v3.head()
def pay_count(default):

    data = default.copy()

    pandas = pd

    from scipy import stats

    for i in range(len(data)):

        temp = pandas.DataFrame.transpose(pandas.DataFrame(data=data.iloc[i]))

        for val in [-2,-1,1,2,3,4,5,6,7,8,9]:

            count = 0

            for j in range(1,7):

                if data.iloc[i]["PAY_" + str(j)] == val:

                    count += 1

                data.at[i, "PAY_COUNT_" + str(val)] = count

    return data

v4 = pay_count(v3)

v4.head()
#VAR

def var(label, default):

    data=default.copy()

    for i in range(len(data)):

        temp = pd.DataFrame.transpose(pd.DataFrame(data=data.iloc[i]))

        for j in range(1,7):

            temp.loc[j] = temp.iloc[0]

        temp["VAL"] = 0

        for j in range(1,7):

            temp.at[j, "VAL"] = data.iloc[i][label + str(j)]

        data.at[i, label + "_VAR"] = temp.VAL.var()

    return data

v5 = var("PAY_AMT", v4)

v5 = var("BILL_AMT", v5)

v5.head()
def f(x):

    return float(1)/float(1+np.exp(x))



def pbr(bill, pay):

    if bill > 0:

        if pay < bill:

            result = pay/bill

        else:

            result = 1 + f(pay)

    elif bill == 0:

        if pay != 0:

            result = 2 + f(pay)

        if pay == 0:

            result = 1

    else:

        if pay == 0:

            result = 3 + f(pay)

        if pay > 0:

            result = 4 + f(pay) 

    return result



#Payback Ratio

def calc_pbr(default):

    data = default

    for i in range(len(data)):

        temp = pd.DataFrame.transpose(pd.DataFrame(data=data.iloc[i]))

        for j in range(1,7):

            temp.loc[j] = temp.iloc[0]

        temp["VAL"] = 0

        for j in range(1,7):

            bill = data.iloc[i]["BILL_AMT" + str(j)]

            pay = data.iloc[i]["PAY_AMT" + str(j)]

            data.at[i, "PBR_" + str(j)] = pbr(bill, pay)

    return data



v6 = calc_pbr(v5)

v6.head()
def PBR_regression(default):

    data = default.copy()

    pandas = pd

    from scipy import stats

    for i in range(len(data)):

        temp = pandas.DataFrame.transpose(pandas.DataFrame(data=data.iloc[i]))

        for j in range(1,7):

            temp.loc[j] = temp.iloc[0]

        temp["AMT"] = 0

        temp["DATE"] = 0

        for j in range(1,7):

            temp.at[j, "AMT"] = data.iloc[i]["PBR_" + str(j)]

            temp.at[j, "DATE"] = j

        slope, intercept, r_value, p_value, std_err = stats.linregress(temp["DATE"],temp["AMT"])

        data.at[i, "PBR_SLOPE"] = slope

        data.at[i, "PBR_INCEPT"] = intercept

        data.at[i, "PBR_STDERR"] = std_err

    return data

v7 = PBR_regression(v6)

v7.head()
def PAY_regression(default):

    data = default.copy()

    pandas = pd

    from scipy import stats

    for i in range(len(data)):

        temp = pandas.DataFrame.transpose(pandas.DataFrame(data=data.iloc[i]))

        for j in range(1,7):

            temp.loc[j] = temp.iloc[0]

        temp["AMT"] = 0

        temp["DATE"] = 0

        for j in range(1,7):

            temp.at[j, "AMT"] = data.iloc[i]["PAY_AMT" + str(j)]

            temp.at[j, "DATE"] = j

        slope, intercept, r_value, p_value, std_err = stats.linregress(temp["DATE"],temp["AMT"])

        data.at[i, "PAY_SLOPE"] = slope

        data.at[i, "PAY_INCEPT"] = intercept

        data.at[i, "PAY_STDERR"] = std_err

    return data

v8 = PAY_regression(v7)

v8.head()
list(v8.columns)
def sub_features(data, alpha):

    features = ["default"]

    lasso = Lasso(alpha=alpha)

    default_train, default_test = data.iloc[:,1:].astype(int), data.iloc[:,0].astype(int)

    lasso.fit(default_train, default_test)

    for i in range(len(lasso.coef_)):

        if lasso.coef_[i] > 0:

            features.append(default_train.columns[i])

    return data.copy()[features]
data_1 = sub_features(v8, 0.1)

list(data_1.columns)
benchmark(data_1)
data_2 = v8.copy()[['default', 'PAY_1',

 'PAY_AMT1',

 'PAY_AMT_COUNT_1000',

 'PAY_AMT_COUNT_5000',

 'PAY_COUNT_-2',

 'PAY_COUNT_2',

 'PAY_COUNT_3',

 'PAY_COUNT_7',

 'PBR_STDERR',

 'BILL_AMT_COUNT_20000']]

benchmark(data_2)
