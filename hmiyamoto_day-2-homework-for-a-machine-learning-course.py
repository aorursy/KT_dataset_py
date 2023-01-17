# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
df_kick = pd.read_csv("../input/ks-projects-201801.csv")
display(df_kick.head())
df_kick['launched'] = pd.to_datetime(df_kick['launched'])
df_kick['laun_month_year'] = df_kick['launched'].dt.to_period("M")
df_kick['laun_year'] = df_kick['launched'].dt.to_period("A")
df_kick['laun_hour'] = df_kick['launched'].dt.hour

df_kick['deadline'] = pd.to_datetime(df_kick['deadline'])
df_kick['dead_month_year'] = df_kick['deadline'].dt.to_period("M")
df_kick['dead_year'] = df_kick['launched'].dt.to_period("A")

#Creating a new columns with Campaign total months
df_kick['time_campaign'] = df_kick['dead_month_year'] - df_kick['laun_month_year']
df_kick['time_campaign'] = df_kick['time_campaign'].astype(int)

df_kick['time_campaign_dummy'] = df_kick['time_campaign']
df_kick['time_campaign_dummy'].loc[df_kick['time_campaign_dummy'] >= 5] = 5

display(df_kick.head())
df_kick['state_dummy'] = df_kick['state']
df_kick['state_dummy'].loc[df_kick['state_dummy'] != 'successful'] = 0
df_kick['state_dummy'].loc[df_kick['state_dummy'] == 'successful'] = 1

# display(df_kick.head())
# epsilon = 1e-8
epsilon = 1

df_kick['backers_log10'] = np.log10(df_kick['backers'] + epsilon)
df_kick['usd_pledged_real_log10'] = np.log10(df_kick['usd_pledged_real'] + epsilon)
df_kick['usd_goal_real_log10'] = np.log10(df_kick['usd_goal_real'] + epsilon)

display(df_kick.head())
y = df_kick["state_dummy"].values
X = df_kick[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"]].values

n_split = 5 # Number of group

cross_valid_log_likelihood = 0
cross_valid_accuracy = 0
split_num = 1

# Cross Validation
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] # Train data
    X_test, y_test = X[test_idx], y[test_idx]     # Test data
    
#     print(X_train.shape)
#     print(X_test.shape)
    df_X_train = pd.DataFrame(X_train,
                             columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
    df_y_train = pd.DataFrame(y_train,
                             columns=["state_dummy"])
    
    df_X_test = pd.DataFrame(X_test,
                             columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
    df_y_test = pd.DataFrame(y_test,
                             columns=["state_dummy"])
    



    # Create dummy variables for category using train data
    # Replace category to category_success_rate
    category_success_rate = {}
    df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
    df_category_all_count = df_X_train['category'].value_counts()
    for category in df_category_all_count.keys():
        category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
    df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)
    
    # Create dummy variables for currency using train data
    # Replace currency to currency_success_rate
    currency_success_rate = {}
    df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
    df_currency_all_count = df_X_train['currency'].value_counts()
    for currency in df_currency_all_count.keys():
        currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
    df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
    df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 
    
    
#     display(df_X_train.head())
#     display(df_y_train.head())

#     display(df_X_test.head())
#     display(df_y_test.head())
    

    print("Fold %s"%split_num)
    
    X_train = df_X_train[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
    X_test = df_X_test[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
    
    clf = SGDClassifier(loss='log', penalty='none', max_iter=100, fit_intercept=True, random_state=1234)
    clf.fit(X_train, y_train)

    # Weight
    w0 = clf.intercept_[0]
    w1 = clf.coef_[0, 0]
    w2 = clf.coef_[0, 1]
    w3 = clf.coef_[0, 2]
    w4 = clf.coef_[0, 3]
    w5 = clf.coef_[0, 4]
    w6 = clf.coef_[0, 5]
    print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))

    # Predict labels
    y_est_test = clf.predict(X_test)
    
    # Log-likelihood
    log_likelihood = - log_loss(y_test, y_est_test)    
    cross_valid_log_likelihood += log_likelihood    
    print('Log-likelihood = {:.3f}'.format(log_likelihood))
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_est_test)
    cross_valid_accuracy += accuracy   
    print('Accuracy = {:.3f}%'.format(100 * accuracy))
    print()
    
#     cross_valid_mae += mae #後で平均を取るためにMAEを加算
    split_num += 1

# Generalization performance
final_log_likelihood_0 = cross_valid_log_likelihood / n_split
print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood_0, 3))
final_accuracy_0 = cross_valid_accuracy / n_split
print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy_0))
penalty = 'l2'

alphas_multiply = np.array(range(-8,0))
alphas = 10.0 ** alphas_multiply
# alpha = 0.0

L2_accuracy = []
L2_log_likelihood = []
L2_weight_abs_max = []
L2_weight_abs_min = []


for alpha in alphas:
    
    print('='*100)
    print('penalty =', penalty)
    print('alpha =', alpha)
    print()

    y = df_kick["state_dummy"].values
    X = df_kick[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"]].values

    n_split = 5 # Number of group

    cross_valid_log_likelihood = 0
    cross_valid_accuracy = 0
    split_num = 1

    # Cross Validation
    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        X_train, y_train = X[train_idx], y[train_idx] # Train data
        X_test, y_test = X[test_idx], y[test_idx]     # Test data

    #     print(X_train.shape)
    #     print(X_test.shape)
        df_X_train = pd.DataFrame(X_train,
                                 columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
        df_y_train = pd.DataFrame(y_train,
                                 columns=["state_dummy"])

        df_X_test = pd.DataFrame(X_test,
                                 columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
        df_y_test = pd.DataFrame(y_test,
                                 columns=["state_dummy"])




        # Create dummy variables for category using train data
        # Replace category to category_success_rate
        category_success_rate = {}
        df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
        df_category_all_count = df_X_train['category'].value_counts()
        for category in df_category_all_count.keys():
            category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
        df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
        df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)

        # Create dummy variables for currency using train data
        # Replace currency to currency_success_rate
        currency_success_rate = {}
        df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
        df_currency_all_count = df_X_train['currency'].value_counts()
        for currency in df_currency_all_count.keys():
            currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
        df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
        df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 


    #     display(df_X_train.head())
    #     display(df_y_train.head())

    #     display(df_X_test.head())
    #     display(df_y_test.head())


        print("Fold %s"%split_num)

        X_train = df_X_train[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
        X_test = df_X_test[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values

        clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
        clf.fit(X_train, y_train)

        # Weight
        w0 = clf.intercept_[0]
        w1 = clf.coef_[0, 0]
        w2 = clf.coef_[0, 1]
        w3 = clf.coef_[0, 2]
        w4 = clf.coef_[0, 3]
        w5 = clf.coef_[0, 4]
        w6 = clf.coef_[0, 5]
        print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))

#         plt.plot(np.abs(clf.coef_.T), marker='o')


        # Predict labels
        y_est_test = clf.predict(X_test)

        # Log-likelihood
        log_likelihood = - log_loss(y_test, y_est_test)    
        cross_valid_log_likelihood += log_likelihood    
        print('Log-likelihood = {:.3f}'.format(log_likelihood))

        # Accuracy
        accuracy = accuracy_score(y_test, y_est_test)
        cross_valid_accuracy += accuracy   
        print('Accuracy = {:.3f}%'.format(100 * accuracy))
        print()

    #     cross_valid_mae += mae #後で平均を取るためにMAEを加算
        split_num += 1

    # Generalization performance
    final_log_likelihood = cross_valid_log_likelihood / n_split
    print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
    final_accuracy = cross_valid_accuracy / n_split
    print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    
    L2_accuracy.append(final_accuracy)
    L2_log_likelihood.append(final_log_likelihood)
    L2_weight_abs_max.append(np.max(np.abs(clf.coef_)))
    L2_weight_abs_min.append(np.min(np.abs(clf.coef_)))
    
plt.plot(alphas_multiply, L2_accuracy, marker='o')
plt.title("L2 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Accuracy")
plt.plot(alphas_multiply, L2_log_likelihood, marker='o')
plt.title("L2 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Log-likelihood")
plt.plot(alphas_multiply, L2_weight_abs_max, marker='o', label='Weight_abs_max')
plt.plot(alphas_multiply, L2_weight_abs_min, marker='o', label='Weight_abs_min')
plt.title("L2 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Weight_abs_max_min")
plt.legend()
penalty = 'l1'

alphas_multiply = np.array(range(-8,0))
alphas = 10.0 ** alphas_multiply
# alpha = 0.0

L1_accuracy = []
L1_log_likelihood = []
L1_weight_abs_max = []
L1_weight_abs_min = []


for alpha in alphas:
    
    print('='*100)
    print('penalty =', penalty)
    print('alpha =', alpha)
    print()

    y = df_kick["state_dummy"].values
    X = df_kick[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"]].values

    n_split = 5 # Number of group

    cross_valid_log_likelihood = 0
    cross_valid_accuracy = 0
    split_num = 1

    # Cross Validation
    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        X_train, y_train = X[train_idx], y[train_idx] # Train data
        X_test, y_test = X[test_idx], y[test_idx]     # Test data

    #     print(X_train.shape)
    #     print(X_test.shape)
        df_X_train = pd.DataFrame(X_train,
                                 columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
        df_y_train = pd.DataFrame(y_train,
                                 columns=["state_dummy"])

        df_X_test = pd.DataFrame(X_test,
                                 columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
        df_y_test = pd.DataFrame(y_test,
                                 columns=["state_dummy"])




        # Create dummy variables for category using train data
        # Replace category to category_success_rate
        category_success_rate = {}
        df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
        df_category_all_count = df_X_train['category'].value_counts()
        for category in df_category_all_count.keys():
            category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
        df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
        df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)

        # Create dummy variables for currency using train data
        # Replace currency to currency_success_rate
        currency_success_rate = {}
        df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
        df_currency_all_count = df_X_train['currency'].value_counts()
        for currency in df_currency_all_count.keys():
            currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
        df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
        df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 


    #     display(df_X_train.head())
    #     display(df_y_train.head())

    #     display(df_X_test.head())
    #     display(df_y_test.head())


        print("Fold %s"%split_num)

        X_train = df_X_train[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
        X_test = df_X_test[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values

        clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
        clf.fit(X_train, y_train)

        # Weight
        w0 = clf.intercept_[0]
        w1 = clf.coef_[0, 0]
        w2 = clf.coef_[0, 1]
        w3 = clf.coef_[0, 2]
        w4 = clf.coef_[0, 3]
        w5 = clf.coef_[0, 4]
        w6 = clf.coef_[0, 5]
        print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))

#         plt.plot(np.abs(clf.coef_.T), marker='o')


        # Predict labels
        y_est_test = clf.predict(X_test)

        # Log-likelihood
        log_likelihood = - log_loss(y_test, y_est_test)    
        cross_valid_log_likelihood += log_likelihood    
        print('Log-likelihood = {:.3f}'.format(log_likelihood))

        # Accuracy
        accuracy = accuracy_score(y_test, y_est_test)
        cross_valid_accuracy += accuracy   
        print('Accuracy = {:.3f}%'.format(100 * accuracy))
        print()

    #     cross_valid_mae += mae #後で平均を取るためにMAEを加算
        split_num += 1

    # Generalization performance
    final_log_likelihood = cross_valid_log_likelihood / n_split
    print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
    final_accuracy = cross_valid_accuracy / n_split
    print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    
    L1_accuracy.append(final_accuracy)
    L1_log_likelihood.append(final_log_likelihood)
    L1_weight_abs_max.append(np.max(np.abs(clf.coef_)))
    L1_weight_abs_min.append(np.min(np.abs(clf.coef_)))
    
plt.plot(alphas_multiply, L1_accuracy, marker='o')
plt.title("L1 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Accuracy")
plt.plot(alphas_multiply, L1_log_likelihood, marker='o')
plt.title("L1 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Log-likelihood")
plt.plot(alphas_multiply, L1_weight_abs_max, marker='o', label='Weight_abs_max')
plt.plot(alphas_multiply, L1_weight_abs_min, marker='o', label='Weight_abs_min')
plt.title("L1 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Weight_abs_max_min")
plt.legend()
penalty = 'elasticnet'

l1_ratios = np.arange(0.1, 1, 0.1)
alpha = 1e-4

EN_accuracy = []
EN_log_likelihood = []
EN_weight_abs_max = []
EN_weight_abs_min = []


for l1_ratio in l1_ratios:
    
    print('='*100)
    print('penalty =', penalty)
    print('alpha =', alpha)
    print('l1_ratio =', l1_ratio)
    print()

    y = df_kick["state_dummy"].values
    X = df_kick[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"]].values

    n_split = 5 # Number of group

    cross_valid_log_likelihood = 0
    cross_valid_accuracy = 0
    split_num = 1

    # Cross Validation
    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        X_train, y_train = X[train_idx], y[train_idx] # Train data
        X_test, y_test = X[test_idx], y[test_idx]     # Test data

    #     print(X_train.shape)
    #     print(X_test.shape)
        df_X_train = pd.DataFrame(X_train,
                                 columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
        df_y_train = pd.DataFrame(y_train,
                                 columns=["state_dummy"])

        df_X_test = pd.DataFrame(X_test,
                                 columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
        df_y_test = pd.DataFrame(y_test,
                                 columns=["state_dummy"])




        # Create dummy variables for category using train data
        # Replace category to category_success_rate
        category_success_rate = {}
        df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
        df_category_all_count = df_X_train['category'].value_counts()
        for category in df_category_all_count.keys():
            category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
        df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
        df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)

        # Create dummy variables for currency using train data
        # Replace currency to currency_success_rate
        currency_success_rate = {}
        df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
        df_currency_all_count = df_X_train['currency'].value_counts()
        for currency in df_currency_all_count.keys():
            currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
        df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
        df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 


    #     display(df_X_train.head())
    #     display(df_y_train.head())

    #     display(df_X_test.head())
    #     display(df_y_test.head())


        print("Fold %s"%split_num)

        X_train = df_X_train[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
        X_test = df_X_test[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values

        clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, max_iter=100, fit_intercept=True, random_state=1234)
        clf.fit(X_train, y_train)

        # Weight
        w0 = clf.intercept_[0]
        w1 = clf.coef_[0, 0]
        w2 = clf.coef_[0, 1]
        w3 = clf.coef_[0, 2]
        w4 = clf.coef_[0, 3]
        w5 = clf.coef_[0, 4]
        w6 = clf.coef_[0, 5]
        print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))

#         plt.plot(np.abs(clf.coef_.T), marker='o')


        # Predict labels
        y_est_test = clf.predict(X_test)

        # Log-likelihood
        log_likelihood = - log_loss(y_test, y_est_test)    
        cross_valid_log_likelihood += log_likelihood    
        print('Log-likelihood = {:.3f}'.format(log_likelihood))

        # Accuracy
        accuracy = accuracy_score(y_test, y_est_test)
        cross_valid_accuracy += accuracy   
        print('Accuracy = {:.3f}%'.format(100 * accuracy))
        print()

    #     cross_valid_mae += mae #後で平均を取るためにMAEを加算
        split_num += 1

    # Generalization performance
    final_log_likelihood = cross_valid_log_likelihood / n_split
    print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
    final_accuracy = cross_valid_accuracy / n_split
    print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    
    EN_accuracy.append(final_accuracy)
    EN_log_likelihood.append(final_log_likelihood)
    EN_weight_abs_max.append(np.max(np.abs(clf.coef_)))
    EN_weight_abs_min.append(np.min(np.abs(clf.coef_)))
    
plt.plot(l1_ratios, EN_accuracy, marker='o')
plt.title("ElasticNet alpha = 1e-4")
plt.xlabel("l1_ratio")
plt.ylabel("Accuracy")
plt.plot(l1_ratios, EN_log_likelihood, marker='o')
plt.title("ElasticNet alpha = 1e-4")
plt.xlabel("l1_ratio")
plt.ylabel("Log-likelihood")
plt.plot(l1_ratios, EN_weight_abs_max, marker='o', label='Weight_abs_max')
plt.plot(l1_ratios, EN_weight_abs_min, marker='o', label='Weight_abs_min')
plt.title("ElasticNet alpha = 1e-4")
plt.xlabel("l1_ratio")
plt.ylabel("Weight_abs_max_min")
plt.legend()
df_kick2 = df_kick[["state_dummy", "backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"]]
df_kick2.head()
# Create dummy variables for category using all data
# Replace category to category_success_rate
category_success_rate = {}
df_kick2_category_successful_count = df_kick2['category'][df_kick2['state_dummy'] == 1].value_counts()
df_kick2_category_all_count = df_kick2['category'].value_counts()
for category in df_category_all_count.keys():
    category_success_rate[category] = df_kick2_category_successful_count[category] / df_kick2_category_all_count[category]
df_kick2['category_dummy'] = df_kick2['category'].replace(category_success_rate)

# Create dummy variables for currency using all data
# Replace currency to currency_success_rate
currency_success_rate = {}
df_kick2_currency_successful_count = df_kick2['currency'][df_kick2['state_dummy'] == 1].value_counts()
df_kick2_currency_all_count = df_kick2['currency'].value_counts()
for currency in df_currency_all_count.keys():
    currency_success_rate[currency] = df_kick2_currency_successful_count[currency] / df_kick2_currency_all_count[currency]
df_kick2['currency_dummy'] = df_kick2['currency'].replace(currency_success_rate) 

df_kick2 = df_kick2[["state_dummy", "backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]]

df_kick2.head()
df_kick2.describe()
import itertools
li_combi = list(itertools.combinations(df_kick2.columns[1:], 2))
for X,Y in li_combi:
    print("X=%s"%X,"Y=%s"%Y)
    print('Correlation coefficient: {:.3f}'.format(np.corrcoef(df_kick2[X], df_kick2[Y])[0,1]))
    df_kick2.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10,c="state_dummy",colormap="winter")#散布図の作成
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.tight_layout()
    plt.show()
X_all = df_kick2[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values

# Normaliztion
stdsc = StandardScaler()
X_all_stand = stdsc.fit_transform(X_all)
df_X_all_stand = pd.DataFrame(X_all_stand,
                             columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"])
df_kick2_stand = pd.concat([df_kick2["state_dummy"], df_X_all_stand], axis=1)
df_kick2_stand.head()
df_kick2_stand.describe()
import itertools
li_combi = list(itertools.combinations(df_kick2_stand.columns[1:], 2))
for X,Y in li_combi:
    print("X=%s"%X,"Y=%s"%Y)
    print('Correlation coefficient: {:.3f}'.format(np.corrcoef(df_kick2_stand[X], df_kick2_stand[Y])[0,1]))
    df_kick2_stand.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10,c="state_dummy",colormap="winter")#散布図の作成
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.tight_layout()
    plt.show()
penalty = 'l1'
alpha = 1e-4

y = df_kick["state_dummy"].values
X = df_kick[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"]].values

n_split = 5 # Number of group

cross_valid_log_likelihood = 0
cross_valid_accuracy = 0
split_num = 1

# Cross Validation
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] # Train data
    X_test, y_test = X[test_idx], y[test_idx]     # Test data

#     print(X_train.shape)
#     print(X_test.shape)
    df_X_train = pd.DataFrame(X_train,
                             columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
    df_y_train = pd.DataFrame(y_train,
                             columns=["state_dummy"])

    df_X_test = pd.DataFrame(X_test,
                             columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
    df_y_test = pd.DataFrame(y_test,
                             columns=["state_dummy"])




    # Create dummy variables for category using train data
    # Replace category to category_success_rate
    category_success_rate = {}
    df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
    df_category_all_count = df_X_train['category'].value_counts()
    for category in df_category_all_count.keys():
        category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
    df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)

    # Create dummy variables for currency using train data
    # Replace currency to currency_success_rate
    currency_success_rate = {}
    df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
    df_currency_all_count = df_X_train['currency'].value_counts()
    for currency in df_currency_all_count.keys():
        currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
    df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
    df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 


#     display(df_X_train.head())
#     display(df_y_train.head())

#     display(df_X_test.head())
#     display(df_y_test.head())


    print("Fold %s"%split_num)

    X_train = df_X_train[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
    X_test = df_X_test[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
    
    # Normaliztion
    stdsc = StandardScaler()
    X_train_stand = stdsc.fit_transform(X_train)
    X_test_stand = stdsc.transform(X_test)
    
    
    
    clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
    clf.fit(X_train_stand, y_train)

    # Weight
    w0 = clf.intercept_[0]
    w1 = clf.coef_[0, 0]
    w2 = clf.coef_[0, 1]
    w3 = clf.coef_[0, 2]
    w4 = clf.coef_[0, 3]
    w5 = clf.coef_[0, 4]
    w6 = clf.coef_[0, 5]
    print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))

#         plt.plot(np.abs(clf.coef_.T), marker='o')


    # Predict labels
    y_est_test = clf.predict(X_test_stand)

    # Log-likelihood
    log_likelihood = - log_loss(y_test, y_est_test)    
    cross_valid_log_likelihood += log_likelihood    
    print('Log-likelihood = {:.3f}'.format(log_likelihood))

    # Accuracy
    accuracy = accuracy_score(y_test, y_est_test)
    cross_valid_accuracy += accuracy   
    print('Accuracy = {:.3f}%'.format(100 * accuracy))
    print()

#     cross_valid_mae += mae #後で平均を取るためにMAEを加算
    split_num += 1

# Generalization performance
final_log_likelihood = cross_valid_log_likelihood / n_split
print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
final_accuracy = cross_valid_accuracy / n_split
print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    

    
X_all = df_kick2[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values


# Normaliztion
stdsc = StandardScaler()
X_all_stand = stdsc.fit_transform(X_all)

# Decorrelation
cov = np.cov(X_all_stand, rowvar=0) # Estimate the covariance matrix
print('cov =', cov)
_, S = np.linalg.eig(cov)           # Compute the eigenvalues and right eigenvectors of the covariance matrix
print('_ =', _)
print('S =', S)
X_all_stand_decorr = np.dot(S.T, X_all_stand.T).T # Decorrelate the data

df_X_all_stand = pd.DataFrame(X_all_stand_decorr,
                             columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"])
df_kick2_stand_decorr = pd.concat([df_kick2["state_dummy"], df_X_all_stand], axis=1)
df_kick2_stand_decorr.head()
df_kick2_stand_decorr.describe()
import itertools
li_combi = list(itertools.combinations(df_kick2_stand_decorr.columns[1:], 2))
for X,Y in li_combi:
    print("X=%s"%X,"Y=%s"%Y)
    print('Correlation coefficient: {:.3f}'.format(np.corrcoef(df_kick2_stand_decorr[X], df_kick2_stand_decorr[Y])[0,1]))
    df_kick2_stand_decorr.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10,c="state_dummy",colormap="winter")#散布図の作成
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.tight_layout()
    plt.show()
penalty = 'l1'
alpha = 1e-4

y = df_kick["state_dummy"].values
X = df_kick[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"]].values

n_split = 5 # Number of group

cross_valid_log_likelihood = 0
cross_valid_accuracy = 0
split_num = 1

# Cross Validation
for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] # Train data
    X_test, y_test = X[test_idx], y[test_idx]     # Test data

#     print(X_train.shape)
#     print(X_test.shape)
    df_X_train = pd.DataFrame(X_train,
                             columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
    df_y_train = pd.DataFrame(y_train,
                             columns=["state_dummy"])

    df_X_test = pd.DataFrame(X_test,
                             columns=["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
    df_y_test = pd.DataFrame(y_test,
                             columns=["state_dummy"])




    # Create dummy variables for category using train data
    # Replace category to category_success_rate
    category_success_rate = {}
    df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
    df_category_all_count = df_X_train['category'].value_counts()
    for category in df_category_all_count.keys():
        category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
    df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)

    # Create dummy variables for currency using train data
    # Replace currency to currency_success_rate
    currency_success_rate = {}
    df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
    df_currency_all_count = df_X_train['currency'].value_counts()
    for currency in df_currency_all_count.keys():
        currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
    df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
    df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 


#     display(df_X_train.head())
#     display(df_y_train.head())

#     display(df_X_test.head())
#     display(df_y_test.head())


    print("Fold %s"%split_num)

    X_train = df_X_train[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
    X_test = df_X_test[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
    
    # Normaliztion
    stdsc = StandardScaler()
    X_train_stand = stdsc.fit_transform(X_train)
    X_test_stand = stdsc.transform(X_test)
    
    # Decorrelation
    cov = np.cov(X_train_stand, rowvar=0) # Estimate the covariance matrix
#     print('cov =', cov)
    _, S = np.linalg.eig(cov)           # Compute the eigenvalues and right eigenvectors of the covariance matrix
#     print('_ =', _)
#     print('S =', S)
    X_train_stand_decorr = np.dot(S.T, X_train_stand.T).T # Decorrelate train data
    X_test_stand_decorr = np.dot(S.T, X_test_stand.T).T # Decorrelate train data
    
    
    
    clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
    clf.fit(X_train_stand_decorr, y_train)

    # Weight
    w0 = clf.intercept_[0]
    w1 = clf.coef_[0, 0]
    w2 = clf.coef_[0, 1]
    w3 = clf.coef_[0, 2]
    w4 = clf.coef_[0, 3]
    w5 = clf.coef_[0, 4]
    w6 = clf.coef_[0, 5]
    print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))

#         plt.plot(np.abs(clf.coef_.T), marker='o')


    # Predict labels
    y_est_test = clf.predict(X_test_stand_decorr)

    # Log-likelihood
    log_likelihood = - log_loss(y_test, y_est_test)    
    cross_valid_log_likelihood += log_likelihood    
    print('Log-likelihood = {:.3f}'.format(log_likelihood))

    # Accuracy
    accuracy = accuracy_score(y_test, y_est_test)
    cross_valid_accuracy += accuracy   
    print('Accuracy = {:.3f}%'.format(100 * accuracy))
    print()

#     cross_valid_mae += mae #後で平均を取るためにMAEを加算
    split_num += 1

# Generalization performance
final_log_likelihood = cross_valid_log_likelihood / n_split
print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
final_accuracy = cross_valid_accuracy / n_split
print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    

    
