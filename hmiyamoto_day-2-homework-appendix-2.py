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

# df_kick['backers_log10'] = np.log10(df_kick['backers'] + epsilon)
# df_kick['usd_pledged_real_log10'] = np.log10(df_kick['usd_pledged_real'] + epsilon)
df_kick['usd_goal_real_log10'] = np.log10(df_kick['usd_goal_real'] + epsilon)

display(df_kick.head())
penalty = 'l1'
alpha = 1e-4

y = df_kick["state_dummy"].values
X = df_kick[["usd_goal_real_log10", "category", "currency", "time_campaign_dummy"]].values

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
                             columns=["usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
    df_y_train = pd.DataFrame(y_train,
                             columns=["state_dummy"])

    df_X_test = pd.DataFrame(X_test,
                             columns=["usd_goal_real_log10", "category", "currency", "time_campaign_dummy"])
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

    X_train = df_X_train[["usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
    X_test = df_X_test[["usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
    
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
    print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}'.format(w0, w1, w2, w3, w4))

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

#     cross_valid_mae += mae 
    split_num += 1

# Generalization performance
final_log_likelihood = cross_valid_log_likelihood / n_split
print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
final_accuracy = cross_valid_accuracy / n_split
print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    

    
penalty = 'l1'
alpha = 1e-4

y = df_kick["state_dummy"].values
X = df_kick[["usd_goal_real_log10", "category"]].values

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
                             columns=["usd_goal_real_log10", "category"])
    df_y_train = pd.DataFrame(y_train,
                             columns=["state_dummy"])

    df_X_test = pd.DataFrame(X_test,
                             columns=["usd_goal_real_log10", "category"])
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

#     # Create dummy variables for currency using train data
#     # Replace currency to currency_success_rate
#     currency_success_rate = {}
#     df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
#     df_currency_all_count = df_X_train['currency'].value_counts()
#     for currency in df_currency_all_count.keys():
#         currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
#     df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
#     df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 


#     display(df_X_train.head())
#     display(df_y_train.head())

#     display(df_X_test.head())
#     display(df_y_test.head())


    print("Fold %s"%split_num)

    X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
    X_test = df_X_test[["usd_goal_real_log10", "category_dummy"]].values
    
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
#     w3 = clf.coef_[0, 2]
#     w4 = clf.coef_[0, 3]
#     print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}'.format(w0, w1, w2, w3, w4))
    print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}'.format(w0, w1, w2))

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

#     cross_valid_mae += mae 
    split_num += 1

# Generalization performance
final_log_likelihood = cross_valid_log_likelihood / n_split
print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
final_accuracy = cross_valid_accuracy / n_split
print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    

    
penalty = 'l1'
alpha = 1e-4

y = df_kick["state_dummy"].values
X = df_kick[["category"]].values

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
                             columns=["category"])
    df_y_train = pd.DataFrame(y_train,
                             columns=["state_dummy"])

    df_X_test = pd.DataFrame(X_test,
                             columns=["category"])
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

#     # Create dummy variables for currency using train data
#     # Replace currency to currency_success_rate
#     currency_success_rate = {}
#     df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
#     df_currency_all_count = df_X_train['currency'].value_counts()
#     for currency in df_currency_all_count.keys():
#         currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
#     df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
#     df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 


#     display(df_X_train.head())
#     display(df_y_train.head())

#     display(df_X_test.head())
#     display(df_y_test.head())


    print("Fold %s"%split_num)

#     X_train = df_X_train[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
#     X_test = df_X_test[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values

#     X_train = df_X_train[["usd_pledged_real_log10", "usd_goal_real_log10"]].values
#     X_test = df_X_test[["usd_pledged_real_log10", "usd_goal_real_log10"]].values
    
    X_train = df_X_train[["category_dummy"]].values
    X_test = df_X_test[["category_dummy"]].values
    
    # Normaliztion
    stdsc = StandardScaler()
    X_train_stand = stdsc.fit_transform(X_train)
    X_test_stand = stdsc.transform(X_test)
    
    
    
    clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
    clf.fit(X_train_stand, y_train)

    # Weight
    w0 = clf.intercept_[0]
    w1 = clf.coef_[0, 0]
#     w2 = clf.coef_[0, 1]
#     w3 = clf.coef_[0, 2]
#     w4 = clf.coef_[0, 3]
#     w5 = clf.coef_[0, 4]
#     w6 = clf.coef_[0, 5]
#     print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))
#     print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}'.format(w0, w1, w2, w3))
#     print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}'.format(w0, w1, w2))
    print('w0 = {:.3f}, w1 = {:.3f}'.format(w0, w1))
    
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

#     cross_valid_mae += mae 
    split_num += 1

# Generalization performance
final_log_likelihood = cross_valid_log_likelihood / n_split
print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
final_accuracy = cross_valid_accuracy / n_split
print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    

    
penalty = 'l1'
alpha = 1e-4

y = df_kick["state_dummy"].values
X = df_kick[["usd_goal_real_log10"]].values

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
                             columns=["usd_goal_real_log10"])
    df_y_train = pd.DataFrame(y_train,
                             columns=["state_dummy"])

    df_X_test = pd.DataFrame(X_test,
                             columns=["usd_goal_real_log10"])
    df_y_test = pd.DataFrame(y_test,
                             columns=["state_dummy"])




#     # Create dummy variables for category using train data
#     # Replace category to category_success_rate
#     category_success_rate = {}
#     df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
#     df_category_all_count = df_X_train['category'].value_counts()
#     for category in df_category_all_count.keys():
#         category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
#     df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
#     df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)

#     # Create dummy variables for currency using train data
#     # Replace currency to currency_success_rate
#     currency_success_rate = {}
#     df_currency_successful_count = df_X_train['currency'][df_y_train['state_dummy'] == 1].value_counts()
#     df_currency_all_count = df_X_train['currency'].value_counts()
#     for currency in df_currency_all_count.keys():
#         currency_success_rate[currency] = df_currency_successful_count[currency] / df_currency_all_count[currency]
#     df_X_train['currency_dummy'] = df_X_train['currency'].replace(currency_success_rate) 
#     df_X_test['currency_dummy'] = df_X_test['currency'].replace(currency_success_rate) 


#     display(df_X_train.head())
#     display(df_y_train.head())

#     display(df_X_test.head())
#     display(df_y_test.head())


    print("Fold %s"%split_num)

#     X_train = df_X_train[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
#     X_test = df_X_test[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values

#     X_train = df_X_train[["usd_pledged_real_log10", "usd_goal_real_log10"]].values
#     X_test = df_X_test[["usd_pledged_real_log10", "usd_goal_real_log10"]].values
    
    X_train = df_X_train[["usd_goal_real_log10"]].values
    X_test = df_X_test[["usd_goal_real_log10"]].values
    
    # Normaliztion
    stdsc = StandardScaler()
    X_train_stand = stdsc.fit_transform(X_train)
    X_test_stand = stdsc.transform(X_test)
    
    
    
    clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
    clf.fit(X_train_stand, y_train)

    # Weight
    w0 = clf.intercept_[0]
    w1 = clf.coef_[0, 0]
#     w2 = clf.coef_[0, 1]
#     w3 = clf.coef_[0, 2]
#     w4 = clf.coef_[0, 3]
#     w5 = clf.coef_[0, 4]
#     w6 = clf.coef_[0, 5]
#     print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))
#     print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}'.format(w0, w1, w2, w3))
#     print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}'.format(w0, w1, w2))
    print('w0 = {:.3f}, w1 = {:.3f}'.format(w0, w1))
    
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

#     cross_valid_mae += mae 
    split_num += 1

# Generalization performance
final_log_likelihood = cross_valid_log_likelihood / n_split
print("Cross Validation Log-likelihood = %s"%round(final_log_likelihood, 3))
final_accuracy = cross_valid_accuracy / n_split
print('Cross Validation Accuracy = {:.3f}%'.format(100 * final_accuracy))
    

    
# Create dummy variables for category using train data
# Replace category to category_success_rate
category_success_rate = {}
df_category_successful_count = df_kick['category'][df_kick['state_dummy'] == 1].value_counts()
df_category_all_count = df_kick['category'].value_counts()
for category in df_category_all_count.keys():
    category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
df_kick['category_dummy'] = df_kick['category'].replace(category_success_rate)

df_kick_2var = df_kick[["state_dummy", "category_dummy", "usd_goal_real_log10"]]


x = df_kick_2var[["category_dummy", "usd_goal_real_log10"]].values #returns a numpy array
standard_scaler = StandardScaler()
x_scaled = standard_scaler.fit_transform(x)
df_kick_2var = pd.DataFrame(x_scaled)
df_kick_2var = pd.concat([df_kick[["state_dummy"]], df_kick_2var], axis=1)

df_kick_2var.columns = ["state_dummy", "category_dummy", "usd_goal_real_log10"]
# df_kick_2var["state_dummy"] = df_kick_2var["state_dummy"].astype(str)
# df_kick_2var.info()
ax = sns.scatterplot(x="category_dummy", y="usd_goal_real_log10", hue="state_dummy", data=df_kick_2var)
pg = sns.pairplot(df_kick_2var)
print(type(pg))
g = sns.FacetGrid(df_kick_2var, hue='state_dummy')
g.map(sns.distplot, "category_dummy", label="state_dummy")
g.add_legend()
plt.show()
g = sns.FacetGrid(df_kick_2var, hue='state_dummy')
g.map(sns.distplot, "usd_goal_real_log10", label="state_dummy")
g.add_legend()
plt.show()
