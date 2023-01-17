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
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ( 'x', '.', 'o', 's', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
                       
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    edgecolor=None,
                    marker=markers[idx], 
                    label=cl)
df_kick = pd.read_csv("../input/ks-projects-201801.csv")
display(df_kick.head())
df_kick['state_dummy'] = df_kick['state']
df_kick['state_dummy'].loc[df_kick['state_dummy'] != 'successful'] = 0
df_kick['state_dummy'].loc[df_kick['state_dummy'] == 'successful'] = 1

# display(df_kick.head())
# epsilon = 1e-8
epsilon = 1

df_kick['usd_goal_real_log10'] = np.log10(df_kick['usd_goal_real'] + epsilon)

display(df_kick.head())
df_ALL = df_kick.loc[:, ['state_dummy', 'usd_goal_real_log10', 'category']]
df_ALL.head()
df_TRAIN, df_TEST = train_test_split(df_ALL, test_size=0.2, random_state=1234)
df_TRAIN.head()
display(df_TRAIN.describe())
display(df_TEST.describe())
penalty = 'l1'

alphas_multiply = np.array(range(-8,0))
alphas = 10.0 ** alphas_multiply

L1_accuracy = []
L1_log_likelihood = []
L1_weight_abs_max = []
L1_weight_abs_min = []


for alpha in alphas:
    
    print('='*100)
    print('penalty =', penalty)
    print('alpha =', alpha)
    print()

    y = df_TRAIN["state_dummy"].values
    X = df_TRAIN[["usd_goal_real_log10", "category"]].values

    n_split = 5 # Number of group

    cross_valid_log_likelihood = 0
    cross_valid_accuracy = 0
    split_num = 1

    # Cross Validation
    for train_idx, valid_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        X_train, y_train = X[train_idx], y[train_idx] # Train data
        X_valid, y_valid = X[valid_idx], y[valid_idx] # Validation data

        df_X_train = pd.DataFrame(X_train,
                                 columns=["usd_goal_real_log10", "category"])
        df_y_train = pd.DataFrame(y_train,
                                 columns=["state_dummy"])

        df_X_valid = pd.DataFrame(X_valid,
                                 columns=["usd_goal_real_log10", "category"])
        df_y_valid = pd.DataFrame(y_valid,
                                 columns=["state_dummy"])




        # Create dummy variables for category using train data
        # Replace category to category_success_rate
        category_success_rate = {}
        df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
        df_category_all_count = df_X_train['category'].value_counts()
        for category in df_category_all_count.keys():
            category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
        df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
        df_X_valid['category_dummy'] = df_X_valid['category'].replace(category_success_rate)


        print("Fold %s"%split_num)

        X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
        X_valid = df_X_valid[["usd_goal_real_log10", "category_dummy"]].values
        
        # Normaliztion
        stdsc = StandardScaler()
        X_train = stdsc.fit_transform(X_train)
        X_valid = stdsc.transform(X_valid)

        clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
        clf.fit(X_train, y_train)

        # Weight
        w0 = clf.intercept_[0]
        w1 = clf.coef_[0, 0]
        w2 = clf.coef_[0, 1]
        print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}'.format(w0, w1, w2))


        # Predict labels
        y_est_valid = clf.predict(X_valid)

        # Log-likelihood
        log_likelihood = - log_loss(y_valid, y_est_valid)    
        cross_valid_log_likelihood += log_likelihood    
        print('Log-likelihood = {:.3f}'.format(log_likelihood))

        # Accuracy
        accuracy = accuracy_score(y_valid, y_est_valid)
        cross_valid_accuracy += accuracy   
        print('Accuracy = {:.3f}%'.format(100 * accuracy))
        print()
        
        if split_num == n_split:
            plot_decision_regions(X_valid, y_valid, classifier=clf)
            plt.title('(Fold %s)  L1 alpha = %s' %(split_num,alpha))
            plt.xlabel('usd_goal_real_log10_stdsc')
            plt.ylabel('category_dummy_stdsc')
            plt.axes().set_aspect('equal', 'datalim')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

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
plt.show()
plt.plot(alphas_multiply, L1_log_likelihood, marker='o')
plt.title("L1 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Log-likelihood")
plt.show()
plt.plot(alphas_multiply, L1_weight_abs_max, marker='o', label='Weight_abs_max')
plt.plot(alphas_multiply, L1_weight_abs_min, marker='o', label='Weight_abs_min')
plt.title("L1 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Weight_abs_max_min")
plt.legend()
plt.show()
penalty = 'l1'

# Best Parameter
alpha = 1e-3


print('penalty =', penalty)
print('alpha =', alpha)
print()

# TRAIN data
df_y_train = df_TRAIN[["state_dummy"]]
df_X_train = df_TRAIN[["usd_goal_real_log10", "category"]]

# TEST data
df_y_test = df_TEST[["state_dummy"]]
df_X_test = df_TEST[["usd_goal_real_log10", "category"]]


# df_X_train = pd.DataFrame(X_train,
#                          columns=["usd_goal_real_log10", "category"])
# df_y_train = pd.DataFrame(y_train,
#                          columns=["state_dummy"])

# df_X_test = pd.DataFrame(X_test,
#                          columns=["usd_goal_real_log10", "category"])
# df_y_test = pd.DataFrame(y_test,
#                          columns=["state_dummy"])


# Create dummy variables for category using train data
# Replace category to category_success_rate
category_success_rate = {}
df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
df_category_all_count = df_X_train['category'].value_counts()
for category in df_category_all_count.keys():
    category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)



X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
X_test = df_X_test[["usd_goal_real_log10", "category_dummy"]].values

y_train = df_y_train[["state_dummy"]].values
y_test = df_y_test[["state_dummy"]].values

# Normaliztion
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
clf.fit(X_train, y_train)

# Weight
w0 = clf.intercept_[0]
w1 = clf.coef_[0, 0]
w2 = clf.coef_[0, 1]
print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}'.format(w0, w1, w2))


# Predict labels
y_est_test = clf.predict(X_test)

# Log-likelihood
log_likelihood = - log_loss(y_test, y_est_test)       
print('Test Log-likelihood = {:.3f}'.format(log_likelihood))

# Accuracy
accuracy = accuracy_score(y_test, y_est_test)  
print('Test Accuracy = {:.3f}%'.format(100 * accuracy))
print()

plot_decision_regions(X_test, y_test.flatten(), classifier=clf)
plt.title('(Final Test)  L1 alpha = %s' %alpha)
plt.xlabel('usd_goal_real_log10_stdsc')
plt.ylabel('category_dummy_stdsc')
plt.axes().set_aspect('equal', 'datalim')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

    
penalty = 'l2'

alphas_multiply = np.array(range(-8,0))
alphas = 10.0 ** alphas_multiply

L2_accuracy = []
L2_log_likelihood = []
L2_weight_abs_max = []
L2_weight_abs_min = []


for alpha in alphas:
    
    print('='*100)
    print('penalty =', penalty)
    print('alpha =', alpha)
    print()

    y = df_TRAIN["state_dummy"].values
    X = df_TRAIN[["usd_goal_real_log10", "category"]].values

    n_split = 5 # Number of group

    cross_valid_log_likelihood = 0
    cross_valid_accuracy = 0
    split_num = 1

    # Cross Validation
    for train_idx, valid_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):
        X_train, y_train = X[train_idx], y[train_idx] # Train data
        X_valid, y_valid = X[valid_idx], y[valid_idx] # Validation data

        df_X_train = pd.DataFrame(X_train,
                                 columns=["usd_goal_real_log10", "category"])
        df_y_train = pd.DataFrame(y_train,
                                 columns=["state_dummy"])

        df_X_valid = pd.DataFrame(X_valid,
                                 columns=["usd_goal_real_log10", "category"])
        df_y_valid = pd.DataFrame(y_valid,
                                 columns=["state_dummy"])




        # Create dummy variables for category using train data
        # Replace category to category_success_rate
        category_success_rate = {}
        df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
        df_category_all_count = df_X_train['category'].value_counts()
        for category in df_category_all_count.keys():
            category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
        df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
        df_X_valid['category_dummy'] = df_X_valid['category'].replace(category_success_rate)


        print("Fold %s"%split_num)

        X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
        X_valid = df_X_valid[["usd_goal_real_log10", "category_dummy"]].values
        
        # Normaliztion
        stdsc = StandardScaler()
        X_train = stdsc.fit_transform(X_train)
        X_valid = stdsc.transform(X_valid)

        clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
        clf.fit(X_train, y_train)

        # Weight
        w0 = clf.intercept_[0]
        w1 = clf.coef_[0, 0]
        w2 = clf.coef_[0, 1]
        print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}'.format(w0, w1, w2))


        # Predict labels
        y_est_valid = clf.predict(X_valid)

        # Log-likelihood
        log_likelihood = - log_loss(y_valid, y_est_valid)    
        cross_valid_log_likelihood += log_likelihood    
        print('Log-likelihood = {:.3f}'.format(log_likelihood))

        # Accuracy
        accuracy = accuracy_score(y_valid, y_est_valid)
        cross_valid_accuracy += accuracy   
        print('Accuracy = {:.3f}%'.format(100 * accuracy))
        print()
        
        if split_num == n_split:
            plot_decision_regions(X_valid, y_valid, classifier=clf)
            plt.title('(Fold %s)  L2 alpha = %s' %(split_num,alpha))
            plt.xlabel('usd_goal_real_log10_stdsc')
            plt.ylabel('category_dummy_stdsc')
            plt.axes().set_aspect('equal', 'datalim')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

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
plt.show()
plt.plot(alphas_multiply, L2_log_likelihood, marker='o')
plt.title("L2 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Log-likelihood")
plt.show()
plt.plot(alphas_multiply, L2_weight_abs_max, marker='o', label='Weight_abs_max')
plt.plot(alphas_multiply, L2_weight_abs_min, marker='o', label='Weight_abs_min')
plt.title("L2 alpha")
plt.xlabel("Log10(alpha)")
plt.ylabel("Weight_abs_max_min")
plt.legend()
plt.show()
penalty = 'l2'

# Best Parameter
alpha = 1e-3


print('penalty =', penalty)
print('alpha =', alpha)
print()

# TRAIN data
df_y_train = df_TRAIN[["state_dummy"]]
df_X_train = df_TRAIN[["usd_goal_real_log10", "category"]]

# TEST data
df_y_test = df_TEST[["state_dummy"]]
df_X_test = df_TEST[["usd_goal_real_log10", "category"]]


# df_X_train = pd.DataFrame(X_train,
#                          columns=["usd_goal_real_log10", "category"])
# df_y_train = pd.DataFrame(y_train,
#                          columns=["state_dummy"])

# df_X_test = pd.DataFrame(X_test,
#                          columns=["usd_goal_real_log10", "category"])
# df_y_test = pd.DataFrame(y_test,
#                          columns=["state_dummy"])


# Create dummy variables for category using train data
# Replace category to category_success_rate
category_success_rate = {}
df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
df_category_all_count = df_X_train['category'].value_counts()
for category in df_category_all_count.keys():
    category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)



X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
X_test = df_X_test[["usd_goal_real_log10", "category_dummy"]].values

y_train = df_y_train[["state_dummy"]].values
y_test = df_y_test[["state_dummy"]].values

# Normaliztion
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

clf = SGDClassifier(loss='log', penalty=penalty, alpha=alpha, max_iter=100, fit_intercept=True, random_state=1234)
clf.fit(X_train, y_train)

# Weight
w0 = clf.intercept_[0]
w1 = clf.coef_[0, 0]
w2 = clf.coef_[0, 1]
print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}'.format(w0, w1, w2))


# Predict labels
y_est_test = clf.predict(X_test)

# Log-likelihood
log_likelihood = - log_loss(y_test, y_est_test)       
print('Test Log-likelihood = {:.3f}'.format(log_likelihood))

# Accuracy
accuracy = accuracy_score(y_test, y_est_test)  
print('Test Accuracy = {:.3f}%'.format(100 * accuracy))
print()

plot_decision_regions(X_test, y_test.flatten(), classifier=clf)
plt.title('(Final Test)  L2 alpha = %s' %alpha)
plt.xlabel('usd_goal_real_log10_stdsc')
plt.ylabel('category_dummy_stdsc')
plt.axes().set_aspect('equal', 'datalim')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

    
kernel = 'linear'
# Cs = [0.001, 0.01, 0.1, 1, 10]
Cs = np.logspace(-8, 3, 12, base=10)

SVM_accuracy = []
SVM_log_likelihood = []
SVM_weight_abs_max = []
SVM_weight_abs_min = []


for C in Cs:
    
    print('='*100)
    print('kernel =', kernel)
    print('C =', C)
    print()
    
    y = df_TRAIN["state_dummy"].values
    X = df_TRAIN[["usd_goal_real_log10", "category"]].values

    # Holdout method
    test_size = 0.2        # 20%
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=1234) # Holdout


    df_X_train = pd.DataFrame(X_train,
                             columns=["usd_goal_real_log10", "category"])
    df_y_train = pd.DataFrame(y_train,
                             columns=["state_dummy"])

    df_X_valid = pd.DataFrame(X_valid,
                             columns=["usd_goal_real_log10", "category"])
    df_y_valid = pd.DataFrame(y_valid,
                             columns=["state_dummy"])


    # Create dummy variables for category using train data
    # Replace category to category_success_rate
    category_success_rate = {}
    df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
    df_category_all_count = df_X_train['category'].value_counts()
    for category in df_category_all_count.keys():
        category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
    df_X_valid['category_dummy'] = df_X_valid['category'].replace(category_success_rate)

    X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
    X_valid = df_X_valid[["usd_goal_real_log10", "category_dummy"]].values
    
    # Normaliztion
    stdsc = StandardScaler()
    X_train = stdsc.fit_transform(X_train)
    X_valid = stdsc.transform(X_valid)

    clf = SVC(C=C,kernel=kernel, max_iter=2000, random_state=1234)
    clf.fit(X_train, y_train)

    # Predict labels
    y_est_valid = clf.predict(X_valid)

    # Log-likelihood
    holdout_log_likelihood = - log_loss(y_valid, y_est_valid)         

    # Accuracy
    holdout_accuracy = accuracy_score(y_valid, y_est_valid)
    
    plot_decision_regions(X_valid, y_valid, classifier=clf)
    plt.title('(Holdout)  SVM Linear C = %s' %C)
    plt.xlabel('usd_goal_real_log10_stdsc')
    plt.ylabel('category_dummy_stdsc')
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Generalization performance
    final_log_likelihood = holdout_log_likelihood
    print("Holdout Log-likelihood = %s"%round(final_log_likelihood, 3))
    final_accuracy = holdout_accuracy
    print('Holdout Accuracy = {:.3f}%'.format(100 * final_accuracy))
    
    SVM_accuracy.append(final_accuracy)
    SVM_log_likelihood.append(final_log_likelihood)
    SVM_weight_abs_max.append(np.max(np.abs(clf.coef_)))
    SVM_weight_abs_min.append(np.min(np.abs(clf.coef_)))
    
plt.plot(Cs, SVM_accuracy, marker='o')
plt.title("SVM kernel = linear")
plt.xlabel("C")
plt.xscale('log')
plt.ylabel("Accuracy")
plt.show()
plt.plot(Cs, SVM_log_likelihood, marker='o')
plt.title("SVM kernel = linear")
plt.xlabel("C")
plt.xscale('log')
plt.ylabel("Log-likelihood")
plt.show()
plt.plot(Cs, SVM_weight_abs_max, marker='o', label='Weight_abs_max')
plt.plot(Cs, SVM_weight_abs_min, marker='o', label='Weight_abs_min')
plt.title("SVM kernel = linear")
plt.xlabel("C")
plt.xscale('log')
plt.ylabel("Weight_abs_max_min")
plt.legend()
plt.show()
kernel = 'linear'

# Best Parameter
C = 1e-4


print('kernel =', kernel)
print('C =', C)
print()

# TRAIN data
df_y_train = df_TRAIN[["state_dummy"]]
df_X_train = df_TRAIN[["usd_goal_real_log10", "category"]]

# TEST data
df_y_test = df_TEST[["state_dummy"]]
df_X_test = df_TEST[["usd_goal_real_log10", "category"]]

# Create dummy variables for category using train data
# Replace category to category_success_rate
category_success_rate = {}
df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
df_category_all_count = df_X_train['category'].value_counts()
for category in df_category_all_count.keys():
    category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)



X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
X_test = df_X_test[["usd_goal_real_log10", "category_dummy"]].values

y_train = df_y_train[["state_dummy"]].values
y_test = df_y_test[["state_dummy"]].values

# print(X_test.shape)
# print(y_test.flatten().shape)

# print(X_valid.shape)
# print(y_valid.shape)

# Normaliztion
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

clf = SVC(C=C,kernel=kernel, max_iter=2000, random_state=1234)
clf.fit(X_train, y_train)

# Predict labels
y_est_test = clf.predict(X_test)

# Log-likelihood
log_likelihood = - log_loss(y_test, y_est_test)       
print('Test Log-likelihood = {:.3f}'.format(log_likelihood))

# Accuracy
accuracy = accuracy_score(y_test, y_est_test)  
print('Test Accuracy = {:.3f}%'.format(100 * accuracy))
print()

plot_decision_regions(X_test, y_test.flatten(), classifier=clf)
plt.title('(Final Test)  SVM Linear C = %s' %C)
plt.xlabel('usd_goal_real_log10_stdsc')
plt.ylabel('category_dummy_stdsc')
plt.axes().set_aspect('equal', 'datalim')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

    
kernel = 'rbf'
Cs = np.logspace(-4, 2, 7, base=10)
gammas = np.logspace(-10, -4, 7, base=10)

SVM_accuracy = []
SVM_log_likelihood = []
SVM_weight_abs_max = []
SVM_weight_abs_min = []


for C in Cs:
    for gamma in gammas:

        print('='*100)
        print('kernel =', kernel)
        print('C =', C)
        print('gamma =', gamma)
        print()

        y = df_TRAIN["state_dummy"].values
        X = df_TRAIN[["usd_goal_real_log10", "category"]].values

        # Holdout method
        test_size = 0.2        # 20%
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=1234) # Holdout


        df_X_train = pd.DataFrame(X_train,
                                 columns=["usd_goal_real_log10", "category"])
        df_y_train = pd.DataFrame(y_train,
                                 columns=["state_dummy"])

        df_X_valid = pd.DataFrame(X_valid,
                                 columns=["usd_goal_real_log10", "category"])
        df_y_valid = pd.DataFrame(y_valid,
                                 columns=["state_dummy"])


        # Create dummy variables for category using train data
        # Replace category to category_success_rate
        category_success_rate = {}
        df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
        df_category_all_count = df_X_train['category'].value_counts()
        for category in df_category_all_count.keys():
            category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
        df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
        df_X_valid['category_dummy'] = df_X_valid['category'].replace(category_success_rate)

        X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
        X_valid = df_X_valid[["usd_goal_real_log10", "category_dummy"]].values

        # Normaliztion
        stdsc = StandardScaler()
        X_train = stdsc.fit_transform(X_train)
        X_valid = stdsc.transform(X_valid)

        clf = SVC(C=C,kernel=kernel, gamma=gamma, max_iter=2000, random_state=1234)
        clf.fit(X_train, y_train)

        # Predict labels
        y_est_valid = clf.predict(X_valid)

        # Log-likelihood
        holdout_log_likelihood = - log_loss(y_valid, y_est_valid)         

        # Accuracy
        holdout_accuracy = accuracy_score(y_valid, y_est_valid)

        plot_decision_regions(X_valid, y_valid, classifier=clf)
        plt.title('(Holdout)  SVM RBF C = %s, gamma = %s' %(C, gamma))
        plt.xlabel('usd_goal_real_log10_stdsc')
        plt.ylabel('category_dummy_stdsc')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        # Generalization performance
        final_log_likelihood = holdout_log_likelihood
        print("Holdout Log-likelihood = %s"%round(final_log_likelihood, 3))
        final_accuracy = holdout_accuracy
        print('Holdout Accuracy = {:.3f}%'.format(100 * final_accuracy))

        SVM_accuracy.append(final_accuracy)
        SVM_log_likelihood.append(final_log_likelihood)

    
SVM_accuracy_2d = np.array(SVM_accuracy).reshape(7, 7)
Cs_str = list(map(str, Cs))
gammas_str = list(map(str, gammas))
df_SVM_accuracy_2d = pd.DataFrame(data=SVM_accuracy_2d, index=Cs_str, columns=gammas_str)
df_SVM_accuracy_2d
sns.heatmap(df_SVM_accuracy_2d, cmap='rainbow')
plt.ylabel("C")
plt.xlabel("gamma")
plt.title('Accuracy')
plt.show()
kernel = 'rbf'

# Best Parameter
C = 100
gamma = 1e-8


print('kernel =', kernel)
print('C =', C)
print('gamma =', gamma)
print()

# TRAIN data
df_y_train = df_TRAIN[["state_dummy"]]
df_X_train = df_TRAIN[["usd_goal_real_log10", "category"]]

# TEST data
df_y_test = df_TEST[["state_dummy"]]
df_X_test = df_TEST[["usd_goal_real_log10", "category"]]

# Create dummy variables for category using train data
# Replace category to category_success_rate
category_success_rate = {}
df_category_successful_count = df_X_train['category'][df_y_train['state_dummy'] == 1].value_counts()
df_category_all_count = df_X_train['category'].value_counts()
for category in df_category_all_count.keys():
    category_success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
df_X_train['category_dummy'] = df_X_train['category'].replace(category_success_rate)
df_X_test['category_dummy'] = df_X_test['category'].replace(category_success_rate)



X_train = df_X_train[["usd_goal_real_log10", "category_dummy"]].values
X_test = df_X_test[["usd_goal_real_log10", "category_dummy"]].values

y_train = df_y_train[["state_dummy"]].values
y_test = df_y_test[["state_dummy"]].values

# print(X_test.shape)
# print(y_test.flatten().shape)

# print(X_valid.shape)
# print(y_valid.shape)

# Normaliztion
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

clf = SVC(C=C,kernel=kernel, gamma=gamma, max_iter=2000, random_state=1234)
clf.fit(X_train, y_train)

# Predict labels
y_est_test = clf.predict(X_test)

# Log-likelihood
log_likelihood = - log_loss(y_test, y_est_test)       
print('Test Log-likelihood = {:.3f}'.format(log_likelihood))

# Accuracy
accuracy = accuracy_score(y_test, y_est_test)  
print('Test Accuracy = {:.3f}%'.format(100 * accuracy))
print()

plot_decision_regions(X_test, y_test.flatten(), classifier=clf)
plt.title('(Final Test)  SVM RBF C = %s, gamma = %s' %(C, gamma))
plt.xlabel('usd_goal_real_log10_stdsc')
plt.ylabel('category_dummy_stdsc')
plt.axes().set_aspect('equal', 'datalim')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

    
