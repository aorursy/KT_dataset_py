import time

start_time = time.time()

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Notice there is a new datafile I have uploaded that we will use in this notebook
class Timing:

    """

    Utility class to time the notebook while running. 

    """

    def __init__(self, start_time):

        self.start_time = start_time

        self.counter = 0



    def timer(self, message=None):

        """

        Timing function that returns the time taken for this step since the starting time. Message is optional otherwise we use a counter. 

        """

        if message:

            print(f"{message} at {time.time()-self.start_time}")

        else:

            print(f"{self.counter} at {time.time()-self.start_time}")

            self.counter += 1

        return

    

timing = Timing(start_time)
def load_data(number_of_rows:int =None, purpose=None)->pd.DataFrame:

    """

    Returns a pandas DataFrame with the loan data inside

    number_of_rows: Controls the number of rows read in, default and maximum is 22,60,668 rows

    restriction: Restricts the columns read in to correct for information you should not have depending on the task at hand

        "time_of_issue": Returns only the data that the lender has access to during the issuing of the loan

    """

    root = "../input/lending-club-loan-data"

    use_cols= None

    if purpose not in [None, 'time_of_issue']:

        raise ValueError(f"Invalid Purpose {purpose}")

    if purpose:

        col_root = "../input/columns-available-at-time-of-loan"

        columnframe = pd.read_csv(os.path.join(col_root, purpose+".csv"))

        illegals = ['sec_app_fico_range_low ', 'sec_app_inq_last_6mths ', 'sec_app_earliest_cr_line ', 'revol_bal_joint ', 'sec_app_mths_since_last_major_derog ', 'sec_app_revol_util ', 'sec_app_collections_12_mths_ex_med ', 'sec_app_open_acc ', 'fico_range_low', 'sec_app_fico_range_high ', 'verified_status_joint', 'last_fico_range_low', 'sec_app_chargeoff_within_12_mths ', 'fico_range_high', 'total_rev_hi_lim \xa0', 'sec_app_mort_acc ', 'sec_app_num_rev_accts ', 'last_fico_range_high']

        use_cols = [x for x in list(columnframe['name']) if x not in illegals]







    path = os.path.join(root, "loan.csv")



    maximum_rows = 2260668

    if not number_of_rows:

        return pd.read_csv(path, low_memory=False, usecols=use_cols)

    else:

        if number_of_rows > maximum_rows or number_of_rows < 1:

            raise ValueError(f"Number of Rows Must be a Number between 1 and {data.shape[0]}")

        else:

            return pd.read_csv(path, low_memory=False, nrows=number_of_rows, usecols=use_cols)
data = load_data(number_of_rows=None, purpose="time_of_issue")
def investigate(data)->None:

    print(data.shape)

    print(data.info())

    print(data.describe())
investigate(data)
def type_list_generator(data, separated=False):

    """

    Prints out 3 list to store which columns are of which type.

    Interest rate can be in the list or not depending on the seperated variable

    """

    numericals = ['loan_amnt','funded_amnt','funded_amnt_inv', 'annual_inc','mort_acc','emp_length', 'int_rate']

    if separated:

        numericals.pop()

    strings = ['issue_d', 'zip_code']

    categoricals = [x for x in data.columns if x not in numericals and x not in strings] # ['term', 'grade', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'purpose', 'title', 'addr_state', 'initial_list_status', 'application_type', 'disbursement_method']

    return numericals, strings, categoricals
numericals, strings, categoricals = type_list_generator(data)
def drop_nan_columns(data, ratio=1.0)->pd.DataFrame:

    """

    The ratio parameter (0.0<=ratio<1.0) lets you drop columns which has 'ratio'% of nans. (i.e if ratio is 0.8 then all columns with 80% or more entries being nan get dropped)

    Returns a new dataframe

    """

    col_list = []

    na_df = data.isna()

    total_size = na_df.shape[0]

    for col in na_df:

        a = na_df[col].value_counts()

        if False not in a.keys():

            col_list.append(col)

        elif True not in a.keys():

            pass

        else:

            if a[True]/total_size >= ratio:

                col_list.append(col)

    print(f"{len(col_list)} columns dropped- {col_list}")

    return data.drop(col_list, axis=1)
data = drop_nan_columns(data, ratio=0.5)
def investigate_nan_columns(data)->None:

    """

    Prints an analysis of the nans in the dataframe

    """

    col_dict = {}

    na_df = data.isna()

    total_size = na_df.shape[0]

    for col in na_df:

        a = na_df[col].value_counts()

        if False not in a.keys():

            col_dict[col] = 1.0

        elif True not in a.keys():

            pass

        else:

            col_dict[col] =  a[True]/total_size

    print(f"{col_dict}")

    return
investigate_nan_columns(data)
data['emp_title'].value_counts()
unemployed = ['unemployed', 'none', 'Unemployed', 'other', 'Other']

for item in unemployed:

    if item in data['emp_title']:

        print("Found It at ", item)
def handle_nans(data)->None:

    """

    Handle the nans induvidually per column

    emp_title: make Nan -> Unemployed

    emp_length: make Nan - > 10+ years this is both mode filling and value filing

    title: make Nan -> Other

    """

    data['emp_title'] = data['emp_title'].fillna("Unemployed")

    data['title'] = data['title'].fillna('Other')

    mode_cols = ['emp_length', 'annual_inc', 'mort_acc', 'zip_code']

    for col in mode_cols:

        data[col] = data[col].fillna(data[col].mode()[0])

    return

handle_nans(data)
any(data.isna().any()) # True iff there some NaN values anywhere in the dataset
investigate(data)
def handle_types(data, numericals, strings, categoricals):

    def helper_emp_length(x):

        if x == "10+ years": return 10

        elif x == "2 years": return 2

        elif x == "< 1 year": return 0

        elif x == "3 years": return 3

        elif x == "1 year": return 1

        elif x == "4 years": return 4

        elif x == "5 years": return 5

        elif x == "6 years": return 6

        elif x == "7 years": return 7

        elif x == "8 years": return 8

        elif x == "9 years": return 9

        else:

            return 10

    data['emp_length'] = data['emp_length'].apply(helper_emp_length)



    for category in categoricals:

        try:

            data[category] = data[category].astype('category')

        except:

            pass

    data['issue_d'] = data['issue_d'].astype('datetime64')

    return

handle_types(data, numericals, strings, categoricals)
def correlation_heatmap(data):

    corrmat = data.corr()

    sns.heatmap(corrmat, vmax=0.9, square=True)

    plt.title("Correlation Heatmap")

    plt.xlabel("Features")

    plt.ylabel("Features")

    plt.show()

    timing.timer("Heatmap")

    return



correlation_heatmap(data)
def distplot(data):

    """

    Reveals a positive skew

    """

    from scipy.stats import norm

    sns.distplot(data['int_rate'], fit=norm)

    plt.title("Distribution and Skew of Interest Rate")

    plt.xlabel("Interest Rate in %")

    plt.ylabel("Occurance in %")

    plt.show()

    timing.timer("Skew with distplot")

    return



distplot(data)
def boxplot(data):

    """

Creates 4 boxplots

            

    """

    fig, axes = plt.subplots(2,2) # create figure and axes

    col_list = ['annual_inc', 'loan_amnt', 'int_rate', 'emp_length']

    by_dict = {0: 'home_ownership', 1:"disbursement_method", 2:"verification_status", 3:"grade"}



    for i,el in enumerate(col_list):

        a = data.boxplot(el, by=by_dict[i], ax=axes.flatten()[i])



    #fig.delaxes(axes[1,1]) # remove empty subplot

    plt.tight_layout()

    plt.title("Various Boxplots")

    plt.show()

    timing.timer("Boxplot")

    return



boxplot(data)
def lines(data):

    """

    Employment length vs interest rate

    """

    sns.lineplot(x=data['emp_length'], y=data['int_rate'])

    plt.title("Employment Length vs Interest Rate")

    plt.xlabel("Employment Length in yrs")

    plt.ylabel("Interest Rate in %")

    plt.show()

    timing.timer("Lines")

    return



lines(data)
def scatter(data):

    """

    Scatter Sub_Grade vs Risk

    """

    info = data.copy()

    a = info.groupby('sub_grade').mean()

    

    sns.scatterplot(x=a.index, y=a['int_rate'])

    plt.title("Subgrade vs Interest Rate ScatterPlot")

    plt.xlabel("Subgrade")

    plt.ylabel("Interest Rate in %")

    plt.show()

    timing.timer("Scatter")

    return



scatter(data)
# 3D Scatterplot

def three_D_scatter(data):

    """

    Loan Amount vs Employment Length vs Interest Rate

    """

    from mpl_toolkits import mplot3d

    import numpy as np

    info = data[:1000]



    fig = plt.figure()

    ax = plt.axes(projection='3d')



    xs = info['loan_amnt']

    zs = info['emp_length']

    ys = info['int_rate']

    ax.scatter(xs, ys, zs, s=1, alpha=1)





    ax.set_xlabel('Loan Amount')

    ax.set_ylabel('Interest Rate')

    ax.set_zlabel('Employment Length')

    plt.title("3D Scatterplot")

    plt.show()

    timing.timer("3D Scatter")

    return



three_D_scatter(data)
# Violin Plot

def violin_plot(data):

    sns.violinplot(x="home_ownership", y="int_rate", data=data, hue="term")

    plt.title("Violin Plot")

    plt.xlabel("Home Ownership")

    plt.ylabel("Interest Rate in %")

    plt.show()

    timing.timer("Violin")

    return



violin_plot(data)
# Bubble Chart

def bubble_chart(data):

    info = data[:1000]

    sns.lmplot(x="loan_amnt", y="int_rate",data=info,  fit_reg=False,scatter_kws={"s": info['annual_inc']*0.005})

    plt.title("Bubble Chart")

    plt.xlabel("Loan Amount")

    plt.ylabel("Interest Rate in %")

    plt.show()

    timing.timer("bubble")

    return



bubble_chart(data)
def load_split_data(number_of_rows=None, purpose=None, column='int_rate', test_size=0.2):

    from sklearn.model_selection import train_test_split

    data = load_data(number_of_rows=number_of_rows, purpose=purpose)

    target = data[column]

    data.drop(column, axis=1, inplace=True)

    return train_test_split(data, target, test_size=test_size)
X_train, X_test, y_train, y_test = load_split_data(50000, purpose="time_of_issue")

numericals, strings, categoricals = type_list_generator(X_train, separated=True)
X_train = drop_nan_columns(X_train, ratio=0.5)

X_test = drop_nan_columns(X_test, ratio=0.5)

handle_nans(X_train)

handle_nans(X_test)

handle_types(X_train, numericals, strings, categoricals)

handle_types(X_test, numericals, strings, categoricals)

# For this notebook we will ignore the string variables, however there are ways to use them using other prepreocessing techniques if desired

X_train = X_train.drop(strings, axis=1)

X_test = X_test.drop(strings, axis=1)

timing.timer("Cleaned Data")
def manage_skews(train_target, test_target):

    """

    Applying Square Root in order

    """

    timing.timer("Unskewed Data")

    return np.sqrt(train_target), np.sqrt(test_target)



y_train, y_test = manage_skews(y_train, y_test)
def scale_numerical_data(X_train, X_test, numericals):

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    X_train[numericals] = sc.fit_transform(X_train[numericals])

    X_test[numericals] = sc.transform(X_test[numericals])

    timing.timer("Scaled Data")

    return



scale_numerical_data(X_train, X_test, numericals)
X_train['emp_title'].value_counts()
def shrink_categoricals(X_train, X_test, categoricals, top=25):

    """

    Mutatues categoricals to only keep the entries which are the top 25 of the daframe otherwise they become other

    """

    for category in categoricals:

        if category not in X_train.columns:

            continue

        tops = X_train[category].value_counts().index[:top]

        def helper(x):

            if x in tops:

                return x

            else:

                return "Other"

        X_train[category] = X_train[category].apply(helper)

        X_test[category] = X_test[category].apply(helper)

    timing.timer("Shrunk Categories")

    return

shrink_categoricals(X_train, X_test, categoricals)



X_train['emp_title'].value_counts()
def encode_categorical_data(X_train, X_test, categoricals):

    from sklearn.preprocessing import LabelEncoder

    for category in categoricals:

        if category not in X_train.columns:

            continue

        le = LabelEncoder()

        X_train[category] = le.fit_transform(X_train[category])

        X_test[category] = le.transform(X_test[category])

    timing.timer("Encoded Categoricals")

    return



encode_categorical_data(X_train, X_test, categoricals)
def dimensionality_reduction(X_train, X_test):

    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.95)

    X_train = pca.fit_transform(X_train)

    X_test = pca.transform(X_test)

    timing.timer("Dimensionality Reduced")

    return X_train, X_test



X_train, X_test = dimensionality_reduction(X_train, X_test)

X_train.shape
def random_forest(X_train, y_train, optimal=False):

    """

    Optimal = True returns an untrained model

    """

    from sklearn.ensemble import RandomForestRegressor

    if optimal:

        return RandomForestRegressor(n_estimators=120, max_depth=25, bootstrap=True, max_features=3)

    from sklearn.model_selection import GridSearchCV

 

    param_grid = [{'n_estimators':[60, 70, 80, 100, 120], 'max_depth':[15, 20, 25, None], 'bootstrap':[True, False], 'max_features':[None, 2, 3]}]

    forest = RandomForestRegressor()

    grid_search = GridSearchCV(forest, param_grid, cv=3, scoring="r2")

    grid_search.fit(X_train, y_train)

    timing.timer("Forest Grid Search Complete")

    final = grid_search.best_params_

    print(final)

    return grid_search.best_estimator_

    



def regression(X_train, y_train, optimal=False):

    """

    Optimal = True returns an untrained model

    """

    from sklearn.linear_model import ElasticNetCV

    if optimal:

        return ElasticNetCV(alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l1_ratio=0.0)

    from sklearn.model_selection import GridSearchCV



    elastic_net = ElasticNetCV()

    param_grid = {'alphas':[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], 'l1_ratio':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

    grid_search = GridSearchCV(elastic_net, param_grid, scoring="r2", cv=3)

    grid_search.fit(X_train, y_train)

    timing.timer("Regression Grid Search Complete")

    print(grid_search.best_params_)

    return grid_search.best_estimator_ 





def knn(X_train, y_train, optimal=False):

    """

    Optimal = True returns an untrained model

    """

    from sklearn.neighbors import KNeighborsRegressor

    if optimal:

        return KNeighborsRegressor(n_neighbors=10, weights='distance')

    

    from sklearn.model_selection import GridSearchCV

    

    model = KNeighborsRegressor()

    param_grid = {'n_neighbors':[2,4,6,8,10,12,14], 'weights':['uniform', 'distance']}

    grid_search = GridSearchCV(model, param_grid, scoring="r2", cv=3)

    grid_search.fit(X_train, y_train)

    timing.timer("Neighbors Grid Search Complete")

    print(grid_search.best_params_)

    return grid_search.best_estimator_ 



def svm(X_train, y_train, optimal=False):

    """

    Optimal = True returns an untrained model

    """

    from sklearn.svm import SVR

    if optimal:

        return SVR()

    from sklearn.model_selection import RandomizedSearchCV



    svr = SVR()



    param_grid = {'kernel':['rbf', 'sigmoid', 'poly', 'linear'], 'C':[0.8, 1.0, 1.2]}

    n_iter = 2

    rsv = RandomizedSearchCV(svr, param_grid, n_iter=n_iter, scoring="r2")

    rsv.fit(X_train, y_train)

    timing.timer("SVR Random Search Complete")

    final = rsv.best_params_

    print(final)

    return rsv.best_estimator_
"""

model_creators = [random_forest,regression, knn, svm]

models = []

for creator in (model_creators):

    models.append(creator(X_train, y_train))

    

for model in models:

    model.score(X_test, y_test)

"""
def ensemble(model_list):

    from sklearn.ensemble import VotingRegressor

    vtr = VotingRegressor(model_list)

    return vtr
class GradientBoost:

    def __init__(self, model_class, example, n_estimators=2):

        self.model_class = model_class

        self.parameters = example.get_params()

        self.n_estimators = n_estimators

        self.estimators = []

        for n in  range(n_estimators):

            model = model_class()

            model.set_params(**self.parameters)

            self.estimators.append(model)



    def fit_helper(self, X, y, i=0):

        if i >= len(self.estimators):

            return

        else:

            self.estimators[i].fit(X, y)

            preds = self.estimators[i].predict(X)

            error = y - preds

            self.fit_helper(X, error, i=i+1)



           

    def fit(self, X, y):

        self.fit_helper(X, y)



    def predict(self, X):

        prediction = self.estimators[0].predict(X)

        for estimator in self.estimators[1:]:

            prediction += estimator.predict(X)

        return prediction



    def score(self, X, y):

        from sklearn.metrics import r2_score

        preds = self.predict(X)

        return r2_score(y, preds)
order = {0: "rfr", 1:"lin_reg", 2:"knn", 3:"svr", 4:"vtr", 5:"gb"}

model_creators = [random_forest,regression, knn, svm]

model_list = []

models = []

for i, creator in enumerate(model_creators):

    model_list.append( (order[i] , creator(X_train, y_train, optimal=True) ) )

    models.append(creator(X_train, y_train, optimal=True))

    timing.timer(f"Appended Model {i}")

models.append(ensemble(model_list))

from sklearn.svm import SVR

models.append(GradientBoost(SVR, models[3], 2))
def train_and_test(models, order):

    for i, model in enumerate(models):

        model.fit(X_train, y_train)

        timing.timer(f"Finished Fitting model {i}")

    scores = []

    for model in models:

        scores.append(model.score(X_test, y_test))

    final = {}

    for score_no in range(len(scores)):

        final[order[score_no]] = scores[score_no]

    return final
train_and_test(models, order)
best_model = models[2]

# We take the first 20 inputs and compare the predictions with the outputs

truths = y_test[0:20]**2 # Squaring to undo the skew solution in order to truly reflect the data

preds = best_model.predict(X_test[0:20])**2

residual_error = truths - preds

print(residual_error)
plt.scatter(truths, preds)

plt.plot([7.0, 22], [7.0, 22], c = "red")

plt.title("Model Analysis")

plt.xlabel("Truth")

plt.ylabel("Prediction")

plt.show()
truths, preds