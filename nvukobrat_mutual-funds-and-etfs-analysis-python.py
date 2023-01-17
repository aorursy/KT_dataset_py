import time

import random



import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from IPython.display import Markdown, display

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn import linear_model, svm



%matplotlib inline

rnd_seed_state = 1
def pre_config():

    """

    Does initial general pre-execution configuration.

    """

    plt.rcParams.update({'font.size': 20})

    random.seed(rnd_seed_state)
pre_config()
def load_dataset():

    """

    Loads ETFs and Mutual Fund's datasets.

    

    Returns:

        ETF and MF DataFrames.

    """

    df_etf = pd.read_csv('/kaggle/input/mutual-funds-and-etfs/ETFs.csv', index_col=False)

    df_mf = pd.read_csv('/kaggle/input/mutual-funds-and-etfs/Mutual Funds.csv', index_col=False)



    return df_etf, df_mf
df_etf, df_mf = load_dataset()
def types(df):

    """

    Prints DataFrame attribute types (numerical or 

    categorical) to the stdout.

    

    Arguments:

        df: Dataset pandas DataFrame.        

    """

    for c, t in zip(df, df.dtypes):

        if np.issubdtype(t, np.number):

            print("Num", c, t)

        else:

            print("Cat", c, t)
# types(df_etf)
def describe(df):

    """

    Describes in more detail each attribute of 

    the referent DataFrame. These descriptions 

    differ based on the attribute type.

    

    Arguments:

        df: Dataset pandas DataFrame.

    """

    print("DataFrame summary:")

    print(df.info())

    print()



    print("Columns summary:")

    for col in list(df):

        print(col)

        print(df[col].dropna().describe())

        print(df[col].dropna().value_counts())

        print()
# describe(df_etf)
def hist_bar_plot(df, rc_num=(2, 6), size=(16 * 3, 9 * 3), unique=True):

    """

    Creates histogram and bar visualizations for 

    a given DataFrame. Depending on the DataFrame 

    attribute type, numerical or categorical, it 

    creates a histogram or bar plot respectively.

    

    Arguments:

        df: Dataset pandas DataFrame.

        rc_num: Number of rows and columns.

        size: Size of the plot.

        unique: If True, it reduces the number of 

        attributes that will be visualized using 

        the number of unique values as a heuristic.

    """

    # Chose adequate columns for visualization

    if unique:

        num_unique_col = df.nunique()

        df = df[[col for col in df if 1 < num_unique_col[col] < 50]]



    # Create Hist and Bar plots

    plt.figure(figsize=size)

    for i, col in enumerate(df):

        plt.subplot(rc_num[0], rc_num[1], i + 1)

        df_column = df[col]

        if np.issubdtype(type(df_column.iloc[0]), np.number):

            df_column.hist()

        else:

            df_column.value_counts().plot.bar()

        plt.title(col, fontsize=25)

        plt.xticks(rotation=30, fontsize=30)

        plt.yticks(fontsize=30)

        plt.ylabel('counts')



        if i == 10:

            break

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    plt.show()
hist_bar_plot(df_etf)
def correlation_clean(df, threshold=(-0.9, -0.3, 0.3, 0.9)):

    """

    This dataset cleaning type reduces the number of 

    columns (attributes) that will be used in future

    analysis or model training. Reduction works by 

    focusing only on columns that have a greater 

    correlation (positive or negative) with other 

    columns. Having this, only high important attributes 

    are left to be utilized. 



    One of the reasons is that in large dimension datasets, 

    it takes too much time to train a model, and doesn't 

    guarantee that the model will have high accuracy. Also, 

    visualization in slow and through to understand in high 

    dimension datasets.

    

    Arguments:

        df: Dataset pandas DataFrame.

        threshold:  Threshold for choosing columns based 

        on correlation score.

    

    Returns:

        Cleaned dataset with high-correlated columns.

    """

    # Fix for non-numerical (or too many nan) columns

    col_corr = {

        # ETFs

        'fund_name', 'fund_extended_name', 'category', 'fund_family', 'net_assets', 'legal_type', 'investment', 'size',

        'currency', 'rating_us_government', 'fund_treynor_ratio_3years', 'category_treynor_ratio_5years',

    }

    try:

        for col in col_corr:

            del df[col]

    except KeyError:

        pass  # legal_type



    # Chose numerical columns

    corr_matrix = df.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if (not threshold[0] <= corr_matrix.iloc[i, j] <= threshold[1] and

                not threshold[2] <= corr_matrix.iloc[i, j] <= threshold[3]

            ) and (corr_matrix.columns[j] not in col_corr):

                colname = corr_matrix.columns[i]

                col_corr.add(colname)

                if colname in df.columns:

                    del df[colname]



    return df
df_etf = correlation_clean(df_etf)
def corr_mtx(df, fig_size=50, unique=True):

    """

    Correlation measures both the strength and direction of 

    the linear relationship between two variables. Depending 

    on the data, correlations could be calculated in a 

    couple of different ways. The most used way is to use 

    the Pearson product-moment correlation coefficient.

    

    Arguments:

        df: Dataset pandas DataFrame.

        fig_size: Size of the plot.

        unique: If True, it reduces the number of 

        attributes that will be visualized using 

        the number of unique values as a heuristic.

    """

    sns.set(font_scale=4.0)

    # Chose adequate columns for visualization

    if unique:

        num_unique_col = df.nunique()

        df = df[[col for col in df if num_unique_col[col] > 1]]



    f, ax = plt.subplots(figsize=(fig_size, fig_size))

    corr = df.corr()

    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

                square=True, ax=ax)

    plt.show()

    sns.set(font_scale=2.0)
corr_mtx(df_etf, unique=False)
def scatter_mtx(df, fig_size=50, cap=10):

    """

    Creates a grid of Axes such that each numeric 

    variable in data will be shared in the y-axis 

    across a single row and in the x-axis across 

    a single column. The diagonal Axes are treated 

    differently, drawing a plot to show the univariate 

    distribution of the data for the variable in 

    that column.

    

    Arguments:

        df: Dataset pandas DataFrame.

        fig_size: Size of the plot.

        cap: Max plots to be plot.

    """

    df = df.select_dtypes(np.number)

    if len(list(df)) > cap:

        df = df[list(df)[:cap]]

    scatter_matrix = pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(fig_size, fig_size), diagonal='kde')

    for ax in scatter_matrix.ravel():

        ax.set_xlabel(ax.get_xlabel(), fontsize=30, rotation=90)

        ax.set_ylabel(ax.get_ylabel(), fontsize=30, rotation=0)

    plt.show()
scatter_mtx(df_etf)
# types(df_mf)
# describe(df_mf)
hist_bar_plot(df_mf)
df_mf = correlation_clean(df_mf, threshold=(-0.9, -0.6, 0.6, 0.9))
corr_mtx(df_mf, unique=False)
scatter_mtx(df_mf, cap=20)
df_etf, df_mf = load_dataset() # Load again because of 'correlation_clean' function
def gaussian_clean(df, dataset_type):

    """

    Prepares a dataset for the training and inference 

    phase. The cleaning phase consists of:

    - Replacing all different NaN values with single 

    consistent NaN indicator

    - Removing columns with low-value information (ones 

    that have many unique values i.e. fund name)

    - Separation of categorical and numerical  attributes

    - Preparing categorical columns:

      - Populate NaN values

      - Label encode columns

      - One-hot encode columns

    - Preparing numerical columns:

      - Determine quantile and standard deviation of 

      each column

      - Populate NaN values using the Normal (Gaussian) 

      distribution

    

    Arguments:

        df: Dataset pandas DataFrame.

        dataset_type: etf or mf.

        

    Returns:

        Cleaned dataset.

    """

    # Fix NaN

    df.replace('', 'NaN', inplace=True)

    df.replace('nan', 'NaN', inplace=True)

    df.replace('NaN', np.nan, inplace=True)



    # As consequence of too many NaN values

    migrate_columns = [

        'fund_treynor_ratio_3years',

    ]

    if dataset_type == 'etf':

        migrate_columns += [

            'category_treynor_ratio_5years',

        ]

    elif dataset_type == 'mf':

        migrate_columns += [

            'price_cashflow',

            'price_sales',

            'price_earnings',

            'median_market_cap',

            'fund_treynor_ratio_5years',

            'fund_treynor_ratio_10years',

        ]



    # Join together numerical columns

    num_mean = df.select_dtypes(np.number)

    for col in migrate_columns:

        num_mean = num_mean.join(pd.to_numeric(df[col], errors="coerce"))



    # Columns with low-value information (ignored during initial analysis)

    low_info_categorical_columns = [

        'category',

        'currency',

        'fund_extended_name',

        'fund_family',

        'fund_name',

    ]

    if dataset_type == 'etf':

        low_info_categorical_columns += [

            'legal_type',

        ]

    elif dataset_type == 'mf':

        low_info_categorical_columns += [

            'inception_date',

        ]



    # Join together categorical (without low-value) columns

    str_mean = df[df.columns.difference(num_mean.columns)]

    for col in low_info_categorical_columns:

        str_mean = str_mean.drop(col, axis=1)



    # Populate empty values in categorical columns

    for col in list(str_mean):

        unique = str_mean[col].unique()

        unique = [x for x in unique if str(x) != 'nan']

        for i, value in enumerate(str_mean[col]):

            if value is np.nan:

                choice = random.choice(unique)

                str_mean[col].iloc[i] = choice



    # Encode categorical columns

    le = LabelEncoder()

    for col in str_mean:

        df[col] = le.fit_transform(str_mean[col])



    # One-hot encode categorical columns

    ohe = OneHotEncoder()

    for col in str_mean:

        unique = str_mean[col].unique()

        unique = [col + "_" + x for x in unique if str(x) != 'nan']

        enc_df = pd.DataFrame(ohe.fit_transform(df[[col]]).toarray(), columns=unique)

        df = df.drop(col, axis=1)

        df = df.join(enc_df)



    # Populate numerical columns

    mu = num_mean.quantile(0)

    sigma = num_mean.std(axis=0)

    for col in num_mean:

        stack = num_mean[col]

        null_stack = stack[pd.isnull(stack)]

        ran = np.random.normal(mu[col], sigma[col], len(null_stack))

        stack.loc[null_stack.index] = ran

        df[col] = stack.values



    # Clean

    for col in low_info_categorical_columns:

        df = df.drop(col, axis=1)



    return df
df_etf = gaussian_clean(df_etf, 'etf')
df_mf = gaussian_clean(df_mf, 'mf')
hist_bar_plot(df_etf)
hist_bar_plot(df_mf)
regressors = [

#     svm.SVR(),

    linear_model.SGDRegressor(),

    linear_model.BayesianRidge(),

    linear_model.LassoLars(),

#     linear_model.ARDRegression(),

    linear_model.PassiveAggressiveRegressor(),

#     linear_model.TheilSenRegressor(),

    linear_model.LinearRegression(),

]
def dataset_split(df, index_col):

    """

    Splits dataset into train and test sets.

    

    Arguments:

        df: Dataset pandas DataFrame.

        index_col: Referent prediction column.

    

    Returns:

        Train and test sets.

    """

    train, test = train_test_split(df, test_size=0.2, random_state=rnd_seed_state)

    y_train = train[index_col].copy()

    del train[index_col]



    y_test = test[index_col].copy()

    del test[index_col]



    return train, test, y_train, y_test
x_train, x_test, y_train, y_test = dataset_split(df_etf, index_col="ytd_return")
def run_models(regressors, x_train, x_test, y_train, y_test, res):

    """

    Trains and run inference on different models.

    

    Arguments:

        regressors: List of all requested regression models.

        x_train: X training data.

        x_test: X testing data.

        y_train: Y (index) training data.

        y_test: Y (index) testing data.

        res: Dictionary for storing results.

    """

    print("Running models...")

    for model in regressors:

        start_time = time.time()

        model_name = type(model).__name__

        print("\tModel: {}".format(model_name))

        clf = model

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        res[model_name] = {}

        result_metrics(y_test, y_pred, res[model_name])

        print("\tExecution time: %s seconds\n" % (round((time.time() - start_time), 3)))
def result_metrics(actual, predicted, res, print_adjust=50):

    """

    Calculates metrics from different models. Used metrics

    are explained in following sections:

    

    ### Explained variance score

    If \hat{y} is the estimated target output,

    y the corresponding (correct) target output, and Var is Variance, the

    square of the standard deviation, then the explained variance is estimated

    as follow:



    explained\_{}variance(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}



    The best possible score is 1.0, lower values are worse.



    ### Max error

    The max_error function computes the maximum residual error, a metric that captures

    the worst case error between the predicted value and the true value. In a perfectly

    fitted single output regression model, max_error would be 0 on the training set

    and though this would be highly unlikely in the real world, this metric shows

    the extent of error that the model had when it was fitted.



    If \hat{y}_i is the predicted value of the i-th sample, and y_i is the

    corresponding true value, then the max error is defined as:



    \text{Max Error}(y, \hat{y}) = max(| y_i - \hat{y}_i |)



    ### Mean absolute error

    The mean_absolute_error function computes mean absolute error,

    a risk metric corresponding to the expected value of the absolute error

    loss or l1-norm loss.



    If \hat{y}_i is the predicted value of the -th sample, and y_i is the

    corresponding true value, then the mean absolute error (MAE) estimated

    over n_{samples} is defined as:



    \text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|.



    ###  Mean squared error

    The mean_squared_error function computes mean square error,

    a risk metric corresponding to the expected value of the squared

    (quadratic) error or loss.



    If \hat{y}_i is the predicted value of the i-th sample,

    and y_i is the corresponding true value, then the mean squared

    error (MSE) estimated over n_{\text{samples}} is defined as:



    \text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.



    ### Median absolute error

    The median_absolute_error is particularly interesting because

    it is robust to outliers. The loss is calculated by taking the

    median of all absolute differences between the target and the

    prediction.



    If \hat{y}_i is the predicted value of the i-th sample and y_i

    is the corresponding true value, then the median absolute

    error (MedAE) estimated over n_{\text{samples}} is defined as:



    \text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid).



    ### R² score, the coefficient of determination

    The r2_score function computes the coefficient of determination, usually denoted as R².



    It represents the proportion of variance (of y) that has been

    explained by the independent variables in the model. It provides

    an indication of goodness of fit and therefore a measure of how

    well unseen samples are likely to be predicted by the model,

    through the proportion of explained variance.



    As such variance is dataset dependent, R² may not be meaningfully

    comparable across different datasets. Best possible score is 1.0

    and it can be negative (because the model can be arbitrarily worse).

    A constant model that always predicts the expected value of y,

    disregarding the input features, would get a R² score of 0.0.



    If \hat{y}_i is the predicted value of the i-th sample and y_i

    is the corresponding true value for total  samples, the estimated

    R² is defined as:



    R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}



    where \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} \epsilon_i^2



    Note that r2_score calculates unadjusted R² without correcting

    for bias in sample variance of y.



    ### Mean Poisson, Gamma, and Tweedie deviances

    The mean_tweedie_deviance function computes the mean Tweedie

    deviance error with a power parameter (p). This is a metric

    that elicits predicted expectation values of regression targets.



    If \hat{y}_i is the predicted value of the i-th sample, and y_i

    is the corresponding true value, then the mean Tweedie deviance

    error (D) for power p, estimated over n_{\text{samples}} is defined as:



    \begin{split}\text{D}(y, \hat{y}) = \frac{1}{n_\text{samples}}

    \sum_{i=0}^{n_\text{samples} - 1}

    \begin{cases}

    (y_i-\hat{y}_i)^2, & \text{for }p=0\text{ (Normal)}\\

    2(y_i \log(y/\hat{y}_i) + \hat{y}_i - y_i),  & \text{for}p=1\text{ (Poisson)}\\

    2(\log(\hat{y}_i/y_i) + y_i/\hat{y}_i - 1),  & \text{for}p=2\text{ (Gamma)}\\

    2\left(\frac{\max(y_i,0)^{2-p}}{(1-p)(2-p)}-

    \frac{y\,\hat{y}^{1-p}_i}{1-p}+\frac{\hat{y}^{2-p}_i}{2-p}\right),

    & \text{otherwise}

    \end{cases}\end{split}



    ### Reference

    https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics



    Arguments:

        actual: True (actual) value.

        predicted: Predicted value.

        print_adjust: Padding size to adjust output to STDOUT.

    """

    evs = metrics.explained_variance_score(actual, predicted)

    print("\t\tExplained variance score ".ljust(print_adjust, '.') + " {}".format(evs))

    res["EVS"] = evs



    me = metrics.max_error(actual, predicted)

    print("\t\tMax error ".ljust(print_adjust, '-') + " {}".format(me))

    res["ME"] = me



    mean_ae = metrics.mean_absolute_error(actual, predicted)

    print("\t\tMean absolute error ".ljust(print_adjust, '.') + " {}".format(mean_ae))

    res["MeanAE"] = mean_ae



    mse = metrics.mean_squared_error(actual, predicted)

    print("\t\tMean squared error ".ljust(print_adjust, '-') + " {}".format(mse))

    res["MSE"] = mse



    median_ae = metrics.median_absolute_error(actual, predicted)

    print("\t\tMedian absolute error ".ljust(print_adjust, '.') + " {}".format(median_ae))

    res["MedianAE"] = median_ae



    r2 = metrics.r2_score(actual, predicted)

    print("\t\tR² score, the coefficient of determination ".ljust(print_adjust, '-') + " {}".format(r2))

    res["R2"] = r2



    mtd = metrics.mean_tweedie_deviance(actual, predicted)

    print("\t\tMean Poisson, Gamma, and Tweedie deviances: ".ljust(50, '.') + " {}".format(mtd))

    res["MTD"] = mtd    
res = {

    'ETF': {},

    'MF': {},

}

run_models(regressors, x_train, x_test, y_train, y_test, res['ETF'])
x_train, x_test, y_train, y_test = dataset_split(df_mf, index_col="ytd_return")
run_models(regressors, x_train, x_test, y_train, y_test, res['MF'])
def visualize_results(res):

    """

    Visualize model results in different graph

    types.

    

    Arguments:

        res: Dictionary that contains model results.

    """

    # Restructure results

    data = []

    for k_fund, v_fund in res.items():

        for k_alg, v_alg in res[k_fund].items():

            for k_met, v_met in res[k_fund][k_alg].items():

                data.append([k_fund, k_alg, k_met, v_met])

    df = pd.DataFrame(data, columns=["Fund Type", "Model Name", "Metric", "Score"])



    # Normalize Score ranges

    min_max_scaler = MinMaxScaler()

    groups = df.groupby("Metric", as_index=False)

    for group in groups.groups.keys():

        val = groups.get_group(group)["Score"].values.reshape(-1, 1)

        scaled = min_max_scaler.fit_transform(val)

        df.loc[df["Metric"] == group, "Score"] = scaled



    visualize_with_focus(df, "Metric")

    visualize_with_focus(df, "Model Name")





def visualize_with_focus(df, focus_index):

    """

    Visualize data fith focus on specific column

    [Model Name or Metric].

    

    Arguments:

        df: Dataset pandas DataFrame.

        focus_index: Focus column.

    """

    print_md("### " + focus_index)

    fund_catplot(df, focus_index)

    fund_lineplot(df, focus_index)

    fund_barplot(df, focus_index)





def fund_catplot(df, focus_index):

    """

    Creates categorical bar plot for comparison of

    different fund types.

    

    Arguments:

        df: Dataset pandas DataFrame.

        focus_index: Focus column.

    """

    sns.catplot(

        x="Metric" if focus_index == "Model Name" else "Model Name",

        y="Score", hue="Fund Type", col=focus_index, kind="bar", data=df, col_wrap=3, height=10)

    plt.show()





def fund_lineplot(df, focus_index):

    """

    Creates line plot for comparison of

    different fund types.

    

    Arguments:

        df: Dataset pandas DataFrame.

        focus_index: Focus column.

    """

    plt.figure(figsize=(40, 30))

    for i, m in enumerate(df[focus_index].unique()):

        plt.subplot(3, 3, i + 1)

        df_group = df[df[focus_index] == m]

        lp = sns.lineplot(x="Metric" if focus_index == "Model Name" else "Model Name",

                          y="Score", hue="Fund Type", data=df_group)

        lp.set_title(m)

        plt.xticks(rotation=30)

    plt.show()





def fund_barplot(df, focus_index):

    """

    Creates bar plot for each fund type.

    

    Arguments:

        df: Dataset pandas DataFrame.

        focus_index: Focus column.

    """

    for fund_type in ["ETF", "MF"]:

        plt.figure(figsize=(40, 30))

        for i, mn in enumerate(df[focus_index].unique()):

            plt.subplot(3, 3, i + 1)



            # Plot config

            df_group = df[(df[focus_index] == mn) & (df["Fund Type"] == fund_type)]

            lp = sns.barplot(x="Metric" if focus_index == "Model Name" else "Model Name",

                             y="Score", hue="Fund Type", data=df_group)

            lp.set_title(mn)

            plt.xticks(rotation=30)

        plt.show()





def print_md(string):

    """

    Print for Jupyter MarkDown.

    

    Arguments:

        string: MarkDown content. 

    """

    display(Markdown(string))
visualize_results(res)
df_etf = correlation_clean(df_etf)
df_mf = correlation_clean(df_mf, threshold=(-0.9, -0.6, 0.6, 0.9))
res = {

    'ETF': {},

    'MF': {},

}

run_models(regressors, x_train, x_test, y_train, y_test, res['ETF'])
x_train, x_test, y_train, y_test = dataset_split(df_mf, index_col="ytd_return")
run_models(regressors, x_train, x_test, y_train, y_test, res['MF'])
visualize_results(res)
x_train, x_test, y_train, y_test = dataset_split(df_etf, index_col="net_assets")
res = {

    'ETF': {},

    'MF': {},

}

run_models(regressors, x_train, x_test, y_train, y_test, res['ETF'])
x_train, x_test, y_train, y_test = dataset_split(df_mf, index_col="net_assets")
run_models(regressors, x_train, x_test, y_train, y_test, res['MF'])
visualize_results(res)
df_etf = correlation_clean(df_etf)
df_mf = correlation_clean(df_mf, threshold=(-0.9, -0.6, 0.6, 0.9))
res = {

    'ETF': {},

    'MF': {},

}

run_models(regressors, x_train, x_test, y_train, y_test, res['ETF'])
x_train, x_test, y_train, y_test = dataset_split(df_mf, index_col="net_assets")
run_models(regressors, x_train, x_test, y_train, y_test, res['MF'])
visualize_results(res)