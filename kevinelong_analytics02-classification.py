# %%bash
# apt install graphviz
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, model_selection, preprocessing, tree
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
%%time
# Resampling data from minute interval to day
bit_df = pd.read_csv('../input/coinbase/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv',
  low_memory=False, error_bad_lines=True)
bit_df['Timestamp'] = bit_df.Timestamp.astype('int', errors='ignore')

# Convert unix time to datetime
bit_df['date'] = pd.to_datetime(bit_df.Timestamp, unit='s', errors='coerce')

# Reset index
bit_df = bit_df.set_index('date')

# Rename columns so easier to code
bit_df = bit_df.rename(columns={'Open':'open', 'High': 'hi', 'Low': 'lo',
                       'Close': 'close', 'Volume_(BTC)': 'vol_btc',
                       'Volume_(Currency)': 'vol_cur',
                       'Weighted_Price': 'wp', 'Timestamp': 'ts'})

# Coerce to numeric types
bit_df['hi'] = pd.to_numeric(bit_df.hi, errors='coerce')
bit_df['lo'] = pd.to_numeric(bit_df.lo, errors='coerce')
bit_df['close'] = pd.to_numeric(bit_df.close, errors='coerce')
bit_df['open'] = pd.to_numeric(bit_df.open, errors='coerce')
bit_df['ts'] = pd.to_numeric(bit_df.ts, errors='coerce')

# Resample and only use recent samples that aren't missing
bit_df = bit_df.resample('d').agg({'open': 'first', 'hi': 'mean',
    'lo': 'mean', 'close': 'last', 'vol_btc': 'sum',
    'vol_cur': 'sum', 'wp': 'mean', 'ts': 'min'}).iloc[-1000:]
bit_df['buy'] = (bit_df.close.shift(-1) > bit_df.close).astype(int)

# drop last row as it is not complete
bit_df = bit_df.iloc[:-1]
# Display the data
bit_df
# Show the data types
bit_df.dtypes
# Create a description of the data
bit_df.describe()
# Partition data in order to train the model
ignore = {'buy'}
cols = [c for c in bit_df.columns if c not in ignore]
X = bit_df[cols]
y = bit_df.buy
# Create model instance, train to create classifier
# Random state is used to seed the initial state of the model
dt_model = tree.DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

# Score against the earlier buy decision
dt_model.score(X, y)
dt_model.predict(X)
# Note that this goes to a Unix path
tree.export_graphviz(dt_model, out_file='/kaggle/working/tree1.dot',
                     feature_names=X.columns, class_names=['Sell', 'Buy'],
                    filled=True
                    )
# %%bash
# Export a visualization of what the algorithm thinks is impotant
# This doesn't run on Windows. Also requires that you have graphviz installed (not a Python module)

# %%bash
# dot -Tpng -o/kaggle/working/tree1.png /tmp/tree1.dot
dt_model.score(X, y)
# Show parameters which were used to create the model
dt_model
# Print a list of the most important parameters used in the creation of the model
print(sorted(zip(X.columns, dt_model.feature_importances_), key=lambda x:x[1], reverse=True))
# Partition out buy column (the data's buy column will be used to judge the model accuracy)
ignore = {'buy'}
cols = [c for c in bit_df.columns if c not in ignore]
X = bit_df[cols]
y = bit_df.buy
X_train, X_test, y_train, y_test = model_selection.\
    train_test_split(X, y, test_size=.3, random_state=42)
# Truncate the depth to which the classifier is allowed to grow
dt2 = tree.DecisionTreeClassifier(random_state=42, max_depth=3)
dt2.fit(X_train, y_train)
dt2.score(X_test, y_test)
# Export the decision tree to a visualization
tree.export_graphviz(dt2, out_file='/tmp/tree2.dot',
                     feature_names=X.columns, class_names=['Sell', 'Buy'],
                    filled=True
                    )
# %%bash
# %%bash
# dot -Tpng -o/data/analytics/img/tree2.png /tmp/tree2.dot
# Introduce Financial Measurements

def rsi(df, num_periods=14):
    """ Relative strength index: technical measure of whether a stock
      is strong or week based on closing prices of a recent trading pool.
    """
    prev = df.close.shift(1)
    change = (df.close - prev) / prev
    change = change.rolling(window=num_periods).mean().fillna(0)
    up, down = change.copy(), change.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    up2 = up.rolling(center=False, window=num_periods).mean()
    down2 = down.rolling(center=False, window=num_periods).mean()
    rs = (up2 / down2).fillna(0)
    res = (100 - 100/(1 + rs))

    return res.replace([np.inf], 0)


def stoc(df, num_periods=14):
    """ Stochastic Oscillator: a "momentum indicator" intended to predict
      whether a stock is on an upswing or downswing.
    """
    cur = df.close
    low = df.close.rolling(center=False, window=num_periods).min()
    high = df.close.rolling(center=False, window=num_periods).max()
    return (100 * (cur - low)/(high - low)).fillna(0)


def williams(df, num_periods=14):
    """ Buy/sell indicator.
        Williams %R ranges from -100 to 0. When its value is above -20,
        it indicates a sell signal and when its value is below -80, it indicates a buy signal.
    """
    cur = df.close#.iloc[-1]
    low = df.close.rolling(center=False, window=num_periods).min() #shift(-num_periods) .iloc[-num_periods:].min()
    high = df.close.rolling(center=False, window=num_periods).max() #df.close.iloc[-num_periods:].max()
    return (-100 * (high - cur) / (high - low)).fillna(-50)


def proc(df, num_periods=14):
    """ It measures the most recent change in price with respect to the price in n days ago.
        https://www.investopedia.com/terms/p/pricerateofchange.asp
    """
    cur = df.close
    prev = df.close.shift(-num_periods)
    return ((cur - prev)/(prev*100)).fillna(0)


def obv(df, vol='vol_btc'):
    """ On balance volume - Use volume flow to predict changes
    if close up add vol, if down subtract
    """
    # -1 if down 1 if up
    close_up_or_down = (bit_df.close.diff().le(0) * 2 - 1)
    obv = (close_up_or_down * bit_df[vol]).cumsum()

    return obv.fillna(0)    
# Code a new column with the metrics
for func in [rsi,
             stoc, williams, proc, obv]:
    bit_df[func.__name__] = func(bit_df)
# Exclude buy (used, to assess the accuracy of the model), generate outcome variable
ignore = {'buy'}
cols2 = [c for c in bit_df.columns if c not in ignore]
X = bit_df[cols2]
y = bit_df.buy

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.\
    train_test_split(X, y, test_size=.3, random_state=42)
# Create a decision tree classifier. Train and score.
dt3 = tree.DecisionTreeClassifier(random_state=42, max_depth=7)
dt3.fit(X_train, y_train)
dt3.score(X_test, y_test)
# Show the important columns
print(sorted(zip(X.columns, dt3.feature_importances_), key=lambda x:x[1], reverse=True))
# Create an alternative algorithm model and generate the overall
rf1 = ensemble.RandomForestClassifier(random_state=3)#, max_depth=7)
rf1.fit(X_train, y_train)
rf1.score(X_test, y_test)
rf1.score(X_train, y_train)
from sklearn.metrics import auc, confusion_matrix, roc_curve

def fig_with_title(ax, title, figkwargs):
    ''' Helper curve for plotting a figure
    '''
    if figkwargs is None:
        figkwargs = {}
    if not ax:
        fig = plt.figure(**figkwargs)
        ax = plt.subplot(111)
    else:
        fig = plt.gcf()
    if title:
        ax.set_title(title)
    return fig, ax


def plot_roc_curve_binary(clf, X, y, label='ROC Curve (area={area:.3})',
                          title="ROC Curve", pos_label=None, sample_weight=None,
                          ax=None, figkwargs=None):
    ax = ax or plt.subplot(111)
    ax.set_xlim([-.1, 1])
    ax.set_ylim([0, 1.1])
    y_score = clf.predict_proba(X)
    if y_score.shape[1] != 2 and not pos_label:
        warnings.warn("Shape is not binary {} and no pos_label".format(y_score.shape))
        return
    try:
        fpr, tpr, thresholds = roc_curve(y, y_score[:,1], pos_label=pos_label,
                                     sample_weight=sample_weight)
    except ValueError as e:
        if 'is not binary' in str(e):
            warnings.warn("Check if y is numeric")
            raise

    roc_auc = auc(fpr, tpr)
    fig, ax = fig_with_title(ax, title, figkwargs)

    ax.plot(fpr, tpr, label=label.format(area=roc_auc))
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Guessing')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    return fig, ax
def plot_roc_curve_multilabels(clf, X, y, labels, label_nums, label='ROC Curve {label} (area={area:.3})',
                          title="ROC Curve", sample_weight=None,
                               ax=None, figkwargs=None, add_avg=True):
    ''' ROC curvefor multiplabel data
    '''
    y_bin = preprocessing.label_binarize(y, label_nums)
    y_score = clf.predict_proba(X)
    fprs = {}
    tprs = {}
    roc_aucs = {}
    for i, l in enumerate(labels):
        try:
            fprs[i], tprs[i], _ = roc_curve(y_bin[:,i], y_score[:,i],
                                          sample_weight=sample_weight)
            roc_aucs[i] = auc(fprs[i], tprs[i])
        except ValueError as e:
            if 'is not binary' in str(e):
                warnings.warn("Check if y is numeric")
                raise
    fig, ax = fig_with_title(ax, title, figkwargs)
    for i, l in enumerate(labels):
        x = fprs[i]
        y = tprs[i]
        text=label.format(area=roc_aucs[i], label=l)
        ax.plot(x, y, label=text)
    if add_avg:
        f, t, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        r = auc(f, t)
        text=label.format(area=r, label='Average')
        ax.plot(f, t, label=text, color='k', linewidth=2)
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Guessing')
    ax.set_xlim([-.1, 1])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    return fig, ax
plot_roc_curve_binary(rf1, X_test, y_test, figkwargs={'figsize':(14,10)})
# yellowbrick version
fig, ax = plt.subplots(figsize=(10, 10))
roc_viz = ROCAUC(rf1)
roc_viz.score(X_test, y_test)
roc_viz.poof()

def plot_confusion_matrix(clf, X, y, labels, random_state=42, annotate=True,
                          cmap=plt.cm.Blues,
                          title="Confusion Matrix", ax=None, figkwargs=None):
    fig, ax = fig_with_title(ax, title, figkwargs)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    if annotate:
        for x in range(len(labels)):
            for y in range(len(labels)):
                plt.annotate(str(cm[x][y]),
                             xy=(y,x),
                             ha='center',va='center',color='red', fontsize=25, fontstyle='oblique')

    return fig, ax
plot_confusion_matrix(rf1, X_test, y_test, ['sell', 'buy'])
# Notice that the training set performs much better!
plot_confusion_matrix(rf1, X_train, y_train, ['sell', 'buy'])
# Yellowbrick - Using percent
mapping = {0:'sell', 1:'buy'}
fig, ax = plt.subplots(figsize=(10, 10))
cm_viz = ConfusionMatrix(rf1, classes=['sell', 'buy'], label_encoder=mapping)
cm_viz.score(X_test, y_test)
cm_viz.poof()
# Yellowbrick - Using count
fig, ax = plt.subplots(figsize=(10, 10))
cm_viz = ConfusionMatrix(rf1, classes=['sell', 'buy'], label_encoder=mapping)
cm_viz.score(X_test, y_test)
# cm_viz.score(X_test, y_test, percent=False)

cm_viz.poof()
cr_viz = ClassificationReport(rf1, classes=['buy', 'sell'])
cr_viz.score(X_test, y_test)
cr_viz.poof()
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)

def plot_calibration_curve(est, name, fig_index,                      
    X_train, X_test, y_train, y_test):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f" % f1_score(y_test, y_pred))
        print("\tScore: %1.3f\n" % clf.score(X_test, y_test))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


plot_calibration_curve(rf1, 'Random Forest', 1,
    X_train, X_test, y_train, y_test)
%%time
rf4 = ensemble.RandomForestClassifier()
params = {'max_features': [.4, 'auto'],
         'n_estimators': [15, 200, 500],
         'min_samples_leaf': [1, .1],
         'random_state':[42]}
cv = model_selection.GridSearchCV(rf4, params).fit(X, y)
print(cv.best_params_)
rf5 = ensemble.RandomForestClassifier(**cv.best_params_)
rf5.fit(X_train, y_train)
rf5.score(X_test, y_test)
rf6 = ensemble.RandomForestClassifier(random_state=41)
rf6.fit(X_train, y_train)
rf6.score(X_test, y_test)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                       fig_opts=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum y values plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    fig_opts = fig_opts or {}
    plt.figure(**fig_opts)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(rf6, 'Random Forest', X, y, fig_opts={'figsize':(14,10)})
%%time
def get_data(filename, resample='d', size=1000):
    bit_df = pd.read_csv(filename)
    # Convert unix time to datetime
    bit_df['date'] = pd.to_datetime(bit_df.Timestamp, unit='s')
    # Reset index
    bit_df = bit_df.set_index('date')
    # Rename columns so easier to code
    bit_df = bit_df.rename(columns={'Open':'open', 'High': 'hi', 'Low': 'lo',
                           'Close': 'close', 'Volume_(BTC)': 'vol_btc',
                           'Volume_(Currency)': 'vol_cur',
                           'Weighted_Price': 'wp', 'Timestamp': 'ts'})
    # Resample and only use recent samples that aren't missing
    bit_df = bit_df.resample(resample).agg({'open': 'first', 'hi': 'mean',
        'lo': 'mean', 'close': 'last', 'vol_btc': 'sum',
        'vol_cur': 'sum', 'wp': 'mean', 'ts': 'min'})

    # drop if open is missing - ADDED!
    bit_df = bit_df[~bit_df.open.isnull()]

    if size:
        bit_df = bit_df.iloc[-size:]
    bit_df['buy'] = (bit_df.close.shift(-1) > bit_df.close).astype(int)
    # drop last row as it is not complete
    bit_df = bit_df.iloc[:-1]
    return bit_df

hour_df = get_data('../data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv',
                   resample='h', size=None)
print(hour_df.shape)

def get_test_train(df):
    for func in [rsi,
             stoc, williams, proc, obv]:
        df[func.__name__] = func(df)

    ignore = {'buy'}
    cols2 = [c for c in df.columns if c not in ignore]
    X = df[cols2]
    X = X.fillna(0)
    y = df.buy
    X_train, X_test, y_train, y_test = model_selection.\
        train_test_split(X, y, test_size=.3, random_state=42)
    return X_train, X_test, y_train, y_test

hX_train, hX_test, hy_train, hy_test = get_test_train(hour_df)
hX_train.isnull().any()
plot_learning_curve(ensemble.RandomForestClassifier(),
                    'Random Forest', hX_train, hy_train, fig_opts={'figsize':(14,10)})
