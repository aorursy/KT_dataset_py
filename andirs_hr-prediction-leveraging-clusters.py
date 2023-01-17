# Read in data & import of packages & frameworks

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np

import warnings



%matplotlib inline



from bokeh.charts import Bar, output_file, show

from bokeh.io import output_notebook

from bokeh.plotting import figure

from bokeh.layouts import gridplot

from bokeh.layouts import column, row

from bokeh.plotting import reset_output

from bokeh.charts.attributes import cat

from collections import Counter

from IPython.display import display
plot_color = 'crimson'

cmap = sns.diverging_palette(222, 10, as_cmap=True)



hr_data = pd.read_csv('../input/HR_comma_sep.csv', header=0)
# Rename column to fit better, get rid of typos and instill consistency

hr_data = hr_data.rename(

    columns = {'sales' : 'department', 

               'average_montly_hours' : 'average_monthly_hours',

               'Work_accident' : 'work_accident'})
hr_data.head()
print("The data set has {} data points and {} variables.".format(*hr_data.shape))
print(hr_data.columns.values)
hr_description = hr_data.describe()

hr_description
hr_data.head()
# Describe categorical data

print("hr_data['department']:", hr_data['department'].unique(), '\n')

print("hr_data['salary']:", hr_data['salary'].unique())
# Mean or median

mean_by_dept = hr_data.groupby('department').mean()

median_by_dept = hr_data.groupby('department').median()



mm_comp = []

index = []

for row_mean, row_median in zip(mean_by_dept.var().iteritems(), median_by_dept.var().iteritems()):

    mm_comp.append([row_mean[1], row_median[1]])

    index.append(row_mean[0])



mm_comp = pd.DataFrame(mm_comp, columns=['mean', 'median'], index=index)

mm_comp
hr_description.transpose()[['std', 'min', 'max']]
cols = ['number_project', 

        'time_spend_company', 

        'department', 

        'salary', 

        'work_accident', 

        'left', 

        'promotion_last_5years']



#Find this code in hr_predict.py on github

#hr_predict.discrete_charts(hr_data, cols, plot_color)
#from math import round

print("{:.2f} % percent of all employees had a promotion within the last 5 years.".format(

    (len(hr_data[hr_data['promotion_last_5years'] == 1]) / 

    float(len(hr_data['promotion_last_5years']))) * 100))

print("{} out of 100 employees had a work related accident.".format(

    int(round((len(hr_data[hr_data['work_accident'] == 1]) / float(len(hr_data))) * 100))))
# Check if there are any NaN values

for name, item in hr_data.isnull().sum().iteritems():

    if item > 0:

        print(name)
people_left = len(hr_data[hr_data['left'] == 1])

print("{} Persons left their company ({:.3} %)".format(

    people_left, ((people_left/float(len(hr_data))) * 100)))
# Average tenure

print("Average tenure: {:.4}".format(hr_data['time_spend_company'].mean()))
#Find this code in hr_predict.py on github

#hr_predict.discrete_dept_charts(mean_by_dept, plot_color)
def preprocess_features(data):

    ''' 

    Preprocesses input data. 

    converts non-numeric binary variables into

    binary (0/1) variables. 

    Converts categorical variables into dummy variables. 

    '''

    output = pd.DataFrame(index = data.index)



    # Investigate each feature column for the data

    for col, col_data in data.iteritems():

        

        # If data type is non-numeric, replace all yes/no values with 1/0

        if col_data.dtype == object:

            col_data = col_data.replace(['yes', 'no'], [1, 0])



        # If data type is categorical, convert to dummy variables

        if col_data.dtype == object:

            col_data = pd.get_dummies(col_data, prefix = col)  

        

        # Collect the revised columns

        output = output.join(col_data)

        # Unify all column names by transforming to lower case

        output.columns = output.columns.str.lower()

    

    return output



hr_data = preprocess_features(hr_data)



print("Processed feature columns ({} total features):\n{}".format(

    len(hr_data.columns), list(hr_data.columns)))
plt.figure(figsize=(16, 14))

cmap = sns.diverging_palette(222, 10, as_cmap=True)

_ = sns.heatmap(hr_data.corr(), annot=True, vmax=.8, square=True, cmap=cmap)
def corr_table(data, features, sig_level=0.05, strength=0.0):

    import math

    import numpy as np

    from scipy.stats import pearsonr



    from operator import itemgetter

    p_val_dict = []

    check_dict = []

    for feature in features:

        feature_first = feature.split('_')[0]

        for label in features:

            # Since these correlations go in both directions, we only need to store on 

            # of the correlations and can discard the secon one

            # i.e. corr(age, Medu) has equal insights to corr(Medu, age)

            feature_comb = label+feature

            label_first = label.split('_')[0]

            

            if feature == label or feature_comb in check_dict or feature_first == label_first:

                #feature is already paired with label or equals label

                #or feature and label are from the same one-hot-encoding category

                continue

            else:

                check_dict.append(feature+label)

                pears = pearsonr(data[feature], data[label])

                p_val = pears[1]

                corr_strength = pears[0]

                cov_strength = np.cov(data[feature], data[label])[0][1]

                

                # Check if correlation is significant and has a high enough correlation

                if p_val < sig_level and math.fabs(corr_strength) > strength:

                    p_val_dict.append([feature, label, cov_strength, corr_strength, p_val])



    p_corr_title = 'Correlation > ' + str(strength)

    p_value_title = 'p-Value < ' + str(sig_level)

    p_val_dict = pd.DataFrame(p_val_dict, columns = ['Feature', 'Label', 'Covariance', p_corr_title, p_value_title])

    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    

    p_val_dict['order'] = abs(p_val_dict[p_corr_title])

    p_val_dict.sort_values(by='order', inplace=True, ascending=False)

    p_val_dict.head()

    p_val_dict = p_val_dict.reset_index(drop=True)

    p_val_dict = p_val_dict.drop('order', axis=1)

    

    return p_val_dict
reg_features = hr_data.columns[~hr_data.columns.str.contains('left')]

label = reg_features



# How many significant correlations are in the data set?

correlations_all = corr_table(hr_data, reg_features, strength=0.1)
correlations_all
# Outlier detection

outliers = {}

outliers_all = []

sum = 0

# For each feature find the data points with extreme high or low values

for feature in hr_data[[0,1,2,3,4]].keys():

    

    # Calculates Q1 for the given feature

    Q1 = np.percentile(hr_data[feature], 25)

    

    # Calculates Q3 for the given feature

    Q3 = np.percentile(hr_data[feature], 75)

    

    # Calculate an outlier step (1.5 times the interquartile range)

    step = 1.5 * (Q3 - Q1)

    

    # Display the outliers

    category_outliers = hr_data[~((hr_data[feature] >= Q1 - step) & (hr_data[feature] <= Q3 + step))]

    for outlier_no in category_outliers.index:

        if outlier_no in outliers:

            outliers[outlier_no] += 1

        else:

            outliers[outlier_no] = 1

            outliers_all.append(outlier_no)

    if len(category_outliers) > 0:

        print("")

        print("Data points considered outliers for the feature '{}' - (Min: {} - Max: {}):".format(

            feature, Q1, Q3))

        print("Outliers: {} ({:.2f} %)".format(len(category_outliers), 

                                              (len(category_outliers) / float(len(hr_data))) * 100 ))

    else:

        print("No outliers for feature '{}'.".format(feature))
steps = [1.5, 2, 2.5, 3, 4, 5]



outlier_index = {}

for step_count in steps:

    outlier_data = hr_data['time_spend_company']

    Q1 = np.percentile(outlier_data, 25)

    Q3 = np.percentile(outlier_data, 75)

    step = step_count * (Q3 - Q1)

    category_outliers = hr_data[~((outlier_data >= Q1 - step) & 

                                    (outlier_data <= Q3 + step))]

    outlier_index[step_count] = category_outliers.index

    print("{}: \t {} \t ({:.2f} %)".format(step_count, 

                                     len(category_outliers), 

                                     (len(category_outliers) / float(len(outlier_data))) * 100))
# In case you want to drop outliers - use this code and change hr_data.index[outlier_index[step]]

# where step can be 1.5, 2, 2.5, 3, 4 or 5.



#hr_data = hr_data.drop(hr_data.index[outlier_index[2]]).reset_index(drop=True)
hr_data = (hr_data - hr_data.min()) / (hr_data.max() - hr_data.min())
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Sets a random state for reproducibility

random_state = 42

data = hr_data.copy().drop('left', axis = 1)

temp = {}



for label in data.columns:

    if ("department" not in label) & ("salary" not in label):

        new_data = data.drop(label, axis = 1)

        new_data_labels = data[label]



        X_train, X_test, y_train, y_test = train_test_split(

            new_data, new_data_labels, test_size=0.25, random_state=random_state)



        regressor = DecisionTreeRegressor(random_state=random_state)

        regressor.fit(X_train, y_train)

        regressor.predict(X_test)



        score = regressor.score(X_test, y_test)

        temp[label] = score

        print("Prediction score for " + label + " (R^2): " + str(score))

temp = pd.DataFrame.from_dict(temp, orient='index')

temp.columns = ['R^2 score']
temp.sort_values('R^2 score', ascending=False)
# Implement learning algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split, cross_val_score

from time import time



# Let's keep 20 % of the data for testing purposes

test_size = .2

random_state = 42



X_all_base = hr_data.drop('left', 1)

y_all_base = hr_data['left']



# Use for testing later and don't touch: X_test_base / y_test_base

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(

    X_all_base, y_all_base, test_size = test_size, random_state = random_state, stratify=y_all_base)



clf_dict_base = {}

clf_report_base = []

clf_feature_relevance_base = []



for clf in [LinearSVC(random_state = random_state),

            LogisticRegression(random_state = random_state),

            DecisionTreeClassifier(random_state = random_state),

            SVC(random_state = random_state),

            RandomForestClassifier(random_state = random_state)]:

    # Extract name of estimator

    clf_name = clf.__class__.__name__

    print("Training", clf_name, "...")

    # Fit model on training data

    clf_dict_base[clf_name] = clf.fit(X_train_base, y_train_base)

    # Predict based on it

    # y_pred = clf.predict(X_train)

    

    # Perform cross validation

    start = time()

    scores = cross_val_score(clf, X_train_base, y_train_base, cv=5, scoring='roc_auc') 

    end = time()

    duration = end - start

    print("Average CV performance for {}: {:.6} (in {:.6} seconds)".format(

        clf_name, scores.mean(), duration))

    clf_report_base.append([clf_name, scores.mean(), duration])



    # Store feature relevance information 

    if clf_name in ["RandomForestClassifier", "DecisionTreeClassifier"]:

        clf_feature_relevance_base.append(clf.feature_importances_.tolist())

    elif clf_name == "LinearSVC":

        clf_feature_relevance_base.append(clf.coef_[0].tolist())

# Store information in list for better visibility



clf_report_base = pd.DataFrame(clf_report_base, columns=['classifier', 'mean_score', 'time'])
pd.DataFrame(

    clf_feature_relevance_base, columns=X_train_base.columns, index=['LinearSVC', 

                                                                     'DecisionTreeClassifier', 

                                                                     'RandomForestClassifier'])
clf_report_base.sort_values(by=['mean_score', 'time'], ascending=False)
predictor_list_base = []

for relevance in clf_dict_base['RandomForestClassifier'].feature_importances_:

    predictor_list_base.append(relevance)

new_base = pd.DataFrame(

    predictor_list_base, columns=['importance'], index=X_all_base.columns.values.tolist())

new_base.sort_values(by='importance', ascending=False, inplace=True)

new_base['features'] = new_base.index



p_base = Bar(new_base,

        values='importance',

        label=cat(columns='features', sort=False),

        title='Feature importance',

        color='crimson',

        plot_width=800, 

        plot_height=500,

        ylabel='importance',

        legend=None,

        toolbar_location=None)

#show(p_base)



# Bokeh doesn't seem to work with kaggle. The output of show(p_base) can be seen below as a picture.
import matplotlib.pyplot as plt

import matplotlib.cm as cm

import pandas as pd

import numpy as np

from sklearn.decomposition import pca



def pca_results(good_data, pca):

    '''

    Create a DataFrame of the PCA results

    Includes dimension feature weights and explained variance

    Visualizes the PCA results

    '''



    # Dimension indexing

    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]



    # PCA components

    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())

    components.index = dimensions



    # PCA explained variance

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)

    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])

    variance_ratios.index = dimensions



    # Create a bar plot visualization

    fig, ax = plt.subplots(figsize = (14,8))



    # Plot the feature weights as a function of the components

    components.plot(ax = ax, kind = 'bar');

    ax.set_ylabel("Feature Weights")

    ax.set_xticklabels(dimensions, rotation=0)





    # Display the explained variance ratios

    for i, ev in enumerate(pca.explained_variance_ratio_):

        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))



    # Return a concatenated DataFrame

    return pd.concat([variance_ratios, components], axis = 1)
from sklearn.decomposition import PCA



pca_data = hr_data.ix[:,0:5]

pca = PCA(n_components=5)

pca = pca.fit(pca_data)



# Generate PCA results plot

pca_results = pca_results(pca_data, pca)
pd.DataFrame(data=[np.cumsum(pca.explained_variance_ratio_)], columns="Add " + 

             pca_results.index.values, index=['Combined Explained Variance'])
from sklearn.decomposition import PCA

pca = PCA(n_components=5)

pca = pca.fit(pca_data)



reduced_data = pca.transform(pca_data)

reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4', 'Dimension 5'])
from sklearn import mixture

from sklearn.metrics import silhouette_score



score_list = []

score_columns = []

preds = {}

centers = {}

sample_preds = {}



# This can be a little time consuming... but fun :)

for n in range(5,1,-1):

    print("Calculating clusters with {} dimensions.".format(n))

    clusterer = mixture.GaussianMixture(n_components=n)

    # Future

    # clusterer = mixture.GaussianMixture(n_components=n)

    clusterer.fit(reduced_data)



    preds[n] = clusterer.predict(reduced_data)

    centers[n] = clusterer.means_

    score = silhouette_score(reduced_data, preds[n], metric='euclidean')

    score_list.append(score)

    score_columns.append(str(n) + " components")



score_list = pd.DataFrame(data=[score_list],columns=score_columns, index=['Silhouette Score'])

score_list
def cluster_results(reduced_data, preds, centers):

    predictions = pd.DataFrame(preds, columns = ['Cluster'])

    plot_data = pd.concat([predictions, reduced_data], axis = 1)



    # Generate the cluster plot

    fig, ax = plt.subplots(figsize = (10,6))



    # Color map

    cmap = cm.get_cmap('gist_rainbow')



    # Color the points based on assigned cluster

    for i, cluster in plot_data.groupby('Cluster'):   

        cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \

                     color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=30);



    # Plot centers with indicators

    for i, c in enumerate(centers):

        ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \

                   alpha = 1, linewidth = 2, marker = 'o', s=200);

        ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);



    # Set plot title

    #plt.legend(loc='upper left', numpoints=1, ncol=4, fontsize=12, bbox_to_anchor=(0, 0))

    ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number",

                 fontsize = 14)
def cluster_results_3d(reduced_data, preds, centers):

    from mpl_toolkits.mplot3d import Axes3D

    

    predictions = pd.DataFrame(preds, columns = ['Cluster'])

    plot_data = pd.concat([predictions, reduced_data], axis = 1)

    cmap = cm.get_cmap('gist_rainbow')

    

    fig = plt.figure(figsize = (10,8))

    ax = fig.add_subplot(111, projection='3d')

    ax2 = fig.add_subplot(111, projection='3d')



    fig = fig.gca(projection='3d')



    for i, c in enumerate(centers):

        ax2.scatter(c[0], c[1], c[2], color = 'white', edgecolors = 'black', \

                   alpha = 1, linewidth = 2, marker = 'o', s=200,

                   zorder=1);

        ax2.scatter(c[0], c[1], c[2], marker='$%d$'%(i), alpha = 1, s=100,

                   zorder=1);

        

    for i, cluster in plot_data.groupby('Cluster'):   

        ax2.scatter(cluster['Dimension 1'], cluster['Dimension 2'], cluster['Dimension 3'],

                   c = cmap((i)*1.0/(len(centers)-1)), alpha=0.2,

                   label = 'Cluster %i'%(i), s=20,

                   zorder=.5)



    fig.set_xlabel('Dimension 1')

    fig.set_ylabel('Dimension 2')

    fig.set_zlabel('Dimension 3')

    plt.legend(loc='upper left', numpoints=1, ncol=4, fontsize=12, bbox_to_anchor=(0, 0))

    ax2.set_title("Cluster Learning on PCA-Reduced Data 3D Plot",

                 fontsize = 14);

    plt.show()
# with preds[n] and centers[n] where n is the number of Dimensions

no_clusters = 4

cluster_results(reduced_data, preds[no_clusters], centers[no_clusters])

cluster_results_3d(reduced_data, preds[no_clusters], centers[no_clusters])
predictions = pd.DataFrame(preds[no_clusters], columns = ['Cluster'])

hr_data['employee_cluster'] = predictions
hr_data.head()
print(hr_data.columns)
# Let's start the prediction

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

from time import time



# Keeps 20 % of the data for testing purposes

test_size = .2

random_state = 42



X_all_clus = hr_data.drop('left', 1)

y_all_clus = hr_data['left']



# Use for testing later and don't touch: X_test_clus / y_test_clus

X_train_clus, X_test_clus, y_train_clus, y_test_clus = train_test_split(

    X_all_clus, y_all_clus, test_size = test_size, random_state = random_state, stratify=y_all_clus)



clf_dict_clus = {}

clf_report_clus = []

clf_feature_relevance_clus = []



for clf_clus in [LinearSVC(random_state = random_state),

                LogisticRegression(random_state = random_state),

                DecisionTreeClassifier(random_state = random_state),

                SVC(random_state = random_state),

                RandomForestClassifier(random_state = random_state)]:

    # Extract name of estimator

    clf_name = clf_clus.__class__.__name__

    print("Training", clf_name, "...")

    # Fit model on training data

    clf_dict_clus[clf_name] = clf_clus.fit(X_train_clus, y_train_clus)

    # Predict based on it

    # y_pred = clf.predict(X_train)

    

    # Perform cross validation

    start = time()

    scores = cross_val_score(

        clf_clus, X_train_clus, y_train_clus, cv=5, scoring='roc_auc') 

    end = time()

    duration = end - start

    print("Average CV performance for {}: {:.6} (in {:.6} seconds)".format(

        clf_name, scores.mean(), duration))

    clf_report_clus.append([clf_name, scores.mean(), duration])



    # Store feature relevance information 

    if clf_name in ["RandomForestClassifier", "DecisionTreeClassifier"]:

        clf_feature_relevance_clus.append(clf_clus.feature_importances_.tolist())

    elif clf_name == "LinearSVC":

        clf_feature_relevance_clus.append(clf_clus.coef_[0].tolist())

# Store information in list for better visibility



clf_report_clus = pd.DataFrame(clf_report_clus, columns=['classifier', 'mean_score', 'time'])
pd.DataFrame(

    clf_feature_relevance_clus, columns=X_train_clus.columns, index=['LinearSVC', 

                                                                     'DecisionTreeClassifier', 

                                                                     'RandomForestClassifier'])
clf_report_clus = clf_report_clus.ix[:,0:3]

clf_report_clus.sort_values('mean_score', ascending=False)
clf_report_clus['mean_score non-cluster'] = clf_report_base['mean_score']

clf_report_clus['time non-cluster'] = clf_report_base['time']

clf_report_clus['score_change'] = clf_report_clus['mean_score'] - clf_report_clus['mean_score non-cluster']

clf_report_clus['time_change'] = clf_report_clus['time'] - clf_report_clus['time non-cluster']

clf_report_clus.sort_values(by=['mean_score', 'score_change', 'time_change'], ascending=False)
predictor_list = []

for relevance in clf_dict_base['RandomForestClassifier'].feature_importances_:

    predictor_list.append(relevance)





columns_list = X_all_base.columns.values.tolist()



new = pd.DataFrame(predictor_list, columns=['importance'], index=columns_list)

new.sort_values(by='importance', ascending=False, inplace=True)

new['features'] = new.index

new = new[:6]



p = Bar(new,

        values='importance',

        label=cat(columns='features', sort=False),

        title='Feature importance without cluster information',

        color='crimson',

        plot_width=800, 

        plot_height=300,

        ylabel='importance',

        legend=None)



predictor_list_clus = []

for relevance in clf_dict_clus['RandomForestClassifier'].feature_importances_:

    predictor_list_clus.append(relevance)

new_clus = pd.DataFrame(

    predictor_list_clus, columns=['importance'], index=X_all_clus.columns.values.tolist())

new_clus.sort_values(by='importance', ascending=False, inplace=True)

new_clus['features'] = new_clus.index



new_clus = new_clus[:6]

p_clus = Bar(new_clus,

        values='importance',

        label=cat(columns='features', sort=False),

        title='Feature importance with cluster information',

        color='crimson',

        plot_width=800, 

        plot_height=300,

        ylabel='importance',

        legend=None)

p.xaxis.major_label_text_font_size = '10pt'

p.title.text_font_size = '12pt'



p_clus.xaxis.major_label_text_font_size = '10pt'

p_clus.title.text_font_size = '12pt'



#show(p)

#show(p_clus)



# Bokeh doesn't seem to work with kaggle. The output of show(p_base) can be seen below as a picture.
def predict_labels(clf, features, target):

    from sklearn import metrics

    ''' Makes predictions using a fit classifier based on AUC score. '''

    

    # Start the clock, make predictions, then stop the clock

    start = time()

    y_pred = clf.predict(features)

    end = time()

    

    # Print and return results

    prediction_duration = end - start

    prediction_auc_score = metrics.roc_auc_score(target.values, y_pred)

    print("Made predictions in {:.4f} seconds.".format(prediction_duration))

    return prediction_auc_score, prediction_duration
# Can take a while as well

# If you feel it's taking too long, you can simply reduce

# the amount of parameters in the parameters_dict dictionary

# Hint: Start with reducing the amount of SVC parameters 

# since this algorithm takes the longest and it performs worse than i.e. RandomForest

# Otherwise reduction in RF parameters will reduce the time significantly as well 

# since its grow is exponential.



from sklearn import model_selection

from sklearn import metrics



parameters_dict = {"LinearSVC" : 

                    {"C": [1, 3, 5],

                     "loss" : ['hinge', 'squared_hinge'],

                     "penalty" : ['l2'],

                     "max_iter" : [100, 200, 300]

                    },

                   "LogisticRegression" :

                    {"C": [1, 3, 5],

                     "penalty" : ['l2'],

                     "solver" : ['sag', 'newton-cg', 'lbfgs', 'liblinear'],

                     "warm_start" : [True, False]

                    },

                   "DecisionTreeClassifier" :

                    {"criterion": ['gini', 'entropy'],

                     "max_features" : ['auto', 'sqrt', 'log2', None],

                     "max_depth" : [None, 2, 5, 10]                    

                    },

                   "SVC" : 

                    {"C": [1, 2],

                     "kernel" : ['poly', 'rbf', 'sigmoid'],

                     "degree" : [2, 3, 4]

                    },

                   "RandomForestClassifier" :

                    {"n_estimators" : [10],

                     "criterion": ['gini', 'entropy'],

                     "bootstrap" : [False, True],

                     "max_features" : ['auto', 'sqrt', 'log2', None],

                     "max_depth" : [None, 2, 5],

                     "min_samples_split" : [2, 4],

                     "warm_start" : [True, False],

                     "max_leaf_nodes" : [None, 4, 6]

                    }

                  }



best_estimator_dict = {}

estimator_test_res = {}



test_size = .2

random_state = 42



X_all_gs = X_train_clus.copy()

y_all_gs = y_train_clus.copy()



X_train_gs, X_test_gs, y_train_gs, y_test_gs = train_test_split(

    X_all_gs, y_all_gs, test_size = test_size, random_state = random_state, stratify=y_all_gs)



# Initialize the classifier

for clf in [LinearSVC(random_state = random_state),

            LogisticRegression(random_state = random_state),

            DecisionTreeClassifier(random_state = random_state),

            SVC(random_state = random_state),

            RandomForestClassifier(random_state = random_state)]:

    # Extract name of estimator

    clf_name = clf.__class__.__name__

    print("Searching Grid for", clf_name, "...")



    # Perform grid search on classifier using roc_auc as scoring method

    grid_obj = model_selection.GridSearchCV(

        clf, param_grid=parameters_dict[clf_name], scoring='roc_auc')



    # Fit the grid search object to the training data and find the optimal parameters

    grid_obj = grid_obj.fit(X_train_gs, y_train_gs)



    # Get the estimator

    clf = grid_obj.best_estimator_

    best_estimator_dict[clf_name] = clf

    estimator_test_res[clf_name] = grid_obj.cv_results_



    # Report the final F1 score for training and testing after parameter tuning

    print("Tuned {} model has a training AUC score of {:.4f}.".format(

        clf_name, predict_labels(clf, X_train_gs, y_train_gs)[0]))

    print("Tuned {} model has a testing AUC score of {:.4f}.".format(

        clf_name, predict_labels(clf, X_test_gs, y_test_gs)[0]))
pd.DataFrame.from_dict(

    estimator_test_res['RandomForestClassifier']).sort_values('mean_test_score', ascending=False) 
for clf_name in ['LinearSVC', 

                 'LogisticRegression', 

                 'DecisionTreeClassifier', 

                 'SVC', 

                 'RandomForestClassifier']:

    print(best_estimator_dict[clf_name])
for clf_name in ['LinearSVC', 

                 'LogisticRegression', 

                 'DecisionTreeClassifier', 

                 'SVC', 

                 'RandomForestClassifier']:

    print("Training", clf_name)

    start = time()

    scores = cross_val_score(

        best_estimator_dict[clf_name], X_train_gs, y_train_gs, cv=5, scoring='roc_auc') 

    end = time()

    duration = end - start

    print("Average CV performance for {}: {:.6} (in {:.6} seconds)".format(

        clf_name, scores.mean(), duration))
clf_dict_tuned = {}

clf_report_tuned = []

clf_feature_relevance_tuned = []



for clf_tuned in ['LinearSVC', 

                  'LogisticRegression', 

                  'DecisionTreeClassifier', 

                  'SVC', 

                  'RandomForestClassifier']:



    # Extract name of estimator

    clf_name = clf_tuned

    clf_tuned = best_estimator_dict[clf_tuned]

    print("Cross-Validation of", clf_name, "...")



    # Perform cross validation

    start = time()

    scores = cross_val_score(

        clf_tuned, X_train_clus, y_train_clus, cv=5, scoring='roc_auc') 

    end = time()

    duration = end - start

    print("Average CV performance for {}: {:.6} (in {:.6} seconds)".format(

        clf_name, scores.mean(), duration))

    clf_report_tuned.append([clf_name, scores.mean(), duration])



    # Store feature relevance information 

    if clf_name in ["RandomForestClassifier", "DecisionTreeClassifier"]:

        clf_feature_relevance_tuned.append(clf_tuned.feature_importances_.tolist())

    elif clf_name == "LinearSVC":

        clf_feature_relevance_tuned.append(clf_tuned.coef_[0].tolist())

# Store information in list for better visibility



clf_report_tuned = pd.DataFrame(clf_report_tuned, columns=['classifier', 'mean_score', 'time'])
pd.DataFrame(clf_feature_relevance_tuned, columns=X_train_clus.columns, index=['LinearSVC', 

                                                                              'DecisionTreeClassifier', 

                                                                              'RandomForestClassifier'])
clf_report_tuned = clf_report_tuned.ix[:,0:3]

clf_report_tuned.sort_values('mean_score', ascending=False)
clf_report_tuned['mean_score non-cluster'] = clf_report_clus['mean_score']

clf_report_tuned['time non-cluster'] = clf_report_clus['time']

clf_report_tuned['score_change'] = (

    clf_report_tuned['mean_score'] - clf_report_tuned['mean_score non-cluster'])

clf_report_tuned['time_change'] = clf_report_tuned['time'] - clf_report_tuned['time non-cluster']

clf_report_tuned.sort_values(by=['mean_score', 'score_change', 'time_change'], ascending=False)
new_index = np.random.permutation(X_train_base.index)

X_train_stab = X_train_clus.reindex(new_index)

y_train_stab = y_train_clus.reindex(new_index)



X_train_stab, X_test_stab, y_train_stab, y_test_stab = train_test_split(

    X_train_stab, y_train_stab, train_size = .4, 

    test_size = .2, 

    random_state = random_state, 

    stratify=y_train_stab)
print("Training RandomForestClassifier ...")

start = time()

scores = cross_val_score(

    best_estimator_dict['RandomForestClassifier'], X_train_stab, y_train_stab, cv=5, scoring='roc_auc') 

end = time()

duration = end - start

print("Average CV performance for {}: {:.6} (in {:.6} seconds)".format(

    'RandomForestClassifier', scores.mean(), duration))
print("Tuned {} model has a testing AUC score of {:.4f}.".format(

    'RandomForestClassifier', predict_labels(

        best_estimator_dict['RandomForestClassifier'], X_test_stab, y_test_stab)[0]))
final_results = {}



for clf_name in ['LinearSVC', 'LogisticRegression', 'DecisionTreeClassifier', 'SVC', 'RandomForestClassifier']:

    print("Tuned {} model has a final testing AUC score of {:.4f}.".format(

        clf_name, predict_labels(best_estimator_dict[clf_name], X_test_clus, y_test_clus)[0]))



    temp = []

    temp.append(

        predict_labels(clf_dict_base[clf_name], X_test_base, y_test_base)[0])

    temp.append(

        predict_labels(clf_dict_clus[clf_name], X_test_clus, y_test_clus)[0])

    temp.append(

        predict_labels(best_estimator_dict[clf_name], X_test_clus, y_test_clus)[0])

    

    print(temp)

    final_results[clf_name] = temp
final_score = pd.DataFrame.from_dict(final_results, orient='index')

final_score.columns = ['base_test_score', 'cluster_test_score', 'tuned_test_score']

final_score['max_score'] = final_score[['base_test_score', 'cluster_test_score', 'tuned_test_score']].max(axis=1)

final_score.sort_values('max_score', ascending=False)
print("The best performing score is {:.4f}, which is {:.4f} better than the benchmark.".format(

    final_score['max_score'].max(), final_score['max_score'].max() - .86))
#show(p_clus)
perc_scores = {}

for clf_name in ['LinearSVC', 

                 'LogisticRegression', 

                 'DecisionTreeClassifier', 

                 'SVC', 

                 'RandomForestClassifier']:

    print('Calculating cv scores for', clf_name, '...')

    clf_scores = list()

    percentage_list = [.1, .2, .4, .6, .8]

    for percentage in percentage_list:

        train_size = percentage

        random_state = 42



        X_perc = hr_data.drop('left', 1)

        y_perc = hr_data['left']



        X_train_perc, X_test_perc, y_train_perc, y_test_perc = train_test_split(

            X_perc, y_perc, train_size = train_size, random_state = random_state, stratify=y_perc)



        perc_score_temp = cross_val_score(

            best_estimator_dict[clf_name], X_train_perc, y_train_perc, cv=5, scoring='roc_auc')

        perc_score_temp = perc_score_temp.mean()

        clf_scores.append(perc_score_temp)

    perc_scores[clf_name] = clf_scores
perc_plot = pd.DataFrame.from_dict(perc_scores, orient='index')

percentage_list = [.1, .2, .4, .6, .8]

perc_plot.columns = [str(x) + '%' for x in percentage_list]

perc_plot = perc_plot.transpose()

perc_plot_ax = perc_plot.plot(

    title = 'ROC AUC Score vs. Training Data Percentage', 

    figsize = (8, 5), 

    fontsize = 13)

perc_plot_ax.set_xlabel("Percentage of used data")

perc_plot_ax.set_ylabel("ROC AUC Score")

_ = perc_plot_ax
def predict_labels_f1(clf, features, target):

    from sklearn import metrics

    ''' Makes predictions using a fit classifier based on AUC score. '''

    

    # Start the clock, make predictions, then stop the clock

    start = time()

    y_pred = clf.predict(features)

    end = time()

    

    # Print and return results

    prediction_duration = end - start

    prediction_f1_score = metrics.f1_score(target.values, y_pred)

    print("Made predictions in {:.4f} seconds.".format(prediction_duration))

    return prediction_f1_score, prediction_duration
final_results_f1 = {}



for clf_name in ['LinearSVC', 

                 'LogisticRegression', 

                 'DecisionTreeClassifier', 

                 'SVC', 

                 'RandomForestClassifier']:

    print("Tuned {} model has a final testing AUC score of {:.4f}.".format(

        clf_name, predict_labels_f1(best_estimator_dict[clf_name], X_test_clus, y_test_clus)[0]))



    temp = []

    temp.append(

        predict_labels_f1(clf_dict_base[clf_name], X_test_base, y_test_base)[0])

    temp.append(

        predict_labels_f1(clf_dict_clus[clf_name], X_test_clus, y_test_clus)[0])

    temp.append(

        predict_labels_f1(best_estimator_dict[clf_name], X_test_clus, y_test_clus)[0])

    

    print(temp)

    final_results_f1[clf_name] = temp
final_score_f1 = pd.DataFrame.from_dict(final_results_f1, orient='index')

final_score_f1.columns = ['base_test_score', 'cluster_test_score', 'tuned_test_score']

final_score_f1['max_f1_score'] = final_score_f1[

    ['base_test_score', 'cluster_test_score', 'tuned_test_score']].max(axis=1)

final_score_f1['max_auc_score'] = final_score['max_score']

final_score_f1['delta'] = final_score_f1['max_f1_score'] - final_score_f1['max_auc_score']

final_score_f1.sort_values('max_f1_score', ascending=False)
final_score_f1.ix[:,3:]