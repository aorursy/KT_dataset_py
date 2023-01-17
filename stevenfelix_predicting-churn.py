# standard

import numpy as np

import pandas as pd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl



%matplotlib inline



# modeling

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve, average_precision_score

from sklearn import preprocessing

import statsmodels.api as sm



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/HR_comma_sep.csv')

orig_vars = 'Satisfaction, Evaluation, Projects, Hours, Tenure, Accident, Left, Promotion, Department, Salary'.split(', ')

data.columns = orig_vars

print(data.shape)

data.head()
def process(data):

    '''Takes in raw data frame and conducts dummy coding and recoding.

    Returns X and y'''

    y = data['Left']

    X = (data.drop(['Left'], axis = 1)

            .replace(['low','medium','high'], [1,2,3],axis=1)

            .pipe(pd.get_dummies,columns=['Department'],drop_first=True)

            #.pipe(preprocessing.scale,axis = 0)) # produces ndarray

            .apply(preprocessing.scale,axis = 0)) # produces pd.DataFrame

    return X, y



# Graphing parameters for seaborn and matplotlib

snsdict = {'axes.titlesize' : 30,

            'axes.labelsize' : 28,

            'xtick.labelsize' : 26,

            'ytick.labelsize' : 26,

            'figure.titlesize': 34}

pltdict = {'axes.titlesize' : 20,

         'axes.labelsize' : 18,

         'xtick.labelsize' : 16,

         'ytick.labelsize' : 16,

         'figure.titlesize': 24}
# Process and split data

X,y = process(data)

X_train, X_test, y_train, y_test = train_test_split(

            X, y, test_size = 0.1, random_state = 50)
## Using statsmodels for its nice output summary

logit = sm.Logit(y_train,X_train)

results = logit.fit();

print(results.summary());
plt.rcParams.update(pltdict)

ORs = np.exp(results.params).sort_values();

g = sns.barplot(x = ORs.index, y = ORs-1, palette = 'RdBu_r');

g.set_xticklabels(ORs.index, rotation = 90);

g.set_title('Percent change in odds of leaving\n(i.e., OR minus 1)');

g.set_ylim(-.6, .6);
lr = LogisticRegression(C = 1, random_state=1)

lr.fit(X_train,y_train)

print('10-fold cross validation accuracy: {0:.2f}% \n\n'

      .format(np.mean(cross_val_score(lr, X_train, y_train,cv = 10))*100))

print('Precision/Recall Table: \n')

print(classification_report(y, lr.predict(X)))

prc = precision_recall_curve(y_train, lr.decision_function(X_train), pos_label=1);

plt.plot(prc[1],prc[0]);

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision-Recall Curve')
    # Set colors for the different groups

current_palette = sns.color_palette()[0:2]

cols = [current_palette[grp] for grp in data.Left]



sns.set(rc = snsdict)

#plt.rcParams.update(pltdict)



fig, ax = plt.subplots(3,1, figsize = (15,20));

sns.barplot(data = data, x = 'Satisfaction', y= 'Left', ax = ax[0], color = sns.xkcd_rgb["pale red"]);

sns.regplot(data = data, x = 'Satisfaction', y= 'Left', y_jitter=.02,

            scatter_kws={'alpha':0.05, 'color':cols}, fit_reg = False, ax = ax[1]);

sns.barplot(data = data, x= 'Left', y = 'Satisfaction', ax = ax[2]);

ax[0].set_ylabel("Proportion Leaving"); ax[0].set_xticks([])

ax[2].set_xticklabels(['Stayed','Left']); ax[2].set_ylabel('Mean Satisfaction');

ax[1].set_yticklabels(['Stayed','Left'], fontsize = 24); ax[1].set_ylabel('');ax[1].set_yticks([0,1]);

fig.suptitle('Satisfaction Plots')

plt.tight_layout()
fig, ax = plt.subplots(3,1, figsize = (15,18));

sns.barplot(data = data, x = 'Evaluation', y= 'Left', ax = ax[0], color = sns.xkcd_rgb["pale red"]);

sns.regplot(data = data, x = 'Evaluation', y= 'Left', y_jitter=.02,

            scatter_kws={'alpha':0.05, 'color':cols}, fit_reg = False, ax = ax[1]);

sns.barplot(data = data, x= 'Left', y = 'Evaluation', ax = ax[2]);

ax[0].set_ylabel('Proportion Leaving'); ax[0].set_xticklabels([])

ax[1].set_xlabel("Evaluation"); ax[1].set_yticks([0,1]); ax[1].set_yticklabels(['Stayed','Left']);

ax[1].set_ylabel('');

ax[2].set_xticklabels(['Stayed','Left']); ax[2].set_ylabel('Mean Evaluation Score')

fig.tight_layout()
#sns.set(rc = snsdict)

plt.rcParams.update(pltdict)

stayed = plt.scatter(x = data.Evaluation[data.Left == 0], y = data.Satisfaction[data.Left == 0], 

                     c= current_palette[0], alpha = .4);

left = plt.scatter(x = data.Evaluation[data.Left == 1], y = data.Satisfaction[data.Left == 1], 

                   c= current_palette[1], alpha = .4)

plt.xlabel("Evaluation"); plt.ylabel("Satisfaction");

plt.title("Satisfaction and Evaluation by Group");

leg = plt.legend((stayed,left), ("Stayed","Left"),fontsize=16, bbox_to_anchor=(.98, .65));

for lh in leg.legendHandles: lh.set_alpha(1)
plt.rcParams.update(pltdict)

data['Predicted'] = ['Left' if pred else 'Remained' for pred in lr.predict(X)]

data['Outcome'] = ['Left' if left else 'Remained' for left in data.Left]

sns.lmplot(x = 'Evaluation', y = 'Satisfaction',hue = 'Predicted', 

           hue_order = ['Remained','Left'],col = 'Outcome', data = data, fit_reg=False,

            scatter_kws={'alpha':0.2}, legend = False);

leg = plt.legend((stayed,left), ("Remained","Left"),fontsize=16, bbox_to_anchor=(.98, .65));

for lh in leg.legendHandles: lh.set_alpha(1);

leg.set_title('Prediction')
sns.set(rc = snsdict)

#plt.rcParams.update(pltdict)

fig, ax = plt.subplots(3,1, figsize = (15,18));

sns.barplot(data = data, x = 'Hours', y= 'Left', ax = ax[0], color = sns.xkcd_rgb["pale red"]);

sns.regplot(data = data, x = 'Hours', y= 'Left', y_jitter=.02,

            scatter_kws={'alpha':0.05, 'color':cols}, fit_reg = False, ax = ax[1]);

sns.barplot(data = data, x= 'Left', y = 'Hours', ax = ax[2]);

ax[0].set_ylabel('Proportion Leaving'); ax[0].set_xticklabels([])

ax[1].set_xlabel("Hours"); ax[1].set_yticks([0,1]); ax[1].set_yticklabels(['Stayed','Left']);

ax[1].set_ylabel('');

ax[2].set_xticklabels(['Stayed','Left']); ax[2].set_ylabel('Mean Hours / Month')

ax[2].set_xlabel("")

fig.tight_layout()
plt.rcParams.update(pltdict)

fig, ax = plt.subplots(2,1, figsize = (10,13));

sns.barplot(data = data, x = 'Projects', y= 'Left', ax = ax[0]);

sns.barplot(data = data, y = 'Projects', x= 'Left', ax = ax[1]);

ax[0].set_xlabel("Number of Projects"); ax[0].set_ylabel("Proportion Leaving");

ax[0].set_title("Proportion of Employees Leaving by Number of Projects");

ax[1].set_xticklabels(['Stayed','Left']); ax[1].set_title("Average Number of Projects by Retention Status");

ax[1].set_ylabel("Average Number of Projects"); ax[1].set_xlabel("");

fig.tight_layout();
#sns.set(rc = snsdict)

plt.rcParams.update(pltdict)

fig, ax = plt.subplots(4,1, figsize = (10,15),sharey = True);

sns.barplot(data = data, x = 'Accident', y= 'Left', ax = ax[0]);

ax[0].set_xticklabels(['No Accident','Accident']); ax[0].set_xlabel('');

sns.barplot(data = data, x= 'Salary', y = 'Left', ax = ax[1]);

sns.barplot(data = data, x= 'Promotion', y = 'Left', ax = ax[2]);

ax[2].set_xticklabels(['No Promortion', 'Promotion']);ax[2].set_xlabel('');

sns.barplot(data = data, x= 'Tenure', y = 'Left', ax = ax[3]);

ax[3].set_xlabel('Tenure (years)')

for i in range(4): ax[i].set_ylabel('Proportion Leaving')

fig.tight_layout()
def newfeatures(data, polyfeats, modeldata = None, return_data = False):

    """Cleans data, computes new features, returns X and y

    In order to properly process simulated data, data = the data to be processed,

    modeldata = the raw training/development data from which the model was developed"""

    #if modeldata is None: modeldata = data  # then data is modeldata

    data = data.replace(['low','medium','high'], [1,2,3],axis=1)

    if modeldata is None:

        polyfit = preprocessing.PolynomialFeatures(2,include_bias = False).fit(data[polyfeats])

        dummyfit = preprocessing.LabelBinarizer().fit(data['Department'])

    else:

        modeldata = modeldata.replace(['low','medium','high'], [1,2,3],axis=1)

        polyfit = preprocessing.PolynomialFeatures(2,include_bias = False).fit(modeldata[polyfeats])

        dummyfit = preprocessing.LabelBinarizer().fit(modeldata['Department'])

    y = data['Left']

    X = (data.drop(['Left'], axis = 1)

         #.replace(['low','medium','high'], [1,2,3],axis=1)

         .merge(computePoly(data[polyfeats], polyfit), left_index=True, right_index=True)

         .merge(computeDummies(data['Department'], dummyfit), left_index=True, right_index = True)

         .assign(sat3 = lambda x: x.Satisfaction**3) # satisfaction third-order polynomial 

         .drop(['Department']+polyfeats,axis = 1))

    if return_data == True:

        return X

    X = scale(X, modeldata, polyfeats)

    return X, y

    

def scale(X, modeldata, polyfeats):

    scalerfit = get_scalerfit(X, modeldata, polyfeats)

    return pd.DataFrame(scalerfit.transform(X), columns = X.columns)



def get_scalerfit(X, modeldata, polyfeats):

    if modeldata is None:

        return preprocessing.StandardScaler().fit(X)

    else:

        X = newfeatures(modeldata, polyfeats, return_data = True)

        return preprocessing.StandardScaler().fit(X)

    

def computePoly(data, polyfit):

    output_nparray = polyfit.transform(data)

    target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(data.columns,p) for p in polyfit.powers_]]

    return pd.DataFrame(output_nparray, columns = target_feature_names)



def computeDummies(data, dummyfit):

    return pd.DataFrame(dummyfit.transform(data), 

                               columns = dummyfit.classes_).iloc[:,:-1] # drop one dummy column
polyfeats = ['Satisfaction','Evaluation','Projects','Hours','Tenure','Salary']



X, y = newfeatures(data[orig_vars], polyfeats)



X_train, X_test, y_train, y_test = train_test_split(

            X, y, test_size = 0.2, random_state = 8)
## Tune hyperparameters

lrcv = LogisticRegressionCV(cv=5, random_state = 20)

lrcv.fit(X_train,y_train)

print('Optimal C = {0:.4f}'.format(lrcv.C_[0]))
## Get cross validation score

lr = LogisticRegression(C = lrcv.C_[0], random_state=5)

lr.fit(X_train,y_train)

print('10-fold cross validation accuracy: {0:.2f}%'

      .format(np.mean(cross_val_score(lr, X_train, y_train,cv = 10))*100))
print('Precision/Recall Table for Training Data: \n')

print(classification_report(y_train, lr.predict(X_train)))

prc = precision_recall_curve(y_train, lr.decision_function(X_train), pos_label=1);

plt.plot(prc[1],prc[0]);

plt.xlabel('Recall');

plt.ylabel('Precision');

plt.title('Precision-Recall Curve:\nTraining Data');
print('Test accuracy: {0:.2f}%'.format(lr.score(X_test,y_test)*100))
print('Precision/Recall Table for Test Data: \n')

print(classification_report(y_test, lr.predict(X_test)))

prc = precision_recall_curve(y_test, lr.decision_function(X_test), pos_label=1);

plt.plot(prc[1],prc[0]);

plt.xlabel('Recall');

plt.ylabel('Precision');

plt.title('Precision-Recall Curve:\nTest Data');
def simulate(x, row, col, data, SKLmodel):

    # simulate data at different levels of x, row, and col

    sim = pd.DataFrame(columns = data.columns)

    for q1 in data[row].quantile([.05,.30,.70,.95]).values:

        for q2 in data[col].quantile([.05,.30,.70,.95]).values:

            sim2 = pd.DataFrame(columns = data.columns).append(data.median(), ignore_index=True)

            sim2 = sim2.append([sim2]*99,ignore_index=True)

            sim2[x] = np.linspace(data[x].min(), data[x].max(), 100)

            sim2[col] = q2

            sim2[row] = q1

            sim = sim.append(sim2, ignore_index = True)

    sim2 = sim.assign(Department = 'IT').assign(Salary = 'low')

    sim = sim.assign(Department = 'IT').assign(Salary = 'high').append(sim2, ignore_index = True)

    Xsim, _ = newfeatures(sim, polyfeats, data)

    

    # generate probabilities using model, add to DF

    probs = SKLmodel.predict_proba(Xsim)

    sim['Probability'] = probs[:,1]

    

    # Plot

    sns.set(style="white", 

            rc = {'axes.titlesize' : 20,

            'axes.labelsize' : 20,

            'xtick.labelsize' : 20,

            'ytick.labelsize' : 20,

            'figure.titlesize': 40})

    grid = sns.FacetGrid(sim, col=col, row=row, hue="Salary", size = 5) # initialize grid

    grid.map(plt.axhline, y=.5, ls=":", c=".5")     # P = .5 horizontal line

    grid.map(plt.plot, x, 'Probability', ls = '-',linewidth = 4) # draw lines

    ## NOTE: make sure the lines for the main variables are plotted LASt, other wise labels will be changed

    grid.set(yticks=[], ylim = (0,1)) #  remove y ticks

    grid.fig.tight_layout(w_pad = 1) # arrangement

    plt.subplots_adjust(top=0.9) # space for title

    grid.fig.suptitle('Probability of leaving by {0}, {1}, and {2}'.format(x,row,col)) # title
data = data[orig_vars]

x, r, c = 'Satisfaction, Hours, Projects'.split(', ')

simulate(x, r, c, data, lr)
x, r, c = 'Tenure, Hours, Projects'.split(', ')

simulate(x, r, c, data, lr)
x, r, c = 'Projects, Evaluation, Hours'.split(', ')

simulate(x, r, c, data, lr)
from sklearn.svm import SVC

X2, y2 = process(data[orig_vars])

X2_train, X2_test, y2_train, y2_test = train_test_split(

            X2, y2, test_size = 0.1, random_state = 50)

clf = SVC()

clf.fit(X2_train, y2_train) 

print('Prediction accuracy on test data with no feature engineering: ',clf.score(X = X2_test, y = y2_test))

X2, y2 = newfeatures(data[orig_vars], polyfeats)

X2_train, X2_test, y2_train, y2_test = train_test_split(

            X2, y2, test_size = 0.1, random_state = 50)

clf = SVC(probability=True)

clf.fit(X2_train, y2_train) 

print('Prediction accuracy on test data WITH feature engineering: ',clf.score(X = X2_test, y = y2_test))
x, r, c = 'Satisfaction, Projects, Hours'.split(', ')

g = simulate(x, r, c, data, clf)