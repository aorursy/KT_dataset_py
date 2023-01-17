import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# %matplotlib notebook
file = '../input/covid19/dataset.xlsx'



data = pd.read_excel(file)



data.tail()
# define the label to be predicted

label = 'SARS-Cov-2 exam result'
# replace positive/negative by 1/0

data[label] = [1 if result=='positive' else 0 for result in data[label]]



data.tail()
ref = data[label].mean()



print(f'Percentage of all COVID-19 positives in data: {round(ref*100)}%')
other_labels = [

    'Patient addmited to regular ward (1=yes, 0=no)'

    'Patient addmited to semi-intensive unit (1=yes, 0=no)',

    'Patient addmited to intensive care unit (1=yes, 0=no)',

]
features = list(set(data.columns) - set(['Patient ID', label] + other_labels))



features_numerical = []

features_categorical = []

features_empties = []



for feature in features:

    hues = set(data[feature][~data[feature].isna()])

    if len(hues)==0:

        features_empties.append(feature)

    else:

        is_string = isinstance(list(hues)[0], str)

        if is_string:

            features_categorical.append(feature)

        else:

            features_numerical.append(feature)

        

        

print(f'{len(features_numerical)} numerical features: ', ', '.join(features_numerical))

print()

print(f'{len(features_categorical)} categorical features: ', ', '.join(features_categorical))

print()

print(f'{len(features_empties)} discarded empty features: ', ', '.join(features_empties))
def plot_age(data, label, save=False):

    

    feature = 'Patient age quantile'

    

    plt.figure()

    

    # plot dashed reference line

    dat = data.groupby(feature).count().reset_index()

    x = list(dat[feature])

    y = [ref*100] * len(x)

    ax = sns.lineplot(x, y, c='darkred', label='Population Reference')

    ax.lines[0].set_linestyle("--")



    # Plot the percentages

    sns.set_color_codes('pastel')

    dat = data.groupby(feature).mean().reset_index()

    dat[label] = dat[label] * 100

    sns.barplot(x=feature, y=label, label='Positive Rate', data=dat, color='r')

    

    plt.legend()

    plt.ylabel('Percentage (%)')

    samples = data.groupby(feature)[label].count().sum()

    plt.title(f'{label} (out of {samples}) per {feature}')



    if save: plt.savefig(f'./plots/{label}/{label}_per_{feature.replace("/", "_")}.png')

        

        

plot_age(data, label)
def plot_categorical(data, feature, label, save=False):

    plt.figure()



    dat = data.groupby(feature).count().reset_index()



    # Plot the total crashes

    sns.set_color_codes('pastel')

    sns.barplot(x=feature, y=label, label='Negative', data=dat, color='b')



    # Plot the total crashes

    sns.set_color_codes('muted')

    dat = data.groupby(feature).sum().reset_index()

    sns.barplot(x=feature, y=label, label='Positive', data=dat, color='r')

    

    plt.legend()

    plt.ylabel('Suspected')

    samples = data.groupby(feature)[label].count().sum()

    plt.title(f'{label} (out of {samples}) per {feature}')



    if save: plt.savefig(f'./plots/{label}/{label}_per_{feature.replace("/", "_")}.png')

        

        

for feature in features_categorical:

    plot_categorical(data, feature, label)
def plot_numerical(data, feature, label, save=False):

    plt.figure()

    hues = list(set(data[label]))

    for hue in hues:

        sns.distplot(data[feature][data[label]==hue].values, norm_hist=False, kde=False)



        

    hues = ['Negative' if hue==0 else 'Positive' if hue==1 else hue for hue in hues]

    plt.legend(hues)

    plt.xlabel(feature)

    plt.ylabel('Suspected')

    samples = data.groupby(feature)[label].count().sum()

    plt.title(f'{label} (out of {samples}) per {feature}')

    if save: plt.savefig(f'./plots/{label}/{label}_per_{feature.replace("/","_")}.png')

    



for feature in features_numerical:

    plot_numerical(data, feature, label)
features_covid = [

    'Leukocytes',

    'Monocytes',

    'Platelets',

    'Patient age quantile',

]



dat = data[features_covid + [label,]].dropna()



print(dat.shape)

dat.head()
plt.figure()

sns.scatterplot(x=features_covid[0], y=features_covid[1], hue=label, data=dat, 

                  linewidth=0, s=16, alpha = 0.8)

plt.title(f'Clusters {features_covid[0]}-{features_covid[1]}')



plt.figure()

sns.scatterplot(x=features_covid[0], y=features_covid[2], hue=label, data=dat, 

                  linewidth=0, s=16, alpha = 0.8)

plt.title(f'Clusters {features_covid[0]}-{features_covid[2]}')
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



X = dat[features_covid].values

y = dat[label].values



X = scaler.fit_transform(X)



print(X.min(), X.max())

print(X)
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



test_ratio = .2



X_train, X_test, y_train, y_test = train_test_split(X, y,

        test_size=test_ratio, shuffle=True)



print(X_train.shape)

print(X_test.shape)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, fbeta_score



avg = y_train.mean()

avg_int = round(avg)



y_naive = [avg] * len(y_test)

y_naive_int = [avg_int] * len(y_test)



score_naive = mean_absolute_error(y_test, y_naive)

score_naive_int = mean_absolute_error(y_test, y_naive_int)



print(f'Mean of test dataset: {avg_int} = round({avg})')

print(f'Score for naive predictions: {score_naive_int} ({score_naive})')
from sklearn.linear_model import SGDClassifier, RidgeClassifier, RidgeClassifierCV, Perceptron, PassiveAggressiveClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis





models = [

    SGDClassifier(),

    RidgeClassifier(),

    RidgeClassifierCV(),

    Perceptron(),

    PassiveAggressiveClassifier(),

    SVC(kernel='rbf'),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    KNeighborsClassifier(2),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    DecisionTreeClassifier(max_depth=6),

    DecisionTreeClassifier(max_depth=7),

    RandomForestClassifier(),

    RandomForestClassifier(max_depth=7, n_estimators=100),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

]





best_score = 999







X_train, X_test, y_train, y_test = train_test_split(X, y,

        test_size=test_ratio, shuffle=True)



for model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = mean_absolute_error(y_test, y_pred)

#     score = accuracy_score(y_test, y_pred)

#     score = fbeta_score(y_test, y_pred, average='weighted', beta=1)

    if score < best_score:

        best_model = model

        best_score = score

        

    model_class = str(model.__class__).split('.')[-1][:-2]

    print(f'{score}\t{model_class}')
from sklearn.metrics import confusion_matrix



model = best_model



y_pred = model.predict(X_test)

all_labels = list(set(y_test))

CM = confusion_matrix(y_test, y_pred, labels=all_labels)

CM = CM / CM.sum(axis=1, keepdims=True)



CM = pd.DataFrame(CM, index=all_labels, columns=all_labels)



FN = CM.values[0,1] / CM.values[0,:].sum()

FP = CM.values[1,0] / CM.values[1,:].sum()

N  = sum(y_pred==0)

n_test = len(y_test)



print(f'False Positives: {round(FP*100,1)}%')

print(f'False Negatives: {round(FN*100,1)}%')

print(f'Negative Results: {round(N/n_test*100,1)}%')

        



plt.figure()

sns.heatmap(CM, annot=True, cmap="Blues")

model_class = str(model.__class__).split('.')[-1][:-2]

plt.title(f'Normalized Confusion Matrix for {model_class}')

plt.xlabel('True')

plt.ylabel('Predicted')
import numpy as np



n_cross_valid = 599

    

n_test = len(y_test)



FP = []

FN = []

NN = []

CM = np.zeros((2,2))



for i in range(n_cross_valid):

    

    X_train, X_test, y_train, y_test = train_test_split(X, y,

        test_size=0.5, shuffle=True)



#     model = RandomForestClassifier(max_depth=2, n_estimators=100)

    model = RidgeClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    CM = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if (CM[:,1].sum() > 0):# and (np.isnan(CM).sum().sum() == 0):

        FN.append(CM[0,1] / CM[0,:].sum())

        FP.append(CM[1,0] / CM[1,:].sum())

        NN.append(CM[0,:].sum() / CM.sum())

    



print(f'Valid cross validations: {len(NN)} out of {n_cross_valid}')

print(f'False Positives: {round(np.mean(FP)*100,1)}% +- {round(np.std(FP, ddof=1)*100,1)}%')

print(f'False Negatives: {round(np.mean(FN)*100,1)}% +- {round(np.std(FN, ddof=1)*100,1)}%')

print(f'Negative Results: {round(np.mean(NN)*100,1)}% +- {round(np.std(NN, ddof=1)/n_test*100,1)}%')