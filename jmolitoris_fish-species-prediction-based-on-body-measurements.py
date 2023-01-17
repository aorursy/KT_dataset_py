# Import relevant modules

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

import seaborn as sns

sns.set(style ='white', palette = 'colorblind')
# Load the Data

df = pd.read_csv("../input/fish-market/Fish.csv")

df.head(10)
print("The species were {}.".format(list(set(df.Species))))

print('There are {} observations in our dataframe.'.format(len(df)))
# Plotting Distributions of Columns

plt.rcParams['xtick.labelsize'] = 16

plt.rcParams['ytick.labelsize'] = 16

fig, ax = plt.subplots(2, 3, sharey = True, figsize = [16, 10])

fig.suptitle('Distribution of Fish Measurements', fontsize = 26)

fig.subplots_adjust(hspace = 0.2)

i = 0

j = 0

for col in df.iloc[:, 1:]:

    ax[i, j].hist(df[col], bins = 20, histtype = 'bar')

    ax[i, j].set_title(col, fontsize = 20)

    ax[i, j].text(0.65, 0.9, r"$\mu$ = {: .1f}".format(df[col].mean()), transform = ax[i, j].transAxes, fontsize = 16)

    ax[i, j].text(0.65, 0.8, r"$\sigma$ = {: .1f}".format(df[col].std()), transform = ax[i, j].transAxes, fontsize = 16)

    if j < 2:

        j += 1

    else:

        i += 1

        j -= 2
df.describe().round(2)
# Replace Zeroes with Median Weight of all fish

median_weight = df['Weight'][df['Weight']!=0].median()

df['Weight'] = df['Weight'].mask(df['Weight'] == 0, median_weight)
df['Weight'].describe().round(2)
# Correlation Matrix of Features

corrs = df.corr().round(2)

plt.figure(figsize = (10, 8))

sns.heatmap(corrs, cmap = 'Greys')
df = df.drop(['Length1', 'Length2'], axis = 1)
df['wid_for_height'] = df['Width'] / df['Height'] 

df['weight_for_height'] = df['Weight'] / df['Height']
x = df.iloc[:, 1:]

y = df.iloc[:, 0]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 123)
#Multinomial Logit

log = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 20000)

log.fit(x_train, y_train)

preds = log.predict(x_test)

species_list = log.classes_.tolist()

conf_mx = confusion_matrix(y_test, preds, species_list)

print("\t\t Model Metrics")

print("Precision: \t", precision_score(y_test, preds, average = 'weighted').round(2))

print("Recall: \t", recall_score(y_test, preds, average = 'weighted').round(2))

print("F1 Score: \t", f1_score(y_test, preds, average = 'weighted').round(2))
randompreds = np.random.choice(list(set(df['Species'])), size = len(y_test))

all_perch = np.full(len(y_test), fill_value = 'Perch')



print("If I randomly guessed, the precision score would be {}.".format(precision_score(y_test, randompreds, average = 'weighted')))

print("If I guessed all fish were perch, it would be {}.".format(precision_score(y_test, all_perch, average = 'weighted')))
# Matrix of errors

row_sums = conf_mx.sum(axis = 1, keepdims = True)

norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)



fig = plt.figure(figsize = (10, 8))

fig.tight_layout()

ax = fig.add_subplot(111)

cax = ax.matshow(norm_conf_mx, cmap = 'gist_heat_r')

plt.title('Proportion of Incorrect Predictions\nfor Fish Classifier', fontsize = 16)

fig.colorbar(cax)

plt.gca().xaxis.tick_bottom()

ax.set_xticklabels([''] + species_list, rotation = 30)

ax.set_yticklabels([''] + species_list)

plt.xlabel('Predicted Class', fontsize = 16)

plt.ylabel('Actual Class', fontsize = 16)