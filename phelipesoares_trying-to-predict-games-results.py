import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier # To build a classifier tree
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn.metrics import confusion_matrix # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix # to draw a confusion matrix
from sklearn import tree
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict
df = pd.read_csv(r"C:\Users\Phelipe\OneDrive\Data Science\Datasets\BBVA League Games 2019-2020.csv", sep=';')
df.info()
df['Dif Pos'] = df['Pos Atual Casa'] - df['Pos Atual Fora']
df['Dif WinRate'] = df['WinRate Casa'] - df['WinRate Fora']
df['Dif Vence'] = df['Vence Casa'] - df['Vence Fora']
df = df.loc[df['Rodada'] > 4]
def dfif(c):
  if c['Pos Atual Casa'] >= 17:
    return 5
  elif 13 <= c['Pos Atual Casa'] <= 16:
    return 4
  elif 9 <= c['Pos Atual Casa'] <= 12:
    return 3
  elif 5 <= c['Pos Atual Casa'] <= 8:
    return 2
  else:
    return 1

df['Pos Casa'] = df.apply(dfif, axis=1)
def lista(d):
  if d['Pos Atual Fora'] >= 17:
    return 5
  elif 13 <= d['Pos Atual Fora'] <= 16:
    return 4
  elif 9 <= d['Pos Atual Fora'] <= 12:
    return 3
  elif 5 <= d['Pos Atual Fora'] <= 8:
    return 2
  else:
    return 1

df['Pos Fora'] = df.apply(lista, axis=1)
def lista(e):
  if e['Ranking Casa'] is 'S':
    return 1
  elif e['Ranking Casa'] is 'A':
    return 2
  elif e['Ranking Casa'] is 'B':
    return 3
  elif e['Ranking Casa'] is 'C':
    return 4
  elif e['Ranking Casa'] is 'D':
    return 5
  else:
    return 6

df['Ranking Casa'] = df.apply(lista, axis=1)
def lista(f):
  if f['Ranking Fora'] is 'S':
    return 1
  elif f['Ranking Fora'] is 'A':
    return 2
  elif f['Ranking Fora'] is 'B':
    return 3
  elif f['Ranking Fora'] is 'C':
    return 4
  elif f['Ranking Fora'] is 'D':
    return 5
  else:
    return 6

df['Ranking Fora'] = df.apply(lista, axis=1)
df['Dif Ranking'] = df['Ranking Casa'] - df['Ranking Fora']
df1 = df[['Dif Pos', 'Dif Ranking', 'Dif Vence', 'Dif WinRate']]
x = df1
y = df[['Resultado.1']].astype(int)
import seaborn as sns
df3 = df[['Dif Pos', 'Dif Ranking', 'Dif Vence', 'Dif WinRate', 'Resultado.1']]
# Correlation Matrix
corr = df3.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.boxplot(y='Dif WinRate', x='Resultado.1', data=df3)
sns.boxplot(y='Dif Vence', x='Resultado.1', data=df3)
df3.hist(column='Dif Pos', by='Resultado.1', bins='sturges', figsize=(15,10))
df3.hist(column='Dif Ranking', by='Resultado.1', bins='sturges', figsize=(15,10))
h = df3.hist(bins='sturges', figsize=(16,16))
X_dummed = x
print(X_dummed)
# In this case, I did'n need to do One-Hot Encoding, because my variables are continuous or discrete.
# Split the data in Train and Test
X_train, X_test, y_train, y_test = train_test_split(X_dummed, y, random_state=0)

# create a decision tree and fit it into the training data
clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt = clf_dt.fit(X_train, y_train)
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=["Casa", "Empate", "Fora"],
          feature_names=X_dummed.columns);
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=["Casa", "Empate", "Fora"])
clf_dt.score(X_test, y_test)
path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # determine values for alpha
ccp_alphas = path.ccp_alphas #extract different values for alpha
ccp_alphas = ccp_alphas[:-1] #exclude the maximum value for alpha

clf_dts = [] #create an array that we will put decision trees into

## now create one decision tree per value for alpha and store it in the array

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
ax.legend()
plt.show()
clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=0.008)
scores = cross_val_score(clf_dt, X_train, y_train, cv=50)
dff = pd.DataFrame(data={'tree': range(50), 'accuracy': scores})

dff.plot(x='tree', y='accuracy', marker='o', linestyle='--')
## create an array to store the results of each fold during cross validation

alpha_loop_values = []

# For each candidate value for alpha, we will run 10-fold cross validation
# Then we will store the mean and standard deviation of the scores (the accuracy) for each call
# to cross_val_score in alpha_loop_values..

for ccp_alpha in ccp_alphas:
    clf_df = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=50)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

# Now we can draw a graph of the means and Standard Deviations of the scores
# for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values,
                            columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                  y='mean_accuracy',
                  yerr='std',
                  marker='o',
                  linestyle='--')
alpha_results[(alpha_results['alpha'] > 0.0025)
                &
            (alpha_results['alpha'] < 0.006)]
ideal_ccp_alpha = float(0.003137)
ideal_ccp_alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=0,
                                      ccp_alpha=ideal_ccp_alpha, min_samples_leaf=1, max_depth=4)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)


plot_confusion_matrix(clf_dt_pruned,
                     X_test,
                     y_test,
                     display_labels=["Casa", "Empate", "Fora"])
clf_dt_pruned.score(X_test, y_test)
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned,
         filled=True,
         rounded=True,
         class_names=["Casa", "Empate", "Fora"],
         feature_names=X_dummed.columns);
from sklearn.ensemble import GradientBoostingClassifier

clf_GBDT = GradientBoostingClassifier(n_estimators=50, min_samples_leaf=1, max_depth=3, random_state=0).fit(X_train, np.ravel(y_train))
clf_GBDT.score(X_test, np.ravel(y_test))
plot_confusion_matrix(clf_GBDT,
                     X_test,
                     y_test,
                     display_labels=["Casa", "Empate", "Fora"])
clf_GBDT.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier
clf_RFC = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0, class_weight='balanced', ccp_alpha=0.01)
clf_RFC.fit(X_train, np.ravel(y_train))

plot_confusion_matrix(clf_RFC,
                     X_test,
                     y_test,
                     display_labels=["Casa", "Empate", "Fora"])
clf_RFC.score(X_test, y_test)
from sklearn.ensemble import AdaBoostClassifier
clf_ab = AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.2)
clf_ab.fit(X_train, np.ravel(y_train))

plot_confusion_matrix(clf_ab,
                     X_test,
                     y_test,
                     display_labels=["Casa", "Empate", "Fora"])
clf_ab.score(X_test, y_test)
df['Prediction'] = clf_RFC.predict(X_dummed)
print(df['Prediction'])
excelfile = df.merge(df['Prediction'], left_index=True, right_index=True)
excelfile.to_excel("Prediction.xlsx", sheet_name="Random Forest")