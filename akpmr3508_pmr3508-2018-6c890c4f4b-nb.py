import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import fbeta_score, make_scorer, roc_curve, auc, fbeta_score, make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import interp
dfTrain = pd.read_csv("../input/spamdata/train_data.csv",
          sep=r'\s*,\s*',
          engine='python',
          na_values="")
dfTest = pd.read_csv("../input/spamdata/test_features.csv",
         sep=r'\s*,\s*',
         engine='python',
         na_values="")
dfTrain.shape
dfTest.shape
dfTrain = dfTrain.drop(columns=['Id'])
dfTrain.head()
dfTrain[dfTrain['ham']==False].describe()
dfTrain[dfTrain['ham']==True].describe()
plt.figure(figsize=(30,30))
plt.title('Matriz de correlação')
sns.heatmap(dfTrain.corr(), annot=True, linewidths=0.1)
threshold = 0.7
bkpHam = dfTrain['ham']
dfWithoutHam = dfTrain.drop(['ham'], axis=1)
selection = np.full((dfWithoutHam.corr().shape[0],), True)
for i in range(dfWithoutHam.corr().shape[0]):
    for j in range(i+1, dfWithoutHam.corr().shape[0]):
        if dfWithoutHam.corr().iloc[i,j] >= threshold:
            selection[j] = False
dfTrain = dfWithoutHam[dfWithoutHam.columns[selection]]
dfTrain = dfTrain.assign(ham=bkpHam)
plt.figure(figsize=(30,30))
plt.title('Matriz de correlação')
sns.heatmap(dfTrain.corr(), annot=True, linewidths=0.1)
col = dfTrain.columns.difference(['ham'])
# Normalizacao dos dados
dfTrainNorm = (dfTrain[col] - dfTrain[col].mean()) / dfTrain[col].std()
dfTrainNorm['ham'] = dfTrain['ham']
dfPlot = pd.melt(dfTrainNorm, id_vars='ham', var_name='features', value_name='value')
plt.figure(figsize=(20,50))
plt.title('Plotagem da distribuição de cada feature para ham x spam')
sns.violinplot(x='features', y='value', hue='ham', data=dfPlot, split=True, inner='quartile')
plt.xticks(rotation=90)
dfPlot = pd.melt(dfTrainNorm[['word_freq_your', 'ham']], id_vars="ham", var_name="features", value_name='value')
plt.figure(figsize=(20,15))
plt.title('Plotagem da distribuição de \'word_freq_your\' para ham x spam')
sns.violinplot(x="features", y="value", hue="ham", data=dfPlot, split=True, inner="quartile")
plt.xticks(rotation=90)
selected_features = [
    'word_freq_remove',
    'word_freq_3d',
    'word_freq_will',
    'word_freq_addresses',
    'word_freq_free',
    'word_freq_business',
    'word_freq_you',
    'word_freq_credit',
    'word_freq_your',
    'word_freq_font',
    'word_freq_000',
    'word_freq_money',
    'word_freq_hp',
    'word_freq_hpl',
    'word_freq_george',
    'word_freq_650',
    'word_freq_lab',
    'word_freq_85',
    'word_freq_parts',
    'word_freq_pm',
    'word_freq_cs',
    'word_freq_meeting',
    'word_freq_original',
    'word_freq_project',
    'word_freq_re',
    'word_freq_edu',
    'word_freq_table',
    'word_freq_conference',
    'char_freq_;',
    'char_freq_[',
    'char_freq_$',
    'capital_run_length_average',
    'capital_run_length_longest',
    'capital_run_length_total'
]
scaler = MinMaxScaler()
dfScaled = scaler.fit_transform(dfTrain[selected_features])
x_train, x_test, y_train, y_test = train_test_split(dfScaled, dfTrain['ham'], test_size=0.20)
mnb = MultinomialNB()
f3_scorer = make_scorer(fbeta_score, beta=3)
f3 = cross_validate(mnb, x_train, y_train, cv=10, scoring=f3_scorer)
metric = cross_validate(mnb, x_train, y_train, cv=10, scoring=['accuracy', 'precision', 'recall'])
print("Teste Accuracy: " + str(metric['test_accuracy'].mean()))
print("Teste Precision: " + str(metric['test_precision'].mean()))
print("Teste Recall: " + str(metric['test_recall'].mean()))
print("Teste F3: " + str(f3['test_score'].mean()))
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)
fbeta_score(y_test, y_pred, beta=3)
x_data = scaler.fit_transform(dfTrain[selected_features])
y_data = dfTrain['ham'].values.reshape(-1, 1).ravel()
cv = StratifiedKFold(n_splits=6)
mnb = MultinomialNB()

x_data = scaler.fit_transform(dfTrain[selected_features])
y_data = dfTrain['ham']

tprs = []
thrs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 8))

i = 0
for train, test in cv.split(x_data, y_data):
    prob = mnb.fit(x_data[train,:], y_data[train]).predict_proba(x_data[test,:])
    fpr, tpr, thresholds = roc_curve(y_data[test], prob[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    thrs.append(interp(mean_fpr, fpr, thresholds))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
mean_threshold = np.mean(thrs, axis=0)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
dist_optimal = np.sqrt(np.power(mean_fpr[0], 2) + np.power(mean_tpr[0] - 1, 2))
best_idx = 0
for i in range(len(mean_tpr)):
    new_dist = np.sqrt(np.power(mean_fpr[i], 2) + np.power(mean_tpr[i] - 1, 2))
    if new_dist < dist_optimal:
        dist_optimal = new_dist
        best_idx = i
print('Resultado:')
print('True Positive Rate: ' + str(mean_tpr[best_idx]))
print('False Positive Rate: ' + str(mean_fpr[best_idx]))
print('Threshold: ' + str(mean_threshold[best_idx]))
mnb = MultinomialNB(class_prior=[1-mean_threshold[best_idx], mean_threshold[best_idx]])
f3_scorer = make_scorer(fbeta_score, beta=3)
f3 = cross_validate(mnb, x_train, y_train, cv=10, scoring=f3_scorer)
metric = cross_validate(mnb, x_train, y_train, cv=10, scoring=['accuracy', 'precision', 'recall'])
print("Teste Accuracy: " + str(metric['test_accuracy'].mean()))
print("Teste Precision: " + str(metric['test_precision'].mean()))
print("Teste Recall: " + str(metric['test_recall'].mean()))
print("Teste F3: " + str(f3['test_score'].mean()))
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)
fbeta_score(y_test, y_pred, beta=3)
x_val_test = scaler.transform(dfTest[selected_features])
y_val_test = mnb.predict(x_val_test)
dfSave = pd.DataFrame(data={"Id" : dfTest["Id"], "ham" : y_val_test})
pd.DataFrame(dfSave[["Id", "ham"]], columns = ["Id", "ham"]).to_csv("Output.csv", index=False)
