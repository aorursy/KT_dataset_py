import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from ipywidgets import interact,  FloatSlider



from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve



from imblearn.under_sampling import TomekLinks

from imblearn.over_sampling import SMOTE
pd.options.display.max_columns = 100

pd.options.display.max_rows = 100

pd.options.display.width=100

plt.style.use('ggplot')
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head()
df.shape
df.isnull().sum().sum()
df.describe()
df.dtypes
diff_class = df['Class'].value_counts()

diff_class.plot(kind='bar', color=['m', 'k'], figsize=(5, 5))

plt.xticks(range(2), ['Normal  [0]', 'Fraud  [1]'], rotation=0)

for i, v in enumerate(diff_class):

    plt.text(i-0.1, v+3000, str(v))

plt.title('Class Count')

plt.show()
ss = StandardScaler()

df['Amount'] = ss.fit_transform(df[['Amount']])

df['Time'] = ss.fit_transform(df[['Time']])
for var in df.columns[:-1]:

    

    sns.boxplot(df[var], hue=df['Class'], palette='Set3')

    mean = df[var].mean()

    std = df[var].std()

    plt.axvline(mean - 3 * std, 0, 1)

    plt.text(mean - 3 * std, -0.55, 'mean - 3* std', rotation=60)

    plt.axvline(mean + 3 * std, 0, 1)

    plt.text(mean + 3 * std, -0.55, 'mean + 3* std', rotation=60)

    



    plt.show()

    
X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df[['Class']].values, test_size=0.3,random_state=1997)
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
def plot_confusion_matrix(y_test, y_pred):

    cf_matrix = confusion_matrix(y_test, y_pred)



    labels_name = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    labels_count = [value for value in cf_matrix.flatten()]

    labels_percentage = [ "{0:.2%}".format(value) for value in cf_matrix.flatten()/ np.sum(cf_matrix)]



    labels = [f'{x}\n {y} \n{z}' for x, y, z in zip(labels_name, labels_count, labels_percentage)]

    labels = np.array(labels, dtype=str).reshape(2,2)



    recall = cf_matrix[1,1]/(cf_matrix[1,0] + cf_matrix[1,1])

    precision = cf_matrix[1,1]/(cf_matrix[0,1] + cf_matrix[1,1])

    accuracy = (cf_matrix[0, 0] + cf_matrix[1,1])/ np.sum(cf_matrix)

    f1_score = (2*precision*recall)/(precision + recall)



    stats = '\n\n Recall:   {0:.03}\n Precision:   {1:.03}\n Accuracy:  {2:.03}\nF1-Score:  {3:.03}'.format(recall, precision, accuracy, f1_score)



    sns.heatmap(cf_matrix, annot=labels, fmt='', center=3, linewidth=3, linecolor='k', cbar=False)

    plt.title('Confusion matrix\n', fontsize=20)

    plt.xlabel('Predicted Label'+stats, fontsize=14)

    plt.ylabel('True Label', fontsize=14)



    plt.show()

plot_confusion_matrix(y_test, y_pred)
y_prob = lr.predict_proba(X_test)

y_prob = y_prob[:, 1] # Probability of getting the output 1 (Fraud)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# gmeans = np.sqrt(tpr*(1-fpr))

# ix = np.argmax(gmeans)

# print("Best thresholds=%f, G-Mean=%.3f" %(thresholds[ix], gmeans[ix]))



plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

plt.plot(fpr, tpr, marker='.', label='Logistic')

# plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='best', sizes=(200, 100))



plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.title('TPR vs FPR', fontsize=20)

plt.show()
lr_precision, lr_recall, lr_thresholds = precision_recall_curve(y_test, y_prob)

no_skill = len(y_test[y_test==1])/ len(y_test)

plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

plt.plot(lr_precision, lr_recall, marker='.', label='Logistic')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.title('Precision vs Recall', fontsize=20)

plt.show()
print("Slide, Range -> (0.001, 0.04)")

def update(var=0.004):

    print("y_prob should be greater than >", var)

    predict_mine = np.where(y_prob > var, 1, 0)

    plot_confusion_matrix(y_test, predict_mine)



interact(update, var=FloatSlider(min=0.001, max=0.04, step=0.001))
lr_b = LogisticRegression(max_iter=1000, class_weight='balanced')

lr_b.fit(X_train, y_train)

y_pred_b = lr_b.predict(X_test)

plot_confusion_matrix(y_test, y_pred_b)
tl = TomekLinks(sampling_strategy='majority')

X_train_tl, y_train_tl = tl.fit_sample(X_train, y_train)

lr_tl = LogisticRegression(max_iter=1000, class_weight='balanced')

lr_tl.fit(X_train_tl, y_train_tl)

y_pred_tl = lr_tl.predict(X_test)

plot_confusion_matrix(y_test, y_pred_tl)
smote  = SMOTE(sampling_strategy='minority')

X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)
lr_sm = LogisticRegression(max_iter=1000)

lr_sm.fit(X_train_sm, y_train_sm)

y_pred_sm = lr_sm.predict(X_test)

plot_confusion_matrix(y_test, y_pred_sm)