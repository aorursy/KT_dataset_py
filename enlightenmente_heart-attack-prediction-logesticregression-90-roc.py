# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')
# Importing relevant libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
# Setting the visual preferance
plt.style.use('dark_background')
df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
df.head()
df.info()
df.shape
df.duplicated().sum()
round(df.isnull().sum()/len(df.index)*100,4)
df.rename(columns = {'cp': 'chest pain', 'trestbps': 'resting BP', 'chol': 'cholestoral', 'fbs': 'fasting Blood sugar',
                    'restecg': 'resting ECG', 'thalach': 'maximum heart rate', 'exang': 'exercise induced angina',
                    'oldpeak': 'ST depression', 'ca': 'no.of major vessels blocked', 'thal': 'defect'}, inplace = True)
df.head()
var = df.drop('target', axis = 1).columns
plt.figure(figsize = (15,15))
for x in enumerate(var):
    plt.subplot(5,3,x[0]+1)
    sns.boxplot(x[1], data = df, palette = 'Purples')
plt.show()
outliers = ['resting BP', 'cholestoral', 'maximum heart rate', 'ST depression', 
            'no.of major vessels blocked', 'defect']
plt.figure(figsize = (15, 10))
for x in enumerate(outliers):
    plt.subplot(2,3,x[0]+1)
    sns.boxplot(x[1], data = df, palette = 'Purples')
plt.show()
df['resting BP'].quantile([0.25,0.50,0.75,0.90,0.95,0.99])
df.loc[df['resting BP']> 160, ['resting BP']] = 160
sns.countplot(x = 'sex', data = df, palette = 'Purples')
plt.show()
bins = [0,10,20,30,40,50,60,70,80]
labels = ['<10', '10-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']
df['age_group'] = pd.cut(x = df['age'], bins = bins, labels = labels)
plt.figure(figsize = (10,5))
sns.countplot(x = df['age_group'], hue= df['target'], palette= 'Purples')
plt.show()
one = df.loc[df['target'] == 1]
zero = df.loc[df['target'] == 0]
var = ['resting BP', 'cholestoral', 'maximum heart rate', 'ST depression']
plt.figure(figsize = (14,7))
for x in enumerate(var):
    plt.subplot(2,2,x[0]+1)
    sns.kdeplot(data = one[x[1]], shade = True, color = 'r')
    
for x in enumerate(var):
    plt.subplot(2,2,x[0]+1)
    sns.kdeplot(data = zero[x[1]], shade = True, color = 'c')
plt.show()
plt.figure(figsize = (10,5))
sns.countplot(x = 'chest pain', hue = 'target', data = df, palette = 'Purples')
plt.show()
# Calculating the imbalance percentage.
label = ['Heart Attack', 'Non-Heart Attack']
explode = [0.1,0]
df['target'].value_counts().plot.pie(explode = explode, labels = label, shadow = True, startangle=60, 
                                      autopct='%1.1f%%', textprops = {'color' : 'k'})
plt.show()
df.drop('age_group', axis = 1, inplace = True)
df.drop_duplicates(inplace = True)
# Splitting the data into train and test.
df_train, df_test = train_test_split(df, train_size = 0.70, random_state = 100)
print(df_train.shape)
print(df_test.shape)
# Rescalling of variable.
var = ['age', 'chest pain', 'resting BP', 'cholestoral', 'resting ECG', 'maximum heart rate', 
       'ST depression', 'slope', 'no.of major vessels blocked', 'defect']
scaler = MinMaxScaler()
df_train[var] = scaler.fit_transform(df_train[var])
df_train.describe()
# Calculating the correlation between variables.
df_train.corr()
# Heatmap
plt.figure(figsize = (15,10))
heat = sns.heatmap(df_train.corr(), annot = True, cmap = 'Purples')
bottom, top = heat.get_ylim()
heat.set_ylim(bottom+0.5, top+0.5)
plt.show()
Y_train = df_train.pop('target')
X_train = df_train
log_reg = LogisticRegression()
rfe = RFE(log_reg, 10)
rfe_model = rfe.fit(X_train, Y_train)
pd.DataFrame(zip(X_train.columns, rfe_model.ranking_)).sort_values(by = 1, ascending = True)
col = X_train.columns[rfe_model.support_]
col
# Model_1
X_train_sm = sm.add_constant(X_train[col])
log_model = sm.GLM(Y_train, X_train_sm, families = sm.families.Binomial).fit()
print(log_model.summary())
vif = pd.DataFrame()
vif['Features'] = col
vif['VIF'] = [variance_inflation_factor(X_train[col].values, x) for x in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
X_2 = X_train[col].drop('cholestoral', axis = 1)
X_2_sm = sm.add_constant(X_2)
log_model_2 = sm.GLM(Y_train, X_2_sm, families = sm.families.Binomial()).fit()
print(log_model_2.summary())
vif = pd.DataFrame()
vif['Features'] = X_2.columns
vif['VIF'] = [variance_inflation_factor(X_2.values, x) for x in range(X_2.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
X_3 = X_2.drop('slope', axis = 1)
X_3_sm = sm.add_constant(X_3)
log_model_3 = sm.GLM(Y_train, X_3_sm, families = sm.families.Binomial()).fit()
print(log_model_3.summary())
vif = pd.DataFrame()
vif['Features'] = X_3.columns
vif['VIF'] = [variance_inflation_factor(X_3.values, x) for x in range(X_3.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
X_4 = X_3.drop('exercise induced angina', axis = 1)
X_4_sm = sm.add_constant(X_4)
log_model_4 = sm.GLM(Y_train, X_4_sm, families = sm.families.Binomial()).fit()
print(log_model_4.summary())
vif = pd.DataFrame()
vif['Features'] = X_4.columns
vif['VIF'] = [variance_inflation_factor(X_4.values, x) for x in range(X_4.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
X_5 = X_4.drop('age', axis = 1)
X_5_sm = sm.add_constant(X_5)
log_model_5 = sm.GLM(Y_train, X_5_sm, families = sm.families.Binomial()).fit()
print(log_model_5.summary())
vif = pd.DataFrame()
vif['Features'] = X_5.columns
vif['VIF'] = [variance_inflation_factor(X_5.values, x) for x in range(X_5.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
X_6 = X_5.drop('defect', axis = 1)
X_6_sm = sm.add_constant(X_6)
log_model_6 = sm.GLM(Y_train, X_6_sm, families = sm.families.Binomial()).fit()
print(log_model_6.summary())
vif = pd.DataFrame()
vif['Features'] = X_6.columns
vif['VIF'] = [variance_inflation_factor(X_6.values, x) for x in range(X_6.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
# Predicting using the log_model_6
Y_train_pred = log_model_6.predict(X_6_sm)
# Conversion at different probability
cutoff = pd.DataFrame()
cutoff['Actual'] = Y_train.values
cutoff['Pred'] = Y_train_pred.values
num = [float(x/10) for x in range(10)]
for x in num:
    cutoff[x] = cutoff['Pred'].map(lambda i: 1 if i > x else 0)
cutoff.head()
# Calculating various measures.
measures = pd.DataFrame(columns = ['Probability', 'Accuracy', 'Sensitivity', 'FPR', 'Specificity', 'FNR'])
for x in num:
    metrix = metrics.confusion_matrix(cutoff['Actual'], cutoff[x])
    total = sum(sum(metrix))
    Accuracy = (metrix[0,0]+metrix[1,1])/total
    Sensitivity = metrix[1,1]/(metrix[1,1]+metrix[1,0])
    FPR = metrix[0,1]/(metrix[0,1]+metrix[0,0])
    Specificity = metrix[0,0]/(metrix[0,0]+metrix[0,1])
    FNR = metrix[1,0]/(metrix[1,0]+metrix[1,1])
    measures.loc[x] = [x, Accuracy, Sensitivity, FPR, Specificity, FNR]
measures
# Plotting the lines to find the optimal Threshold limit.
measures.plot.line(x = 'Probability', y = ['Accuracy', 'Sensitivity', 'Specificity'])
plt.show()
def roc (actual, prob):
    FPR, TPR, threshold = metrics.roc_curve(actual, prob, drop_intermediate = False)
    auc_score = metrics.roc_auc_score(actual, prob)
    plt.plot(FPR, TPR, label = 'ROC curve (area = %0.2f)' %auc_score)
    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.show()
    return None
FPR, TPR, threshold = metrics.roc_curve(cutoff['Actual'], cutoff['Pred'], drop_intermediate = False)
roc(cutoff['Actual'], cutoff['Pred'])
measures.loc[measures['Probability']== 0.5]
df_test[var] = scaler.transform(df_test[var])
df_test.describe()
# Assigning X and Y
Y_test = df_test.pop('target')
X_test = df_test
# Matching the test data with Log_model_6 columns
cols = X_6.columns
X_test = X_test[cols]
# Prediction on test data
X_test_sm = sm.add_constant(X_test)
Y_test_pred = log_model_6.predict(X_test_sm)
test = pd.DataFrame()
test['Actual'] = Y_test.values
test['Pred'] = Y_test_pred.values
test['Final'] = test['Pred'].map(lambda x: 1 if x >= 0.5 else 0)
test.head()
con = metrics.confusion_matrix(test['Actual'], test['Final'])
con
sensitivity = con[1,1]/(con[1,1]+con[1,0])
specificity = con[0,0]/(con[0,0]+con[0,1])
FNR = con[1,0]/(con[1,0]+con[1,1])
print({'Accuracy': round(metrics.accuracy_score(test['Actual'], test['Final']),2)})
print({'Sensitivity': round(sensitivity, 2)})
print({'Specificity': round(specificity, 2)})
print({'FNR': round(FNR, 2)})
df_test['target'] = Y_test.values
df_test['probability'] = test['Pred'].values
df_test['final'] = test['Final'].values
df_test.head(10)
