import numpy as np
import numpy as np
import pandas as pd
import pandas_profiling as pf
from collections import Counter

# VisualiZation
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# Configure visualisations
%matplotlib inline
plt.style.use('ggplot')

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_colwidth = 130
pima_dictionary = pd.read_excel('../input/diabetespima/pima_dictionary.xlsx', index_col='Column Name')
pima_dictionary
pima = pd.read_csv('../input/pima-indians/PimaIndians.csv') 
pima.head(2)
#checking for total null values
pima.isna().sum() 
pima.dtypes
# update data dictionary
pima_dictionary2 = pima_dictionary.copy()
pima_dictionary2['Data Type'] = pima.dtypes
pima_dictionary2
# All functions that are used in this notebook

# function for countplot
def bplot_perc(col, data, title, xlabel, ax=None, hue=None):
    sns.set(font_scale=1.3)
    sns.countplot(col, data=data, ax=ax, hue=hue)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    total = len(pima)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2, height + 4, '{:1.2f}%'.format((height/total)*100), ha="center", size=12) 
    plt.tight_layout()   
    plt.show()

    
def value_count_plot(data, col, sort, ax, xlabel, title):
    data[col].value_counts(sort).plot.bar(ax=ax)
    ax.set_xlabel(xlabel, labelpad = 10)
    ax.set_xticklabels(data[col].unique(), rotation = 0)
    ax.set_title(title)

    
# function for distribution plot
def dplot(col, target, data, title, xlabel, x=None, ax=None, hue=None):
    sns.set(font_scale=1.1)
    sns.distplot(data[data[target]==data[target].unique()[0]][col], ax=ax, color= 'blue', label=data[target].unique()[0])
    sns.distplot(data[pima[target]==data[target].unique()[1]][col], ax=ax, color='green', label=data[target].unique()[1])
    ax.set_title(title, size=20)
    ax.set_xlabel(xlabel, size=14)
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()   
    plt.show()  

    
# function for relational plot (between features)
def rel_plots(x, data, cols, ax, hue, pos, loc):
    ax = ax.ravel()
    for i in range(len(cols)):
        sns.pointplot(x, cols[i], hue=hue, data=data, ax=ax[i])
        if i != 1:         
            ax[i].legend_.remove()
        if i == 1:
            ax[i].legend(loc=loc, bbox_to_anchor=pos)
    plt.tight_layout()
    plt.show()
    
    
# function to detect outliers
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)  
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers   


# function for model quality prediction
def model_quality(model):
    for i in range(len(a)):
        model.fit(a[i], b[i])
        accuracy_model = round(model.score(c[i], d[i]) *100,2)
        print ("\n\n", '≡'*18, "Model for Pima",i+1, '≡'*18, "\n")
        print('accuracy_model :',accuracy_model, "\n")
        auc = roc_auc_score(d[i], model.predict(c[i]))
        print ("AUC = %2.2f" % auc)
        print (classification_report(d[i], model.predict(c[i])))
        
        
# function for model accuracy prediction
def accuracy(models, a, b, c, d):
    pi1 = []
    for i in range(len(models)):
        models[i].fit(a,b)
        accs = round(models[i].score(c, d) *100,2)
        pi1.append(accs)
    return pi1


# function for classification report
def clf_report(model,m,n,p,q):
    model.fit(m, n)
    true_label = q
    pred_label = model.predict(p)
    clf_rep = list(metrics.precision_recall_fscore_support(true_label, pred_label, average='weighted'))[:-1]
    precision.append(clf_rep[0])
    recall.append(clf_rep[1])
    F1_score.append(clf_rep[2])
    

# function for Area Under Curve
def auc(X_train, y_train, X_test, y_test):
    auc_scores = []
    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        auc_scores.append(roc_auc_score(y_test, models[i].predict(X_test)))
    return (auc_scores)


# function to plot Area Under Curve
def auc_plots(models, a, b, c, d, label):
    for i in range(len(models)):
        models[i].fit(a, b)
        fpr, tpr, thresholds = roc_curve(d, models[i].predict_proba(c)[:,1])
        gaussian1_auc = roc_auc_score(d, models[i].predict(c))
        ax.set_title('ROC Graph- '+ label)
        plt.plot([0,1], [0,1])
        ax.plot(fpr, tpr, label=label + ': ' + str(models_name[i])+ ' (area = %0.2f)' % gaussian1_auc)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        
        ax.legend(loc="lower right", prop={'size': 10})
    return plt.show()

# function for cross-validation
def cross_validation(X,y):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    kfold = KFold(n_splits=10, shuffle=False, random_state=0) # k=10, split the data into 10 equal parts
    xyz=[] 
    accuracy_cv=[]
    classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest', 'Extra Tree']
    models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),
        DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100), ExtraTreesClassifier(n_estimators=10)]
    for i in models:
        model = i
        cv_result = cross_val_score(model, X, y, cv=kfold, scoring = "accuracy")
        cv_result=cv_result
        xyz.append(cv_result.mean())
        accuracy_cv.append(cv_result)

    new_models_dataframe2 = pd.DataFrame({'CV Mean':xyz},index=classifiers) 
    new_models_dataframe2.loc['avg'] = new_models_dataframe2.mean()
    return new_models_dataframe2
    return accuracy_cv


# function to plot confusion matrix
def confusion_matrix_plot(X, y):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    f,ax = plt.subplots(2,4, figsize=(16,7), gridspec_kw={'wspace': 0.25, 'hspace': 0.4})
    ax = ax.ravel()
    for i in range(len(models)):
        y_pred = cross_val_predict(models[i], X, y, cv=10)
        sns.heatmap(confusion_matrix(y1,y_pred),ax=ax[i],annot=True,fmt='2.0f')
        ax[i].set_title('Matrix for '+ model_names[i])
# plot for target variable
f,ax=plt.subplots(figsize = (10,6))
bplot_perc(col='test', data=pima, title='negatif vs positif', xlabel='Test', ax=ax)
pf.ProfileReport(pima)
# select features with <15 unique elements as categories 
pima.nunique()
# update data dictionary about feature type
c, o = 'continuous', 'ordinal'
d = {'pregnant':c, 'glucose':c, 'diastolic':c, 'triceps':c, 'insulin':c, 'bmi':c,
       'diabetes':c, 'age':c, 'test':o}
pima_dictionary2['Feature Info'] = pd.Series(d)
pima_dictionary2
pd.crosstab(pima['test'], pima['pregnant']).style.background_gradient(cmap='summer_r')
sns.set(font_scale=1.6) 
f,ax=plt.subplots(1, 2, figsize=(14,5))
pima.groupby('test').agg({'pregnant':'mean'}).plot.bar(ax=ax[0])
ax[0].set_title('Pregnancy vs Test', size=18)
ax[0].set_ylabel('Pregnancy rate', size=14)
ax[0].set_xlabel('Test', size=14)
xticklabels = ['negatif', 'positif']
ax[0].set_xticklabels(xticklabels, rotation = 0)

sns.countplot('pregnant',hue='test',data=pima,ax=ax[1])
ax[1].set_title('Pregnancy vs Test', size=18)
ax[1].set_xlabel('Pregnancy', size=14)
ax[1].legend(loc='upper right')
plt.tight_layout()   
plt.show()
#f.savefig('test_rotation.png', dpi=300, format='png', bbox_inches='tight')
#plt.style.use('classic')
# feature engineering_pregnencies
pima['pregnency_group'] = pd.cut(pima['pregnant'],[0, 6, 12, 18], include_lowest=True, labels=['1', '2', '3'])
pd.crosstab(pima['pregnency_group'], pima['test']).style.background_gradient(cmap='summer_r')
def percConvert(ser):
    return ser/float(ser[-1])
#pd.crosstab(pima['pregnency_group'], pima['test'],margins=True).apply(percConvert, axis=1)
pd.crosstab(pima['pregnency_group'], pima['test'], normalize=True, margins=True).round(2)
sns.set(font_scale=1.5) 
f,ax=plt.subplots(1,2,figsize=(14,5))
value_count_plot(data=pima, col='pregnency_group', sort=False, ax=ax[0], xlabel='Pregnency group', title='Pregnency group frequency')
bplot_perc(col='pregnency_group', data=pima, title='Pregnency group vs Test', xlabel='Pregnency group', ax=ax[1], hue='test')
# exploring pregnant vs non-pregnant groups
cols= ['glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'diabetes', 'age', 'test']
filt= pima['pregnant']==0
s = pima.loc[filt, cols].mean()
s1 = pima.loc[~filt, cols].mean()
s2 = pima[pima['pregnant']>6][cols].mean()
pd.DataFrame(data = {'Never Pregnant':s, 'Pregnant':s1,
                     'Pregnant >6':s2 }).round(2).T.style.background_gradient(cmap='summer_r')
print(pima.loc[filt, 'test'].value_counts())
pima.loc[~filt, 'test'].value_counts()
print(f"Highest glucose level:{pima['glucose'].max()}")
print(f"Lowest glucose level:{pima['glucose'].min()}")
print(f"Average glucose level:{round(pima['glucose'].mean(),2)}")

f,ax=plt.subplots(figsize=(10,5))
dplot(col='glucose', target='test', data=pima, title='negatif vs positif',
      x = list(range(50,201,20)), ax=ax, xlabel='Glucose level')
# feature engineering_glucose
pima['glucose_group']=pd.cut(pima['glucose'],[50,139.99,199.99,250],
                            include_lowest=False, labels=['normal', 'prediabetes', 'diabetes'])
pd.crosstab(pima['glucose_group'], pima['test']).style.background_gradient(cmap='summer_r')
sns.set(font_scale=1.5) 
f,ax=plt.subplots(1,2,figsize=(14,4))
value_count_plot(data=pima, col='glucose_group', sort=False, ax=ax[0], 
                 xlabel='Glucose group', title='Glucose group frequency')
sns.countplot('glucose_group',hue='test',data=pima,ax=ax[1])
ax[1].set_title('Glucose group vs Test', size=18)
ax[1].set_xlabel('Glucose group', size=14)
ax[1].legend(loc='upper right')
plt.tight_layout()   
plt.show()
# feature engineering_glucose
pima['glucose_group_N']=pd.cut(pima['glucose'],[50,80,110,140,170,200],
                            include_lowest=False, labels=['1', '2', '3', '4', '5'])

pd.crosstab(pima['test'], pima['glucose_group_N']).style.background_gradient(cmap='summer_r')
sns.set(font_scale=1.3) 
f,ax=plt.subplots(1,2,figsize=(14,5))
value_count_plot(data=pima, col='glucose_group_N', ax=ax[0], sort=True, 
                 xlabel='Glucose group_N', title='Glucose group_N frequency')
bplot_perc(col='glucose_group_N', data=pima, title='Glucose group vs Test', 
           xlabel='Glucose group_N', hue='test', ax=ax[1])
sns.set(font_scale=1.5) 
f,ax = plt.subplots(2,2, figsize=(16,8), gridspec_kw={'wspace': 0.25, 'hspace': 0.4})
cols = ['age', 'bmi', 'triceps', 'pregnant']
rel_plots(x = 'glucose_group_N', cols = cols, data=pima, ax=ax, hue='test', loc=0, pos=(1,1))
pima.groupby(['test','glucose_group_N']).agg({'diastolic': 'mean', 'triceps': 'mean',
'bmi':'mean', 'age':'mean'}).round(2).style.background_gradient(cmap='summer_r')
print(f"Highest diastolic count:{pima['diastolic'].max()}")
print(f"Lowest diastolic count:{pima['diastolic'].min()}")
print(f"Average diastolic count:{round(pima['diastolic'].mean(),2)}")
pima.groupby('test').agg({'diastolic':['mean','median', 'std']}).round(2).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(14,5))
color = ['red', 'green']
for i in range(len(pima['test'].unique())):
    sns.distplot(pima[pima['test']==pima['test'].unique()[i]]['diastolic'],ax=ax[i], color= color[i])
    ax[i].set_title(pima['test'].unique()[i], size=20)
    ax[i].set_xlabel('Diastolic level', size=20)
    x1=list(range(20,120,20))
    ax[i].set_xticks(x1)
# feature engineering_diastolic
pima['diastolic_group'] = pd.cut(pima['diastolic'],[20,79.99,89.99,115],
                            include_lowest=False, labels=['normal', 'prehypertension', 'high'])
pd.crosstab(pima['diastolic_group'], pima['test'], margins=False).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(14,5))
value_count_plot(data=pima, col='diastolic_group', ax=ax[0], sort=False, 
                 xlabel='Diastolic group', title='Diastolic group frequency')
bplot_perc(col='diastolic_group', data=pima, title='Diastolic group vs Test', 
           xlabel='Diastolic group', ax=ax[1], hue='test')
f,ax=plt.subplots(1,3,figsize=(17,4), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
cols = ['pregnant', 'glucose', 'triceps']
rel_plots(x ='diastolic_group', cols = cols, data=pima, ax=ax, hue='test', loc=3, pos=(1.9,1))
pima.groupby(['test','diastolic_group']).agg({'glucose':'mean','triceps': 'mean', 
'bmi':'mean','age':'mean'}).round(2).style.background_gradient(cmap='summer_r')
print(f"Highest triceps count:{pima['triceps'].max()}")
print(f"Lowest triceps count:{pima['triceps'].min()}")
print(f"Average triceps count:{round(pima['triceps'].mean(),2)}")
pima.pivot_table(index='glucose_group_N', columns='test', values='triceps', 
    aggfunc='mean').astype('int').style.background_gradient(cmap='summer_r')
sns.set(font_scale=1.1) 
f,ax=plt.subplots(figsize=(10,5))
dplot(col='triceps', target='test', data=pima, title='negatif vs positif', x = list(range(5,65,10)), ax=ax, xlabel='Triceps')
# feature engineering_triceps
pima['triceps_group']=pd.qcut(pima['triceps'],4,labels=['1', '2', '3', '4'])                            
print(pima['triceps_group'].value_counts())
pd.crosstab(pima['triceps_group'], pima['test'], margins=True).style.background_gradient(cmap='summer_r')
sns.set(font_scale=1.4) 
f,ax=plt.subplots(1,2,figsize=(14,5))
value_count_plot(data=pima, col='triceps_group', ax=ax[0], sort=True,
                 xlabel='Triceps group', title='Triceps group frequency')
bplot_perc(col='triceps_group', data=pima, title='Triceps group vs Test', xlabel='Triceps group', ax=ax[1], hue='test')
f,ax=plt.subplots(1,3,figsize=(17,4), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
cols = ['glucose', 'diabetes', 'diastolic']
rel_plots(x = 'triceps_group', cols = cols, data=pima, ax=ax, hue='test', loc=3, pos=(1.9,1))
pima.groupby(['test','triceps_group']).agg({'glucose':'mean','diastolic': 'mean', 'insulin': 'mean',
'bmi':'mean', 'diabetes':'mean'}).round(2).style.background_gradient(cmap='summer_r')
print(f"Highest insulin count:{pima['insulin'].max()}")
print(f"Lowest insulin count:{pima['insulin'].min()}")
print(f"Average insulin count:{round(pima['insulin'].mean(),2)}")
pima.groupby('test').agg({'insulin':['mean','median']}).round(2).style.background_gradient(cmap='summer_r')
pima.pivot_table(index='test', columns='triceps_group', values='insulin', 
    aggfunc='mean').round(2).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(figsize=(10,5))
dplot(col='insulin', target='test', data=pima, title='negatif vs positif', x = list(range(10,850,100)), ax=ax, xlabel='Insulin')
pima['insulin_group'] = pd.qcut(pima['insulin'], 4, labels=['1', '2', '3', '4'])                            
print(pima['insulin_group'].value_counts(sort=False))
pd.crosstab(pima['insulin_group'], pima['test'], margins=True).style.background_gradient(cmap='summer_r')
sns.set(font_scale=1.4)
f,ax=plt.subplots(1,2,figsize=(14,5))
value_count_plot(data=pima, col='insulin_group', sort=False, ax=ax[0], xlabel='Insulin group', title='Insulin group frequency')
bplot_perc(col='insulin_group', data=pima, title='Insulin group vs Test', xlabel='Insulin group', ax=ax[1], hue='test')
f,ax=plt.subplots(2,2,figsize=(14,8), gridspec_kw={'wspace': 0.25, 'hspace': 0.4})
cols = ['diastolic', 'glucose', 'triceps', 'bmi']
rel_plots(x = 'insulin_group', cols = cols, data=pima, ax=ax, hue='test', loc=0, pos=(1,1))
sns.set(font_scale=1.2)
f,ax=plt.subplots(1,2,figsize=(18,5), gridspec_kw={'wspace': 0.2, 'hspace': 0.28})
sns.countplot('insulin_group',hue='glucose_group_N',data=pima,ax=ax[0])
ax[0].set_title('Insulin vs Glucose group', size=18)
ax[0].legend(loc=0, bbox_to_anchor=(0.83,.53))
sns.countplot('insulin_group',hue='diastolic_group',data=pima,ax=ax[1])
ax[1].set_title('Insulin vs Diastolic group', size=18)
ax[1].legend(loc=0, bbox_to_anchor=(0.6,.736))
plt.tight_layout()
plt.show()
pima.groupby(['test','insulin_group']).agg({'glucose': 'mean', 'diastolic': 'mean', 'triceps': 'mean',
'insulin': 'mean', 'diabetes':'mean', 'age':'mean'}).round(2).style.background_gradient(cmap='summer_r')
print(f"Highest BMI:{pima['bmi'].max()}")
print(f"Lowest BMI:{pima['bmi'].min()}")
print(f"Average BMI:{round(pima['bmi'].mean(),2)}")
pima.groupby('test').agg({'bmi':['mean', 'std']}).round(2).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(figsize=(16,6))
pima[pima['test']=='negatif'].bmi.plot.hist(ax=ax,bins=20,edgecolor='black',color='blue', alpha=0.2)
pima[pima['test']=='positif'].bmi.plot.hist(ax=ax,color='green',bins=20,edgecolor='black', alpha=0.8)
ax.set_title('negatif vs positif', size=20)
ax.set_xlabel('BMI', size=20)
x1=list(range(15,70,10))
ax.set_xticks(x1)
plt.tight_layout()   
plt.show()
pima['bmi_group'] = pd.qcut(pima['bmi'], 4, labels=['1', '2', '3', '4'])                            
print(pima['bmi_group'].value_counts(sort=False))
pd.crosstab(pima['bmi_group'], pima['test'], normalize=True).round(2).style.background_gradient(cmap='summer_r')
sns.set(font_scale=1.2)
f,ax=plt.subplots(1,2,figsize=(14,5))
value_count_plot(data=pima, col='bmi_group', sort=False, ax=ax[0], 
                 xlabel='BMI group', title='BMI group frequency')
bplot_perc(col='bmi_group', data=pima, title='BMI group vs Test', xlabel='BMI group', ax=ax[1], hue='test')
f,ax=plt.subplots(1,2,figsize=(18,5), gridspec_kw={'wspace': 0.2, 'hspace': 0.5})
sns.violinplot('bmi_group', 'glucose',hue='test', split=True, inner='quartile',data=pima, ax=ax[0])
ax[0].set_title('BMI vs Glucose', size=18)
ax[0].legend_.remove()
sns.boxplot('bmi_group', 'age',hue='test',data=pima, ax=ax[1])
ax[1].set_title('BMI vs Age', size=18)
ax[1].legend(loc=0)
plt.tight_layout()
plt.show()
f,ax=plt.subplots(2,2,figsize=(20,10), gridspec_kw={'wspace': 0.15, 'hspace': 0.50})
sns.countplot('bmi_group',hue='insulin_group',data=pima,ax=ax[0,0])
ax[0,0].set_title('BMI vs Insulin group', size=22)
ax[0,0].legend(loc=1, bbox_to_anchor=(.93,1.02))

sns.countplot('bmi_group',hue='glucose_group_N',data=pima,ax=ax[0,1])
ax[0,1].set_title('BMI vs Glucose group', size=22)
ax[0,1].legend(loc=1, bbox_to_anchor=(1.03,1.04))

sns.countplot('bmi_group',hue='diastolic_group',data=pima,ax=ax[1,0])
ax[1,0].set_title('BMI vs Diastolic group', size=22)
ax[1,0].legend(loc=0, bbox_to_anchor=(1.02,1.04))

sns.countplot('bmi_group',hue='triceps_group',data=pima,ax=ax[1,1])
ax[1,1].set_title('BMI vs Tricep group', size=22)
ax[1,1].legend(loc=1, bbox_to_anchor=(0.93,1.04))

plt.tight_layout()
plt.show()
pima.groupby(['test','bmi_group']).agg({'glucose': 'mean', 'diastolic': 'mean', 'triceps': 'mean',
'insulin': 'mean', 'diabetes':'mean', 'age':'mean'}).round(2).style.background_gradient(cmap='summer_r')
print(f"Highest diabetes count:{pima['diabetes'].max()}")
print(f"Lowest diabetes count:{pima['diabetes'].min()}")
print(f"Average diabetes count:{round(pima['diabetes'].mean(),2)}")
pima.groupby('test').agg({'diabetes':'mean'}).round(2).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(figsize=(10,5))
dplot(col='diabetes', target='test', data=pima, title='negatif vs positif', x = list(range(0,4,1)), ax=ax, xlabel='Diabetes')
pima['diabetes_group'] = pd.qcut(pima['diabetes'], 3, labels=['1', '2', '3'])                            
print(pima['diabetes_group'].value_counts(sort=False))
pd.crosstab(pima['diabetes_group'], pima['test'], normalize=True).round(2).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(14,6))
value_count_plot(data=pima, col='diabetes_group', sort=False, ax=ax[0], 
                 xlabel='Diabetes group', title='Diabetes group frequency')
bplot_perc(col='diabetes_group', data=pima, title='Diabetes group vs Test', 
           xlabel='Diabetes group', ax=ax[1], hue='test')
f,ax=plt.subplots(1,3,figsize=(16,4), gridspec_kw={'wspace': 0.45, 'hspace': 0.45})
cols = ['glucose', 'triceps', 'age']
rel_plots(x = 'diabetes_group', cols = cols, data=pima, ax=ax, hue='test', loc=3, pos=(2,1))
f,ax=plt.subplots(2,2,figsize=(22,10), gridspec_kw={'wspace': 0.35, 'hspace': 0.50})
sns.countplot('diabetes_group',hue='insulin_group',data=pima,ax=ax[0,0])
ax[0,0].set_title('Diabetes vs Insulin group', size=24)
ax[0,0].legend(loc=0, bbox_to_anchor=(1.02,1.04))
sns.countplot('diabetes_group',hue='glucose_group_N',data=pima,ax=ax[0,1])
ax[0,1].set_title('Diabetes vs Glucose group', size=24)
ax[0,1].legend(loc=1, bbox_to_anchor=(1.16,1.04))
sns.countplot('diabetes_group',hue='bmi_group',data=pima,ax=ax[1,0])
ax[1,0].set_title('Diabetes vs BMI group', size=24)
ax[1,0].legend(loc=0, bbox_to_anchor=(1.02,1.04))
sns.countplot('diabetes_group',hue='triceps_group',data=pima,ax=ax[1,1])
ax[1,1].set_title('Diabetes vs Tricep group', size=24)
ax[1,1].legend(loc=0, bbox_to_anchor=(1.02,1.04))
plt.tight_layout()
plt.show()
pima.groupby(['test','diabetes_group']).agg({'glucose': 'mean', 'diastolic': 'mean', 'triceps': 'mean',
'insulin': 'mean', 'bmi':'mean', 'age':'mean'}).round(2).style.background_gradient(cmap='summer_r')
print(f"Olderst individual:{pima['age'].max()}")
print(f"Youngest individual:{pima['age'].min()}")
print(f"Average age:{round(pima['age'].mean(),2)}")
pima.groupby('test').agg({'age':'mean'}).round(2).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(16,6))
color = ['red', 'green']
for i in range(pima['test'].nunique()):
    pima[pima['test']==pima['test'].unique()[i]]['age'].plot.hist(ax=ax[i],bins=20,edgecolor='black',color=color[i], alpha=0.6)
    ax[i].set_title(pima['test'].unique()[i], size=20)
    ax[i].set_xlabel('Age', size=20)
    x1=list(range(20,85,10))
    ax[i].set_xticks(x1)
pima['age_group']=pd.qcut(pima['age'], 3, labels=['1', '2', '3'])                            
print(pima['age_group'].value_counts(sort=False))
pd.crosstab(pima['age_group'], pima['test'], normalize=True).round(2).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(13,5))
value_count_plot(data=pima, col='age_group', ax=ax[0], sort=False,
                 xlabel='Age group', title='Age group frequency')
bplot_perc(col='age_group', data=pima, title='Age group vs Test', xlabel='Age group', ax=ax[1], hue='test')
f,ax=plt.subplots(1,3,figsize=(16,4), gridspec_kw={'wspace': 0.3, 'hspace': 0.45})
cols = ['glucose', 'diastolic', 'insulin']
rel_plots(x = 'age_group', cols = cols, data=pima, ax=ax, hue='test', loc=2, pos=(1.85,1.25))
f,ax=plt.subplots(2,2,figsize=(22,12), gridspec_kw={'wspace': 0.28, 'hspace': 0.40})
sns.countplot('age_group',hue='insulin_group',data=pima,ax=ax[0,0])
ax[0,0].set_title('BMI vs Insulin group', size=24)
ax[0,0].legend(loc=0, bbox_to_anchor=(1.15,1.04))

sns.countplot('age_group',hue='glucose_group_N',data=pima,ax=ax[0,1])
ax[0,1].set_title('BMI vs Glucose group', size=24)
ax[0,1].legend(loc=0, bbox_to_anchor=(1.15,1.04))

sns.countplot('age_group',hue='diastolic_group',data=pima,ax=ax[1,0])
ax[1,0].set_title('BMI vs Diastolic group', size=24)
ax[1,0].legend(loc=1, bbox_to_anchor=(1.02,1.02))

sns.countplot('age_group',hue='diabetes_group',data=pima,ax=ax[1,1])
ax[1,1].set_title('BMI vs Diastolic group', size=24)
ax[1,1].legend(loc=0, bbox_to_anchor=(1.01,1.02))

plt.tight_layout()
plt.show()
pima.groupby(['test','age_group']).agg({'glucose': 'mean', 'diastolic': 'mean', 'triceps': 'mean',
'insulin': 'mean', 'bmi':'mean', 'diabetes':'mean'}).round(2).style.background_gradient(cmap='summer_r')
sns.pairplot(pima, diag_kind='kde', 
             vars=['age', 'pregnant', 'glucose', 'diabetes', 'insulin', 'triceps', 'diastolic', 'bmi'],
            hue = 'test', plot_kws = {'s':14}, size=1.7)
# conversion of target to numeric value will allow correlating with other features
pima_x = pima
pima_x['test_n']=pima_x['test'].map({'positif':1, 'negatif':0})

sns.set(font_scale=1.3)
g=sns.clustermap(pima_x.corr(),annot=True, cmap='viridis', linewidths=0.2,annot_kws={'size':14}) 
fig=plt.gcf()
fig.set_size_inches(14,12)
plt.xticks(fontsize=14)
plt.show()
fig, axes = plt.subplots(2,4, figsize = (14,8), sharex=False, sharey=False)
axes = axes.ravel()
cols = ['age', 'pregnant', 'glucose', 'diabetes', 'insulin', 'triceps', 'diastolic', 'bmi']
for i in range(len(cols)):
    sns.boxplot(y=cols[i],data=pima, ax=axes[i])
plt.tight_layout()
filt = (pima['triceps']>60) & (pima['diabetes']>2.4) & (pima['bmi']>55)
pima[filt].index
# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(pima,2,['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'diabetes', 'age'])
pima.loc[Outliers_to_drop]
pima['glucose_group'].replace(['normal', 'prediabetes', 'diabetes'], [0,1,2], inplace=True)
pima['diastolic_group'].replace(['normal', 'prehypertension', 'high'], [0,1,2], inplace=True)
pima1 = pima.drop(['pregnency_group', 'glucose_group',
       'glucose_group_N', 'diastolic_group', 'triceps_group', 'insulin_group',
       'bmi_group', 'diabetes_group', 'age_group'],axis=1)
pima2 = pima.drop(['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'diabetes', 'age'],axis=1)
pima2.dtypes
# convert categorical columns into numericals
cols = ['pregnency_group', 'glucose_group_N', 'triceps_group', 'insulin_group',
       'bmi_group', 'diabetes_group', 'age_group']
pima2[cols] = pima2[cols].apply(pd.to_numeric,errors='ignore')
pima2.dtypes
sns.heatmap(pima1.corr(),annot=True,cmap='viridis',linewidths=0.2,annot_kws={'size':15})
fig=plt.gcf()
fig.set_size_inches(16,12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
sns.heatmap(pima2.corr(),annot=True,cmap='viridis',linewidths=0.2,annot_kws={'size':15})
fig=plt.gcf()
fig.set_size_inches(16,12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
# glucose_group is dropped as glucose_group_N has higher correlation with target
pima2 = pima2.drop(columns='glucose_group')
pima2.columns
pima3 = pima[['pregnency_group','glucose', 'diastolic',
              'triceps','insulin_group', 'bmi', 'diabetes', 'age_group','test_n']]
pima2 = pd.get_dummies(pima2, columns = ["pregnency_group"])
pima2 = pd.get_dummies(pima2, columns = ["glucose_group_N"])
pima2 = pd.get_dummies(pima2, columns = ["diastolic_group"])
pima2 = pd.get_dummies(pima2, columns = ["triceps_group"])
pima2 = pd.get_dummies(pima2, columns = ["insulin_group"])
pima2 = pd.get_dummies(pima2, columns = ["bmi_group"])
pima2 = pd.get_dummies(pima2, columns = ["diabetes_group"])
pima2 = pd.get_dummies(pima2, columns = ["age_group"])
pima3 = pd.get_dummies(pima3, columns = ["pregnency_group"])
pima3 = pd.get_dummies(pima3, columns = ["insulin_group"])
pima3 = pd.get_dummies(pima3, columns = ["age_group"])
pima1 = pima1.drop(columns='test')
pima2 = pima2.drop(columns='test')
pima1.shape, pima2.shape, pima3.shape
pima1.columns
pima2.columns
pima3.columns
# importing all the required ML packages
from sklearn.model_selection import train_test_split # training and testing data split
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn import svm # support vector Machine
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier # K-NN
from sklearn.naive_bayes import GaussianNB # naive bayes
from sklearn.tree import DecisionTreeClassifier # decision Tree
from sklearn.ensemble import RandomForestClassifier # random Forest
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics #accuracy measure
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix # confusion matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold # for K-fold cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score # score evaluation
from sklearn.model_selection import cross_val_predict # prediction
# Models
gaussian = GaussianNB() # Gaussian Naive Bayes
logreg = LogisticRegression(random_state=0) # Logistic Regression
svc = SVC(random_state=0) # Support Vector classifier
decisiontree = DecisionTreeClassifier(random_state=0) # Decision Tree
randomforest = RandomForestClassifier(n_estimators=10, random_state=0) # Random Forest
extraTrees = ExtraTreesClassifier(n_estimators=10, random_state=0) # Extra Tree
knn = KNeighborsClassifier(n_neighbors = 5) # KNN or k-Nearest Neighbors
gbk = GradientBoostingClassifier(random_state=0) # Gradient Boosting Classifier

# pima1 split
X1 = pima1.drop(columns='test_n')
y1 = pima1['test_n']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.22, random_state = 0)

# pima2(only engineered features) split
X2 = pima2.drop(columns='test_n')
y2 = pima2['test_n']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.22, random_state = 0)

# Mix of original and engineered data (pima3) split
X3 = pima3.drop(columns='test_n')
y3 = pima3['test_n']
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size = 0.22, random_state = 0)

a = [X_train1, X_train2, X_train3]
b = [y_train1, y_train2, y_train3]
c = [X_test1, X_test2, X_test3]
d = [y_test1, y_test2, y_test3]

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

print(X1.shape, y1.shape)
print(X2.shape, y2.shape)
print(X3.shape, y3.shape)
# Automating model quality prediction
model_names = ['Gaussian', 'LogisticRegression', 'SVC', 'DecisionTree', 'RandomForest', 'ExtraTrees', 'KNN', 'GradientBoosting']
models = [gaussian, logreg, svc, decisiontree, randomforest, extraTrees, knn, gbk]

for i in range(len(models)):
    print('■'*22, '\033[1m' + colored(model_names[i], 'green'), '■'*23)
    model_quality(models[i])
    print ("\n\n")
# Accuracy comparison between different datasets
pima_accuracy = {}
for i in range(len(a)):
    pimai_accuracy = accuracy(models, a=a[i], b=b[i], c=c[i], d=d[i])
    pima_accuracy['pima'+str(i+1)+'_accuracy'] = pimai_accuracy
    
accuracy_models = pd.DataFrame({
    'Model': model_names, 'pima1_Acc': pima_accuracy['pima1_accuracy'], 'pima2_Acc': pima_accuracy['pima2_accuracy'], 
    'pima3_Acc': pima_accuracy['pima3_accuracy']})
accuracy_models['Acc_mean'] = accuracy_models.mean(axis=1).round(2)
accuracy_models.set_index('Model', inplace=True)
accuracy_models.loc['avg'] = accuracy_models.mean()
accuracy_models
# Classification report comparison between different datasets

# PIMA_1
precision = []
recall = []
F1_score = []
for i in range(len(models)):
    clf_report(models[i],X_train1, y_train1, X_test1, y_test1)
    
dict_1 = {'Model':model_names, 'F1_score': F1_score, 'Precision': precision, 'Recall': recall}           
df_pima1 = pd.DataFrame(dict_1, columns=['Model', 'Precision', 'Recall', 'F1_score'])
df_pima1.set_index('Model', inplace=True)
df_pima1.loc['avg'] = df_pima1.mean()
print('▬'*20, '\033[1m' + colored('Pima_1', 'green'), '▬'*20)
print(df_pima1[['Precision', 'Recall', 'F1_score']], "\n\n")

# PIMA_2
precision = []
recall = []
F1_score = []
for i in range(len(models)):
    clf_report(models[i],X_train2, y_train2, X_test2, y_test2)
    
dict_2 =  {'Model':model_names, 'F1_score': F1_score, 'Precision': precision, 'Recall': recall}           
df_pima2 = pd.DataFrame(dict_2, columns=['Model', 'Precision', 'Recall', 'F1_score'])
df_pima2.set_index('Model', inplace=True)
df_pima2.loc['avg'] = df_pima2.mean()
print('▬'*20, '\033[1m' + colored('Pima_2', 'green'), '▬'*20)
print(df_pima2[['Precision', 'Recall', 'F1_score']], "\n\n")

# PIMA_3
precision = []
recall = []
F1_score = []
for i in range(len(models)):
    clf_report(models[i],X_train3, y_train3, X_test3, y_test3)
    
dict_3 =  {'Model':model_names, 'F1_score': F1_score, 'Precision': precision, 'Recall': recall}             
df_pima3 = pd.DataFrame(dict_3, columns=['Model', 'Precision', 'Recall', 'F1_score'])
df_pima3.set_index('Model', inplace=True)
df_pima3.loc['avg'] = df_pima3.mean()
print('▬'*20, '\033[1m' + colored('Pima_3', 'green'), '▬'*20)
print(df_pima3[['Precision', 'Recall', 'F1_score']])
# AUC comparison between different datasets
pima_auc = {}
for i in range(len(a)):
    pimai_auc = auc(X_train=a[i], y_train=b[i], X_test=c[i], y_test=d[i])
    pima_auc['pima'+str(i+1)+'_auc'] = pimai_auc

auc_models = pd.DataFrame({
    'Model': model_names, 'pima_1': pima_auc['pima1_auc'], 'pima_2': pima_auc['pima2_auc'], 
    'pima_3': pima_auc['pima3_auc']})
auc_models.set_index('Model', inplace=True)
auc_models.loc['avg'] = auc_models.mean()
auc_models=auc_models.round(2)
auc_models
# AUC plots among best models based on accuracy, F1-score and AUC
pima1_models = [gaussian, logreg, randomforest]
pima2_models = [extraTrees, knn, gbk]
pima3_models = [gaussian, logreg, randomforest]

# Pima_1
pima1_models = [gaussian, logreg, randomforest]
models_name = ['Gaussian', 'LogisticRegression', 'RandomForest']
f,ax=plt.subplots(figsize=(8,5))
auc_plots(pima1_models, X_train1, y_train1, X_test1, y_test1, label='Pima 1')

# Pima_2
pima2_models = [extraTrees, knn, gbk]
models_name = ['ExtraTrees', 'KNN', 'GradientBoosting']
f,ax=plt.subplots(figsize=(8,5))
auc_plots(pima2_models, X_train2, y_train2, X_test2, y_test2, label='Pima 2')

# Pima_3
pima3_models = [gaussian, logreg, randomforest]
models_name = ['Gaussian', 'LogisticRegression', 'RandomForest']
f,ax=plt.subplots(figsize=(8,5))
auc_plots(pima3_models, X_train3, y_train3, X_test3, y_test3, label='Pima 3')
# Visualization of the gaussian classifier w.r.t the dataset 
from yellowbrick.classifier import ClassificationReport
classes = ['positif', 'negatif']
visualizer = ClassificationReport(gaussian, classes=classes, support=False, with_avg_total=True)
visualizer.fit(X_train3, y_train3)
visualizer.fit(X_train3, y_train3)  # Fit the visualizer and the model
visualizer.score(X_test3, y_test3)  # Evaluate the model on the test data
g = visualizer.poof() 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1_stndz = sc.fit_transform(X_train1)
X_test1_standz = sc.transform(X_test1)
X_train2_stndz = sc.fit_transform(X_train2)
X_test2_standz = sc.transform(X_test2)
X_train3_stndz = sc.fit_transform(X_train3)
X_test3_standz = sc.transform(X_test3)

a_stdz = [X_train1_stndz, X_train2_stndz, X_train3_stndz]
b_stdz = [y_train1, y_train2, y_train3]
c_stdz = [X_test1_standz, X_test2_standz, X_test3_standz]
d_stdz = [y_test1, y_test2, y_test3]
# Accuracy comparison between different standardized datasets
print(list(accuracy_models.iloc[-1,:]))

pima_stndz_accuracy = {}
for i in range(len(a_stdz)):
    pimai_accuracy = accuracy(models, a=a_stdz[i], b=b_stdz[i], c=c_stdz[i], d=d_stdz[i])
    pima_stndz_accuracy['pima'+str(i+1)+'_accuracy'] = pimai_accuracy
    
accuracy_stndz_models = pd.DataFrame({
    'Model': model_names, 'pima1_Acc': pima_stndz_accuracy['pima1_accuracy'], 'pima2_Acc': pima_stndz_accuracy['pima2_accuracy'], 
    'pima3_Acc': pima_stndz_accuracy['pima3_accuracy']})
accuracy_stndz_models['Acc_mean'] = accuracy_stndz_models.mean(axis=1).round(2)
accuracy_stndz_models.set_index('Model', inplace=True)
accuracy_stndz_models.loc['avg'] = accuracy_stndz_models.mean()
accuracy_stndz_models.round(2)
sns.set_style("darkgrid")
f,ax=plt.subplots(1,2,figsize=(16,4))
accuracy_models.boxplot(ax = ax[0])
ax[0].set_title('Accuracy comparison between original datasets')
accuracy_stndz_models.boxplot(ax = ax[1])
ax[1].set_title('Accuracy comparison between standarized_datasets')
# AUC comparison between different standardized datasets
print(list(auc_models.iloc[-1,:]))

pima_stndz_auc = {}
for i in range(len(a)):
    pimai_auc = auc(X_train=a_stdz[i], y_train=b_stdz[i], X_test=c_stdz[i], y_test=d_stdz[i])
    pima_stndz_auc['pima'+str(i+1)+'_auc'] = pimai_auc

auc_stndz_models = pd.DataFrame({
    'Model': model_names, 'pima1_stndz': pima_stndz_auc['pima1_auc'], 'pima2_stndz': pima_stndz_auc['pima2_auc'], 
    'pima3_stndz': pima_stndz_auc['pima3_auc']})
auc_stndz_models.set_index('Model', inplace=True)
auc_stndz_models.loc['avg'] = auc_stndz_models.mean()
auc_stndz_models=auc_stndz_models.round(2)
auc_stndz_models
f,ax=plt.subplots(1,2,figsize=(16,4))
auc_models.boxplot(ax = ax[0])
ax[0].set_title('AUC comparison between original datasets')

auc_stndz_models.boxplot(ax = ax[1])
ax[1].set_title('AUC comparison between standarized_datasets')
# Classification report comparison between standardized datasets

# PIMA_1
precision = []
recall = []
F1_score = []
for i in range(len(models)):
    clf_report(models[i], X_train1_stndz, y_train1, X_test1_standz, y_test1)
    
dict_1 = {'Model':model_names, 'F1_score': F1_score, 'Precision': precision, 'Recall': recall}           
df_pima1_stndz = pd.DataFrame(dict_1, columns=['Model', 'Precision', 'Recall', 'F1_score'])
df_pima1_stndz.set_index('Model', inplace=True)
df_pima1_stndz.loc['avg'] = df_pima1_stndz.mean()
print('▬'*20, '\033[1m' + colored('Pima_1', 'green'), '▬'*20)
print(df_pima1_stndz[['Precision', 'Recall', 'F1_score']], "\n\n")

# PIMA_2
precision = []
recall = []
F1_score = []
for i in range(len(models)):
    clf_report(models[i], X_train2_stndz, y_train2, X_test2_standz, y_test2)
    
dict_2 =  {'Model':model_names, 'F1_score': F1_score, 'Precision': precision, 'Recall': recall}           
df_pima2_stndz = pd.DataFrame(dict_2, columns=['Model', 'Precision', 'Recall', 'F1_score'])
df_pima2_stndz.set_index('Model', inplace=True)
df_pima2_stndz.loc['avg'] = df_pima2_stndz.mean()
print('▬'*20, '\033[1m' + colored('Pima_2', 'green'), '▬'*20)
print(df_pima2_stndz[['Precision', 'Recall', 'F1_score']], "\n\n")

# PIMA_3
precision = []
recall = []
F1_score = []
for i in range(len(models)):
    clf_report(models[i], X_train3_stndz, y_train3, X_test3_standz, y_test3)
    
dict_3 =  {'Model':model_names, 'F1_score': F1_score, 'Precision': precision, 'Recall': recall}             
df_pima3_stndz = pd.DataFrame(dict_3, columns=['Model', 'Precision', 'Recall', 'F1_score'])
df_pima3_stndz.set_index('Model', inplace=True)
df_pima3_stndz.loc['avg'] = df_pima3_stndz.mean()
print('▬'*20, '\033[1m' + colored('Pima_3', 'green'), '▬'*20)
print(df_pima3_stndz[['Precision', 'Recall', 'F1_score']], "\n\n")
print('▬'*50, "\n")

# original dataset
pima_1= pd.DataFrame({'dataset': ['pima1']*3, 'Model':df_pima1.idxmax(), 'Values':df_pima1.max()})
pima_2= pd.DataFrame({'dataset': ['pima2']*3, 'Model':df_pima2.idxmax(), 'Values':df_pima2.max()})
pima_3= pd.DataFrame({'dataset': ['pima3']*3, 'Model':df_pima3.idxmax(), 'Values':df_pima3.max()})
df_summary = pd.concat([pima_1, pima_2, pima_3])
df_summary.reset_index(inplace=True)
df_summary.set_index(['dataset', 'index'], inplace=True)
print('Classification report summary of the ORIGINAL dataset', "\n")
print(df_summary)#.T
print("\n",'▬'*50, "\n")

# standardized dataset
pima1_stndz= pd.DataFrame({'dataset': ['pima1']*3, 'Model':df_pima1_stndz.idxmax(), 'Values':df_pima1_stndz.max()})
pima2_stndz= pd.DataFrame({'dataset': ['pima2']*3, 'Model':df_pima2_stndz.idxmax(), 'Values':df_pima2_stndz.max()})
pima3_stndz= pd.DataFrame({'dataset': ['pima3']*3, 'Model':df_pima3_stndz.idxmax(), 'Values':df_pima3_stndz.max()})
df_summary_stndz = pd.concat([pima1_stndz, pima2_stndz, pima3_stndz])
df_summary_stndz.reset_index(inplace=True)
df_summary_stndz.set_index(['dataset', 'index'], inplace=True)
print('Classification report summary of the STANDARDIZED dataset', "\n")
print(df_summary_stndz) #.T
cv_1 = cross_validation(X1,y1)#.head(2)
cv_2 = cross_validation(X2,y2)
cv_3 = cross_validation(X3,y3)

cv_comp = pd.concat([cv_1, cv_2, cv_3], axis=1).round(2)
cv_comp.columns = ['Pima1_CV_mean', 'Pima2_CV_mean', 'Pima3_CV_mean']
cv_comp.sort_values(by='Pima1_CV_mean', ascending=False)
cv_comp.loc['avg'] =cv_comp.mean()
cv_comp.round(2)
# CV visulaization
f,ax=plt.subplots(1,3,figsize=(16,4))
for i in range(len(cv_comp.columns)):
    cv_comp.iloc[:,i].plot.bar(width=0.8, ax=ax[i])
    ax[i].set_title(cv_comp.columns[i])
    ax[i].set_xticklabels(cv_comp.index, rotation=80)
svc_r = svm.SVC(kernel='rbf',random_state=0)
svc_l = svm.SVC(kernel='linear',random_state=0)
model_names = ['Gaussian', 'Log-Regression', 'SVC_radial', 'SVC_linear', 'Decision-Tree', 'Random-Forest', 'ExtraTrees', 'KNN']
models = [gaussian, logreg, svc_r, svc_l, decisiontree, randomforest, extraTrees, knn]
      
# Pima 1
confusion_matrix_plot(X1, y1)
# Pima 2
confusion_matrix_plot(X2, y2)
# Pima 3
confusion_matrix_plot(X3, y3)
X1_stndz = sc.fit_transform(X1)
X2_stndz = sc.fit_transform(X2)
X3_stndz = sc.fit_transform(X3)
# Hyperparameter tuning_SVM
from sklearn.model_selection import GridSearchCV
kfold = KFold(n_splits=10, shuffle=False, random_state=0)
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd_SVC=GridSearchCV(estimator=svm.SVC(random_state=0),param_grid=hyper, cv=kfold, verbose=True)
gd_SVC.fit(X1,y1)
print(gd_SVC.best_score_)
print(gd_SVC.best_estimator_)
# Hyperparameter tuning_SVM
from sklearn.model_selection import GridSearchCV
kfold = KFold(n_splits=10, shuffle=False, random_state=0)
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
C = [0.1, 0.2, 0.3, 0.25, 0.4, 0.5]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5]
kernel = ['rbf', 'linear']
hyper={'kernel':kernel, 'C':C, 'gamma':gamma}

gd_SVC = GridSearchCV(estimator=svm.SVC(random_state=0),param_grid=hyper, cv=kfold, n_jobs= 4, verbose=True)
gd_SVC.fit(X1_stndz,y1)
print(gd_SVC.best_score_)
print(gd_SVC.best_estimator_)
# comparing confusion matrix before and after hyperparameter tuning
f,ax=plt.subplots(1,2,figsize=(14,5), gridspec_kw={'wspace': 0.5, 'hspace': 0.95})
y_pred = cross_val_predict(svm.SVC(kernel='linear',random_state=0),X1_stndz,y1,cv=kfold)
sns.heatmap(confusion_matrix(y1,y_pred),ax=ax[0],annot=True,fmt='2.0f')
ax[0].set_title('Confusion Matrix: linear-SVM before hyperparameter tuning', pad=15)

y_pred = cross_val_predict(svm.SVC(C=0.1, gamma=0.1, kernel='linear',random_state=0),X1_stndz,y1,cv=kfold)
sns.heatmap(confusion_matrix(y1,y_pred),annot=True,ax=ax[1],fmt='2.0f')
ax[1].set_title('Confusion Matrix: linear-SVM after hyperparameter tuning', pad=15)
plt.show()
y_pred = cross_val_predict(svm.SVC(kernel='linear',random_state=0),X1_stndz,y1,cv=kfold)
print('▬'*10, 'Classification Report BEFORE Hyper-Parameter Tuning', '▬'*10, "\n")
print(classification_report(y1, y_pred), "\n")
print( "\n", '▬'*10, 'Classification Report AFTER Hyper-Parameter Tuning', '▬'*10, "\n")
y_pred = cross_val_predict(svm.SVC(C=0.1, gamma=0.1, kernel='linear',random_state=0),X1_stndz,y1,cv=kfold)
print(classification_report(y1, y_pred))
# Hyperparameter tuning_RFC
leaf = np.arange(1, 5)
depths = np.arange(2, 5)
splits = np.arange(4, 7)
n_estimators=range(1,100,10)
hyper={'max_depth': depths, 
       'min_samples_split': splits,
       'min_samples_leaf': leaf,
       'n_estimators':n_estimators,
       }
gd_RFC = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=hyper, 
                cv=10, scoring="accuracy", n_jobs= 4, verbose=True)
gd_RFC.fit(X1_stndz,y1)
print(gd_RFC.best_score_)
print(gd_RFC.best_estimator_)
# comparing confusion matrix before and after hyperparameter tuning
f,ax=plt.subplots(1,2,figsize=(14,5), gridspec_kw={'wspace': 0.5, 'hspace': 0.95})
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100,random_state=0),X1_stndz,y1,cv=kfold)
sns.heatmap(confusion_matrix(y1,y_pred),ax=ax[0],annot=True,fmt='2.0f')
ax[0].set_title('Confusion Matrix: RFC before hyperparameter tuning', pad=15)

y_pred = cross_val_predict(RandomForestClassifier(max_depth=4,min_samples_leaf=2, min_samples_split=5,n_estimators=71,random_state=0),X1_stndz,y1,cv=10)
sns.heatmap(confusion_matrix(y1,y_pred),annot=True,ax=ax[1],fmt='2.0f')
ax[1].set_title('Confusion Matrix: RFC after hyperparameter tuning', pad=15)
plt.show()
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100,random_state=0),X1_stndz,y1,cv=kfold)
print('▬'*10, 'Classification Report BEFORE Hyper-Parameter Tuning', '▬'*10, "\n")
print(classification_report(y1, y_pred))
print( "\n", '▬'*10, 'Classification Report AFTER Hyper-Parameter Tuning', '▬'*10, "\n")
y_pred = cross_val_predict(RandomForestClassifier(max_depth=4,min_samples_leaf=2, min_samples_split=5,n_estimators=71,random_state=0),X1_stndz,y1,cv=10)
print(classification_report(y1, y_pred))
from sklearn.ensemble import BaggingClassifier

# Decision Tree
model=BaggingClassifier(base_estimator = DecisionTreeClassifier(),random_state=0,n_estimators=200)
model.fit(X_train1_stndz,y_train1)
y_pred=model.predict(X_test1_standz)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(y_pred,y_test1))
result=cross_val_score(model,X1_stndz,y1,cv=kfold,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())
print ('-'*75)

# Logistic Regression
model=BaggingClassifier(base_estimator = LogisticRegression(),random_state=0,n_estimators=1000)
model.fit(X_train1_stndz,y_train1)
y_pred=model.predict(X_test1_standz)
print('The accuracy for bagged Logistic Regression is:',metrics.accuracy_score(y_pred,y_test1))
result=cross_val_score(model,X1_stndz,y1,cv=kfold,scoring='accuracy')
print('The cross validated score for bagged Logistic Regression is:',result.mean())
print ('-'*75)

# K-NN
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors=9),random_state=0,n_estimators=700)
model.fit(X_train1_stndz,y_train1)
y_pred=model.predict(X_test1_standz)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(y_pred,y_test1))
result=cross_val_score(model,X1_stndz,y1,cv=kfold,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())
print ('-'*75)

# linear SVC
model=BaggingClassifier(base_estimator = SVC(C=0.1, gamma=0.1, kernel='linear',random_state=0),random_state=0,n_estimators=100)
model.fit(X_train1_stndz,y_train1)
y_pred=model.predict(X_test1_standz)
print('The accuracy for bagged Linear SVC is:',metrics.accuracy_score(y_pred,y_test1))
result=cross_val_score(model,X1_stndz,y1,cv=kfold,scoring='accuracy')
print('The cross validated score for bagged Linear SVC is:',result.mean())
print ('-'*75)

# Naive Bayes
model=BaggingClassifier(base_estimator = GaussianNB(),random_state=0,n_estimators=1000)
model.fit(X_train1_stndz,y_train1)
y_pred=model.predict(X_test1_standz)
print('The accuracy for bagged Gaussian Naive Bayes is:',metrics.accuracy_score(y_pred,y_test1))
result=cross_val_score(model,X1_stndz,y1,cv=kfold,scoring='accuracy')
print('The cross validated score for bagged Gaussian Naive Bayes is:',result.mean())
# Adaboost tuning
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=0)
n_estimators = range(1,100,10)
learning_rate = [ 0.1, 0.2, 0.5, 1, 1.4, 1.5]
ada_param_grid = { "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" : n_estimators,
              "learning_rate":  learning_rate}
gd_ada= GridSearchCV(ada, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose=1)
gd_ada.fit(X_train1_stndz, y_train1)
print(gd_ada.best_estimator_)
print(gd_ada.best_score_)
print('▬'*60)

y_pred = gd_ada.predict(X_test1_standz)
print(classification_report(y_test1, y_pred))
# Random Forest and SVC already done

#ExtraTrees tuning
ExtC = ExtraTreesClassifier()
leaf = np.arange(1, 6)
depths = np.arange(5, 9)
splits = np.arange(6, 12)
n_estimators=range(1,50,10)
## Search grid for optimal parameters
ex_param_grid = {"max_depth": depths,
              "min_samples_split": splits,
              "min_samples_leaf": leaf,
              "n_estimators" :n_estimators}
gd_ExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gd_ExtC.fit(X_train1_stndz,y_train1)
print(gd_ExtC.best_estimator_)
print(gd_ExtC.best_score_)
print('▬'*60)

y_pred = gd_ExtC.predict(X_test1_standz)
print(classification_report(y_test1, y_pred))
# Gradient boosting tunning
GBC = GradientBoostingClassifier()
leaf = np.arange(1, 4)
depths = np.arange(20, 21, 23)
learning_rate = [0.04, 0.05, 0.06, 0.07]

## Search grid for optimal parameters
param_grid = {'loss' : ['deviance'],
              'n_estimators' : [100, 200, 300, 400],
              'learning_rate': learning_rate,
              'max_depth': depths,
              'min_samples_leaf': leaf,
              'max_features': [0.4, 0.5, 0.6]
              }
gd_GBC  = GridSearchCV(GBC,param_grid = param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gd_GBC.fit(X_train1_stndz,y_train1)
print(gd_GBC .best_estimator_)
print(gd_GBC .best_score_)
print('▬'*60)

y_pred = gd_GBC.predict(X_test1_standz)
print(classification_report(y_test1, y_pred))
# XGBoost tunning
import xgboost as xg
xgboost=xg.XGBClassifier()
depths = np.arange(1,5)
learning_rate = [0.0001, 0.001]
## Search grid for optimal parameters
param_grid = {'gamma':[0.00001, 0.0001, 0.001, 0.01],
              'n_estimators' : [300, 400, 500],
              'learning_rate': learning_rate,
              'max_depth': depths}           
gd = GridSearchCV(xgboost, param_grid = param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gd.fit(X_train1_stndz,y_train1)
print(gd.best_estimator_)
print(gd.best_score_)
print('▬'*60)

y_pred = gd.predict(X_test1_standz)
print(classification_report(y_test1, y_pred))
gd_models = [gd_RFC, gd_ExtC, gd_SVC, gd_ada, gd_GBC, logreg, gaussian, decisiontree, knn]
gd_models_name = ["RFC", "ExtC", "SVC", "Ada", "GBC", "LR", "NB", "DTC", "KNN"]
series = []
for i in range(len(gd_models)):
    gd_models[i].fit(X1_stndz, y1)
    series.append(pd.Series(gd_models[i].predict(X_test1_standz),name=gd_models_name[i]))
    
# Concatenate all classifier results
a = pd.concat(series, axis=1)
g = sns.heatmap(a.corr(),annot=True,cmap='viridis',linewidths=0.2,annot_kws={'size':12})
fig=plt.gcf()
fig.set_size_inches(14,10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[
                                            #('SVC',svm.SVC(probability=True,kernel='rbf',C=0.3,gamma=0.1)),
                                            #('RFC',RandomForestClassifier(max_depth=4,min_samples_leaf=2, min_samples_split=8,n_estimators=11,random_state=0)),
                                            ('LR',LogisticRegression(C=0.05)),
                                            #('DT',DecisionTreeClassifier(random_state=0)),
                                            #('NB',GaussianNB()),
                                            #('Ada',AdaBoostClassifier(n_estimators=200,learning_rate=0.05, random_state=0)),
                                            #('Ext',ExtraTreesClassifier( max_depth=7,min_samples_leaf=2, min_samples_split=11,n_estimators=21)),
                                            ('GBC',GradientBoostingClassifier(learning_rate=0.03, max_depth=21,max_features=0.5,min_samples_leaf=2,n_estimators=300)),
                                            #('Xg',xg.XGBClassifier(gamma=1e-05, learning_rate=0.0001, n_estimators=400)),
                                             ], voting='soft').fit(X_train1_stndz,y_train1)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(X_test1_standz, y_test1))
cross = cross_val_score(ensemble_lin_rbf, X1_stndz, y1, cv = kfold, scoring = "accuracy", n_jobs= 4, verbose=True)
print('The cross validated score is', cross.mean())
print('▬'*60)
y_pred = cross_val_predict(ensemble_lin_rbf,X_test1_standz,y_test1, cv = kfold, n_jobs= 4, verbose=True)
print(classification_report(y_test1, y_pred))
# Finding most important features
print(X1.columns)
models = [RandomForestClassifier(max_depth=4,min_samples_leaf=2, min_samples_split=5,n_estimators=91,random_state=0),
         AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.2, n_estimators=41, random_state=0),
         GradientBoostingClassifier(learning_rate=0.04, max_depth=20, max_features=0.4, min_samples_leaf=2, n_estimators=400),
         xg.XGBClassifier(gamma=1e-05, learning_rate=0.001, n_estimators=300)]
model_name = ['RandomForest', 'AdaBoost', 'Gradient Boosting', 'XgBoost']

f,ax=plt.subplots(2,2,figsize=(14,10), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
ax = ax.ravel()
for i in range(len(models)):
    #models[i].fit(X_train1_stndz, y_train1)
    models[i].fit(X1_stndz, y1)
    pd.Series(models[i].feature_importances_, X1.columns).sort_values(ascending=True).\
                                                            plot.barh(width=0.8,ax=ax[i])
    ax[i].set_title('Feature Importance in ' + model_name[i])