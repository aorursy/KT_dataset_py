from IPython.display import HTML
import pandas as pd
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import collections
from lightgbm import LGBMClassifier, plot_importance
import seaborn as snss
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 200})
%matplotlib inline
train_application_df = pd.read_csv('../input/application_train.csv')
test_application_df = pd.read_csv('../input/application_test.csv')
# print(train_application_df.shape, test_application_df.shape)
all_application_df = pd.concat([train_application_df, test_application_df], axis=0)
# print(all_application_df.shape)
target_distribution = train_application_df['TARGET'].value_counts()
target_distribution.plot.pie(figsize=(10, 10),
                             title='Target Distribution',
                             fontsize=15, 
                             legend=True, 
                             autopct=lambda v: "{:0.1f}%".format(v))
total_nans = all_application_df.isna().sum()
nan_precents = (all_application_df.isna().sum()/all_application_df.isna().count()*100)
feature_overview_df  = pd.concat([total_nans, nan_precents], axis=1, keys=['NaN Count', 'NaN Pencent'])
feature_overview_df['Type'] = [all_application_df[c].dtype for c in feature_overview_df.index]
pd.set_option('display.max_rows', None)
display(feature_overview_df)
pd.set_option('display.max_rows', 20)
all_application_is_nan_df = pd.DataFrame()
for column in all_application_df.columns:
    if all_application_df[column].isna().sum() == 0:
        continue
    all_application_is_nan_df['is_nan_'+column] = all_application_df[column].isna()
    all_application_is_nan_df['is_nan_'+column] = all_application_is_nan_df['is_nan_'+column].map(lambda v: 1 if v else 0)
all_application_is_nan_df['target'] = all_application_df['TARGET']
all_application_is_nan_df = all_application_is_nan_df[pd.notna(all_application_is_nan_df['target'])]
display(all_application_is_nan_df)
Y = all_application_is_nan_df.pop('target')
X = all_application_is_nan_df

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2, random_state=2018)
clf = LGBMClassifier(n_estimators=200, learning_rate=0.01)
clf.fit(
        train_X,
        train_Y,
        eval_set=[(train_X, train_Y), (valid_X, valid_Y)],
        eval_metric='auc',
        early_stopping_rounds=50,
        verbose=False
       )
plot_importance(clf, figsize=(10,10))
# add noise to y axis to avoid overlapping
def rand_jitter(arr):
    nosie = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr))

def draw_feature_distribution(df, column):
    column_values = df[df[column].notna()][column]
    # group by target
    class_0_values = df[df[column].notna() & (df['TARGET']==0)][column]
    class_1_values = df[df[column].notna() & (df['TARGET']==1)][column]
    class_t_values = df[df[column].notna() & (df['TARGET'].isna())][column]        
    print('\n\n', column)
    # for features with unique values >= 10
    if len(df[column].value_counts().keys()) >= 10:
        fig, ax = plt.subplots(1, figsize=(15, 4))
        if df[column].dtype == 'object':
            label_encoder = LabelEncoder()
            label_encoder.fit(column_values)
            class_0_values = label_encoder.transform(class_0_values)
            class_1_values = label_encoder.transform(class_1_values)
            class_t_values = label_encoder.transform(class_t_values)
            column_values = label_encoder.transform(column_values)
            plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, fontsize=12, rotation='vertical')

        ax.scatter(class_0_values, rand_jitter([0]*class_0_values.shape[0]), label='Class0', s=10, marker='o', color='#7ac143', alpha=1)
        ax.scatter(class_1_values, rand_jitter([10]*class_1_values.shape[0]), label='Class1', s=10, marker='o', color='#fd5c63', alpha=1)
        ax.scatter(class_t_values, rand_jitter([20]*class_t_values.shape[0]), label='Test', s=10, marker='o', color='#037ef3', alpha=0.4)
        ax.set_title(column +' group by target', fontsize=16)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(column +' distribution', fontsize=16)
    else:      
        all_categories = list(df[df[column].notna()][column].value_counts().keys())
        bar_width = 0.25
        
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.set_title(column, fontsize=16)
        plt.xlabel('Categories', fontsize=16)
        plt.ylabel('Counts', fontsize=16)

        value_counts = class_0_values.value_counts()
        x_0 = np.arange(len(all_categories))
        y_0 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_0, y_0, color='#7ac143', width=bar_width, label='class0')

        value_counts = class_1_values.value_counts()
        x_1 = np.arange(len(all_categories))
        y_1 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_1+bar_width, y_1, color='#fd5c63', width=bar_width, label='class1')
        
        value_counts = class_t_values.value_counts()
        x_2 = np.arange(len(all_categories))
        y_2 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_2+2*bar_width, y_2, color='#037ef3', width=bar_width, label='test')
        
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        
        for i, v in enumerate(y_0):
            if y_0[i]+y_1[i] == 0:
                ax.text(i - .08, max(y_0)//1.25,  'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i - .08, max(y_0)//1.25,  "{:0.1f}%".format(100*y_0[i]/(y_0[i]+y_1[i])), fontsize=14, rotation='vertical')
        
        for i, v in enumerate(y_1):
            if y_0[i]+y_1[i] == 0:
                ax.text(i - .08, max(y_0)//1.25,  'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i + bar_width - .08, max(y_0)//1.25, "{:0.1f}%".format(100*y_1[i]/(y_0[i]+y_1[i])), fontsize=14, rotation='vertical')
 
        for i, v in enumerate(y_2):
            if y_2[i] == 0:
                ax.text(i + 2*bar_width - .08, max(y_0)//1.25, 'Missing in Test', fontsize=14, rotation='vertical')
            else:
                ax.text(i + 2*bar_width - .08, max(y_0)//1.25, str(y_2[i]), fontsize=14, rotation='vertical')
        
        plt.xticks(x_0 + 2*bar_width/3, all_categories, fontsize=16)
        
    plt.show()
print("only showing the distribution for the first few columns, edit the counter to show all distribution")
show_feature_count = 10
for column in all_application_df.columns:
    if show_feature_count == 0:
        break
    show_feature_count -= 1
    draw_feature_distribution(all_application_df, column)
draw_feature_distribution(all_application_df, 'DAYS_EMPLOYED')
# the organizer used 365243 to represent missing value in this column
temp_df = all_application_df[all_application_df['DAYS_EMPLOYED'] != 365243]
draw_feature_distribution(temp_df, 'DAYS_EMPLOYED')
print(all_application_df['AMT_INCOME_TOTAL'].describe())
draw_feature_distribution(all_application_df, 'AMT_INCOME_TOTAL')
temp_df = all_application_df[all_application_df['AMT_INCOME_TOTAL'] != 1.170000e+08]
draw_feature_distribution(temp_df, 'AMT_INCOME_TOTAL')
print(all_application_df['AMT_REQ_CREDIT_BUREAU_QRT'].describe())
draw_feature_distribution(all_application_df, 'AMT_REQ_CREDIT_BUREAU_QRT')
temp_df = all_application_df[all_application_df['AMT_REQ_CREDIT_BUREAU_QRT'] != 261]
draw_feature_distribution(temp_df, 'AMT_REQ_CREDIT_BUREAU_QRT')
draw_feature_distribution(all_application_df, 'NONLIVINGAPARTMENTS_MODE')
draw_feature_distribution(all_application_df, 'OBS_30_CNT_SOCIAL_CIRCLE')
