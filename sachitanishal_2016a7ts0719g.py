import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt





from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, classification_report

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

# Import Data



train_benign = pd.read_csv('../input/dm-assignment-3/train_benign.csv', index_col=0)

train_malware = pd.read_csv('../input/dm-assignment-3/train_malware.csv', index_col=0)

test = pd.read_csv('../input/dm-assignment-3/Test_data.csv', index_col=0)





print("Shape of Train Benign: ", train_benign.shape)

print("Shape of Train Malware: ", train_malware.shape)

print("Shape of Test: ", test.shape)



test.head()
# Make x_train, y_train, x_test, y_test 



train_benign['label'] = [0]*train_benign.shape[0]

train_malware['label'] = [1]*train_malware.shape[0]



x_train = pd.concat([train_benign,

                     train_malware]).reset_index(drop=True)

y_train = x_train['label']

x_train = x_train.drop(columns = ['label'])

x_test = test.drop(columns = ['Unnamed: 1809']).reset_index(drop=True)

# Chack for NaN and ? in Numerical Data in x_train and x_test



print("Testing for NaN")

# Check NA

for col in x_test.columns:

    if x_test[col].isna().sum() > 0:

        print(col, x_test[col].isna().sum())

        

print("Testing for ?")        

if x_test.isin(['?']).any().any():

    print('? exists in Dataframe')
# Check if all values are int type for x_test and x_train

for row in x_test.dtypes:

    if(row!='int64'):

        print(row)
x_train.shape
# Function to identify non unique columns in training data

def non_unique_cols(df):

    a = df.to_numpy() 

    return (a[0] == a).all(0)

# List of same value columns

same_value_cols_train = x_train.columns[non_unique_cols(x_train)].tolist()

same_value_cols_test = x_test.columns[non_unique_cols(x_test)].tolist()



same_value_cols_all = same_value_cols_train + same_value_cols_test

same_value_cols_all = list(set(same_value_cols_all))



# Drop them from test and train

x_train = x_train.drop(columns = same_value_cols_all)

x_test = x_test.drop(columns = same_value_cols_all)



# Shapes after dropping same value columns

print("Shapes after dropping same value columns")

print("x_train: ", x_train.shape)

print("x_test: ", x_test.shape)
print("Same Value cols dropped: ", len(same_value_cols_all))
# Concatenate df to check corr

x_train_corr_check = pd.concat([x_train, y_train], axis=1)



# Calculate corr

corr_df = pd.DataFrame(x_train_corr_check.corr()['label'][:]).sort_values(by='label', 

                                                                          ascending=False).iloc[1:, :].reset_index()

corr_df.columns=['feature', 'corr_w_class']

# Check for NaN in corr_df

for col in corr_df.columns:

    if corr_df[col].isna().sum() > 0:

        print(col, corr_df[col].isna().sum())
# Plot corr_df



plt.figure(figsize=(16, 6))

plt.title('Feature Correlation with Class Label', size=20, weight="bold", pad=20)



plt.grid('slategray', axis='y', linestyle='dashed');



plt.bar(x=corr_df['feature'],

        height=corr_df['corr_w_class'],

        color='gold')



plt.xticks([]);



plt.xlabel('Feature Labels', size=15, weight="bold", labelpad=15);

plt.ylabel('Feature Correlation with Class Label', size=15, weight="bold", labelpad=15);





#plt.savefig('figs/feature_corr.png', dpi=300, bbox_inches = 'tight')



plt.show()
# Plot histogram of feature correlations

# the histogram of the data

plt.figure(figsize=(12, 8))





n, bins, patches = plt.hist(x=corr_df['corr_w_class'], 

                            bins=np.arange(-0.5, 0.8, 0.05),

                            facecolor='g', 

                            alpha=0.75)



plt.xticks(bins);



plt.xlabel('Correlation Bin', size=15, weight="bold", labelpad=15)

plt.ylabel('Number of Features', size=15, weight="bold", labelpad=15)

plt.title('Histogram of Feature Correlations with Label', size=20, weight="bold", pad=20)

plt.grid(True)



#plt.savefig('figs/feature_corr_hist.png', dpi=300, bbox_inches = 'tight')



plt.show()
# Df with corr > 0.25 with class

corr_025 = corr_df[(corr_df['corr_w_class']>=0.25)|(corr_df['corr_w_class']<=-0.25)].sort_values(by='corr_w_class', 

                                                                                                 ascending=False).reset_index(drop=True)



print("Number of features w/ correlation to 'label' >= 0.25 or <=0.25: ", corr_025.shape[0])

corr_025.tail()
# Get these features as a list

corr_025_features = corr_025['feature'].values.tolist()



# Keep only the high correlation features

x_train = x_train[corr_025_features]

x_test = x_test[corr_025_features]



# Shapes after dropping low correlation features

print("Shapes after dropping low correlation features")

print("x_train: ", x_train.shape)

print("x_test: ", x_test.shape)
# Make IQR Ranges



Q1 = x_train.quantile(0.25)

Q3 = x_train.quantile(0.75)

IQR = Q3 - Q1

outliers_true_iqr_df = (x_train < (Q1 - 1.5 * IQR)) | (x_train > (Q3 + 1.5 * IQR))
outliers_true_iqr_df.head()
# Count of outliers

outliers_true_iqr_df['num_outliers'] = outliers_true_iqr_df.sum(axis=1)
# Plot histogram of feature correlations

# the histogram of the data

plt.figure(figsize=(12, 8))





n, bins, patches = plt.hist(x=outliers_true_iqr_df['num_outliers'], 

                            bins=np.arange(0, 330, 10),

                            cumulative=False,

                            facecolor='dodgerblue', 

                            alpha=0.75)



plt.xticks(bins);



plt.xlabel('Outlier Values Range', size=15, weight="bold", labelpad=15)

plt.ylabel('Number of Training Samples', size=15, weight="bold", labelpad=15)

plt.title('Distribution of Number of Outlier Attributes in a Training Sample', size=20, weight="bold", pad=20)



plt.grid(True)

#plt.savefig('figs/outlier_cols_per_row.png', dpi=300, bbox_inches = 'tight')







plt.show()
# Experiemnt with various thresholds of outliers and look at resulting sample size

outliers_true_iqr_df['to_drop'] = outliers_true_iqr_df['num_outliers'].apply(lambda x: 1 if x >=90 else 0)

outliers_true_iqr_df['to_drop'].value_counts()
# Drop rows with outliers above threshold

x_train_y_train = pd.concat([x_train, 

                             y_train], axis=1)



x_train_y_train = pd.concat([x_train_y_train,

                            outliers_true_iqr_df['to_drop']], 

                            axis=1)



# Get names of indexes for which column value is True

index_names = x_train_y_train[ x_train_y_train['to_drop']==1].index



# Delete these row indexes from dataFrame

x_train_y_train.drop(index_names , inplace=True)



# Now drop the 'to_drop' column

x_train_y_train.drop('to_drop' , inplace=True, axis=1)



# Re-assign x_train and y_train

x_train = x_train_y_train.iloc[:, :-1]

y_train = x_train_y_train.iloc[:, -1]



# Shapes after dropping outlier rows

print("Shapes after dropping outlier rows")

print("x_train: ", x_train.shape)

print("y_train: ", y_train.shape)

print("x_test: ", x_test.shape)
# Sanity Check



# Shapes after dropping low correlation features

print("Most recent df shapes:")

print("x_train: ", x_train.shape, "| y_train: ", y_train.shape)

print("x_test: ", x_test.shape)

# TTS

x_train_subset, x_test_subset, y_train_subset, y_test_subset = train_test_split(x_train, y_train, 

                                                                                test_size=0.25, 

                                                                                random_state=12)



print("From the data for which labels are availabale, the following datasets are generated: ")

print("x_train_subset: ", x_train_subset.shape)

print("y_train_subset: ", y_train_subset.shape)

print("x_test_subset: ", x_test_subset.shape)

print("y_test_subset: ", y_test_subset.shape)





# Scaling

sc = RobustScaler()

x_train_subset = sc.fit_transform(x_train_subset)

x_test_subset = sc.transform(x_test_subset)

y_train_subset.value_counts()
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    #plt.savefig('figs/roc_auc_rf.png', dpi=300, bbox_inches = 'tight')

    plt.legend()

    plt.show()
# Evaluate the tuned rf on train data

rf_tuned = RandomForestClassifier(oob_score=True,

                                  bootstrap = True,

                                  max_depth= 70,

                                  max_features= 'auto',

                                  min_samples_leaf= 1,

                                  min_samples_split= 2,

                                  n_estimators= 2000)



rf_tuned.fit(x_train_subset, y_train_subset)

y_pred_subset_RF_tuned = rf_tuned.predict(x_test_subset)



print(confusion_matrix(y_test_subset, y_pred_subset_RF_tuned))

print(classification_report(y_test_subset, y_pred_subset_RF_tuned))



print("OOB Score: ", rf_tuned.oob_score_)



# ROC AUC Stuff



# Predict probabilities

probs_rf_tuned = rf_tuned.predict_proba(x_test_subset)



# Keep Probabilities of the positive class only.

probs_rf_tuned = probs_rf_tuned[:, 1]



# Compute the AUC Score.

auc = roc_auc_score(y_test_subset, probs_rf_tuned)

print('AUC: %.4f' % auc)



# Get the ROC Curve.

fpr, tpr, thresholds = roc_curve(y_test_subset, probs_rf_tuned)



# Plot ROC Curve using our defined function

plot_roc_curve(fpr, tpr)
# Scaling

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)

x_test_scaled = sc.transform(x_test)

# Random Forest

rf_final = RandomForestClassifier(oob_score=True,

                                  bootstrap = True,

                                  max_depth= 70,

                                  max_features= 'auto',

                                  min_samples_leaf= 1,

                                  min_samples_split= 2,

                                  n_estimators= 2000)



rf_final.fit(x_train_scaled, y_train)

y_pred_RF = rf_final.predict(x_test_scaled)
rf_final.oob_score_

# Get indices

y_pred_indices = np.arange(1, x_test.shape[0]+1)



# Convert to df

y_pred_df = pd.DataFrame(y_pred_RF, columns=['Class'])

y_pred_df['FileName'] = y_pred_indices

y_pred_df = y_pred_df[['FileName', 'Class']]



y_pred_df.head()
from IPython.display import HTML 

import base64

import pandas as pd

import numpy as np



def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    

    return HTML(html) 



create_download_link(y_pred_df)