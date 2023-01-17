import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, RepeatedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#from sklearn.cross_validation import cross_val_score

from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC

from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.metrics import classification_report_imbalanced

plt.rcParams['font.family']='Avenir'
plt.rcParams['font.weight']='bold'
pd.set_option('display.max_columns', None)  
# Import Data

train = pd.read_csv("../input/data-mining-assignment-2/train.csv")
test = pd.read_csv("../input/data-mining-assignment-2/test.csv")

print("Shape of Train: ", train.shape)
print("Shape of Test: ", test.shape)

train.head()
# Make x_train, x_test, y_test by dropping certain columns

x_train = train.drop(columns = ['Class', 'ID'])
y_train = train['Class']
x_test = test.drop(columns = ['ID'])
id_list = test['ID']

# Make list of categorical columns and num_columns
cat_columns = ['col2', 'col11', 'col37', 'col44', 'col56']
num_columns = list(set(x_train.columns) - set(cat_columns))

# Which categorical columns are nominal and which are ordinal
x_train[cat_columns].head()
# One-Hot encode col11, col37, col44

x_train = pd.get_dummies(x_train, columns=['col11', 'col37', 'col44'], drop_first=True)
x_test = pd.get_dummies(x_test, columns=['col11', 'col37', 'col44'], drop_first=True)

# Ordinally Encode col2 and col56 in both train and test

x_train.col2 = x_train.col2.map({'Silver': 1, 'Gold': 2, 'Platinum':3, 'Diamond':4})
x_train.col56 = x_train.col56.map({'Low': 1, 'Medium': 2, 'High':3})
x_test.col2 = x_test.col2.map({'Silver': 1, 'Gold': 2, 'Platinum':3, 'Diamond':4})
x_test.col56 = x_test.col56.map({'Low': 1, 'Medium': 2, 'High':3})

x_train.head()
# Create a list of all columns
cols_after_encoding = x_train.columns

names_cat_cols_after_encoding = ['col2', 'col56', 'col11_Yes', 'col37_Male', 'col44_Yes']

# Shapes After Encoding
print("Shapes After Encoding")
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

# Concatenate df to check corr
x_train_corr_check = pd.concat([x_train, y_train], axis=1)

# Calculate corr
corr_df = pd.DataFrame(x_train_corr_check.corr()['Class'][:]).sort_values(by='Class', ascending=False).iloc[1:, :].reset_index()
corr_df.columns=['feature', 'corr_w_class']

# Df with corr > 0.24 with class
corr_024 = corr_df[(corr_df['corr_w_class']>=0.25)|(corr_df['corr_w_class']<=-0.24)].sort_values(by='corr_w_class', ascending=False).reset_index(drop=True)
#corr_025.columns=['feature', 'corr_w_class']

print("Number of features w/ correlation to 'Class' >= 0.24 or <=0.24: ", corr_024.shape[0])
print(corr_024.shape)
corr_024.tail()

# Get these features as a list
corr_024_features = corr_024['feature'].values.tolist()

corr_024_features[0]
# Plot corr_df

plt.figure(figsize=(16, 6))
plt.title('Feature Correlation with Class Label', size=20, weight="bold", pad=20)

plt.grid('slategray', axis='y', linestyle='dashed');

plt.bar(x=corr_df['feature'],
        height=corr_df['corr_w_class'],
        color='gold')

plt.xticks(corr_df['feature'].values.tolist(),
           rotation=90, size=11);


plt.yticks(np.arange(-0.3, 0.4, 0.05), 
           size=11);

plt.xlabel('Feature Labels', size=15, weight="bold", labelpad=15);
plt.ylabel('Feature Correlation with Class Label', size=15, weight="bold", labelpad=15);

#plt.savefig('figure/feature_corr.png', dpi=300, bbox_inches = 'tight')

plt.show()

x_train = x_train.drop(columns = corr_024_features)
x_test = x_test.drop(columns = corr_024_features)
x_train.shape

#Random Forest - for a base model
x_train_subset, x_test_subset, y_train_subset, y_test_subset = train_test_split(x_train, y_train, 
                                                                                test_size=0.2, 
                                                                                )

print("From the data for which labels are availabale, the following datasets are generated: ")
print("x_train_subset: ", x_train_subset.shape)
print("y_train_subset: ", y_train_subset.shape)
print("x_test_subset: ", x_test_subset.shape)
print("y_test_subset: ", y_test_subset.shape)

# Scaling
sc = StandardScaler()
x_train_subset = sc.fit_transform(x_train_subset)
x_test_subset = sc.transform(x_test_subset)

clf_rf = RandomForestClassifier(max_depth = 11)

clf_rf.fit(x_train_subset, y_train_subset)
y_test_subset_predicted = clf_rf.predict(x_test_subset)


print("\nResults on Train/Validation: \n")

print(confusion_matrix(y_test_subset, y_test_subset_predicted))
print(classification_report_imbalanced(y_test_subset, y_test_subset_predicted))

# f1_micro
print("F1 Micro: ", f1_score(y_test_subset, y_test_subset_predicted, average='micro'))
# Scaling
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

clf_rf = RandomForestClassifier(n_estimators=500,
                                max_depth = 11,
                                min_samples_split=3)

clf_rf.fit(x_train_scaled, y_train)
y_predicted_RF = clf_rf.predict(x_test_scaled)

y_predicted_RF = pd.concat((id_list, pd.DataFrame(y_predicted_RF, 
                                               columns=['Class'])), 
                        axis=1)

y_predicted_RF.head()


from IPython.display import HTML 
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html) 

create_download_link(y_predicted_RF)
